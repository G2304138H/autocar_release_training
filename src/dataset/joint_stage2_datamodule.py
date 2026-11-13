"""Joint RCA/LCA training with artery-specific geometry and patient-safe splits."""
from __future__ import annotations

from bisect import bisect_right

from torch.utils.data import ConcatDataset

from src.dataset.case_splits import apply_training_case_exclusions, load_case_splits
from src.dataset.stage2_datamodule import Stage2NPZDataModule


def align_patient_splits(sources, policy="heldout_priority"):
    """Keep each physical patient in one split across all artery sources.

    heldout_priority promotes conflicting assignments: test > val > train.
    strict rejects conflicts, useful when retaining published cohorts exactly.
    """
    if policy not in ("heldout_priority", "strict"):
        raise ValueError("split_policy must be heldout_priority or strict")
    rank = {"train": 0, "val": 1, "test": 2}
    assignments = {}
    for splits in sources.values():
        for split, ids in splits.items():
            for case_id in ids:
                previous = assignments.get(case_id, split)
                if previous != split and policy == "strict":
                    raise ValueError(f"Cross-artery patient leakage: {case_id}: {previous}/{split}")
                assignments[case_id] = max((previous, split), key=rank.get)
    result = {}
    for artery, splits in sources.items():
        result[artery] = {name: [] for name in rank}
        for ids in splits.values():
            for case_id in ids:
                result[artery][assignments[case_id]].append(case_id)
        for name, ids in result[artery].items():
            if not ids:
                raise ValueError(f"{artery}/{name} is empty after patient split alignment")
    return result


class ArteryConcatDataset(ConcatDataset):
    """One sample per artery/case, preserving native geometry and volume shape."""

    def __init__(self, datasets, arteries):
        super().__init__(datasets)
        self.arteries = tuple(arteries)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source = bisect_right(self.cumulative_sizes, index)
        sample = dict(super().__getitem__(index))
        sample["artery"] = self.arteries[source]
        sample["patient_id"] = sample["case_id"]
        sample["case_id"] = f"{self.arteries[source]}:{sample['case_id']}"
        return sample

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


class JointStage2NPZDataModule(Stage2NPZDataModule):
    """Shuffle both arteries through a shared model, once per sample per epoch."""

    def __init__(self, sources, split_policy="heldout_priority", batch_size=1,
                 num_workers=0, random_seed=42, min_train_views=2,
                 max_train_views=2, minimum_train_pair_angle_deg=30.0,
                 pin_memory=True):
        super().__init__(
            projection_source="", voxel_source="", split_json="",
            batch_size=batch_size, num_workers=num_workers,
            random_seed=random_seed, min_train_views=min_train_views,
            max_train_views=max_train_views,
            minimum_train_pair_angle_deg=minimum_train_pair_angle_deg,
            pin_memory=pin_memory,
        )
        self.save_hyperparameters(logger=False)
        if set(sources) != {"rca", "lca"}:
            raise ValueError("sources must contain exactly rca and lca")
        self.sources = dict(sources)
        self.split_policy = split_policy
        self.modules = {}
        self.effective_splits = {}

    def setup(self, stage=None):
        filtered = {}
        self.modules = {}
        for artery, config in self.sources.items():
            options = dict(config)
            options.pop("_target_", None)
            options.update(
                batch_size=self.batch_size, num_workers=self.num_workers,
                random_seed=self.random_seed, min_train_views=self.min_train_views,
                max_train_views=self.max_train_views,
                minimum_train_pair_angle_deg=self.minimum_train_pair_angle_deg,
                pin_memory=self.pin_memory,
            )
            module = Stage2NPZDataModule(**options)
            if module.case_id_mode != "imagecas_numeric":
                raise ValueError("Joint ImageCAS sources require imagecas_numeric IDs")
            splits = load_case_splits(module.split_json, case_id_mode=module.case_id_mode)
            filtered[artery], removed, absent = apply_training_case_exclusions(
                splits, module.excluded_train_case_ids, case_id_mode=module.case_id_mode,
            )
            module.applied_train_case_exclusions = removed
            module.unmatched_train_case_exclusions = absent
            self.modules[artery] = module
        self.effective_splits = align_patient_splits(filtered, self.split_policy)
        for artery, module in self.modules.items():
            module._build_datasets(self.effective_splits[artery], stage)
        for attribute in ("data_train", "data_val", "data_test"):
            datasets = [getattr(module, attribute) for module in self.modules.values()]
            if all(dataset is not None for dataset in datasets):
                setattr(self, attribute, ArteryConcatDataset(datasets, self.modules))
