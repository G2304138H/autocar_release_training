"""Lightning data module for case-level Stage-2 NPZ splits."""

from __future__ import annotations

from typing import Any, Sequence

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.dataset.case_splits import load_case_splits
from src.dataset.stage2_npz import Stage2NPZDataset


class Stage2NPZDataModule(LightningDataModule):
    """Create deterministic train, validation, and test NPZ loaders.

    Training samples a reproducible pair from all available views. Validation
    and test use one predeclared pair so the primary AutoCAR result is not an
    implicitly cherry-picked multi-pair ensemble.
    """

    def __init__(
        self,
        projection_source: str | Sequence[str],
        voxel_source: str | Sequence[str],
        split_json: str,
        *,
        evaluation_view_indices: Sequence[int] = (0, 6),
        evaluation_view_labels: Sequence[str] | None = (
            "RAO 25, CAU 35",
            "LAO 5, CRA 40",
        ),
        batch_size: int = 1,
        num_workers: int = 0,
        random_seed: int = 42,
        minimum_train_pair_angle_deg: float = 30.0,
        case_id_mode: str = "literal",
        expected_imager_pixel_spacing_mm: float | None = None,
        fallback_imager_pixel_spacing_mm: float | None = None,
        fallback_sid_mm: float | None = None,
        source_to_isocenter_mm: float = 750.0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError(
                "Stage2NPZDataModule currently requires batch_size=1 because "
                "native GT volumes may have different shapes."
            )
        if int(num_workers) < 0:
            raise ValueError("num_workers cannot be negative.")
        evaluation_view_indices = tuple(int(v) for v in evaluation_view_indices)
        if len(evaluation_view_indices) != 2 or len(set(evaluation_view_indices)) != 2:
            raise ValueError("evaluation_view_indices must contain two distinct views.")
        if evaluation_view_labels is None:
            normalised_view_labels = None
        else:
            normalised_view_labels = tuple(
                str(value).strip() for value in evaluation_view_labels
            )
            if len(normalised_view_labels) != 2 or any(
                not value for value in normalised_view_labels
            ):
                raise ValueError(
                    "evaluation_view_labels must be None or contain two "
                    "non-empty labels."
                )
        self.save_hyperparameters(logger=False)
        self.projection_source = projection_source
        self.voxel_source = voxel_source
        self.split_json = split_json
        self.evaluation_view_indices = evaluation_view_indices
        self.evaluation_view_labels = normalised_view_labels
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.random_seed = int(random_seed)
        self.minimum_train_pair_angle_deg = float(
            minimum_train_pair_angle_deg
        )
        self.case_id_mode = str(case_id_mode)
        self.expected_imager_pixel_spacing_mm = (
            None
            if expected_imager_pixel_spacing_mm is None
            else float(expected_imager_pixel_spacing_mm)
        )
        self.fallback_imager_pixel_spacing_mm = (
            None
            if fallback_imager_pixel_spacing_mm is None
            else float(fallback_imager_pixel_spacing_mm)
        )
        self.fallback_sid_mm = (
            None if fallback_sid_mm is None else float(fallback_sid_mm)
        )
        self.source_to_isocenter_mm = float(source_to_isocenter_mm)
        self.pin_memory = bool(pin_memory)
        self.data_train: Stage2NPZDataset | None = None
        self.data_val: Stage2NPZDataset | None = None
        self.data_test: Stage2NPZDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        splits = load_case_splits(
            self.split_json, case_id_mode=self.case_id_mode
        )
        common: dict[str, Any] = {
            "projection_source": self.projection_source,
            "voxel_source": self.voxel_source,
            "output_type": "torch",
            "case_id_mode": self.case_id_mode,
            "expected_imager_pixel_spacing_mm": (
                self.expected_imager_pixel_spacing_mm
            ),
            "fallback_imager_pixel_spacing_mm": (
                self.fallback_imager_pixel_spacing_mm
            ),
            "fallback_sid_mm": self.fallback_sid_mm,
            "source_to_isocenter_mm": self.source_to_isocenter_mm,
        }
        if stage in (None, "fit", "validate"):
            self.data_train = Stage2NPZDataset(
                **common,
                case_ids=splits["train"],
                view_mode="random_pair",
                random_seed=self.random_seed,
                minimum_pair_angle_deg=self.minimum_train_pair_angle_deg,
            )
            self.data_val = Stage2NPZDataset(
                **common,
                case_ids=splits["val"],
                view_mode="fixed",
                fixed_view_indices=self.evaluation_view_indices,
                fixed_view_labels=self.evaluation_view_labels,
            )
        if stage in (None, "test", "predict"):
            self.data_test = Stage2NPZDataset(
                **common,
                case_ids=splits["test"],
                view_mode="fixed",
                fixed_view_indices=self.evaluation_view_indices,
                fixed_view_labels=self.evaluation_view_labels,
            )

    def set_train_epoch(self, epoch: int) -> None:
        if self.data_train is not None:
            self.data_train.set_epoch(epoch)

    def _loader(self, dataset: Stage2NPZDataset | None, *, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting a loader.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # Worker copies must be recreated so epoch-dependent random pairs
            # observe Stage2NPZDataset.set_epoch().
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        # Lightning restores ``current_epoch`` from a checkpoint before asking
        # for the training loader. Seed the main-process dataset here so the
        # first worker copies in a resumed run do not silently reuse epoch 0.
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            self.set_train_epoch(int(trainer.current_epoch))
        return self._loader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.data_test, shuffle=False)
