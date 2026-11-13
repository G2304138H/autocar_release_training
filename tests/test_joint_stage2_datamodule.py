from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lightning")
from src.dataset.joint_stage2_datamodule import (
    ArteryConcatDataset, JointStage2NPZDataModule, align_patient_splits,
)
from test_stage2_npz_dataset import _write_projection, _write_voxel


def test_patient_alignment_promotes_heldout_and_strict_rejects_conflicts():
    sources = {
        "rca": {"train": ["1", "4", "5"], "val": ["2"], "test": ["3"]},
        "lca": {"train": ["1", "2"], "val": ["4"], "test": ["3", "5"]},
    }
    result = align_patient_splits(sources)
    assert result["rca"] == {"train": ["1"], "val": ["4", "2"], "test": ["5", "3"]}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (set().union(*(set(s[left]) for s in result.values())) &
                    set().union(*(set(s[right]) for s in result.values())))
    with pytest.raises(ValueError, match="patient leakage"):
        align_patient_splits(sources, "strict")


def test_alignment_rejects_empty_training_split():
    with pytest.raises(ValueError, match="empty"):
        align_patient_splits({
            "rca": {"train": ["1"], "val": ["2"], "test": ["3"]},
            "lca": {"train": ["2"], "val": ["1"], "test": ["3"]},
        })


def test_joint_loader_preserves_geometry_identity_and_epoch(tmp_path, monkeypatch):
    sources = {}
    for artery, spacing in (("rca", 0.55), ("lca", 0.65)):
        root = tmp_path / artery
        projections, voxels = root / "projections", root / "voxels"
        projections.mkdir(parents=True)
        voxels.mkdir()
        for case in ("1", "2", "3"):
            _write_projection(projections / f"{case}.npz", case_id=case, pixel_spacing_mm=spacing)
            _write_voxel(voxels / f"{case}.npz", case_id=case)
        manifest = root / "split.json"
        manifest.write_text(json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}))
        sources[artery] = dict(
            projection_source=str(projections), voxel_source=str(voxels), split_json=str(manifest),
            case_id_mode="imagecas_numeric", expected_imager_pixel_spacing_mm=spacing,
            evaluation_view_indices=[0, 3], evaluation_view_labels=None,
        )
    module = JointStage2NPZDataModule(sources, minimum_train_pair_angle_deg=0)
    module.setup()
    batches = list(module.train_dataloader())
    assert {batch["case_id"][0] for batch in batches} == {"rca:1", "lca:1"}
    for batch in batches:
        assert batch["patient_id"] == ["1"]
        assert float(batch["imager_pixel_spacing_mm"][0]) == pytest.approx(
            0.55 if batch["artery"] == ["rca"] else 0.65)
    assert len(module.data_val) == len(module.data_test) == 2
    from types import SimpleNamespace
    module.trainer = SimpleNamespace(current_epoch=7)
    module.train_dataloader()
    for dataset in module.data_train.datasets:
        assert dataset.epoch == 7

    # Check the shared loader with an actual Lightning optimization/evaluation loop.
    import lightning as L
    import torch

    class TinyModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = {"train": set(), "val": set(), "test": set()}

        def on_train_epoch_start(self):
            self.trainer.datamodule.set_train_epoch(self.current_epoch)

        def training_step(self, batch, batch_idx):
            self.seen["train"].update(batch["artery"])
            return (self.weight * batch["images"].float().mean()).square()

        def validation_step(self, batch, batch_idx):
            self.seen["val"].update(batch["artery"])

        def test_step(self, batch, batch_idx):
            self.seen["test"].update(batch["artery"])

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = TinyModel()
    trainer = L.Trainer(accelerator="cpu", devices=1, max_epochs=1,
                        logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=module)
    trainer.test(model, datamodule=module)
    assert model.seen == {name: {"rca", "lca"} for name in ("train", "val", "test")}

    # Exercise the real launcher against synthetic NPZs and check its split files.
    from scripts.train_joint_imagecas_npz import main
    import sys
    report = tmp_path / "effective.json"
    overrides = ["data.minimum_train_pair_angle_deg=0"]
    for artery, options in sources.items():
        for name, value in options.items():
            encoded = "null" if value is None else str(value)
            overrides.append(f"data.sources.{artery}.{name}={encoded}")
        overrides.append(f"data.sources.{artery}.excluded_train_case_ids=[]")
    monkeypatch.setattr(sys, "argv", ["train_joint", "--preflight-only", "--split-report", str(report), "--", *overrides])
    main()
    assert json.loads(report.read_text()) == module.effective_splits
    assert json.loads((tmp_path / "effective_rca.json").read_text())["test"] == ["3"]


def test_joint_hydra_configuration_instantiates_without_nested_modules():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="train", overrides=["experiment=stage2_npz_joint"])
    module = instantiate(config.data)
    assert isinstance(module, JointStage2NPZDataModule)
    assert module.sources["rca"]["expected_imager_pixel_spacing_mm"] == 0.55
    assert module.sources["lca"]["expected_imager_pixel_spacing_mm"] == 0.65
    assert config.model.recon_net.ray_casting.LODs == [1.0]
