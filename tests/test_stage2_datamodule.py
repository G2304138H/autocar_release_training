from __future__ import annotations

import json

import pytest

from src.dataset.case_splits import load_case_splits


def test_split_manifest_rejects_case_leakage(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["1"], "val": ["1"], "test": ["2"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leakage"):
        load_case_splits(path)


def test_split_manifest_preserves_declared_case_order(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {"train": ["2", "1"], "val": ["3"], "test": ["4"]}
        ),
        encoding="utf-8",
    )
    assert load_case_splits(path)["train"] == ("2", "1")


def test_datamodule_rejects_variable_shape_batching():
    pytest.importorskip("lightning")
    from src.dataset.stage2_datamodule import Stage2NPZDataModule

    with pytest.raises(ValueError, match="batch_size=1"):
        Stage2NPZDataModule(
            projection_source="projections",
            voxel_source="voxels",
            split_json="splits.json",
            batch_size=2,
        )


def test_train_loader_seeds_dataset_from_resumed_epoch(monkeypatch):
    pytest.importorskip("lightning")
    from types import SimpleNamespace

    from src.dataset.stage2_datamodule import Stage2NPZDataModule

    class EpochDataset:
        epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

    datamodule = Stage2NPZDataModule(
        projection_source="projections",
        voxel_source="voxels",
        split_json="splits.json",
    )
    dataset = EpochDataset()
    datamodule.data_train = dataset
    datamodule._trainer = SimpleNamespace(current_epoch=7)
    sentinel = object()
    monkeypatch.setattr(datamodule, "_loader", lambda value, *, shuffle: sentinel)

    assert datamodule.train_dataloader() is sentinel
    assert dataset.epoch == 7
