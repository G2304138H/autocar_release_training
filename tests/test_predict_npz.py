from __future__ import annotations

import math
from functools import partial

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.predict_npz import _aggregate_metrics, _device, build_parser
from src.modules import autocar_voxel_pl


class _DummyAutoCAR(torch.nn.Module):
    """Minimal constructor stand-in for checkpoint hyperparameter tests."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.sparse_backend = "raw"


def test_aggregate_metrics_reports_mean_sd_and_standard_error():
    summary = _aggregate_metrics(
        [
            {"masked_dice_3d": 0.5, "ssim_3d": 0.75},
            {"masked_dice_3d": 1.0, "ssim_3d": 0.25},
        ]
    )

    dice = summary["metrics"]["masked_dice_3d"]
    assert dice["mean"] == pytest.approx(0.75)
    assert dice["standard_deviation"] == pytest.approx(math.sqrt(0.125))
    assert dice["standard_error"] == pytest.approx(0.25)


def test_cpu_device_is_accepted():
    assert _device("cpu") == torch.device("cpu")


def test_primary_cli_defaults_to_test_split_anchor_pair():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.ckpt",
            "--projections",
            "projections",
            "--voxels",
            "voxels",
            "--output-directory",
            "predictions",
            "--split-json",
            "splits.json",
        ]
    )

    assert args.split == "test"
    assert args.view_indices == (0, 6)
    assert args.expected_view_labels == (
        "RAO 25, CAU 35",
        "LAO 5, CRA 40",
    )


def test_unavailable_cuda_device_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="only 1 device"):
        _device("cuda:1")


def test_voxel_lightning_checkpoint_reconstructs_saved_config(
    monkeypatch, tmp_path
):
    lightning = pytest.importorskip("lightning")
    monkeypatch.setattr(autocar_voxel_pl, "AutoCAR", _DummyAutoCAR)
    model = autocar_voxel_pl.AutoCARVoxelLit(
        recon_net={"profile": "test"},
        optimizer=partial(torch.optim.Adam, lr=3e-4),
        scheduler=None,
    )
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint_path,
    )

    restored = autocar_voxel_pl.AutoCARVoxelLit.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )

    assert restored.recon_net.config == {"profile": "test"}
    assert restored.recon_net.scale.item() == pytest.approx(2.0)
    configured = restored.configure_optimizers()["optimizer"]
    assert isinstance(configured, torch.optim.Adam)
    assert configured.param_groups[0]["lr"] == pytest.approx(3e-4)
