from __future__ import annotations

import json

import pytest

from scripts import visualize_imagecas_npz_case as launcher


@pytest.mark.parametrize(
    ("artery", "spacing", "labels"),
    [
        ("lca", 0.65, ["RAO 25, CAU 35", "LAO 5, CRA 40"]),
        ("rca", 0.55, None),
    ],
)
def test_single_case_dry_run_uses_artery_training_run(
    tmp_path,
    artery,
    spacing,
    labels,
):
    experiment_dir = tmp_path / f"train_autocar_{artery}" / "runs" / "run_1"
    checkpoint_dir = experiment_dir / "checkpoints"
    projections = tmp_path / artery / "projections"
    voxels = tmp_path / artery / "voxels"
    split = tmp_path / artery / "split.json"
    checkpoint_dir.mkdir(parents=True)
    projections.mkdir(parents=True)
    voxels.mkdir()
    checkpoint = checkpoint_dir / "epoch_009.ckpt"
    checkpoint.touch()
    split.write_text(
        json.dumps(
            {
                "train": ["1"],
                "val": [f"{artery}_0017.npz"],
                "test": ["18"],
            }
        ),
        encoding="utf-8",
    )

    result = launcher.main(
        [
            "--artery",
            artery,
            "--case-number",
            "17",
            "--checkpoint",
            str(checkpoint),
            "--projection-source",
            str(projections),
            "--voxel-source",
            str(voxels),
            "--split-json",
            str(split),
            "--dry-run",
        ]
    )

    assert result == 0
    config_path = (
        experiment_dir
        / "evaluation_configs"
        / "visualisation_epoch_009_case_17.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["checkpoint_path"] == str(checkpoint.resolve())
    assert config["evaluation_mode"] == "visualisation"
    assert config["eval_split"] == "val_test"
    assert config["eval_case_ids"] == ["17"]
    assert config["max_visualizations"] == 1
    assert config["evaluation_view_indices"] == [0, 6]
    assert config["evaluation_view_labels"] == labels
    assert config["expected_imager_pixel_spacing_mm"] == spacing
    assert config["fallback_imager_pixel_spacing_mm"] == spacing
    assert config["fallback_sid_mm"] == 900.0
    assert config["eval_output_dir"] == str(
        (experiment_dir / "evaluation" / "epoch_009" / "case_17").resolve()
    )


def test_single_case_visualization_rejects_training_case(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    projections = tmp_path / "projections"
    voxels = tmp_path / "voxels"
    split = tmp_path / "split.json"
    checkpoint_dir.mkdir(parents=True)
    projections.mkdir()
    voxels.mkdir()
    checkpoint = checkpoint_dir / "epoch_001.ckpt"
    checkpoint.touch()
    split.write_text(
        json.dumps({"train": ["7"], "val": ["8"], "test": ["9"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present in the validation or test"):
        launcher.main(
            [
                "--artery",
                "rca",
                "--case-number",
                "7",
                "--checkpoint",
                str(checkpoint),
                "--projection-source",
                str(projections),
                "--voxel-source",
                str(voxels),
                "--split-json",
                str(split),
                "--dry-run",
            ]
        )
