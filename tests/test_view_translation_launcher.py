from __future__ import annotations

import json

from scripts import evaluate_imagecas_npz_view_translation_robustness as launcher


def test_both_artery_dry_run_writes_fixed_nine_condition_protocol(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    arguments = ["--dry-run", "--output-root", str(tmp_path / "evaluation")]
    for artery in ("lca", "rca"):
        checkpoint = checkpoint_dir / f"{artery}.ckpt"
        checkpoint.touch()
        projections = tmp_path / artery / "projections"
        voxels = tmp_path / artery / "voxels"
        split = tmp_path / artery / "split.json"
        projections.mkdir(parents=True)
        voxels.mkdir(parents=True)
        split.write_text(
            json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
            encoding="utf-8",
        )
        arguments.extend(
            [
                f"--{artery}-checkpoint",
                str(checkpoint),
                f"--{artery}-projection-source",
                str(projections),
                f"--{artery}-voxel-source",
                str(voxels),
                f"--{artery}-split-json",
                str(split),
            ]
        )

    assert launcher.main(arguments) == 0

    config_dir = tmp_path / "evaluation" / "configs"
    for artery in ("lca", "rca"):
        config = json.loads(
            (config_dir / f"{artery}_view_translation.json").read_text()
        )
        robustness = config["view_translation_robustness"]
        assert config["evaluation_mode"] == "view_translation_robustness"
        assert config["evaluation_view_indices"] == [0, 1]
        assert config["evaluation_view_directions"] == {
            "accurate": True,
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
        }
        assert robustness["patterns"] == ["y", "xz", "xyz"]
        assert robustness["magnitudes_mm"] == [5.0, 10.0, 20.0]
        assert robustness["minimum_clean_rerender_dice"] == 0.98
        assert robustness["record_paper_metrics"] is True

    plan = json.loads(
        (tmp_path / "evaluation" / "evaluation_plan.json").read_text()
    )
    assert plan["evaluation_description"] == (
        "fixed two-view translational calibration robustness evaluation"
    )
    assert [run["artery"] for run in plan["runs"]] == ["lca", "rca"]
