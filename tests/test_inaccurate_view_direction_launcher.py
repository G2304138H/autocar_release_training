from __future__ import annotations

import json

from scripts import evaluate_imagecas_npz_inaccurate_view_direction as launcher


def test_both_artery_dry_run_writes_24_condition_configs(tmp_path):
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
    lca = json.loads(
        (config_dir / "lca_inaccurate_view_direction.json").read_text()
    )
    rca = json.loads(
        (config_dir / "rca_inaccurate_view_direction.json").read_text()
    )
    plan = json.loads(
        (tmp_path / "evaluation" / "evaluation_plan.json").read_text()
    )
    for config in (lca, rca):
        robustness = config["view_direction_robustness"]
        assert config["evaluation_mode"] == "inaccurate_view_direction"
        assert config["eval_split"] == "val_test"
        assert config["evaluation_view_indices"] == [0, 6]
        assert robustness["axis_degrees"] == [2.0, 5.0, 10.0, 15.0]
        assert robustness["combined_degrees"] == [5.0, 10.0]
        assert len(robustness["visualization_conditions_deg"]) == 8
        assert robustness["record_paper_metrics"] is True
        assert robustness["accurate_baseline_summary"] == "auto"
    assert lca["expected_imager_pixel_spacing_mm"] == 0.65
    assert rca["expected_imager_pixel_spacing_mm"] == 0.55
    assert [run["artery"] for run in plan["runs"]] == ["lca", "rca"]


def test_custom_small_grid_filters_representative_visualizations(tmp_path):
    checkpoint = tmp_path / "lca.ckpt"
    projections = tmp_path / "projections"
    voxels = tmp_path / "voxels"
    split = tmp_path / "split.json"
    checkpoint.touch()
    projections.mkdir()
    voxels.mkdir()
    split.write_text(
        json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
        encoding="utf-8",
    )
    args = launcher._parser().parse_args(
        [
            "--artery",
            "lca",
            "--lca-checkpoint",
            str(checkpoint),
            "--lca-projection-source",
            str(projections),
            "--lca-voxel-source",
            str(voxels),
            "--lca-split-json",
            str(split),
            "--axis-degrees",
            "2",
            "--combined-degrees",
            "5",
        ]
    )

    plan = launcher._build_plans(args, output_root=tmp_path / "out")[0]

    assert plan.config["view_direction_robustness"][
        "visualization_conditions_deg"
    ] == []

