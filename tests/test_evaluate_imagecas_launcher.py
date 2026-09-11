from __future__ import annotations

import json

import pytest

from scripts import evaluate_imagecas_npz as launcher


def test_training_task_names_are_read_from_artery_configs():
    assert launcher._training_task_name("lca") == "train_autocar_lca"
    assert launcher._training_task_name("rca") == "train_autocar_rca"


def test_best_checkpoint_prefers_unique_non_last_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = checkpoint_dir / "epoch_017.ckpt"
    best.touch()
    (checkpoint_dir / "last.ckpt").touch()

    assert launcher._best_checkpoint_in_run(tmp_path / "run") == best.resolve()


def test_best_checkpoint_refuses_ambiguous_training_run(tmp_path):
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch_003.ckpt").touch()
    (checkpoint_dir / "epoch_017.ckpt").touch()

    with pytest.raises(ValueError, match="pass the intended checkpoint explicitly"):
        launcher._best_checkpoint_in_run(tmp_path / "run")


def test_latest_checkpoint_skips_new_run_before_first_validation(tmp_path):
    runs_dir = tmp_path / "train_autocar_lca" / "runs"
    old_run = runs_dir / "2026-09-06_10-00-00"
    new_run = runs_dir / "2026-09-07_10-00-00"
    old_checkpoints = old_run / "checkpoints"
    old_checkpoints.mkdir(parents=True)
    new_run.mkdir(parents=True)
    best = old_checkpoints / "epoch_042.ckpt"
    best.touch()

    checkpoint, experiment_dir = launcher._latest_trained_checkpoint(
        tmp_path, "lca"
    )

    assert checkpoint == best.resolve()
    assert experiment_dir == old_run.resolve()


def test_both_artery_dry_run_writes_paper_metric_configs(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    lca_checkpoint = checkpoints / "lca.ckpt"
    rca_checkpoint = checkpoints / "rca.ckpt"
    lca_checkpoint.touch()
    rca_checkpoint.touch()

    arguments = ["--dry-run", "--output-root", str(tmp_path / "evaluation")]
    for artery, checkpoint in (
        ("lca", lca_checkpoint),
        ("rca", rca_checkpoint),
    ):
        projection_source = tmp_path / artery / "projections"
        voxel_source = tmp_path / artery / "voxels"
        split_json = tmp_path / artery / "split.json"
        projection_source.mkdir(parents=True)
        voxel_source.mkdir(parents=True)
        split_json.write_text(
            json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
            encoding="utf-8",
        )
        arguments.extend(
            [
                f"--{artery}-checkpoint",
                str(checkpoint),
                f"--{artery}-projection-source",
                str(projection_source),
                f"--{artery}-voxel-source",
                str(voxel_source),
                f"--{artery}-split-json",
                str(split_json),
            ]
        )

    assert launcher.main(arguments) == 0

    config_dir = tmp_path / "evaluation" / "configs"
    lca = json.loads((config_dir / "lca_paper_metric.json").read_text())
    rca = json.loads((config_dir / "rca_paper_metric.json").read_text())
    plan = json.loads(
        (tmp_path / "evaluation" / "evaluation_plan.json").read_text()
    )

    for config in (lca, rca):
        assert config["evaluation_mode"] == "paper_metric"
        assert config["eval_split"] == "val_test"
        assert config["num_eval_cases"] == "all"
        assert config["evaluation_view_indices"] == [0, 1]
        assert config["case_id_mode"] == "imagecas_numeric"
        assert config["save_prediction_npz_files"] is True
        assert config["paper_metric_save_centerline_graphs"] is True
    assert lca["expected_imager_pixel_spacing_mm"] == 0.65
    assert lca["fallback_imager_pixel_spacing_mm"] == 0.65
    assert lca["fallback_sid_mm"] == 900.0
    assert lca["evaluation_view_labels"] == [
        "RAO 25, CAU 35",
        "LAO 5, CAU 30",
    ]
    assert rca["expected_imager_pixel_spacing_mm"] == 0.55
    assert rca["fallback_imager_pixel_spacing_mm"] == 0.55
    assert rca["fallback_sid_mm"] == 900.0
    assert rca["evaluation_view_labels"] is None
    assert [run["artery"] for run in plan["runs"]] == ["lca", "rca"]


def test_default_output_is_inside_the_training_experiment(tmp_path):
    experiment_dir = tmp_path / "train_autocar_lca" / "runs" / "run_1"
    checkpoint_dir = experiment_dir / "checkpoints"
    projections = tmp_path / "projections"
    voxels = tmp_path / "voxels"
    split = tmp_path / "split.json"
    checkpoint_dir.mkdir(parents=True)
    projections.mkdir()
    voxels.mkdir()
    checkpoint = checkpoint_dir / "epoch_021.ckpt"
    checkpoint.touch()
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
        ]
    )

    plan = launcher._build_plans(args, output_root=None)[0]

    assert plan.experiment_dir == experiment_dir.resolve()
    assert plan.output_dir == (
        experiment_dir / "evaluation_paper_metric" / "epoch_021"
    ).resolve()
    assert plan.config_path == (
        experiment_dir / "evaluation_configs" / "paper_metric_epoch_021.json"
    ).resolve()


def test_combined_summary_keeps_arteries_separate(tmp_path):
    plans = []
    for artery, dice in (("lca", 0.8), ("rca", 0.6)):
        output_dir = tmp_path / artery
        metric_dir = output_dir / "metrics"
        metric_dir.mkdir(parents=True)
        (metric_dir / "paper_metric_summary.json").write_text(
            json.dumps({"macro_dice_3d": dice}), encoding="utf-8"
        )
        (output_dir / "performance_summary.json").write_text(
            json.dumps(
                {
                    "evaluation": {},
                    "timing": {"mean_processing_elapsed_ms": 25.0},
                }
            ),
            encoding="utf-8",
        )
        timing_dir = output_dir / "timings" / "processing"
        timing_dir.mkdir(parents=True)
        (timing_dir / "summary.json").write_text(
            json.dumps({"mean_processing_elapsed_ms": 25.0}),
            encoding="utf-8",
        )
        checkpoint = tmp_path / f"{artery}.ckpt"
        checkpoint.touch()
        plans.append(
            launcher.EvaluationPlan(
                artery=artery,
                checkpoint=checkpoint,
                checkpoint_source="explicit",
                experiment_dir=tmp_path,
                config_path=tmp_path / f"{artery}.json",
                output_dir=output_dir,
                config={},
            )
        )

    summary = launcher._combined_summary(plans)

    assert summary["arteries"]["lca"]["paper_metric"]["macro_dice_3d"] == 0.8
    assert summary["arteries"]["rca"]["paper_metric"]["macro_dice_3d"] == 0.6
    assert summary["arteries"]["lca"]["timing"][
        "mean_processing_elapsed_ms"
    ] == 25.0
    assert "pooled" in summary["aggregation_policy"]
