from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.view_direction_robustness_npz import (
    condition_id,
    fit_quadratic_response_surface,
    is_view_direction_robustness_mode,
    resolve_view_direction_robustness_plan,
    run_view_direction_robustness,
)


def test_default_plan_matches_parametric_24_condition_protocol():
    plan = resolve_view_direction_robustness_plan({})

    assert len(plan["conditions"]) == 24
    assert len(set(plan["conditions"])) == 24
    assert len(plan["visualization_conditions"]) == 8
    assert (-15.0, 0.0) in plan["conditions"]
    assert (10.0, -10.0) in plan["conditions"]
    assert plan["record_paper_metrics"] is True
    assert plan["save_mask_npz_files"] is False


def test_single_condition_run_compares_to_accurate_baseline(monkeypatch, tmp_path):
    checkpoint = tmp_path / "run" / "checkpoints" / "epoch_1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    baseline_dir = (
        tmp_path / "run" / "evaluation_paper_metric" / "epoch_1"
    )
    baseline_dir.mkdir(parents=True)
    baseline_cases = [
        {"case_id": "2", "split": "validation", "final": {}},
        {"case_id": "3", "split": "test", "final": {}},
    ]
    (baseline_dir / "performance_per_case.json").write_text(
        json.dumps(baseline_cases), encoding="utf-8"
    )
    baseline = {
        "comparison_condition": {
            "checkpoint": str(checkpoint),
            "evaluation_split": "val_test",
            "eval_num_views": 2,
            "eval_view_selection": "fixed",
            "evaluation_view_indices": [0, 6],
            "view_directions_accurate": True,
        },
        "roles": {
            "final": {
                "paper_mask_dice_3d": 0.8,
                "paper_mask_cldice_3d": 0.75,
                "paper_mask_ssim_3d": 0.7,
            }
        },
        "timing": {"mean_processing_elapsed_ms": 20.0},
        "per_case_metrics_file": "performance_per_case.json",
    }
    (baseline_dir / "performance_summary.json").write_text(
        json.dumps(baseline), encoding="utf-8"
    )
    config_path = tmp_path / "sweep.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "robustness"
    eval_config = {
        "eval_output_dir": str(output_dir),
        "evaluation_mode": "inaccurate_view_direction",
        "projection_source": str(tmp_path / "projections"),
        "voxel_source": str(tmp_path / "voxels"),
        "split_json_path": str(tmp_path / "split.json"),
        "eval_split": "val_test",
        "eval_num_views": 2,
        "eval_view_selection": "fixed",
        "evaluation_view_indices": [0, 6],
        "view_direction_robustness": {
            "changes_deg": [[5.0, 0.0]],
            "visualization_conditions_deg": [],
        },
    }

    def fake_run(command, cwd, check):
        child = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        child_output = Path(child["eval_output_dir"])
        child_output.mkdir(parents=True)
        (child_output / "performance_per_case.json").write_text(
            json.dumps(baseline_cases), encoding="utf-8"
        )
        performance = {
            "comparison_condition": {
                **baseline["comparison_condition"],
                "view_directions_accurate": False,
                "theta_change_deg": 5.0,
                "phi_change_deg": 0.0,
            },
            "roles": {
                "final": {
                    "paper_mask_dice_3d": 0.7,
                    "paper_mask_cldice_3d": 0.6,
                    "paper_mask_ssim_3d": 0.6,
                }
            },
            "timing": {"mean_processing_elapsed_ms": 25.0},
            "per_case_metrics_file": "performance_per_case.json",
        }
        (child_output / "performance_summary.json").write_text(
            json.dumps(performance), encoding="utf-8"
        )
        metrics_dir = child_output / "metrics"
        metrics_dir.mkdir()
        (metrics_dir / "paper_metric_per_case.json").write_text(
            "[]", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "src.view_direction_robustness_npz.subprocess.run", fake_run
    )

    summary_path = run_view_direction_robustness(
        eval_config=eval_config,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_choice="explicit",
        experiment_dir=tmp_path / "run",
        output_override=None,
        project_root=tmp_path,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    comparison = summary["results"][0]["comparison_to_accurate"]["final"]
    assert summary["status"] == "complete"
    assert summary["num_conditions"] == 1
    assert comparison["paper_mask_dice_3d"]["signed_change"] == pytest.approx(-0.1)
    assert comparison["paper_mask_cldice_3d"]["signed_change"] == pytest.approx(
        -0.15
    )
    assert (output_dir / "view_direction_robustness_metrics.csv").is_file()
    assert (output_dir / "view_direction_robustness_timing.csv").is_file()


def test_explicit_plan_rejects_accurate_condition_and_unknown_visualization():
    with pytest.raises(ValueError, match=r"cannot contain \[0, 0\]"):
        resolve_view_direction_robustness_plan(
            {"view_direction_robustness": {"changes_deg": [[0.0, 0.0]]}}
        )

    with pytest.raises(ValueError, match="must occur"):
        resolve_view_direction_robustness_plan(
            {
                "view_direction_robustness": {
                    "changes_deg": [[5.0, 0.0]],
                    "visualization_conditions_deg": [[10.0, 0.0]],
                }
            }
        )


def test_mode_alias_and_condition_id_are_stable():
    assert is_view_direction_robustness_mode("inaccurate-view-directions")
    assert is_view_direction_robustness_mode("view direction robustness")
    assert condition_id(-2.5, 10.0) == (
        "theta_change_minus2p5deg_phi_change_10deg"
    )


def test_quadratic_surface_recovers_known_coefficients():
    samples = []
    for theta, phi in (
        (0.0, 0.0),
        (-2.0, 0.0),
        (2.0, 0.0),
        (0.0, -3.0),
        (0.0, 3.0),
        (-2.0, -3.0),
        (-2.0, 3.0),
        (2.0, -3.0),
        (2.0, 3.0),
    ):
        value = 1.0 + 2.0 * theta - 3.0 * phi + 0.5 * theta**2
        value += 0.25 * theta * phi + 0.75 * phi**2
        samples.append((theta, phi, value))

    fitted = fit_quadratic_response_surface(samples)

    assert fitted is not None
    assert fitted["r_squared"] == pytest.approx(1.0)
    assert fitted["coefficients"]["theta_linear"] == pytest.approx(2.0)
    assert fitted["coefficients"]["phi_squared"] == pytest.approx(0.75)
