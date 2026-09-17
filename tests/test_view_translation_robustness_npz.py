from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.view_translation_robustness_npz import run_view_translation_robustness


def test_single_translation_condition_compares_cldice_to_control(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "run" / "checkpoints" / "epoch_1.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    baseline_dir = tmp_path / "run" / "evaluation_paper_metric" / "epoch_1"
    baseline_dir.mkdir(parents=True)
    cases = [
        {"case_id": "2", "split": "validation", "final": {}},
        {"case_id": "3", "split": "test", "final": {}},
    ]
    (baseline_dir / "performance_per_case.json").write_text(
        json.dumps(cases), encoding="utf-8"
    )
    baseline = {
        "comparison_condition": {
            "checkpoint": str(checkpoint),
            "evaluation_split": "val_test",
            "eval_num_views": 2,
            "eval_view_selection": "fixed",
            "evaluation_view_indices": [0, 1],
            "view_directions_accurate": True,
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
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
    config_path = tmp_path / "translation.json"
    config_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "translation"
    eval_config = {
        "eval_output_dir": str(output_dir),
        "evaluation_mode": "view_translation_robustness",
        "projection_source": str(tmp_path / "projections"),
        "voxel_source": str(tmp_path / "voxels"),
        "split_json_path": str(tmp_path / "split.json"),
        "eval_split": "val_test",
        "eval_num_views": 2,
        "eval_view_selection": "fixed",
        "evaluation_view_indices": [0, 1],
        "view_translation_robustness": {
            "magnitudes_mm": [5.0],
            "patterns": ["y"],
        },
    }

    def fake_run(command, cwd, check):
        child = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        child_output = Path(child["eval_output_dir"])
        child_output.mkdir(parents=True)
        (child_output / "performance_per_case.json").write_text(
            json.dumps(cases), encoding="utf-8"
        )
        diagnostics = child_output / "metrics" / "view_translation_per_case.json"
        diagnostics.parent.mkdir()
        diagnostics.write_text("[]", encoding="utf-8")
        performance = {
            "comparison_condition": {
                **baseline["comparison_condition"],
                "view_translation_applied": True,
                "artery_translation_xyz_mm": [0.0, 5.0, 0.0],
                "translation_magnitude_mm": 5.0,
            },
            "evaluation": {
                "view_translation_perturbation": {
                    "enabled": True,
                    "per_case_file": str(diagnostics),
                    "summary": {
                        "clean_rerender_dice_vs_stored": {"minimum": 0.99}
                    },
                }
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
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "src.view_translation_robustness_npz.subprocess.run", fake_run
    )
    summary_path = run_view_translation_robustness(
        eval_config=eval_config,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_choice="explicit",
        experiment_dir=tmp_path / "run",
        output_override=None,
        project_root=tmp_path,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["results"][0]
    assert summary["status"] == "complete"
    assert summary["num_conditions"] == 1
    assert summary["plan"]["theta_change_deg"] == 0.0
    assert summary["plan"]["phi_change_deg"] == 0.0
    assert result["comparison_to_accurate"]["final"][
        "paper_mask_cldice_3d"
    ]["signed_change"] == pytest.approx(-0.15)
    assert result["coarse_metrics"] is None
    assert (output_dir / "view_translation_robustness_metrics.csv").is_file()
    assert (output_dir / "view_translation_robustness_timing.csv").is_file()
