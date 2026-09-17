"""Orchestrate fixed positive second-view translation stress tests."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.view_direction_robustness_npz import (
    _baseline_path,
    _case_keys,
    _load_json,
    _resolve_relative,
    _validate_baseline,
    _write_json,
)
from src.view_translation_npz import TRANSLATION_SIGN_CONVENTION


DEFAULT_TRANSLATION_MAGNITUDES_MM = (5.0, 10.0, 20.0)
DEFAULT_TRANSLATION_PATTERNS = ("y", "xz", "xyz")


def is_view_translation_robustness_mode(raw: Any) -> bool:
    if raw is None:
        return False
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return value in {
        "inaccurate_view_translation",
        "view_translation_robustness",
        "second_view_translation_robustness",
        "translational_calibration_robustness",
    }


def _finite_number(raw: Any, *, label: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return value


def translation_vector_mm(
    pattern: str, magnitude_mm: float
) -> tuple[float, float, float]:
    normalized = str(pattern).strip().lower().replace("-", "_")
    normalized = {"y_only": "y", "xz_equal": "xz", "xyz_equal": "xyz"}.get(
        normalized, normalized
    )
    if normalized not in DEFAULT_TRANSLATION_PATTERNS:
        raise ValueError(
            f"translation pattern must be one of {DEFAULT_TRANSLATION_PATTERNS}."
        )
    magnitude = _finite_number(magnitude_mm, label="magnitude_mm")
    if magnitude <= 0.0:
        raise ValueError("magnitude_mm must be positive.")
    if normalized == "y":
        return (0.0, magnitude, 0.0)
    if normalized == "xz":
        component = magnitude / math.sqrt(2.0)
        return (component, 0.0, component)
    component = magnitude / math.sqrt(3.0)
    return (component, component, component)


def translation_condition_id(pattern: str, magnitude_mm: float) -> str:
    normalized = str(pattern).strip().lower().replace("-", "_")
    number = format(float(magnitude_mm), ".12g").replace(".", "p")
    return f"{normalized}_translation_{number}mm"


def _condition_label(pattern: str, magnitude_mm: float) -> str:
    return f"{pattern.upper()}-{format(float(magnitude_mm), '.12g')}"


def resolve_view_translation_robustness_plan(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config.get("view_translation_robustness", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("view_translation_robustness must be a JSON object.")
    magnitudes_raw = raw.get(
        "magnitudes_mm", list(DEFAULT_TRANSLATION_MAGNITUDES_MM)
    )
    if not isinstance(magnitudes_raw, (list, tuple)) or not magnitudes_raw:
        raise ValueError(
            "view_translation_robustness.magnitudes_mm must be non-empty."
        )
    magnitudes: list[float] = []
    for index, item in enumerate(magnitudes_raw):
        value = _finite_number(
            item, label=f"view_translation_robustness.magnitudes_mm[{index}]"
        )
        if value <= 0.0:
            raise ValueError("Translation magnitudes must be positive.")
        if value not in magnitudes:
            magnitudes.append(value)
    patterns_raw = raw.get("patterns", list(DEFAULT_TRANSLATION_PATTERNS))
    if not isinstance(patterns_raw, (list, tuple)) or not patterns_raw:
        raise ValueError("view_translation_robustness.patterns must be non-empty.")
    patterns: list[str] = []
    aliases = {"y_only": "y", "xz_equal": "xz", "xyz_equal": "xyz"}
    for item in patterns_raw:
        pattern = str(item).strip().lower().replace("-", "_")
        pattern = aliases.get(pattern, pattern)
        if pattern not in DEFAULT_TRANSLATION_PATTERNS:
            raise ValueError(
                "view_translation_robustness.patterns entries must be y, xz, "
                "or xyz."
            )
        if pattern not in patterns:
            patterns.append(pattern)
    if int(config.get("eval_num_views", 2)) != 2:
        raise ValueError("Translation robustness requires eval_num_views=2.")
    if int(raw.get("perturbed_input_position", 1)) != 1:
        raise ValueError(
            "view_translation_robustness.perturbed_input_position must be 1."
        )
    circle_points = raw.get("renderer_num_circle_points", 120)
    if (
        isinstance(circle_points, bool)
        or int(circle_points) < 3
        or float(circle_points) != float(int(circle_points))
    ):
        raise ValueError("renderer_num_circle_points must be an integer >= 3.")
    minimum_dice = _finite_number(
        raw.get("minimum_clean_rerender_dice", 0.98),
        label="view_translation_robustness.minimum_clean_rerender_dice",
    )
    visibility_threshold = _finite_number(
        raw.get("visibility_warning_threshold", 0.95),
        label="view_translation_robustness.visibility_warning_threshold",
    )
    if not 0.0 <= minimum_dice <= 1.0:
        raise ValueError("minimum_clean_rerender_dice must lie in [0,1].")
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_warning_threshold must lie in [0,1].")
    booleans: dict[str, bool] = {}
    for key, default in (
        ("require_accurate_baseline", True),
        ("record_paper_metrics", True),
        ("save_mask_npz_files", False),
        ("save_centerline_graph_npz_files", False),
        ("fail_below_visibility_threshold", False),
    ):
        value = raw.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"view_translation_robustness.{key} must be boolean.")
        booleans[key] = value
    if not booleans["record_paper_metrics"]:
        raise ValueError(
            "record_paper_metrics must be true because translation robustness "
            "requires Dice, clDice, and the other paper metrics."
        )
    conditions = [
        {
            "pattern": pattern,
            "magnitude_mm": magnitude,
            "translation_xyz_mm": translation_vector_mm(pattern, magnitude),
        }
        for pattern in patterns
        for magnitude in magnitudes
    ]
    return {
        "conditions": conditions,
        "patterns": patterns,
        "magnitudes_mm": magnitudes,
        "renderer_num_circle_points": int(circle_points),
        "minimum_clean_rerender_dice": minimum_dice,
        "visibility_warning_threshold": visibility_threshold,
        "accurate_baseline_summary": raw.get(
            "accurate_baseline_summary", "auto"
        ),
        "output_dir": raw.get("output_dir"),
        **booleans,
    }


def _validate_translation_condition(
    performance: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    identifier: str,
    expected_vector: Sequence[float],
) -> None:
    current = performance.get("comparison_condition")
    accurate = baseline.get("comparison_condition")
    if not isinstance(current, Mapping) or not isinstance(accurate, Mapping):
        raise ValueError("Performance summaries must record comparison_condition.")
    if not bool(current.get("view_directions_accurate", False)):
        raise ValueError(f"Condition {identifier} changed camera directions.")
    for key in ("theta_change_deg", "phi_change_deg"):
        if not math.isclose(float(current.get(key, math.nan)), 0.0, abs_tol=1e-9):
            raise ValueError(f"Condition {identifier} recorded non-zero {key}.")
    if not bool(current.get("view_translation_applied", False)):
        raise ValueError(f"Condition {identifier} did not record a translation.")
    actual = current.get("artery_translation_xyz_mm")
    if not isinstance(actual, list) or len(actual) != 3 or not all(
        math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-6)
        for got, want in zip(actual, expected_vector)
    ):
        raise ValueError(
            f"Condition {identifier} recorded translation {actual!r}, expected "
            f"{list(expected_vector)!r}."
        )
    for key in (
        "checkpoint",
        "evaluation_split",
        "eval_num_views",
        "eval_view_selection",
        "evaluation_view_indices",
    ):
        current_value = current.get(key)
        baseline_value = accurate.get(key)
        matches = (
            Path(str(current_value)).resolve()
            == Path(str(baseline_value)).resolve()
            if key == "checkpoint" and current_value and baseline_value
            else current_value == baseline_value
        )
        if not matches:
            raise ValueError(
                f"Condition {identifier} differs from its accurate control for "
                f"{key}: {current_value!r} != {baseline_value!r}."
            )


def _numeric_translation_comparison(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, float | None]]]:
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for role, current_metrics in current.items():
        baseline_metrics = baseline.get(role)
        if not isinstance(current_metrics, Mapping) or not isinstance(
            baseline_metrics, Mapping
        ):
            continue
        role_result: dict[str, dict[str, float | None]] = {}
        for metric, current_raw in current_metrics.items():
            baseline_raw = baseline_metrics.get(metric)
            if (
                isinstance(current_raw, bool)
                or isinstance(baseline_raw, bool)
                or not isinstance(current_raw, (int, float))
                or not isinstance(baseline_raw, (int, float))
            ):
                continue
            translated = float(current_raw)
            accurate = float(baseline_raw)
            signed_change = translated - accurate
            role_result[str(metric)] = {
                "accurate": accurate,
                "translated": translated,
                "signed_change": signed_change,
                "relative_change_percent": (
                    None
                    if accurate == 0.0
                    else 100.0 * signed_change / abs(accurate)
                ),
            }
        if role_result:
            result[str(role)] = role_result
    return result


def _write_flat_metrics(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        comparison = result.get("comparison_to_accurate", {})
        for role, metrics in result.get("roles", {}).items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                baseline = comparison.get(role, {}).get(metric, {})
                rows.append(
                    {
                        "condition_id": result["condition_id"],
                        "condition_label": result["condition_label"],
                        "theta_change_deg": 0.0,
                        "phi_change_deg": 0.0,
                        "translation_x_mm": result["translation_xyz_mm"][0],
                        "translation_y_mm": result["translation_xyz_mm"][1],
                        "translation_z_mm": result["translation_xyz_mm"][2],
                        "translation_magnitude_mm": result["magnitude_mm"],
                        "role": role,
                        "metric": metric,
                        "value": float(value),
                        "accurate_value": baseline.get("accurate"),
                        "signed_change_from_accurate": baseline.get(
                            "signed_change"
                        ),
                        "relative_change_from_accurate_percent": baseline.get(
                            "relative_change_percent"
                        ),
                    }
                )
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_timing_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        {
            "condition_id": result["condition_id"],
            "condition_label": result["condition_label"],
            "translation_x_mm": result["translation_xyz_mm"][0],
            "translation_y_mm": result["translation_xyz_mm"][1],
            "translation_z_mm": result["translation_xyz_mm"][2],
            "translation_magnitude_mm": result["magnitude_mm"],
            **{
                key: value
                for key, value in result.get("timing", {}).items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
        }
        for result in results
    ]
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_view_translation_robustness(
    *,
    eval_config: Mapping[str, Any],
    config_path: Path,
    checkpoint: Path,
    checkpoint_choice: str,
    experiment_dir: Path,
    output_override: str | None,
    project_root: Path,
) -> Path:
    plan = resolve_view_translation_robustness_plan(eval_config)
    output_raw = output_override or plan["output_dir"] or eval_config.get(
        "eval_output_dir"
    )
    output_dir = (
        _resolve_relative(str(output_raw), config_path=config_path)
        if output_raw
        else experiment_dir
        / "evaluation_view_translation_robustness"
        / checkpoint.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "run_configs"
    conditions_dir = output_dir / "conditions"
    config_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = _baseline_path(
        plan["accurate_baseline_summary"],
        experiment_dir=experiment_dir,
        checkpoint=checkpoint,
        config_path=config_path,
    )
    baseline: dict[str, Any] | None = None
    baseline_cases: list[tuple[str, str]] | None = None
    if baseline_path is not None and baseline_path.is_file():
        baseline = _load_json(baseline_path, label="Accurate performance summary")
        _validate_baseline(baseline_path, baseline, checkpoint=checkpoint)
        baseline_cases = _case_keys(baseline_path, baseline)
    elif plan["require_accurate_baseline"]:
        raise FileNotFoundError(
            "Run accurate two-view paper-metric evaluation first, or configure "
            "view_translation_robustness.accurate_baseline_summary. Expected: "
            f"{baseline_path}"
        )
    baseline_roles = {} if baseline is None else baseline.get("roles", {})
    if not isinstance(baseline_roles, dict):
        raise ValueError("The accurate baseline has invalid role metrics.")

    base_config = dict(eval_config)
    base_config.pop("view_translation_robustness", None)
    base_config["checkpoint_path"] = str(checkpoint)
    base_config["checkpoint_choice"] = checkpoint_choice
    base_config["experiment_dir"] = str(experiment_dir)
    for key in (
        "projection_source",
        "projections",
        "voxel_source",
        "voxels",
        "split_json",
        "split_json_path",
    ):
        if base_config.get(key):
            base_config[key] = str(
                _resolve_relative(base_config[key], config_path=config_path)
            )

    aggregate_path = output_dir / "view_translation_robustness_summary.json"
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "evaluation_description": (
            "fixed two-view translational calibration robustness evaluation"
        ),
        "stress_test_scope": "fixed-positive-direction",
        "checkpoint": str(checkpoint),
        "checkpoint_choice": checkpoint_choice,
        "accurate_baseline_summary": (
            None if baseline_path is None else str(baseline_path)
        ),
        "accurate_baseline": (
            None
            if baseline is None
            else {
                "comparison_condition": baseline.get("comparison_condition"),
                "roles": baseline_roles,
                "timing": baseline.get("timing"),
                "case_count": len(baseline_cases or []),
            }
        ),
        "plan": {
            "error_model": "fixed_positive_second_view_artery_translation",
            "num_input_views": 2,
            "ordered_source_view_indices": list(
                base_config.get("evaluation_view_indices", [0, 1])
            ),
            "accurate_input_positions": [0],
            "perturbed_input_positions": [1],
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
            "camera_angles_changed": False,
            "camera_matrices_changed": False,
            "ground_truth_target_changed": False,
            "model_weights_changed": False,
            "source_dataset_files_changed": False,
            "image_features_recomputed": True,
            "image_feature_cache_used": False,
            "patterns": plan["patterns"],
            "magnitudes_mm": plan["magnitudes_mm"],
            "conditions": [
                {
                    "condition_id": translation_condition_id(
                        condition["pattern"], condition["magnitude_mm"]
                    ),
                    "condition_label": _condition_label(
                        condition["pattern"], condition["magnitude_mm"]
                    ),
                    **condition,
                    "theta_change_deg": 0.0,
                    "phi_change_deg": 0.0,
                }
                for condition in plan["conditions"]
            ],
            "total_magnitude_normalized": True,
            "coordinate_convention": {
                "+x": "patient left",
                "+y": "patient anterior/away from table",
                "+z": "patient superior/toward head",
            },
            "sign_convention": TRANSLATION_SIGN_CONVENTION,
            "minimum_clean_rerender_dice": plan[
                "minimum_clean_rerender_dice"
            ],
            "recorded_metrics_include_cldice": True,
            "model_output_roles": ["final"],
            "coarse_refined_effect": (
                "not_applicable: AutoCAR exposes one supervised final volume; "
                "its ray-casting tensor is an internal feature representation"
            ),
        },
        "results": [],
    }
    _write_json(aggregate_path, aggregate)

    expected_cases = baseline_cases
    for condition in plan["conditions"]:
        pattern = str(condition["pattern"])
        magnitude = float(condition["magnitude_mm"])
        vector = [float(value) for value in condition["translation_xyz_mm"]]
        identifier = translation_condition_id(pattern, magnitude)
        condition_output = conditions_dir / identifier
        child = dict(base_config)
        child.update(
            {
                "evaluation_mode": "metric",
                "view_translation_robustness_condition": True,
                "record_view_translation_paper_metrics": True,
                "paper_metric_save_masks": plan["save_mask_npz_files"],
                "paper_metric_save_centerline_graphs": plan[
                    "save_centerline_graph_npz_files"
                ],
                "evaluation_view_directions": {
                    "accurate": True,
                    "theta_change_deg": 0.0,
                    "phi_change_deg": 0.0,
                },
                "evaluation_view_translation": {
                    "enabled": True,
                    "translation_xyz_mm": vector,
                    "perturbed_input_position": 1,
                    "renderer_num_circle_points": plan[
                        "renderer_num_circle_points"
                    ],
                    "minimum_clean_rerender_dice": plan[
                        "minimum_clean_rerender_dice"
                    ],
                    "visibility_warning_threshold": plan[
                        "visibility_warning_threshold"
                    ],
                    "fail_below_visibility_threshold": plan[
                        "fail_below_visibility_threshold"
                    ],
                },
                "eval_output_dir": str(condition_output),
            }
        )
        child_path = config_dir / f"{identifier}.json"
        _write_json(child_path, child)
        completed = subprocess.run(
            [sys.executable, "-m", "src.eval_npz", "--config", str(child_path)],
            cwd=project_root,
            check=False,
        )
        if completed.returncode != 0:
            aggregate.update(
                {
                    "status": "failed",
                    "failed_condition": identifier,
                    "failed_return_code": int(completed.returncode),
                }
            )
            _write_json(aggregate_path, aggregate)
            raise RuntimeError(
                f"Translation condition {identifier} failed with exit code "
                f"{completed.returncode}. Partial results: {aggregate_path}"
            )

        performance_path = condition_output / "performance_summary.json"
        performance = _load_json(
            performance_path, label=f"Performance summary for {identifier}"
        )
        if baseline is not None:
            _validate_translation_condition(
                performance,
                baseline,
                identifier=identifier,
                expected_vector=vector,
            )
        condition_cases = _case_keys(performance_path, performance)
        if expected_cases is None:
            expected_cases = condition_cases
        elif condition_cases != expected_cases:
            raise ValueError(
                f"Condition {identifier} evaluated different cases or order."
            )
        roles = performance.get("roles")
        if not isinstance(roles, dict) or not isinstance(roles.get("final"), dict):
            raise ValueError(f"Condition {identifier} has no final role metrics.")
        for metric in ("paper_mask_dice_3d", "paper_mask_cldice_3d"):
            if metric not in roles["final"]:
                raise RuntimeError(
                    f"Condition {identifier} did not record required {metric}."
                )
        diagnostics = performance.get("evaluation", {}).get(
            "view_translation_perturbation"
        )
        if not isinstance(diagnostics, dict) or not diagnostics.get(
            "per_case_file"
        ):
            raise RuntimeError(
                f"Condition {identifier} did not record translation diagnostics."
            )
        result = {
            "condition_id": identifier,
            "condition_label": _condition_label(pattern, magnitude),
            "pattern": pattern,
            "magnitude_mm": magnitude,
            "translation_xyz_mm": vector,
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
            "output_dir": str(condition_output),
            "performance_summary": str(performance_path),
            "translation_diagnostics": diagnostics,
            "roles": roles,
            "coarse_metrics": None,
            "refined_minus_coarse": None,
            "coarse_refined_not_applicable_reason": (
                "AutoCAR exposes one supervised final reconstruction volume."
            ),
            "timing": performance.get("timing", {}),
            "comparison_to_accurate": (
                None
                if baseline is None
                else _numeric_translation_comparison(roles, baseline_roles)
            ),
        }
        aggregate["results"].append(result)
        _write_json(aggregate_path, aggregate)

    aggregate["status"] = "complete"
    aggregate["num_conditions"] = len(aggregate["results"])
    aggregate["case_count"] = len(expected_cases or [])
    _write_json(aggregate_path, aggregate)
    _write_flat_metrics(
        output_dir / "view_translation_robustness_metrics.csv",
        aggregate["results"],
    )
    _write_timing_csv(
        output_dir / "view_translation_robustness_timing.csv",
        aggregate["results"],
    )
    return aggregate_path


__all__ = [
    "DEFAULT_TRANSLATION_MAGNITUDES_MM",
    "DEFAULT_TRANSLATION_PATTERNS",
    "is_view_translation_robustness_mode",
    "resolve_view_translation_robustness_plan",
    "run_view_translation_robustness",
    "translation_condition_id",
    "translation_vector_mm",
]
