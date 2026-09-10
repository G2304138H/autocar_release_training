"""Orchestrate fixed signed camera-angle robustness sweeps for AutoCAR."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_DEFAULT_AXIS_DEGREES = (2.0, 5.0, 10.0, 15.0)
_DEFAULT_COMBINED_DEGREES = (5.0, 10.0)
_DEFAULT_VISUALIZATION_CONDITIONS = (
    (-10.0, 0.0),
    (10.0, 0.0),
    (0.0, -10.0),
    (0.0, 10.0),
    (-10.0, -10.0),
    (-10.0, 10.0),
    (10.0, -10.0),
    (10.0, 10.0),
)


def is_view_direction_robustness_mode(raw: Any) -> bool:
    if raw is None:
        return False
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return value in {
        "inaccurate_view_direction",
        "inaccurate_view_directions",
        "view_direction_robustness",
        "view_directions_robustness",
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


def _positive_degrees(raw: Any, *, label: str) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{label} must be a non-empty list of positive degrees.")
    values: list[float] = []
    for index, item in enumerate(raw):
        value = _finite_number(item, label=f"{label}[{index}]")
        if value <= 0.0:
            raise ValueError(f"{label}[{index}] must be > 0.")
        if value not in values:
            values.append(value)
    return values


def _angle_pairs(raw: Any, *, label: str) -> list[tuple[float, float]]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of [theta, phi] pairs.")
    pairs: list[tuple[float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label}[{index}] must be [theta_deg, phi_deg].")
        pair = (
            _finite_number(item[0], label=f"{label}[{index}][0]"),
            _finite_number(item[1], label=f"{label}[{index}][1]"),
        )
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def resolve_view_direction_robustness_plan(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = config.get("view_direction_robustness", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("view_direction_robustness must be a JSON object.")
    if raw.get("changes_deg") is not None:
        conditions = _angle_pairs(
            raw["changes_deg"], label="view_direction_robustness.changes_deg"
        )
    else:
        axis = _positive_degrees(
            raw.get("axis_degrees", list(_DEFAULT_AXIS_DEGREES)),
            label="view_direction_robustness.axis_degrees",
        )
        combined = _positive_degrees(
            raw.get("combined_degrees", list(_DEFAULT_COMBINED_DEGREES)),
            label="view_direction_robustness.combined_degrees",
        )
        conditions = []
        for degrees in axis:
            conditions.extend(
                [
                    (-degrees, 0.0),
                    (degrees, 0.0),
                    (0.0, -degrees),
                    (0.0, degrees),
                ]
            )
        for degrees in combined:
            conditions.extend(
                [
                    (-degrees, -degrees),
                    (-degrees, degrees),
                    (degrees, -degrees),
                    (degrees, degrees),
                ]
            )
    if not conditions:
        raise ValueError("The view-direction robustness plan has no conditions.")
    if (0.0, 0.0) in conditions:
        raise ValueError(
            "The inaccurate-view-direction mode cannot contain [0, 0]; use "
            "paper_metric mode for the accurate baseline."
        )
    raw_visualizations = raw.get("visualization_conditions_deg")
    visualizations = (
        [pair for pair in _DEFAULT_VISUALIZATION_CONDITIONS if pair in conditions]
        if raw_visualizations is None
        else _angle_pairs(
            raw_visualizations,
            label="view_direction_robustness.visualization_conditions_deg",
        )
    )
    missing = [pair for pair in visualizations if pair not in conditions]
    if missing:
        raise ValueError(
            "Every visualization condition must occur in the robustness plan; "
            f"missing={missing}."
        )
    require_baseline = raw.get("require_accurate_baseline", True)
    record_paper_metrics = raw.get("record_paper_metrics", True)
    save_masks = raw.get("save_mask_npz_files", False)
    for label, value in (
        ("require_accurate_baseline", require_baseline),
        ("record_paper_metrics", record_paper_metrics),
        ("save_mask_npz_files", save_masks),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"view_direction_robustness.{label} must be boolean.")
    if save_masks and not record_paper_metrics:
        raise ValueError(
            "view_direction_robustness.save_mask_npz_files=true requires "
            "record_paper_metrics=true."
        )
    return {
        "conditions": conditions,
        "visualization_conditions": visualizations,
        "accurate_baseline_summary": raw.get(
            "accurate_baseline_summary", "auto"
        ),
        "require_accurate_baseline": require_baseline,
        "record_paper_metrics": record_paper_metrics,
        "save_mask_npz_files": save_masks,
        "output_dir": raw.get("output_dir"),
    }


def _format_number(value: float) -> str:
    return format(float(value), ".12g").replace("-", "minus").replace(".", "p")


def condition_id(theta_change_deg: float, phi_change_deg: float) -> str:
    return (
        f"theta_change_{_format_number(theta_change_deg)}deg_"
        f"phi_change_{_format_number(phi_change_deg)}deg"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_relative(raw: str | Path, *, config_path: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _baseline_path(
    raw: Any,
    *,
    experiment_dir: Path,
    checkpoint: Path,
    config_path: Path,
) -> Path | None:
    if raw is None or raw is False:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return (
            experiment_dir
            / "evaluation_paper_metric"
            / checkpoint.stem
            / "performance_summary.json"
        )
    return _resolve_relative(str(raw), config_path=config_path)


def _case_keys(summary_path: Path, summary: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw = summary.get("per_case_metrics_file")
    if not raw:
        raise ValueError(
            f"Performance summary does not identify per-case metrics: {summary_path}"
        )
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Per-case performance must be a JSON list: {path}")
    return [
        (str(record["case_id"]), str(record["split"]))
        for record in payload
        if isinstance(record, dict)
    ]


def _validate_baseline(
    path: Path,
    baseline: Mapping[str, Any],
    *,
    checkpoint: Path,
) -> None:
    condition = baseline.get("comparison_condition")
    if not isinstance(condition, dict):
        raise ValueError(f"Accurate baseline lacks comparison_condition: {path}")
    if not bool(condition.get("view_directions_accurate", False)):
        raise ValueError(
            "Baseline does not explicitly record accurate view directions. "
            f"Rerun the current paper-metric evaluator: {path}"
        )
    recorded = condition.get("checkpoint")
    if recorded and Path(str(recorded)).resolve() != checkpoint.resolve():
        raise ValueError(
            f"Baseline checkpoint does not match: {recorded} != {checkpoint}"
        )
    roles = baseline.get("roles")
    final = roles.get("final") if isinstance(roles, dict) else None
    if not isinstance(final, dict):
        raise ValueError(f"Accurate baseline has no final role metrics: {path}")
    for metric in (
        "paper_mask_dice_3d",
        "paper_mask_cldice_3d",
        "paper_mask_ssim_3d",
    ):
        if metric not in final:
            raise ValueError(
                f"Accurate baseline lacks {metric}; run paper_metric mode first."
            )


def _validate_condition(
    condition_summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    identifier: str,
    theta_change_deg: float,
    phi_change_deg: float,
) -> None:
    current = condition_summary.get("comparison_condition")
    accurate = baseline.get("comparison_condition")
    if not isinstance(current, dict) or not isinstance(accurate, dict):
        raise ValueError("Performance summaries must record comparison_condition.")
    if bool(current.get("view_directions_accurate", True)):
        raise ValueError(f"Condition {identifier} was recorded as accurate.")
    for key, expected in (
        ("theta_change_deg", theta_change_deg),
        ("phi_change_deg", phi_change_deg),
    ):
        actual = current.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Condition {identifier} recorded {key}={actual!r}, expected "
                f"{expected}."
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
            Path(str(current_value)).resolve() == Path(str(baseline_value)).resolve()
            if key == "checkpoint" and current_value and baseline_value
            else current_value == baseline_value
        )
        if not matches:
            raise ValueError(
                f"Condition {identifier} differs from the baseline for {key}: "
                f"{current_value!r} != {baseline_value!r}."
            )


def _numeric_comparison(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, float | None]]]:
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for role, current_metrics in current.items():
        baseline_metrics = baseline.get(role)
        if not isinstance(current_metrics, dict) or not isinstance(
            baseline_metrics, dict
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
            current_value = float(current_raw)
            baseline_value = float(baseline_raw)
            signed_change = current_value - baseline_value
            role_result[str(metric)] = {
                "accurate": baseline_value,
                "inaccurate": current_value,
                "signed_change": signed_change,
                "relative_change_percent": (
                    None
                    if baseline_value == 0.0
                    else 100.0 * signed_change / abs(baseline_value)
                ),
            }
        if role_result:
            result[str(role)] = role_result
    return result


def fit_quadratic_response_surface(
    samples: Sequence[tuple[float, float, float]],
) -> dict[str, Any] | None:
    finite = [
        (float(theta), float(phi), float(value))
        for theta, phi, value in samples
        if all(math.isfinite(float(item)) for item in (theta, phi, value))
    ]
    if len(finite) < 6:
        return None
    design = np.asarray(
        [
            [1.0, theta, phi, theta * theta, theta * phi, phi * phi]
            for theta, phi, _ in finite
        ],
        dtype=np.float64,
    )
    if int(np.linalg.matrix_rank(design)) < 6:
        return None
    observed = np.asarray([value for _, _, value in finite], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ coefficients
    residual_sum_squares = float(np.sum(residual * residual))
    total_sum_squares = float(np.sum((observed - np.mean(observed)) ** 2))
    names = (
        "intercept",
        "theta_linear",
        "phi_linear",
        "theta_squared",
        "theta_phi_interaction",
        "phi_squared",
    )
    return {
        "model": (
            "metric = intercept + theta_linear*theta + phi_linear*phi + "
            "theta_squared*theta^2 + theta_phi_interaction*theta*phi + "
            "phi_squared*phi^2"
        ),
        "coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
        "num_samples": len(finite),
        "r_squared": (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else 1.0 if residual_sum_squares == 0.0 else None
        ),
        "rmse": float(math.sqrt(residual_sum_squares / len(finite))),
        "theta_range_deg": [min(item[0] for item in finite), max(item[0] for item in finite)],
        "phi_range_deg": [min(item[1] for item in finite), max(item[1] for item in finite)],
    }


def _response_surfaces(
    results: Sequence[Mapping[str, Any]], baseline_roles: Mapping[str, Any]
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    role_names = set(str(role) for role in baseline_roles)
    role_names.update(
        str(role) for result in results for role in result.get("roles", {})
    )
    for role in sorted(role_names):
        baseline_metrics = baseline_roles.get(role, {})
        metric_names = set(str(metric) for metric in baseline_metrics)
        metric_names.update(
            str(metric)
            for result in results
            for metric in result.get("roles", {}).get(role, {})
        )
        role_models: dict[str, Any] = {}
        for metric in sorted(metric_names):
            samples: list[tuple[float, float, float]] = []
            baseline_value = baseline_metrics.get(metric)
            if isinstance(baseline_value, (int, float)) and not isinstance(
                baseline_value, bool
            ):
                samples.append((0.0, 0.0, float(baseline_value)))
            for result in results:
                value = result.get("roles", {}).get(role, {}).get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    samples.append(
                        (
                            float(result["theta_change_deg"]),
                            float(result["phi_change_deg"]),
                            float(value),
                        )
                    )
            fitted = fit_quadratic_response_surface(samples)
            if fitted is not None:
                role_models[metric] = fitted
        if role_models:
            models[role] = role_models
    return {
        "roles": models,
        "interpretation": (
            "Descriptive fixed-grid response surfaces; do not extrapolate beyond "
            "the recorded theta/phi ranges."
        ),
    }


def _write_flat_metrics(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        comparison = result.get("comparison_to_accurate", {})
        for role, metrics in result.get("roles", {}).items():
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                baseline = comparison.get(role, {}).get(metric, {})
                rows.append(
                    {
                        "condition_id": result["condition_id"],
                        "theta_change_deg": result["theta_change_deg"],
                        "phi_change_deg": result["phi_change_deg"],
                        "evaluation_mode": result["evaluation_mode"],
                        "role": role,
                        "metric": metric,
                        "value": float(value),
                        "accurate_value": baseline.get("accurate"),
                        "signed_change_from_accurate": baseline.get("signed_change"),
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
            "theta_change_deg": result["theta_change_deg"],
            "phi_change_deg": result["phi_change_deg"],
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


def run_view_direction_robustness(
    *,
    eval_config: Mapping[str, Any],
    config_path: Path,
    checkpoint: Path,
    checkpoint_choice: str,
    experiment_dir: Path,
    output_override: str | None,
    project_root: Path,
) -> Path:
    plan = resolve_view_direction_robustness_plan(eval_config)
    output_raw = output_override or plan["output_dir"] or eval_config.get(
        "eval_output_dir"
    )
    output_dir = (
        _resolve_relative(str(output_raw), config_path=config_path)
        if output_raw
        else experiment_dir
        / "evaluation_inaccurate_view_direction_robustness"
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
            "Run accurate paper-metric evaluation first, or configure "
            "view_direction_robustness.accurate_baseline_summary. Expected: "
            f"{baseline_path}"
        )
    baseline_roles = {} if baseline is None else baseline.get("roles", {})
    if not isinstance(baseline_roles, dict):
        raise ValueError("The accurate baseline has invalid role metrics.")

    base_config = dict(eval_config)
    base_config.pop("view_direction_robustness", None)
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

    aggregate_path = output_dir / "view_direction_robustness_summary.json"
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
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
            "error_model": "fixed_signed_theta_phi_grid",
            "model_interface": "world2pix4x4_recomputed_from_perturbed_angles",
            "images_changed": False,
            "both_selected_views_perturbed": True,
            "conditions": [list(pair) for pair in plan["conditions"]],
            "visualization_conditions": [
                list(pair) for pair in plan["visualization_conditions"]
            ],
            "require_accurate_baseline": plan["require_accurate_baseline"],
            "record_paper_metrics": plan["record_paper_metrics"],
            "save_prediction_npz_files": True,
            "save_mask_npz_files": plan["save_mask_npz_files"],
        },
        "results": [],
    }
    _write_json(aggregate_path, aggregate)

    expected_cases = baseline_cases
    for theta_change, phi_change in plan["conditions"]:
        identifier = condition_id(theta_change, phi_change)
        condition_output = conditions_dir / identifier
        condition_mode = (
            "visualisation"
            if (theta_change, phi_change) in plan["visualization_conditions"]
            else "metric"
        )
        child = dict(base_config)
        child.update(
            {
                "evaluation_mode": condition_mode,
                "view_direction_robustness_condition": True,
                "record_inaccurate_view_direction_paper_metrics": plan[
                    "record_paper_metrics"
                ],
                "paper_metric_save_masks": plan["save_mask_npz_files"],
                "evaluation_view_directions": {
                    "accurate": False,
                    "theta_change_deg": theta_change,
                    "phi_change_deg": phi_change,
                },
                "eval_output_dir": str(condition_output),
            }
        )
        child_path = config_dir / f"{identifier}.json"
        _write_json(child_path, child)
        command = [sys.executable, "-m", "src.eval_npz", "--config", str(child_path)]
        completed = subprocess.run(command, cwd=project_root, check=False)
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
                f"View-direction condition {identifier} failed with exit code "
                f"{completed.returncode}. Partial results: {aggregate_path}"
            )

        performance_path = condition_output / "performance_summary.json"
        performance = _load_json(
            performance_path, label=f"Performance summary for {identifier}"
        )
        if baseline is not None:
            _validate_condition(
                performance,
                baseline,
                identifier=identifier,
                theta_change_deg=theta_change,
                phi_change_deg=phi_change,
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
        paper_metric_path = (
            condition_output / "metrics" / "paper_metric_per_case.json"
        )
        if plan["record_paper_metrics"]:
            if not paper_metric_path.is_file():
                raise RuntimeError(
                    f"Condition {identifier} did not write {paper_metric_path}."
                )
            for metric in (
                "paper_mask_dice_3d",
                "paper_mask_cldice_3d",
                "paper_mask_ssim_3d",
            ):
                if metric not in roles["final"]:
                    raise RuntimeError(
                        f"Condition {identifier} did not record {metric}."
                    )
        timing = performance.get("timing", {})
        result = {
            "condition_id": identifier,
            "theta_change_deg": theta_change,
            "phi_change_deg": phi_change,
            "angular_offset_magnitude_deg": math.hypot(
                theta_change, phi_change
            ),
            "evaluation_mode": condition_mode,
            "output_dir": str(condition_output),
            "performance_summary": str(performance_path),
            "paper_metric_per_case": (
                str(paper_metric_path)
                if plan["record_paper_metrics"]
                else None
            ),
            "roles": roles,
            "timing": timing,
            "comparison_to_accurate": (
                _numeric_comparison(roles, baseline_roles)
                if baseline is not None
                else None
            ),
        }
        aggregate["results"].append(result)
        _write_json(aggregate_path, aggregate)

    aggregate["status"] = "complete"
    aggregate["num_conditions"] = len(aggregate["results"])
    aggregate["case_count"] = len(expected_cases or [])
    aggregate["quadratic_response_surfaces"] = _response_surfaces(
        aggregate["results"], baseline_roles
    )
    _write_json(aggregate_path, aggregate)
    _write_flat_metrics(
        output_dir / "view_direction_robustness_metrics.csv",
        aggregate["results"],
    )
    _write_timing_csv(
        output_dir / "view_direction_robustness_timing.csv",
        aggregate["results"],
    )
    return aggregate_path
