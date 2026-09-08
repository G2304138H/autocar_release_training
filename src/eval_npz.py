"""Run config-driven AutoCAR evaluation on validation and/or test NPZ cases.

The runner mirrors the paired evaluation interface used by the parametric
baseline while keeping AutoCAR's existing Stage-2 volume and metric contracts:

* a JSON configuration selects the checkpoint, data, split, and fixed views;
* every selected case is reconstructed from its input views;
* the dense probability volume is saved as a self-describing compressed NPZ;
* paper-metric mode writes per-case CSV/JSON and aggregate summaries; and
* visualisation mode additionally writes an input-view/volume monitor bundle;
* inaccurate-view mode sweeps fixed signed camera-angle errors while keeping
  the two source projection images unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.dataset.case_ids import normalize_case_id
from src.dataset.case_splits import load_case_splits
from src.dataset.stage2_npz import Stage2NPZDataset
from src.evaluate_npz import evaluate_case
from src.geometry.projection_geometry import ProjectionGeometry
from src.geometry.voxel_grid import VoxelGrid, resample_binary_volume_nearest
from src.metrics import compute_volume_metrics, masked_ssim_3d
from src.modules.autocar_voxel_pl import AutoCARVoxelLit
from src.modules.sparse_utils import rasterize_sparse_channel
from src.predict_npz import _aggregate_metrics, _device


_EVALUATION_SPLITS = frozenset({"val", "test", "val_test"})
_EVALUATION_MODES = frozenset({"visualisation", "metric", "paper_metric"})
_SPLIT_LABELS = {"val": "validation", "test": "test"}
_CHECKPOINT_FILENAMES = {
    "best": "best.ckpt",
    "last": "last.ckpt",
    "latest": "last.ckpt",
}
_PAPER_METRIC_SHAPE_ZYX = (128, 128, 128)


@dataclass(frozen=True)
class EvaluationOptions:
    config_path: Path
    checkpoint: Path
    checkpoint_choice: str
    projection_source: Path
    voxel_source: Path
    split_json: Path
    output_dir: Path
    evaluation_mode: str
    eval_split: str
    case_ids: tuple[str, ...]
    case_splits: Mapping[str, str]
    case_id_mode: str
    expected_imager_pixel_spacing_mm: float | None
    fallback_imager_pixel_spacing_mm: float | None
    fallback_sid_mm: float | None
    view_indices: tuple[int, int]
    view_labels: tuple[str, str] | None
    view_direction_options: Mapping[str, Any]
    compute_paper_metrics: bool
    device: str
    precision: str
    output_dtype: str
    prediction_threshold: float
    paper_metric_save_masks: bool
    ssim_window_size: int
    ssim_chunk_depth: int
    ground_truth_origin_xyz_mm: tuple[float, float, float] | None
    source_to_isocenter_mm: float
    max_visualizations: int
    visualization_gif_frames: int
    visualization_gif_fps: int
    visualization_max_points: int
    overwrite: bool
    resolved_config: Mapping[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation configuration does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation configuration must contain a JSON object: {path}")
    return value


def _resolve_path(raw: Any, *, config_path: Path, label: str) -> Path:
    if raw is None or not str(raw).strip():
        raise ValueError(f"{label} must be set in the evaluation configuration.")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_checkpoint(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    cli_checkpoint: str | None,
) -> tuple[Path, str]:
    raw_checkpoint = cli_checkpoint or config.get(
        "checkpoint_path", config.get("checkpoint")
    )
    choice = str(config.get("checkpoint_choice", "best")).strip().lower()
    choice = {"latest": "last"}.get(choice, choice)
    if raw_checkpoint is not None and str(raw_checkpoint).strip():
        checkpoint_text = str(raw_checkpoint).strip()
        if checkpoint_text.lower() in _CHECKPOINT_FILENAMES:
            choice = {"latest": "last"}.get(
                checkpoint_text.lower(), checkpoint_text.lower()
            )
        else:
            checkpoint = _resolve_path(
                checkpoint_text,
                config_path=config_path,
                label="checkpoint_path",
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint}")
            return checkpoint, "explicit"

    if choice not in {"best", "last"}:
        raise ValueError("checkpoint_choice must be 'best' or 'last'.")
    experiment_dir = _resolve_path(
        config.get("experiment_dir"),
        config_path=config_path,
        label="experiment_dir",
    )
    filename = _CHECKPOINT_FILENAMES[choice]
    candidates = [
        experiment_dir / "checkpoints" / filename,
        experiment_dir / filename,
    ]
    if choice == "best":
        candidates.extend(
            path
            for path in sorted((experiment_dir / "checkpoints").glob("*.ckpt"))
            if path.name != "last.ckpt"
        )
    existing = list(
        dict.fromkeys(path.resolve() for path in candidates if path.is_file())
    )
    if len(existing) != 1:
        searched = ", ".join(str(path) for path in candidates)
        if not existing:
            raise FileNotFoundError(
                f"Could not resolve checkpoint_choice={choice!r}; searched: {searched}"
            )
        raise ValueError(
            f"checkpoint_choice={choice!r} is ambiguous; set checkpoint_path explicitly."
        )
    return existing[0], choice


def _evaluation_mode(raw: Any) -> str:
    value = str(raw or "paper_metric").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    aliases = {
        "visualization": "visualisation",
        "visualization_demo": "visualisation",
        "visualisation_demo": "visualisation",
        "evaluation": "visualisation",
        "metrics": "metric",
        "metrics_only": "metric",
        "paper_metrics": "paper_metric",
    }
    value = aliases.get(value, value)
    if value not in _EVALUATION_MODES:
        raise ValueError(
            "evaluation_mode must be 'visualisation', 'metric', or "
            f"'paper_metric', got {raw!r}."
        )
    return value


def _eval_split(raw: Any) -> str:
    value = str(raw or "val_test").strip().lower()
    value = value.replace("-", "_").replace("+", "_")
    value = {
        "validation": "val",
        "validation_test": "val_test",
        "test_validation": "val_test",
    }.get(value, value)
    if value not in _EVALUATION_SPLITS:
        raise ValueError("eval_split must be 'val', 'test', or 'val_test'.")
    return value


def resolve_evaluation_view_directions(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fixed evaluation-only camera-angle perturbations."""

    raw = config.get("evaluation_view_directions", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("evaluation_view_directions must be a JSON object.")
    legacy_keys = {
        "max_theta_change_deg",
        "max_phi_change_deg",
        "seed",
    }.intersection(raw)
    if legacy_keys:
        raise ValueError(
            "Random maximum-bounded view-direction changes are not supported; "
            "use fixed theta_change_deg and phi_change_deg values."
        )
    options = dict(raw)
    options.setdefault("accurate", True)
    options.setdefault("theta_change_deg", 0.0)
    options.setdefault("phi_change_deg", 0.0)
    if not isinstance(options["accurate"], bool):
        raise ValueError("evaluation_view_directions.accurate must be boolean.")
    for key in ("theta_change_deg", "phi_change_deg"):
        raw_value = options[key]
        if isinstance(raw_value, bool):
            raise ValueError(
                f"evaluation_view_directions.{key} must be a finite number."
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"evaluation_view_directions.{key} must be a finite number."
            )
        options[key] = value
    has_change = bool(
        options["theta_change_deg"] != 0.0
        or options["phi_change_deg"] != 0.0
    )
    if options["accurate"] and has_change:
        raise ValueError(
            "Accurate evaluation view directions require theta_change_deg=0 "
            "and phi_change_deg=0."
        )
    if not options["accurate"] and not has_change:
        raise ValueError(
            "Inaccurate evaluation view directions require a non-zero fixed "
            "theta_change_deg or phi_change_deg."
        )
    options["distribution"] = "fixed_per_view"
    options["model_interface"] = "world2pix4x4"
    return options


def _case_limit(raw: Any) -> int | None:
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "all"):
        return None
    if isinstance(raw, bool):
        raise ValueError("num_eval_cases must be a positive integer, null, or 'all'.")
    value = _integer(raw, label="num_eval_cases")
    if value < 1:
        raise ValueError("num_eval_cases must be >= 1, null, or 'all'.")
    return value


def _visualization_limit(raw: Any, *, num_cases: int) -> int:
    if raw is None:
        return min(20, num_cases)
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return num_cases
    if isinstance(raw, bool):
        raise ValueError(
            "max_visualizations must be a non-negative integer, null, or 'all'."
        )
    value = _integer(raw, label="max_visualizations")
    if value < 0:
        raise ValueError(
            "max_visualizations must be >= 0, null, or 'all'."
        )
    return min(value, num_cases)


def _integer(raw: Any, *, label: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(numeric)


def _parse_case_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [str(value).strip() for value in raw if str(value).strip()]
    else:
        values = [str(raw).strip()]
    if len(set(values)) != len(values):
        raise ValueError("eval_case_ids must not contain duplicates.")
    return values


def _selected_split_cases(
    splits: Mapping[str, Sequence[str]],
    *,
    eval_split: str,
    requested_case_ids: Sequence[str],
    limit: int | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    split_names = ("val", "test") if eval_split == "val_test" else (eval_split,)
    available = [
        (str(case_id), split_name)
        for split_name in split_names
        for case_id in splits[split_name]
    ]
    split_by_case = {case_id: split_name for case_id, split_name in available}
    if requested_case_ids:
        missing = [
            case_id
            for case_id in requested_case_ids
            if case_id not in split_by_case
        ]
        if missing:
            raise ValueError(
                "eval_case_ids are outside the selected evaluation split or missing "
                "from the split manifest: " + ", ".join(missing)
            )
        selected = [
            (case_id, split_by_case[case_id]) for case_id in requested_case_ids
        ]
        if limit is not None and limit < len(selected):
            raise ValueError(
                "num_eval_cases would discard explicitly requested eval_case_ids."
            )
    else:
        selected = available
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Evaluation selected no validation or test cases.")
    return (
        tuple(case_id for case_id, _ in selected),
        {case_id: _SPLIT_LABELS[split_name] for case_id, split_name in selected},
    )


def _finite_three(raw: Any, *, label: str) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be null or contain three finite values.")
    return tuple(float(value) for value in values)


def resolve_evaluation_options(
    config_path: str | Path,
    *,
    cli_checkpoint: str | None = None,
    cli_output_dir: str | None = None,
    cli_split: str | None = None,
    cli_case_ids: Sequence[str] | None = None,
    cli_max_cases: int | None = None,
    metrics_only: bool = False,
) -> EvaluationOptions:
    config_path = Path(config_path).expanduser().resolve()
    config = _load_json_object(config_path)
    checkpoint, checkpoint_choice = _resolve_checkpoint(
        config,
        config_path=config_path,
        cli_checkpoint=cli_checkpoint,
    )
    projection_source = _resolve_path(
        config.get("projection_source", config.get("projections")),
        config_path=config_path,
        label="projection_source",
    )
    voxel_source = _resolve_path(
        config.get("voxel_source", config.get("voxels")),
        config_path=config_path,
        label="voxel_source",
    )
    split_json = _resolve_path(
        config.get("split_json_path", config.get("split_json")),
        config_path=config_path,
        label="split_json_path",
    )
    for path, label in (
        (projection_source, "projection_source"),
        (voxel_source, "voxel_source"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not split_json.is_file():
        raise FileNotFoundError(f"split_json_path does not exist: {split_json}")

    output_value = cli_output_dir or config.get("eval_output_dir")
    output_dir = _resolve_path(
        output_value,
        config_path=config_path,
        label="eval_output_dir",
    )
    mode = (
        "metric"
        if metrics_only
        else _evaluation_mode(config.get("evaluation_mode"))
    )
    selected_split = _eval_split(cli_split or config.get("eval_split"))
    case_id_mode = str(config.get("case_id_mode", "literal"))
    requested_case_ids = [
        normalize_case_id(value, mode=case_id_mode)
        for value in _parse_case_ids(
            list(cli_case_ids) if cli_case_ids else config.get("eval_case_ids")
        )
    ]
    if len(set(requested_case_ids)) != len(requested_case_ids):
        raise ValueError(
            "eval_case_ids contain duplicates after case-ID normalization."
        )
    case_ids, case_splits = _selected_split_cases(
        load_case_splits(split_json, case_id_mode=case_id_mode),
        eval_split=selected_split,
        requested_case_ids=requested_case_ids,
        limit=_case_limit(
            cli_max_cases
            if cli_max_cases is not None
            else config.get("num_eval_cases", config.get("max_cases"))
        ),
    )

    raw_view_indices = config.get("evaluation_view_indices", (0, 6))
    if not isinstance(raw_view_indices, (list, tuple)):
        raise ValueError("evaluation_view_indices must contain exactly two integers.")
    view_indices = tuple(
        _integer(value, label="evaluation_view_indices")
        for value in raw_view_indices
    )
    if len(view_indices) != 2 or len(set(view_indices)) != 2 or min(view_indices) < 0:
        raise ValueError(
            "evaluation_view_indices must contain two distinct non-negative integers."
        )
    raw_eval_num_views = config.get("eval_num_views", len(view_indices))
    if isinstance(raw_eval_num_views, (list, tuple)):
        raise ValueError(
            "AutoCAR checkpoints have a fixed input width; eval_num_views sweeps "
            "are not supported. Run one compatible checkpoint/config per view count."
        )
    if _integer(raw_eval_num_views, label="eval_num_views") != len(view_indices):
        raise ValueError(
            "eval_num_views must match the number of evaluation_view_indices."
        )
    eval_view_selection = str(
        config.get("eval_view_selection", "fixed")
    ).strip().lower()
    if eval_view_selection != "fixed":
        raise ValueError(
            "AutoCAR evaluation requires eval_view_selection='fixed'."
        )
    raw_view_labels = config.get(
        "evaluation_view_labels",
        ("RAO 25, CAU 35", "LAO 5, CRA 40"),
    )
    view_labels: tuple[str, str] | None
    if raw_view_labels is None:
        view_labels = None
    else:
        if not isinstance(raw_view_labels, (list, tuple)):
            raise ValueError(
                "evaluation_view_labels must be null or contain exactly two strings."
            )
        labels = tuple(str(value).strip() for value in raw_view_labels)
        if len(labels) != 2 or any(not label for label in labels):
            raise ValueError(
                "evaluation_view_labels must be null or contain exactly two non-empty strings."
            )
        view_labels = labels

    precision = str(config.get("precision", "32"))
    if precision not in {"16-mixed", "32"}:
        raise ValueError("precision must be '16-mixed' or '32'.")
    output_dtype = str(config.get("output_dtype", "float16"))
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("output_dtype must be 'float16' or 'float32'.")
    raw_threshold = config.get(
        "paper_metric_volume_threshold",
        config.get("prediction_threshold", 0.5),
    )
    if isinstance(raw_threshold, bool):
        raise ValueError("prediction_threshold must lie in [0,1].")
    threshold = float(
        raw_threshold
    )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("prediction_threshold must lie in [0,1].")
    window_size = _integer(
        config.get(
            "paper_metric_ssim_window_size",
            config.get("ssim_window_size", 7),
        ),
        label="ssim_window_size",
    )
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("ssim_window_size must be an odd integer >= 3.")
    chunk_depth = _integer(
        config.get("ssim_chunk_depth", 8),
        label="ssim_chunk_depth",
    )
    if chunk_depth < 1:
        raise ValueError("ssim_chunk_depth must be a positive integer.")
    source_to_isocenter_mm = float(config.get("source_to_isocenter_mm", 750.0))
    if not math.isfinite(source_to_isocenter_mm) or source_to_isocenter_mm <= 0:
        raise ValueError("source_to_isocenter_mm must be finite and positive.")
    raw_expected_pixel_spacing = config.get("expected_imager_pixel_spacing_mm")
    expected_pixel_spacing = (
        None
        if raw_expected_pixel_spacing is None
        else float(raw_expected_pixel_spacing)
    )
    if expected_pixel_spacing is not None and (
        not math.isfinite(expected_pixel_spacing) or expected_pixel_spacing <= 0
    ):
        raise ValueError(
            "expected_imager_pixel_spacing_mm must be null or finite and positive."
        )
    raw_fallback_pixel_spacing = config.get("fallback_imager_pixel_spacing_mm")
    fallback_pixel_spacing = (
        None
        if raw_fallback_pixel_spacing is None
        else float(raw_fallback_pixel_spacing)
    )
    if fallback_pixel_spacing is not None and (
        not math.isfinite(fallback_pixel_spacing) or fallback_pixel_spacing <= 0
    ):
        raise ValueError(
            "fallback_imager_pixel_spacing_mm must be null or finite and positive."
        )
    if (
        fallback_pixel_spacing is not None
        and expected_pixel_spacing is not None
        and not math.isclose(
            fallback_pixel_spacing,
            expected_pixel_spacing,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    ):
        raise ValueError(
            "fallback_imager_pixel_spacing_mm must match "
            "expected_imager_pixel_spacing_mm when both are configured."
        )
    raw_fallback_sid = config.get("fallback_sid_mm")
    fallback_sid = None if raw_fallback_sid is None else float(raw_fallback_sid)
    if fallback_sid is not None and (
        not math.isfinite(fallback_sid) or fallback_sid <= source_to_isocenter_mm
    ):
        raise ValueError(
            "fallback_sid_mm must be null, finite, and greater than "
            "source_to_isocenter_mm."
        )
    if config.get("save_prediction_npz_files", True) is not True:
        raise ValueError(
            "save_prediction_npz_files must be true for the AutoCAR evaluation runner."
        )
    view_direction_options = resolve_evaluation_view_directions(config)
    robustness_condition = config.get(
        "view_direction_robustness_condition", False
    )
    if not isinstance(robustness_condition, bool):
        raise ValueError("view_direction_robustness_condition must be boolean.")
    record_robustness_paper_metrics = config.get(
        "record_inaccurate_view_direction_paper_metrics", False
    )
    if not isinstance(record_robustness_paper_metrics, bool):
        raise ValueError(
            "record_inaccurate_view_direction_paper_metrics must be boolean."
        )
    if record_robustness_paper_metrics and not robustness_condition:
        raise ValueError(
            "record_inaccurate_view_direction_paper_metrics=true is reserved "
            "for child runs created by evaluation_mode='inaccurate_view_direction'."
        )
    if robustness_condition and view_direction_options["accurate"]:
        raise ValueError(
            "A view-direction robustness child condition must use inaccurate "
            "evaluation_view_directions."
        )
    compute_paper_metrics = bool(
        mode == "paper_metric" or record_robustness_paper_metrics
    )
    paper_metric_save_masks = config.get("paper_metric_save_masks", True)
    if not isinstance(paper_metric_save_masks, bool):
        raise ValueError("paper_metric_save_masks must be boolean.")

    max_visualizations = 0
    if mode == "visualisation":
        max_visualizations = _visualization_limit(
            config.get("max_visualizations"),
            num_cases=len(case_ids),
        )
    gif_frames = _integer(
        config.get("visualization_gif_frames", 24),
        label="visualization_gif_frames",
    )
    gif_fps = _integer(
        config.get("visualization_gif_fps", 6),
        label="visualization_gif_fps",
    )
    max_points = _integer(
        config.get("visualization_max_points", 20_000),
        label="visualization_max_points",
    )
    if gif_frames < 0 or gif_fps < 1 or max_points < 1:
        raise ValueError(
            "visualization_gif_frames must be >= 0 and visualization_gif_fps/"
            "visualization_max_points must be >= 1."
        )

    resolved_config = dict(config)
    resolved_config.update(
        {
            "evaluation_config": str(config_path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_choice_resolved": checkpoint_choice,
            "projection_source": str(projection_source),
            "voxel_source": str(voxel_source),
            "split_json_path": str(split_json),
            "eval_output_dir": str(output_dir),
            "evaluation_mode": mode,
            "eval_split": selected_split,
            "eval_case_ids": list(case_ids),
            "case_id_mode": case_id_mode,
            "expected_imager_pixel_spacing_mm": expected_pixel_spacing,
            "fallback_imager_pixel_spacing_mm": fallback_pixel_spacing,
            "fallback_sid_mm": fallback_sid,
            "evaluation_view_indices": list(view_indices),
            "evaluation_view_labels": None if view_labels is None else list(view_labels),
            "evaluation_view_directions": view_direction_options,
            "eval_num_views": len(view_indices),
            "eval_view_selection": "fixed",
            "save_prediction_npz_files": True,
            "max_visualizations_effective": max_visualizations,
            "prediction_threshold": threshold,
            "paper_metric_save_masks": paper_metric_save_masks,
            "compute_paper_metrics": compute_paper_metrics,
            "ssim_window_size": window_size,
            "ssim_chunk_depth": chunk_depth,
        }
    )
    overwrite = config.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean.")
    return EvaluationOptions(
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_choice=checkpoint_choice,
        projection_source=projection_source,
        voxel_source=voxel_source,
        split_json=split_json,
        output_dir=output_dir,
        evaluation_mode=mode,
        eval_split=selected_split,
        case_ids=case_ids,
        case_splits=case_splits,
        case_id_mode=case_id_mode,
        expected_imager_pixel_spacing_mm=expected_pixel_spacing,
        fallback_imager_pixel_spacing_mm=fallback_pixel_spacing,
        fallback_sid_mm=fallback_sid,
        view_indices=(view_indices[0], view_indices[1]),
        view_labels=view_labels,
        view_direction_options=view_direction_options,
        compute_paper_metrics=compute_paper_metrics,
        device=str(config.get("device", "auto")),
        precision=precision,
        output_dtype=output_dtype,
        prediction_threshold=threshold,
        paper_metric_save_masks=paper_metric_save_masks,
        ssim_window_size=window_size,
        ssim_chunk_depth=chunk_depth,
        ground_truth_origin_xyz_mm=_finite_three(
            config.get("ground_truth_origin_xyz_mm"),
            label="ground_truth_origin_xyz_mm",
        ),
        source_to_isocenter_mm=source_to_isocenter_mm,
        max_visualizations=max_visualizations,
        visualization_gif_frames=gif_frames,
        visualization_gif_fps=gif_fps,
        visualization_max_points=max_points,
        overwrite=overwrite,
        resolved_config=resolved_config,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_dense(
    model: AutoCARVoxelLit,
    sample: Mapping[str, Any],
    *,
    device: torch.device,
    use_mixed_precision: bool,
) -> tuple[np.ndarray, float]:
    masks = sample["images"][None, :, 0].to(device=device, dtype=torch.float32)
    matrices = sample["world2pix4x4"][None].to(device=device, dtype=torch.float32)
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_mixed_precision
        else nullcontext()
    )
    _synchronize(device)
    started = time.perf_counter()
    with precision_context:
        prediction, _ = model.recon_net(masks, matrices)
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    projection = model.recon_net.ray_casting
    dense_tensor = rasterize_sparse_channel(
        prediction,
        backend=model.sparse_backend,
        spatial_shape_xyz=projection.spatial_shape_xyz,
        batch_size=1,
        output_device="cpu",
    )[0]
    if dense_tensor.dtype == torch.bfloat16:
        dense_tensor = dense_tensor.float()
    return dense_tensor.detach().numpy(), elapsed_ms


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _array_like(reference: Any, value: np.ndarray) -> Any:
    """Return ``value`` using the tensor/array representation of ``reference``."""

    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(
            value,
            dtype=reference.dtype,
            device=reference.device,
        )
    reference_array = np.asarray(reference)
    return np.ascontiguousarray(value, dtype=reference_array.dtype)


def _wrapped_degrees(values: np.ndarray) -> np.ndarray:
    wrapped = (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return wrapped.astype(np.float32)


def evaluation_model_camera_sample(
    sample: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Return one sample with evaluation-only model camera matrices.

    The projection images and source metadata are retained.  For an inaccurate
    condition, the model-facing camera geometry is regenerated after adding the
    same fixed signed theta/phi changes to both selected views.
    """

    original_theta = np.asarray(_numpy(sample["theta_deg"]), dtype=np.float64)
    original_phi = np.asarray(_numpy(sample["phi_deg"]), dtype=np.float64)
    if original_theta.ndim != 1 or original_phi.shape != original_theta.shape:
        raise ValueError(
            "Evaluation theta_deg and phi_deg must be matching one-dimensional arrays."
        )
    theta_change = 0.0 if options["accurate"] else float(
        options["theta_change_deg"]
    )
    phi_change = 0.0 if options["accurate"] else float(
        options["phi_change_deg"]
    )
    evaluated_theta = _wrapped_degrees(original_theta + theta_change)
    evaluated_phi = _wrapped_degrees(original_phi + phi_change)
    result = dict(sample)
    result["original_world2pix4x4"] = sample["world2pix4x4"]

    if not options["accurate"]:
        geometry = ProjectionGeometry.from_angles(
            theta_deg=evaluated_theta,
            phi_deg=evaluated_phi,
            image_dim=int(np.asarray(_numpy(sample["image_dim"])).item()),
            sid_mm=float(np.asarray(_numpy(sample["sid_mm"])).item()),
            pixel_spacing_mm=float(
                np.asarray(_numpy(sample["imager_pixel_spacing_mm"])).item()
            ),
            source_to_isocenter_mm=float(
                np.asarray(_numpy(sample["source_to_isocenter_mm"])).item()
            ),
        )
        replacements = {
            "world2pix4x4": geometry.world2pix4x4,
            "camera_source_xyz_mm": geometry.source_xyz_mm,
            "detector_center_xyz_mm": geometry.detector_center_xyz_mm,
            "detector_x_xyz": geometry.detector_x_xyz,
            "detector_y_xyz": geometry.detector_y_xyz,
            "view_directions_world": geometry.view_directions_world,
        }
        for key, value in replacements.items():
            result[key] = _array_like(sample[key], value)
        if geometry.num_views == 2:
            cosine = float(
                np.dot(
                    geometry.view_directions_world[0],
                    geometry.view_directions_world[1],
                )
            )
            evaluated_pair_angle = float(
                np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
        else:
            evaluated_pair_angle = None
    else:
        evaluated_pair_angle = (
            float(np.asarray(_numpy(sample["pair_angle_deg"])).item())
            if sample.get("pair_angle_deg") is not None
            else None
        )

    result["evaluated_theta_deg"] = _array_like(
        sample["theta_deg"], evaluated_theta
    )
    result["evaluated_phi_deg"] = _array_like(sample["phi_deg"], evaluated_phi)
    result["theta_change_deg"] = _array_like(
        sample["theta_deg"],
        np.full(original_theta.shape, theta_change, dtype=np.float32),
    )
    result["phi_change_deg"] = _array_like(
        sample["phi_deg"],
        np.full(original_phi.shape, phi_change, dtype=np.float32),
    )
    result["view_directions_accurate"] = bool(options["accurate"])
    result["evaluated_pair_angle_deg"] = evaluated_pair_angle

    selected_indices = np.asarray(_numpy(sample["view_indices"]), dtype=np.int64)
    records = [
        {
            "model_view_position": int(position),
            "selected_view_index": int(selected_indices[position]),
            "original_theta_deg": float(original_theta[position]),
            "theta_change_deg": theta_change,
            "evaluated_theta_deg": float(evaluated_theta[position]),
            "original_phi_deg": float(original_phi[position]),
            "phi_change_deg": phi_change,
            "evaluated_phi_deg": float(evaluated_phi[position]),
        }
        for position in range(len(original_theta))
    ]
    return result, records


def _view_direction_change_statistics(
    records_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    applied: bool,
) -> dict[str, Any]:
    entries = [
        entry
        for case_records in records_by_case.values()
        for entry in case_records
    ]

    def axis_statistics(key: str) -> dict[str, float]:
        values = [float(entry[key]) for entry in entries]
        if not values:
            return {
                "minimum": 0.0,
                "maximum": 0.0,
                "mean_signed": 0.0,
                "mean_absolute": 0.0,
                "maximum_absolute": 0.0,
            }
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean_signed": sum(values) / len(values),
            "mean_absolute": sum(abs(value) for value in values) / len(values),
            "maximum_absolute": max(abs(value) for value in values),
        }

    return {
        "applied": applied,
        "num_cases": len(records_by_case) if applied else 0,
        "num_perturbed_views": len(entries) if applied else 0,
        "theta_change_deg": axis_statistics("theta_change_deg"),
        "phi_change_deg": axis_statistics("phi_change_deg"),
    }


def _aligned_ground_truth(
    sample: Mapping[str, Any],
    *,
    prediction_shape_zyx: Sequence[int],
    bbox_min_xyz_mm: Sequence[float],
    voxel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_grid = VoxelGrid(
        shape_zyx=tuple(int(value) for value in _numpy(sample["gt_volume_zyx"]).shape),
        spacing_xyz_mm=tuple(float(value) for value in _numpy(sample["gt_spacing_xyz_mm"])),
        origin_xyz_mm=tuple(float(value) for value in _numpy(sample["gt_origin_xyz_mm"])),
    )
    target_grid = VoxelGrid(
        shape_zyx=tuple(int(value) for value in prediction_shape_zyx),
        spacing_xyz_mm=(float(voxel_size_mm),) * 3,
        origin_xyz_mm=tuple(float(value) for value in bbox_min_xyz_mm),
    )
    return resample_binary_volume_nearest(
        _numpy(sample["gt_volume_zyx"]) != 0,
        source_grid,
        target_grid,
        target_to_source_offset_xyz_mm=_numpy(
            sample["projection_center_offset_xyz_mm"]
        ),
    )


def _paper_metric_case(
    prediction_zyx: np.ndarray,
    sample: Mapping[str, Any],
    *,
    bbox_min_xyz_mm: Sequence[float],
    voxel_size_mm: float,
    prediction_threshold: float,
    ssim_window_size: int,
    ssim_chunk_depth: int,
) -> dict[str, Any]:
    """Score one prediction on the parametric evaluator's native-FOV 128³ grid."""

    native_mask = _numpy(sample["gt_volume_zyx"]) != 0
    native_spacing = np.asarray(
        _numpy(sample["gt_spacing_xyz_mm"]),
        dtype=np.float64,
    )
    native_origin = np.asarray(
        _numpy(sample["gt_origin_xyz_mm"]),
        dtype=np.float64,
    )
    native_shape_xyz = np.asarray(native_mask.shape[::-1], dtype=np.int64)
    if np.any(native_shape_xyz < 2):
        raise ValueError(
            "paper_metric endpoint-aligned resampling requires every native "
            "volume dimension to be >= 2."
        )
    target_shape_xyz = np.asarray(_PAPER_METRIC_SHAPE_ZYX[::-1], dtype=np.int64)
    target_spacing = (
        native_spacing
        * (native_shape_xyz.astype(np.float64) - 1.0)
        / (target_shape_xyz.astype(np.float64) - 1.0)
    )
    native_first_center = native_origin + 0.5 * native_spacing
    target_origin = native_first_center - 0.5 * target_spacing
    native_grid = VoxelGrid(
        shape_zyx=tuple(int(value) for value in native_mask.shape),
        spacing_xyz_mm=tuple(float(value) for value in native_spacing),
        origin_xyz_mm=tuple(float(value) for value in native_origin),
    )
    target_grid = VoxelGrid(
        shape_zyx=_PAPER_METRIC_SHAPE_ZYX,
        spacing_xyz_mm=tuple(float(value) for value in target_spacing),
        origin_xyz_mm=tuple(float(value) for value in target_origin),
    )
    ground_truth_mask, native_valid = resample_binary_volume_nearest(
        native_mask,
        native_grid,
        target_grid,
    )
    if not np.all(native_valid):
        raise RuntimeError(
            "The endpoint-aligned paper-metric grid unexpectedly left the "
            "native ground-truth field of view."
        )
    prediction_grid = VoxelGrid(
        shape_zyx=tuple(int(value) for value in prediction_zyx.shape),
        spacing_xyz_mm=(float(voxel_size_mm),) * 3,
        origin_xyz_mm=tuple(float(value) for value in bbox_min_xyz_mm),
    )
    center_offset = np.asarray(
        _numpy(sample["projection_center_offset_xyz_mm"]),
        dtype=np.float64,
    )
    predicted_mask, prediction_valid = resample_binary_volume_nearest(
        prediction_zyx >= prediction_threshold,
        prediction_grid,
        target_grid,
        # Target centers are absolute/native; AutoCAR's grid is centered.
        target_to_source_offset_xyz_mm=-center_offset,
    )
    metrics = compute_volume_metrics(
        ground_truth_mask,
        predicted_mask,
        valid_fov_mask=np.ones(_PAPER_METRIC_SHAPE_ZYX, dtype=np.bool_),
        prediction_threshold=0.5,
        ssim_window_size=ssim_window_size,
        ssim_chunk_depth=ssim_chunk_depth,
    )
    vessel_window = ground_truth_mask | predicted_mask
    try:
        paper_masked_ssim = masked_ssim_3d(
            ground_truth_mask,
            predicted_mask,
            mask=vessel_window,
            data_range=1.0,
            window_size=ssim_window_size,
            chunk_depth=ssim_chunk_depth,
        )
    except ValueError:
        paper_masked_ssim = None
    return {
        "predicted_mask_zyx": predicted_mask,
        "ground_truth_mask_zyx": ground_truth_mask,
        "metrics": {
            "paper_mask_dice_3d": metrics["masked_dice_3d"],
            "paper_mask_ssim_3d": metrics["ssim_3d"],
            "paper_mask_masked_ssim_3d": paper_masked_ssim,
            "paper_mask_intersection_voxels": metrics["intersection_voxels"],
            "paper_mask_predicted_foreground_voxels": metrics[
                "prediction_foreground_voxels"
            ],
            "paper_mask_ground_truth_foreground_voxels": metrics[
                "ground_truth_foreground_voxels"
            ],
            "paper_mask_ground_truth_native_foreground_voxels": int(
                np.count_nonzero(native_mask)
            ),
            "paper_mask_prediction_fov_voxels": int(
                np.count_nonzero(prediction_valid)
            ),
        },
        "source_volume_shape_xyz": native_shape_xyz.astype(int).tolist(),
        "source_spacing_xyz_mm": native_spacing.astype(float).tolist(),
        "target_shape_xyz": target_shape_xyz.astype(int).tolist(),
        "target_spacing_xyz_mm": target_spacing.astype(float).tolist(),
        "target_origin_xyz_mm": native_first_center.astype(float).tolist(),
        "projection_center_offset_xyz_mm": center_offset.astype(float).tolist(),
    }


def _save_paper_mask_artifact(
    path: Path,
    result: Mapping[str, Any],
    *,
    case_id: str,
    split: str,
    eval_num_views: int,
) -> None:
    metrics = result["metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        # Match the parametric paper-metric artifact's declared XYZ order.
        predicted_mask=np.asarray(
            result["predicted_mask_zyx"], dtype=np.uint8
        ).transpose(2, 1, 0),
        ground_truth_mask=np.asarray(
            result["ground_truth_mask_zyx"], dtype=np.uint8
        ).transpose(2, 1, 0),
        case_id=np.asarray(case_id),
        split=np.asarray(split),
        prediction_role=np.asarray("final"),
        evaluation_num_views=np.asarray(eval_num_views, dtype=np.int64),
        ground_truth_array_axis_order=np.asarray("XYZ"),
        source_volume_shape=np.asarray(
            result["source_volume_shape_xyz"], dtype=np.int32
        ),
        source_spacing_mm=np.asarray(
            result["source_spacing_xyz_mm"], dtype=np.float32
        ),
        target_voxel_shape=np.asarray(
            result["target_shape_xyz"], dtype=np.int32
        ),
        target_spacing_mm=np.asarray(
            result["target_spacing_xyz_mm"], dtype=np.float64
        ),
        target_origin_mm=np.asarray(
            result["target_origin_xyz_mm"], dtype=np.float64
        ),
        stored_projection_center_offset_mm=np.asarray(
            result["projection_center_offset_xyz_mm"], dtype=np.float32
        ),
        paper_mask_dice_3d=np.asarray(
            metrics["paper_mask_dice_3d"], dtype=np.float64
        ),
        paper_mask_ssim_3d=np.asarray(
            metrics["paper_mask_ssim_3d"], dtype=np.float64
        ),
    )


def _volume_mips(volume_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = np.asarray(volume_zyx)
    return (
        np.max(volume, axis=0),
        np.max(volume, axis=1),
        np.max(volume, axis=2),
    )


def _save_input_views(
    sample: Mapping[str, Any],
    path: Path,
    *,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = _numpy(sample["images"])[:, 0]
    labels = list(sample.get("view_labels", ()))
    indices = [int(value) for value in _numpy(sample["view_indices"])]
    original_theta = _numpy(sample["theta_deg"])
    original_phi = _numpy(sample["phi_deg"])
    evaluated_theta = _numpy(
        sample.get("evaluated_theta_deg", sample["theta_deg"])
    )
    evaluated_phi = _numpy(
        sample.get("evaluated_phi_deg", sample["phi_deg"])
    )
    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), squeeze=False)
    for index, image in enumerate(images):
        axes[0, index].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        label = labels[index] if index < len(labels) else f"view {indices[index]}"
        axes[0, index].set_title(
            f"Input {indices[index]}: {label}\n"
            f"source θ/φ={original_theta[index]:.1f}/{original_phi[index]:.1f}°, "
            f"model θ/φ={evaluated_theta[index]:.1f}/{evaluated_phi[index]:.1f}°"
        )
        axes[0, index].axis("off")
    figure.suptitle(f"AutoCAR input views — case {case_id}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _overlay_rgb(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    overlay = np.zeros((*target.shape, 3), dtype=np.float32)
    overlay[..., 0] = prediction
    overlay[..., 1] = target
    overlay[..., 2] = prediction
    overlay[target & prediction] = 1.0
    return overlay


def _save_volume_comparison(
    prediction_zyx: np.ndarray,
    target_zyx: np.ndarray,
    valid_fov: np.ndarray,
    path: Path,
    *,
    threshold: float,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probability_mips = _volume_mips(prediction_zyx)
    prediction_mips = _volume_mips(
        (prediction_zyx >= threshold) & valid_fov
    )
    target_mips = _volume_mips(target_zyx & valid_fov)
    plane_names = ("axial (XY)", "coronal (XZ)", "sagittal (YZ)")
    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    for index, plane_name in enumerate(plane_names):
        axes[0, index].imshow(probability_mips[index], cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, index].set_title(f"Prediction probability — {plane_name}")
        axes[1, index].imshow(_overlay_rgb(target_mips[index], prediction_mips[index]))
        axes[1, index].set_title(f"GT (green) / prediction (magenta) — {plane_name}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle(f"AutoCAR volume monitor — case {case_id}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _sample_points(mask_zyx: np.ndarray, maximum: int) -> np.ndarray:
    points_zyx = np.argwhere(mask_zyx)
    if len(points_zyx) > maximum:
        indices = np.linspace(0, len(points_zyx) - 1, maximum, dtype=np.int64)
        points_zyx = points_zyx[indices]
    return points_zyx[:, [2, 1, 0]] if len(points_zyx) else np.empty((0, 3))


def _save_3d_overlay_gif(
    prediction_zyx: np.ndarray,
    target_zyx: np.ndarray,
    valid_fov: np.ndarray,
    path: Path,
    *,
    threshold: float,
    frames: int,
    fps: int,
    maximum_points: int,
    case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    predicted_points = _sample_points(
        (prediction_zyx >= threshold) & valid_fov,
        maximum_points,
    )
    target_points = _sample_points(target_zyx & valid_fov, maximum_points)
    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="3d")
    if len(target_points):
        axis.scatter(
            target_points[:, 0],
            target_points[:, 1],
            target_points[:, 2],
            c="#27ae60",
            s=1.0,
            alpha=0.35,
            label="ground truth",
        )
    if len(predicted_points):
        axis.scatter(
            predicted_points[:, 0],
            predicted_points[:, 1],
            predicted_points[:, 2],
            c="#d81b60",
            s=1.0,
            alpha=0.35,
            label="prediction",
        )
    size_z, size_y, size_x = prediction_zyx.shape
    axis.set_xlim(0, max(size_x - 1, 1))
    axis.set_ylim(0, max(size_y - 1, 1))
    axis.set_zlim(0, max(size_z - 1, 1))
    axis.set_box_aspect((size_x, size_y, size_z))
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(f"Case {case_id}: GT / AutoCAR prediction")
    if len(target_points) or len(predicted_points):
        axis.legend(loc="upper right")

    def rotate(frame_index: int):
        axis.view_init(elev=25.0, azim=360.0 * frame_index / frames)
        return (axis,)

    animation = FuncAnimation(figure, rotate, frames=frames, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)


def _save_visualization_bundle(
    sample: Mapping[str, Any],
    prediction_zyx: np.ndarray,
    metrics: Mapping[str, Any],
    path: Path,
    *,
    bbox_min_xyz_mm: Sequence[float],
    voxel_size_mm: float,
    threshold: float,
    gif_frames: int,
    gif_fps: int,
    maximum_points: int,
    prediction_path: Path,
) -> None:
    case_id = str(sample["case_id"])
    target, valid_fov = _aligned_ground_truth(
        sample,
        prediction_shape_zyx=prediction_zyx.shape,
        bbox_min_xyz_mm=bbox_min_xyz_mm,
        voxel_size_mm=voxel_size_mm,
    )
    _save_input_views(sample, path / "input_views.png", case_id=case_id)
    _save_volume_comparison(
        prediction_zyx,
        target,
        valid_fov,
        path / "volume_comparison.png",
        threshold=threshold,
        case_id=case_id,
    )
    gif_name = None
    if gif_frames > 0:
        gif_name = "3d_overlay.gif"
        _save_3d_overlay_gif(
            prediction_zyx,
            target,
            valid_fov,
            path / gif_name,
            threshold=threshold,
            frames=gif_frames,
            fps=gif_fps,
            maximum_points=maximum_points,
            case_id=case_id,
        )
    _write_json(path / "metrics.json", metrics)
    _write_json(
        path / "evaluation_result_manifest.json",
        {
            "case_id": case_id,
            "input_views": "input_views.png",
            "volume_comparison": "volume_comparison.png",
            "3d_overlay": gif_name,
            "metrics": "metrics.json",
            "prediction_npz": str(prediction_path),
            "view_directions": {
                "accurate": bool(sample.get("view_directions_accurate", True)),
                "original_theta_deg": _numpy(sample["theta_deg"]).tolist(),
                "original_phi_deg": _numpy(sample["phi_deg"]).tolist(),
                "theta_change_deg": _numpy(sample["theta_change_deg"]).tolist(),
                "phi_change_deg": _numpy(sample["phi_change_deg"]).tolist(),
                "evaluated_theta_deg": _numpy(
                    sample["evaluated_theta_deg"]
                ).tolist(),
                "evaluated_phi_deg": _numpy(
                    sample["evaluated_phi_deg"]
                ).tolist(),
            },
        },
    )


def _prediction_protocol(model: AutoCARVoxelLit) -> dict[str, Any]:
    projection = model.recon_net.ray_casting
    return {
        "expected_view_count": model.recon_net.expected_view_count,
        "candidate_mode": projection.candidate_mode,
        "distance_sampling": projection.distance_sampling,
        "max_pixel_distance": projection.max_pixel_distance,
        "support_views": projection.support_views,
        "fusion": projection.fusion,
        "include_distance_feature": projection.include_distance_feature,
        "voxel_chunk_size": projection.voxel_chunk_size,
        "projection_pixel_order": projection.projection_pixel_order,
        "bbox_min_xyz_mm": [
            float(value) for value in projection.bbox_min.detach().cpu()
        ],
        "bbox_max_xyz_mm": [
            float(value) for value in projection.bbox_max.detach().cpu()
        ],
        "voxel_size_mm": float(projection.voxel_size),
    }


def _save_prediction_npz(
    path: Path,
    prediction_zyx: np.ndarray,
    sample: Mapping[str, Any],
    *,
    dataset_split: str,
    protocol: Mapping[str, Any],
    output_dtype: str,
) -> None:
    dtype = np.float16 if output_dtype == "float16" else np.float32
    labels = list(sample.get("view_labels", ()))
    original_theta = _numpy(sample["theta_deg"]).astype(np.float32, copy=False)
    original_phi = _numpy(sample["phi_deg"]).astype(np.float32, copy=False)
    evaluated_theta = _numpy(
        sample.get("evaluated_theta_deg", sample["theta_deg"])
    ).astype(np.float32, copy=False)
    evaluated_phi = _numpy(
        sample.get("evaluated_phi_deg", sample["phi_deg"])
    ).astype(np.float32, copy=False)
    theta_change = _numpy(
        sample.get("theta_change_deg", np.zeros_like(original_theta))
    ).astype(np.float32, copy=False)
    phi_change = _numpy(
        sample.get("phi_change_deg", np.zeros_like(original_phi))
    ).astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        prediction_volume_zyx=prediction_zyx.astype(dtype, copy=False),
        case_id=np.asarray(str(sample["case_id"])),
        dataset_split=np.asarray(dataset_split),
        evaluation_role=np.asarray("final"),
        view_indices=_numpy(sample["view_indices"]).astype(np.int64, copy=False),
        pair_angle_deg=np.asarray(float(sample["pair_angle_deg"]), dtype=np.float32),
        evaluated_pair_angle_deg=np.asarray(
            float(sample.get("evaluated_pair_angle_deg", sample["pair_angle_deg"])),
            dtype=np.float32,
        ),
        view_labels=np.asarray(labels),
        view_directions_accurate=np.asarray(
            bool(sample.get("view_directions_accurate", True))
        ),
        original_theta_deg=original_theta,
        original_phi_deg=original_phi,
        theta_change_deg=theta_change,
        phi_change_deg=phi_change,
        evaluated_theta_deg=evaluated_theta,
        evaluated_phi_deg=evaluated_phi,
        evaluated_world2pix4x4=_numpy(sample["world2pix4x4"]).astype(
            np.float32, copy=False
        ),
        original_world2pix4x4=_numpy(
            sample.get("original_world2pix4x4", sample["world2pix4x4"])
        ).astype(np.float32, copy=False),
        volume_axis_order=np.asarray("zyx"),
        bbox_min_xyz_mm=np.asarray(protocol["bbox_min_xyz_mm"], dtype=np.float32),
        bbox_max_xyz_mm=np.asarray(protocol["bbox_max_xyz_mm"], dtype=np.float32),
        voxel_size_mm=np.asarray(protocol["voxel_size_mm"], dtype=np.float32),
    )


def _metric_values(report: Mapping[str, Any]) -> dict[str, float | int | None]:
    excluded = {
        "case_id",
        "split",
        "prediction",
        "visualization_dir",
        "view_indices",
        "view_labels",
        "pair_angle_deg",
        "evaluated_pair_angle_deg",
        "inference_elapsed_ms",
        "processing_elapsed_ms",
    }
    return {
        key: value
        for key, value in report.items()
        if key not in excluded
        and (
            value is None
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
        )
    }


def _mean_standard_error(values: Sequence[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    error = (
        float(np.std(array, ddof=1) / math.sqrt(len(array)))
        if len(array) >= 2
        else None
    )
    return mean, error


def _processing_timing_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    warmup_performed: bool = False,
) -> dict[str, Any]:
    """Summarize synchronized inference and end-to-end case processing times."""

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        inference_mean, inference_se = _mean_standard_error(
            [float(row["inference_elapsed_ms"]) for row in selected]
        )
        processing_mean, processing_se = _mean_standard_error(
            [float(row["processing_elapsed_ms"]) for row in selected]
        )
        return {
            "num_cases": len(selected),
            "mean_inference_elapsed_ms": inference_mean,
            "inference_elapsed_ms_standard_error": inference_se,
            "mean_processing_elapsed_ms": processing_mean,
            "processing_elapsed_ms_standard_error": processing_se,
        }

    summary = summarize(rows)
    summary.update(
        {
            "inference_scope": "model_forward_only_with_device_synchronization",
            "processing_scope": (
                "dataset_item_load_through_prediction_npz_write_and_case_metric_"
                "artifact_generation; excludes_visualization_and_cross_case_"
                "aggregation"
            ),
            "warmup_policy": (
                "one_untimed_model_forward_on_first_selected_case_before_all_"
                "case_timings"
                if warmup_performed
                else "none"
            ),
            "per_case_file": "timings/processing/per_case.csv",
            "by_split": {
                split: summarize(
                    [row for row in rows if str(row["split"]) == split]
                )
                for split in dict.fromkeys(str(row["split"]) for row in rows)
            },
        }
    )
    return summary


def _paper_metric_summary(
    reports: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    intersection = sum(
        int(report["paper_mask_intersection_voxels"]) for report in reports
    )
    predicted = sum(
        int(report["paper_mask_predicted_foreground_voxels"])
        for report in reports
    )
    target = sum(
        int(report["paper_mask_ground_truth_foreground_voxels"])
        for report in reports
    )
    denominator = predicted + target
    dice, dice_se = _mean_standard_error(
        [float(report["paper_mask_dice_3d"]) for report in reports]
    )
    ssim, ssim_se = _mean_standard_error(
        [float(report["paper_mask_ssim_3d"]) for report in reports]
    )
    masked_ssim_values = [
        float(report["paper_mask_masked_ssim_3d"])
        for report in reports
        if report.get("paper_mask_masked_ssim_3d") is not None
    ]
    masked_ssim = (
        _mean_standard_error(masked_ssim_values)
        if masked_ssim_values
        else (None, None)
    )
    return {
        "protocol": protocol,
        "final_model_role": "final",
        "num_cases": len(reports),
        "micro_dice_3d": (
            1.0 if denominator == 0 else 2.0 * intersection / denominator
        ),
        "macro_dice_3d": dice,
        "macro_dice_3d_standard_error": dice_se,
        "macro_dice_3d_num_cases": len(reports),
        "macro_ssim_3d": ssim,
        "macro_ssim_3d_standard_error": ssim_se,
        "macro_ssim_3d_num_cases": len(reports),
        "macro_masked_ssim_3d": masked_ssim[0],
        "macro_masked_ssim_3d_standard_error": masked_ssim[1],
        "macro_masked_ssim_3d_num_cases": len(masked_ssim_values),
    }


def _flat_rows(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                key: value
                for key, value in report.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _role_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = (
        "masked_dice_3d",
        "ssim_3d",
        "masked_ssim_3d",
        "paper_mask_dice_3d",
        "paper_mask_ssim_3d",
        "paper_mask_masked_ssim_3d",
    )
    for name in metric_names:
        values = [
            float(report[name])
            for report in reports
            if report.get(name) is not None
        ]
        if not values:
            continue
        mean, standard_error = _mean_standard_error(values)
        standard_deviation = (
            float(np.std(values, ddof=1)) if len(values) >= 2 else None
        )
        summary[name] = mean
        summary[f"{name}_standard_deviation"] = standard_deviation
        summary[f"{name}_standard_error"] = standard_error
        summary[f"{name}_num_cases"] = len(values)
    return summary


def _save_metric_histogram(reports: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reports and "paper_mask_dice_3d" in reports[0]:
        names = (
            "paper_mask_dice_3d",
            "paper_mask_ssim_3d",
            "paper_mask_masked_ssim_3d",
        )
        labels = ("Paper Dice", "Paper SSIM", "Paper vessel-window SSIM")
    else:
        names = ("masked_dice_3d", "ssim_3d", "masked_ssim_3d")
        labels = ("Masked Dice", "Global SSIM", "Vessel-window SSIM")
    split_names = list(
        dict.fromkeys(str(report["split"]) for report in reports)
    )
    x = np.arange(len(names), dtype=np.float64)
    width = 0.8 / max(len(split_names), 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    for split_index, split_name in enumerate(split_names):
        means = []
        errors = []
        for name in names:
            values = [
                float(report[name])
                for report in reports
                if report["split"] == split_name and report.get(name) is not None
            ]
            mean, error = _mean_standard_error(values) if values else (0.0, None)
            means.append(mean)
            errors.append(0.0 if error is None else error)
        offset = (split_index - (len(split_names) - 1) / 2.0) * width
        axis.bar(x + offset, means, width, yerr=errors, capsize=3, label=split_name)
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Mean score ± standard error")
    axis.set_title("AutoCAR evaluation metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_evaluation(options: EvaluationOptions) -> dict[str, Any]:
    output_dir = options.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"eval_output_dir is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not options.overwrite:
        raise FileExistsError(
            f"Evaluation output directory is not empty: {output_dir}. "
            "Set overwrite=true to replace matching artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", options.resolved_config)

    device = _device(options.device)
    model = AutoCARVoxelLit.load_from_checkpoint(options.checkpoint, map_location="cpu")
    if model.sparse_backend == "spconv" and device.type != "cuda":
        raise RuntimeError(
            "This checkpoint uses the spconv backend, which requires an NVIDIA "
            "CUDA device for evaluation."
        )
    if model.recon_net.expected_view_count != len(options.view_indices):
        raise ValueError(
            f"The checkpoint expects {model.recon_net.expected_view_count} views, "
            f"but evaluation_view_indices contains {len(options.view_indices)}."
        )
    model.to(device)
    model.freeze()
    protocol = _prediction_protocol(model)
    metric_protocol = {
        "voxel_shape": list(_PAPER_METRIC_SHAPE_ZYX[::-1]),
        "target_grid_policy": "full_native_fov_endpoint_aligned",
        "ground_truth_resampling": "nearest_neighbor",
        "prediction_resampling": "threshold_then_nearest_neighbor",
        "prediction_domain": "sigmoid_probability_on_checkpoint_grid",
        "prediction_threshold": options.prediction_threshold,
        "ssim_protocol": "uniform_valid_window_sample_covariance",
        "ssim_window_size": options.ssim_window_size,
        "coordinate_convention": (
            "prediction_array_zyx; grid_coordinates_xyz_mm; native_ground_truth_"
            "array_xyz"
        ),
        "prediction_grid": protocol,
    }
    use_mixed_precision = device.type == "cuda" and options.precision == "16-mixed"
    dataset = Stage2NPZDataset(
        options.projection_source,
        options.voxel_source,
        view_mode="fixed",
        fixed_view_indices=options.view_indices,
        fixed_view_labels=options.view_labels,
        output_type="torch",
        case_ids=options.case_ids,
        case_id_mode=options.case_id_mode,
        expected_imager_pixel_spacing_mm=(
            options.expected_imager_pixel_spacing_mm
        ),
        fallback_imager_pixel_spacing_mm=(
            options.fallback_imager_pixel_spacing_mm
        ),
        fallback_sid_mm=options.fallback_sid_mm,
        gt_origin_xyz_mm=options.ground_truth_origin_xyz_mm,
        source_to_isocenter_mm=options.source_to_isocenter_mm,
    )

    reports: list[dict[str, Any]] = []
    paper_metric_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    view_direction_changes_by_case: dict[str, list[dict[str, float | int]]] = {}
    warmup_performed = options.compute_paper_metrics
    with torch.inference_mode():
        if warmup_performed:
            print("Running one untimed model warmup on the first selected case")
            warmup_sample, _ = evaluation_model_camera_sample(
                dataset[0], options.view_direction_options
            )
            _forward_dense(
                model,
                warmup_sample,
                device=device,
                use_mixed_precision=use_mixed_precision,
            )
            del warmup_sample
        for index in range(len(dataset)):
            processing_started = time.perf_counter()
            source_sample = dataset[index]
            sample, view_direction_records = evaluation_model_camera_sample(
                source_sample, options.view_direction_options
            )
            case_id = str(sample["case_id"])
            view_direction_changes_by_case[case_id] = view_direction_records
            dataset_split = options.case_splits[case_id]
            print(
                f"[{index + 1}/{len(dataset)}] Reconstructing case {case_id} "
                f"({dataset_split})"
            )
            dense, elapsed_ms = _forward_dense(
                model,
                sample,
                device=device,
                use_mixed_precision=use_mixed_precision,
            )
            prediction_path = (
                output_dir / "predictions" / "final" / dataset_split / f"{case_id}.npz"
            )
            _save_prediction_npz(
                prediction_path,
                dense,
                sample,
                dataset_split=dataset_split,
                protocol=protocol,
                output_dtype=options.output_dtype,
            )
            metrics = evaluate_case(
                dense,
                Path(sample["voxel_path"]),
                Path(sample["projection_path"]),
                bbox_min_xyz_mm=protocol["bbox_min_xyz_mm"],
                voxel_size_mm=float(protocol["voxel_size_mm"]),
                ground_truth_origin_xyz_mm=options.ground_truth_origin_xyz_mm,
                prediction_threshold=options.prediction_threshold,
                ssim_window_size=options.ssim_window_size,
                ssim_chunk_depth=options.ssim_chunk_depth,
            )
            report: dict[str, Any] = {
                "case_id": case_id,
                "split": dataset_split,
                "eval_num_views": len(options.view_indices),
                "view_indices": [int(value) for value in _numpy(sample["view_indices"])],
                "view_labels": list(sample.get("view_labels", ())),
                "pair_angle_deg": float(sample["pair_angle_deg"]),
                "evaluated_pair_angle_deg": float(
                    sample.get("evaluated_pair_angle_deg", sample["pair_angle_deg"])
                ),
                "prediction": str(prediction_path),
                "inference_elapsed_ms": elapsed_ms,
                **metrics,
            }
            paper_record: dict[str, Any] | None = None
            if options.compute_paper_metrics:
                paper_result = _paper_metric_case(
                    dense,
                    sample,
                    bbox_min_xyz_mm=protocol["bbox_min_xyz_mm"],
                    voxel_size_mm=float(protocol["voxel_size_mm"]),
                    prediction_threshold=options.prediction_threshold,
                    ssim_window_size=options.ssim_window_size,
                    ssim_chunk_depth=options.ssim_chunk_depth,
                )
                mask_path = None
                if options.paper_metric_save_masks:
                    mask_path = (
                        output_dir
                        / "metrics"
                        / "voxel_masks"
                        / "final"
                        / dataset_split
                        / f"case_{case_id}_{dataset_split}.npz"
                    )
                    _save_paper_mask_artifact(
                        mask_path,
                        paper_result,
                        case_id=case_id,
                        split=dataset_split,
                        eval_num_views=len(options.view_indices),
                    )
                paper_record = {
                    "case_id": case_id,
                    "split": dataset_split,
                    "eval_num_views": len(options.view_indices),
                    "role": "final",
                    "is_final_model_role": True,
                    "mask_artifact": None if mask_path is None else str(mask_path),
                    "prediction": str(prediction_path),
                    "projection": str(sample["projection_path"]),
                    "ground_truth": str(sample["voxel_path"]),
                    **paper_result["metrics"],
                }
                paper_metric_records.append(paper_record)
                report.update(paper_result["metrics"])
            processing_elapsed_ms = (
                time.perf_counter() - processing_started
            ) * 1000.0
            report["processing_elapsed_ms"] = processing_elapsed_ms
            if paper_record is not None:
                paper_record.update(
                    {
                        "inference_elapsed_ms": elapsed_ms,
                        "processing_elapsed_ms": processing_elapsed_ms,
                    }
                )
            if index < options.max_visualizations:
                visualization_path = (
                    output_dir / "visualization" / case_id / dataset_split / "final"
                )
                _save_visualization_bundle(
                    sample,
                    dense,
                    report,
                    visualization_path,
                    bbox_min_xyz_mm=protocol["bbox_min_xyz_mm"],
                    voxel_size_mm=float(protocol["voxel_size_mm"]),
                    threshold=options.prediction_threshold,
                    gif_frames=options.visualization_gif_frames,
                    gif_fps=options.visualization_gif_fps,
                    maximum_points=options.visualization_max_points,
                    prediction_path=prediction_path,
                )
                report["visualization_dir"] = str(visualization_path)
            reports.append(report)
            prediction_records.append(
                {
                    "case_id": case_id,
                    "split": dataset_split,
                    "eval_num_views": len(options.view_indices),
                    "role": "final",
                    "path": str(prediction_path),
                }
            )
            timing_rows.append(
                {
                    "case_id": case_id,
                    "split": dataset_split,
                    "inference_elapsed_ms": elapsed_ms,
                    "processing_elapsed_ms": processing_elapsed_ms,
                }
            )

    timing_summary = _processing_timing_summary(
        timing_rows,
        warmup_performed=warmup_performed,
    )
    combined = _aggregate_metrics(reports)
    combined["split_counts"] = {
        split_name: sum(report["split"] == split_name for report in reports)
        for split_name in dict.fromkeys(report["split"] for report in reports)
    }
    metrics_by_split = {
        split_name: _aggregate_metrics(
            [report for report in reports if report["split"] == split_name]
        )
        for split_name in combined["split_counts"]
    }
    role_summary = _role_summary(reports)
    paper_summary = (
        _paper_metric_summary(paper_metric_records, protocol=metric_protocol)
        if paper_metric_records
        else None
    )
    if paper_summary is not None:
        paper_summary["by_split"] = {
            split_name: _paper_metric_summary(
                [
                    record
                    for record in paper_metric_records
                    if record["split"] == split_name
                ],
                protocol=metric_protocol,
            )
            for split_name in dict.fromkeys(
                str(record["split"]) for record in paper_metric_records
            )
        }
        paper_summary["timing"] = timing_summary
        for split_name, split_summary in paper_summary["by_split"].items():
            split_summary["timing"] = timing_summary["by_split"][split_name]
    performance_cases = [
        {
            "case_id": report["case_id"],
            "split": report["split"],
            "eval_num_views": report["eval_num_views"],
            "timing": {
                "inference_elapsed_ms": report["inference_elapsed_ms"],
                "processing_elapsed_ms": report["processing_elapsed_ms"],
            },
            "final": _metric_values(report),
        }
        for report in reports
    ]
    performance_summary = {
        "schema_version": 1,
        "comparison_condition": {
            "checkpoint": str(options.checkpoint),
            "checkpoint_choice": options.checkpoint_choice,
            "evaluation_mode": options.evaluation_mode,
            "evaluation_split": options.eval_split,
            "eval_num_views": len(options.view_indices),
            "eval_view_selection": "fixed",
            "evaluation_view_indices": list(options.view_indices),
            "view_directions_accurate": bool(
                options.view_direction_options["accurate"]
            ),
            "theta_change_deg": float(
                options.view_direction_options["theta_change_deg"]
            ),
            "phi_change_deg": float(
                options.view_direction_options["phi_change_deg"]
            ),
        },
        "evaluation": {
            **combined,
            "metrics_by_split": metrics_by_split,
            "paper_metric": paper_summary,
            "evaluation_view_directions": dict(options.view_direction_options),
            "view_direction_changes_by_case": view_direction_changes_by_case,
            "view_direction_change_statistics": _view_direction_change_statistics(
                view_direction_changes_by_case,
                applied=not bool(options.view_direction_options["accurate"]),
            ),
        },
        "timing": timing_summary,
        "roles": {"final": role_summary},
        "per_case_metrics_file": "performance_per_case.json",
    }
    _write_json(output_dir / "performance_per_case.json", performance_cases)
    _write_json(output_dir / "performance_summary.json", performance_summary)
    _write_json(
        output_dir / "predictions" / "manifest.json",
        {
            "format": "compressed_npz",
            "volume_key": "prediction_volume_zyx",
            "volume_axis_order": "zyx",
            "roles": ["final"],
            "num_files": len(prediction_records),
            "files": prediction_records,
        },
    )

    inference_rows = [
        {
            "case_id": row["case_id"],
            "split": row["split"],
            "inference_elapsed_ms": row["inference_elapsed_ms"],
        }
        for row in timing_rows
    ]
    _write_csv(
        output_dir / "timings" / "inference" / "per_case.csv",
        inference_rows,
    )
    _write_json(
        output_dir / "timings" / "inference" / "summary.json",
        {
            "num_cases": timing_summary["num_cases"],
            "mean_inference_elapsed_ms": timing_summary[
                "mean_inference_elapsed_ms"
            ],
            "inference_elapsed_ms_standard_error": timing_summary[
                "inference_elapsed_ms_standard_error"
            ],
            "scope": "model_forward_only_with_device_synchronization",
        },
    )
    _write_csv(
        output_dir / "timings" / "processing" / "per_case.csv",
        timing_rows,
    )
    _write_json(
        output_dir / "timings" / "processing" / "summary.json",
        timing_summary,
    )

    if options.evaluation_mode in {"metric", "paper_metric"} or options.compute_paper_metrics:
        metrics_dir = output_dir / "metrics"
        flat_rows = _flat_rows(reports)
        _write_json(metrics_dir / "per_case_metrics.json", reports)
        _write_csv(metrics_dir / "flat_per_case_metrics.csv", flat_rows)
        _write_json(
            metrics_dir / "summary.json",
            {
                "num_cases": len(reports),
                "roles": {"final": role_summary},
                "evaluation": performance_summary["evaluation"],
                "timing": timing_summary,
            },
        )
        _save_metric_histogram(reports, metrics_dir / "comparison_histogram.png")
        if options.compute_paper_metrics:
            assert paper_summary is not None
            _write_json(
                metrics_dir / "paper_metric_per_case.json",
                paper_metric_records,
            )
            _write_csv(
                metrics_dir / "paper_metric_per_case.csv",
                paper_metric_records,
            )
            _write_json(metrics_dir / "paper_metric_summary.json", paper_summary)
            _write_json(
                metrics_dir / "voxel_masks" / "manifest.json",
                {
                    "format": "compressed_npz",
                    "mask_shape_xyz": list(_PAPER_METRIC_SHAPE_ZYX[::-1]),
                    "protocol": metric_protocol,
                    "saved": options.paper_metric_save_masks,
                    "num_files": sum(
                        record["mask_artifact"] is not None
                        for record in paper_metric_records
                    ),
                    "files": [
                        {
                            "case_id": record["case_id"],
                            "split": record["split"],
                            "eval_num_views": record["eval_num_views"],
                            "role": record["role"],
                            "path": record["mask_artifact"],
                        }
                        for record in paper_metric_records
                        if record["mask_artifact"] is not None
                    ],
                },
            )

    split_sha256 = hashlib.sha256(options.split_json.read_bytes()).hexdigest()
    evaluation_record = {
        "evaluation_config": str(options.config_path),
        "checkpoint": str(options.checkpoint),
        "checkpoint_choice": options.checkpoint_choice,
        "projection_source": str(options.projection_source),
        "voxel_source": str(options.voxel_source),
        "mode": options.evaluation_mode,
        "eval_split": options.eval_split,
        "split_json": str(options.split_json),
        "split_json_sha256": split_sha256,
        "case_split_by_case": dict(options.case_splits),
        "case_id_mode": options.case_id_mode,
        "eval_case_ids": list(options.case_ids),
        "num_eval_cases": len(reports),
        "evaluation_view_indices": list(options.view_indices),
        "evaluation_view_labels": (
            None if options.view_labels is None else list(options.view_labels)
        ),
        "evaluation_view_directions": dict(options.view_direction_options),
        "view_direction_changes_by_case": view_direction_changes_by_case,
        "view_direction_change_statistics": _view_direction_change_statistics(
            view_direction_changes_by_case,
            applied=not bool(options.view_direction_options["accurate"]),
        ),
        "expected_imager_pixel_spacing_mm": (
            options.expected_imager_pixel_spacing_mm
        ),
        "fallback_imager_pixel_spacing_mm": (
            options.fallback_imager_pixel_spacing_mm
        ),
        "fallback_sid_mm": options.fallback_sid_mm,
        "max_visualizations": options.max_visualizations,
        "device": str(device),
        "inference_precision": "16-mixed" if use_mixed_precision else "32",
        "output_dtype": options.output_dtype,
        "sparse_backend": model.sparse_backend,
        "projection_protocol": protocol,
        "paper_metric": paper_summary,
        "timing": timing_summary,
        "prediction_npz_export": {
            "enabled": True,
            "roles": ["final"],
            "format": "compressed_npz",
            "directory": str(output_dir / "predictions"),
            "manifest": str(output_dir / "predictions" / "manifest.json"),
            "num_files": len(prediction_records),
        },
        "output_layout": {
            "predictions": "predictions/final/{validation,test}/<case_id>.npz",
            "timings": "timings/inference",
            "processing_timings": "timings/processing",
            "visualization": (
                "visualization/<case_id>/<split>/final" if options.max_visualizations else None
            ),
            "metrics": (
                "metrics"
                if options.evaluation_mode in {"metric", "paper_metric"}
                or options.compute_paper_metrics
                else None
            ),
            "paper_metric_masks": (
                "metrics/voxel_masks/final/<split>/case_<id>_<split>.npz"
                if options.compute_paper_metrics
                and options.paper_metric_save_masks
                else None
            ),
            "performance_summary": "performance_summary.json",
            "performance_per_case": "performance_per_case.json",
        },
    }
    _write_json(output_dir / "evaluation_record.json", evaluation_record)
    print(json.dumps(performance_summary, indent=2, sort_keys=True, allow_nan=False))
    return performance_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Evaluation JSON configuration.")
    parser.add_argument("--checkpoint", help="Override checkpoint_path from the JSON.")
    parser.add_argument("--output_dir", help="Override eval_output_dir from the JSON.")
    parser.add_argument(
        "--case_id",
        action="append",
        dest="case_ids",
        help="Evaluate a specific case ID. Repeat to select multiple cases.",
    )
    parser.add_argument("--split", choices=("val", "test", "val_test"))
    parser.add_argument(
        "--max_cases",
        type=int,
        help="Override num_eval_cases from the JSON.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Override evaluation_mode with metric mode and skip visualizations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    raw_config = _load_json_object(config_path)
    from src.view_direction_robustness_npz import (
        is_view_direction_robustness_mode,
        run_view_direction_robustness,
    )

    if is_view_direction_robustness_mode(raw_config.get("evaluation_mode")):
        if args.metrics_only:
            raise ValueError(
                "--metrics_only cannot replace the inaccurate-view-direction "
                "orchestrator mode."
            )
        sweep_config = dict(raw_config)
        if args.split is not None:
            sweep_config["eval_split"] = args.split
        if args.case_ids:
            sweep_config["eval_case_ids"] = list(args.case_ids)
        if args.max_cases is not None:
            sweep_config["num_eval_cases"] = args.max_cases
        checkpoint, checkpoint_choice = _resolve_checkpoint(
            sweep_config,
            config_path=config_path,
            cli_checkpoint=args.checkpoint,
        )
        experiment_raw = sweep_config.get("experiment_dir")
        if experiment_raw is not None and str(experiment_raw).strip():
            experiment_dir = _resolve_path(
                experiment_raw,
                config_path=config_path,
                label="experiment_dir",
            )
        else:
            experiment_dir = (
                checkpoint.parent.parent
                if checkpoint.parent.name == "checkpoints"
                else checkpoint.parent
            )
        aggregate_path = run_view_direction_robustness(
            eval_config=sweep_config,
            config_path=config_path,
            checkpoint=checkpoint,
            checkpoint_choice=checkpoint_choice,
            experiment_dir=experiment_dir,
            output_override=args.output_dir,
            project_root=Path(__file__).resolve().parents[1],
        )
        print(f"Completed inaccurate view-direction evaluation: {aggregate_path}")
        return 0
    options = resolve_evaluation_options(
        config_path,
        cli_checkpoint=args.checkpoint,
        cli_output_dir=args.output_dir,
        cli_split=args.split,
        cli_case_ids=args.case_ids,
        cli_max_cases=args.max_cases,
        metrics_only=args.metrics_only,
    )
    run_evaluation(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
