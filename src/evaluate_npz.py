"""Evaluate a reconstructed AutoCAR volume against Stage-2 NPZ ground truth.

This entry point is deliberately independent of the training framework and
CUDA.  Both AutoCAR and comparison methods can therefore export a probability
volume and use exactly the same physical alignment and metric implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.geometry.voxel_grid import VoxelGrid, resample_binary_volume_nearest
from src.metrics import compute_volume_metrics, masked_ssim_3d


_MILLIMETRE_UNITS = {
    "mm",
    "millimetre",
    "millimetres",
    "millimeter",
    "millimeters",
}
_METRE_UNITS = {"m", "metre", "metres", "meter", "meters"}


def _three_floats(values: Sequence[str]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Expected exactly three values in XYZ order.")
    result = tuple(float(value) for value in values)
    if not np.isfinite(result).all():
        raise ValueError("XYZ values must be finite.")
    return result


def _metadata_scalar(data: Any, key: str, path: Path) -> Any:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Prediction metadata {key!r} in {path} must be scalar.")
    return value.item()


def _load_prediction(path: Path, key: str) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if path.suffix.lower() == ".npy":
        prediction = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(
                    f"Prediction NPZ {path} has no {key!r} key; available keys: "
                    f"{sorted(data.files)}"
                )
            prediction = np.asarray(data[key])
            if "volume_axis_order" in data:
                axis_order = _scalar_text(
                    _metadata_scalar(data, "volume_axis_order", path)
                ).lower()
                if axis_order not in {"xyz", "zyx"}:
                    raise ValueError(
                        "volume_axis_order must be 'xyz' or 'zyx', got "
                        f"{axis_order!r}."
                    )
                metadata["volume_axis_order"] = axis_order
            for name in ("bbox_min_xyz_mm", "bbox_max_xyz_mm"):
                if name not in data:
                    continue
                value = np.asarray(data[name], dtype=np.float64)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(
                        f"Prediction metadata {name!r} must contain three "
                        "finite XYZ values."
                    )
                metadata[name] = [float(item) for item in value]
            if "voxel_size_mm" in data:
                voxel_size = float(
                    _metadata_scalar(data, "voxel_size_mm", path)
                )
                if not math.isfinite(voxel_size) or voxel_size <= 0.0:
                    raise ValueError(
                        "Prediction metadata voxel_size_mm must be positive "
                        "and finite."
                    )
                metadata["voxel_size_mm"] = voxel_size
            if "case_id" in data:
                case_id = _scalar_text(_metadata_scalar(data, "case_id", path))
                if not case_id:
                    raise ValueError("Prediction metadata case_id cannot be empty.")
                metadata["case_id"] = case_id
            if "view_indices" in data:
                raw_indices = np.asarray(data["view_indices"])
                try:
                    indices = raw_indices.astype(np.int64)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Prediction metadata view_indices must contain integers."
                    ) from error
                if (
                    indices.ndim != 1
                    or indices.size == 0
                    or not np.array_equal(raw_indices, indices)
                ):
                    raise ValueError(
                        "Prediction metadata view_indices must be a non-empty "
                        "one-dimensional integer array."
                    )
                metadata["view_indices"] = [int(item) for item in indices]
    else:
        raise ValueError("Prediction must be a .npy or .npz file.")
    prediction = np.asarray(prediction).squeeze()
    if prediction.ndim != 3:
        raise ValueError(
            f"Prediction must reduce to one ZYX volume, got {prediction.shape}."
        )
    if not (
        np.issubdtype(prediction.dtype, np.number)
        or np.issubdtype(prediction.dtype, np.bool_)
    ):
        raise ValueError("Prediction must contain numeric values.")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values.")
    return np.ascontiguousarray(prediction), metadata


def _scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _length_array_to_mm(values: np.ndarray, units: str, label: str) -> np.ndarray:
    normalised = units.strip().lower()
    if normalised in _MILLIMETRE_UNITS:
        scale = 1.0
    elif normalised in _METRE_UNITS:
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported {label} unit {units!r}.")
    return np.asarray(values, dtype=np.float64) * scale


def _load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "vol" not in data or "spacing" not in data:
            raise KeyError(f"Ground-truth NPZ {path} requires vol and spacing keys.")
        volume_xyz = np.asarray(data["vol"])
        spacing_xyz_mm = np.asarray(data["spacing"], dtype=np.float64)
        if "spacing_units" in data:
            units_value = np.asarray(data["spacing_units"])
            if units_value.shape != ():
                raise ValueError("spacing_units must be scalar when supplied.")
            spacing_xyz_mm = _length_array_to_mm(
                spacing_xyz_mm,
                _scalar_text(units_value.item()),
                "spacing",
            )
    if volume_xyz.ndim != 3 or min(volume_xyz.shape, default=0) <= 0:
        raise ValueError(
            f"Ground-truth vol must be a non-empty XYZ 3-D array, got "
            f"{volume_xyz.shape}."
        )
    if not (
        np.issubdtype(volume_xyz.dtype, np.number)
        or np.issubdtype(volume_xyz.dtype, np.bool_)
    ):
        raise ValueError("Ground-truth vol must contain numeric values.")
    if not np.isfinite(volume_xyz).all():
        raise ValueError("Ground-truth vol contains NaN or infinite values.")
    if (
        spacing_xyz_mm.shape != (3,)
        or not np.isfinite(spacing_xyz_mm).all()
        or not np.all(spacing_xyz_mm > 0)
    ):
        raise ValueError("Ground-truth spacing must be three positive XYZ values.")
    volume_zyx = np.ascontiguousarray((volume_xyz != 0).transpose(2, 1, 0))
    return volume_zyx, spacing_xyz_mm


def _load_center_offset_mm(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "projection_center_offset" not in data:
            raise KeyError(
                f"Projection NPZ {path} requires projection_center_offset."
            )
        offset = np.asarray(data["projection_center_offset"], dtype=np.float64)
        units = None
        if "projection_center_offset_units" in data:
            units_value = np.asarray(data["projection_center_offset_units"])
            if units_value.shape != ():
                raise ValueError(
                    "projection_center_offset_units must be scalar when supplied."
                )
            units = _scalar_text(units_value.item())
        if units is None:
            scale = (
                float(np.asarray(data["input_scale_to_mm"]).item())
                if "input_scale_to_mm" in data
                else 1000.0
            )
        else:
            scale = None
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("projection_center_offset must contain three finite values.")
    if units is not None:
        return _length_array_to_mm(offset, units, "projection center offset")
    if scale is None or not math.isfinite(scale) or scale <= 0:
        raise ValueError("input_scale_to_mm must be finite and positive.")
    return offset * scale


def _optional_case_id(path: Path, *, fallback_to_stem: bool) -> str | None:
    with np.load(path, allow_pickle=False) as data:
        if "case_id" in data:
            value = np.asarray(data["case_id"])
            if value.shape != ():
                raise ValueError(f"case_id in {path} must be scalar.")
            case_id = _scalar_text(value.item())
            if not case_id:
                raise ValueError(f"case_id in {path} cannot be empty.")
            return case_id
    return path.stem if fallback_to_stem else None


def _resolve_prediction_grid(
    prediction: np.ndarray,
    metadata: dict[str, Any],
    *,
    explicit_axis_order: str | None,
    explicit_bbox_min_xyz_mm: Sequence[str] | None,
    explicit_voxel_size_mm: float | None,
) -> tuple[np.ndarray, tuple[float, float, float], float, str]:
    """Resolve and cross-check prediction-array axis/grid metadata."""

    metadata_axis = metadata.get("volume_axis_order")
    if (
        explicit_axis_order is not None
        and metadata_axis is not None
        and explicit_axis_order != metadata_axis
    ):
        raise ValueError(
            f"--prediction-axis-order={explicit_axis_order!r} conflicts with "
            f"NPZ metadata {metadata_axis!r}."
        )
    axis_order = explicit_axis_order or metadata_axis
    if axis_order is None:
        raise ValueError(
            "Prediction axis order is missing. Export NPZ metadata or pass "
            "--prediction-axis-order explicitly."
        )
    if axis_order not in {"xyz", "zyx"}:
        raise ValueError("Prediction axis order must be 'xyz' or 'zyx'.")

    metadata_bbox = metadata.get("bbox_min_xyz_mm")
    if explicit_bbox_min_xyz_mm is None:
        if metadata_bbox is None:
            raise ValueError(
                "Prediction grid origin is missing. Export bbox_min_xyz_mm "
                "metadata or pass --bbox-min-xyz-mm explicitly."
            )
        bbox_min = tuple(float(value) for value in metadata_bbox)
    else:
        bbox_min = _three_floats(explicit_bbox_min_xyz_mm)
        if metadata_bbox is not None and not np.allclose(
            bbox_min, metadata_bbox, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "--bbox-min-xyz-mm conflicts with prediction NPZ metadata."
            )

    metadata_voxel_size = metadata.get("voxel_size_mm")
    if explicit_voxel_size_mm is None:
        if metadata_voxel_size is None:
            raise ValueError(
                "Prediction voxel size is missing. Export voxel_size_mm "
                "metadata or pass --voxel-size-mm explicitly."
            )
        voxel_size = float(metadata_voxel_size)
    else:
        voxel_size = float(explicit_voxel_size_mm)
        if metadata_voxel_size is not None and not math.isclose(
            voxel_size,
            float(metadata_voxel_size),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ValueError("--voxel-size-mm conflicts with prediction NPZ metadata.")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("Prediction voxel size must be positive and finite.")

    shape_xyz = np.asarray(
        prediction.shape if axis_order == "xyz" else prediction.shape[::-1],
        dtype=np.float64,
    )
    metadata_bbox_max = metadata.get("bbox_max_xyz_mm")
    if metadata_bbox_max is not None:
        expected_max = np.asarray(bbox_min) + shape_xyz * voxel_size
        if not np.allclose(
            expected_max,
            metadata_bbox_max,
            rtol=0.0,
            atol=max(1e-6, voxel_size * 1e-5),
        ):
            raise ValueError(
                "Prediction shape/bbox_min/voxel_size are inconsistent with "
                "bbox_max_xyz_mm metadata."
            )

    prediction_zyx = (
        prediction.transpose(2, 1, 0) if axis_order == "xyz" else prediction
    )
    return (
        np.ascontiguousarray(prediction_zyx),
        bbox_min,
        voxel_size,
        axis_order,
    )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    result = np.empty(values.shape, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def evaluate_case(
    prediction_zyx: np.ndarray,
    ground_truth_xyz_npz: Path,
    projection_npz: Path,
    *,
    bbox_min_xyz_mm: Sequence[float] = (-100.0, -100.0, -100.0),
    voxel_size_mm: float = 0.5,
    ground_truth_origin_xyz_mm: Sequence[float] | None = None,
    prediction_threshold: float = 0.5,
    ssim_window_size: int = 7,
    ssim_chunk_depth: int = 8,
) -> dict[str, Any]:
    """Physically align one case and return audited 3-D metrics.

    ``ground_truth_origin_xyz_mm`` is the lower boundary of native voxel zero.
    When omitted, native index zero is assumed to be centred at physical zero,
    giving a per-case lower boundary of ``-0.5 * spacing``.
    """

    # Preserve float16 inputs so evaluating a 400^3 export does not allocate
    # an unnecessary full-volume float32 copy. SSIM promotes one depth slab at
    # a time in ``src.metrics.volume``.
    prediction = np.asarray(prediction_zyx)
    if prediction.ndim != 3:
        raise ValueError("prediction_zyx must be a 3-D array.")
    if not (
        np.issubdtype(prediction.dtype, np.number)
        or np.issubdtype(prediction.dtype, np.bool_)
    ):
        raise ValueError("prediction_zyx must contain numeric values.")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN or infinite values.")
    if np.min(prediction) < 0.0 or np.max(prediction) > 1.0:
        raise ValueError("Prediction probabilities must lie in [0,1].")
    voxel_size_mm = float(voxel_size_mm)
    if not math.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive and finite.")
    bbox_min = np.asarray(tuple(bbox_min_xyz_mm), dtype=np.float64)
    if bbox_min.shape != (3,) or not np.isfinite(bbox_min).all():
        raise ValueError("bbox_min_xyz_mm must contain three finite XYZ values.")

    ground_truth_zyx, gt_spacing_xyz_mm = _load_ground_truth(
        Path(ground_truth_xyz_npz)
    )
    if ground_truth_origin_xyz_mm is None:
        ground_truth_origin = -0.5 * gt_spacing_xyz_mm
    else:
        ground_truth_origin = np.asarray(
            tuple(ground_truth_origin_xyz_mm), dtype=np.float64
        )
        if ground_truth_origin.shape != (3,) or not np.isfinite(
            ground_truth_origin
        ).all():
            raise ValueError(
                "ground_truth_origin_xyz_mm must be None or contain three "
                "finite lower-bound values."
            )
    center_offset_xyz_mm = _load_center_offset_mm(Path(projection_npz))
    prediction_grid = VoxelGrid(
        shape_zyx=prediction.shape,
        spacing_xyz_mm=(voxel_size_mm,) * 3,
        origin_xyz_mm=tuple(float(value) for value in bbox_min),
    )
    ground_truth_grid = VoxelGrid(
        shape_zyx=ground_truth_zyx.shape,
        spacing_xyz_mm=tuple(float(value) for value in gt_spacing_xyz_mm),
        origin_xyz_mm=tuple(float(value) for value in ground_truth_origin),
    )
    aligned_ground_truth, valid_fov = resample_binary_volume_nearest(
        ground_truth_zyx,
        ground_truth_grid,
        prediction_grid,
        target_to_source_offset_xyz_mm=center_offset_xyz_mm,
    )
    metrics = compute_volume_metrics(
        aligned_ground_truth,
        prediction,
        valid_fov_mask=valid_fov,
        prediction_threshold=prediction_threshold,
        ssim_window_size=ssim_window_size,
        ssim_chunk_depth=ssim_chunk_depth,
    )
    vessel_window_mask = aligned_ground_truth | (prediction >= prediction_threshold)
    try:
        metrics["masked_ssim_3d"] = masked_ssim_3d(
            aligned_ground_truth,
            prediction,
            mask=vessel_window_mask,
            roi_mask=valid_fov,
            data_range=1.0,
            window_size=ssim_window_size,
            chunk_depth=ssim_chunk_depth,
        )
    except ValueError:
        # Empty foreground has a well-defined global SSIM but no foreground
        # windows. Keep the absence explicit in JSON rather than hiding it.
        metrics["masked_ssim_3d"] = None
    metrics.update(
        {
            "prediction_shape_zyx": list(prediction.shape),
            "prediction_spacing_xyz_mm": [voxel_size_mm] * 3,
            "prediction_origin_xyz_mm": [float(value) for value in bbox_min],
            "ground_truth_shape_zyx": list(ground_truth_zyx.shape),
            "ground_truth_spacing_xyz_mm": [
                float(value) for value in gt_spacing_xyz_mm
            ],
            "ground_truth_origin_xyz_mm": [
                float(value) for value in ground_truth_origin
            ],
            "projection_center_offset_xyz_mm": [
                float(value) for value in center_offset_xyz_mm
            ],
            "prediction_threshold": float(prediction_threshold),
            "ssim_window_size": int(ssim_window_size),
        }
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-key", default="prediction_volume_zyx")
    parser.add_argument(
        "--prediction-domain", choices=("probability", "logit"), default="probability"
    )
    parser.add_argument(
        "--prediction-axis-order",
        choices=("zyx", "xyz"),
        help="Required for bare arrays; exported NPZ files record this value.",
    )
    parser.add_argument(
        "--bbox-min-xyz-mm",
        nargs=3,
        help="Prediction-grid lower XYZ boundary; otherwise read from NPZ metadata.",
    )
    parser.add_argument(
        "--voxel-size-mm",
        type=float,
        help="Prediction voxel size; otherwise read from NPZ metadata.",
    )
    parser.add_argument(
        "--ground-truth-origin-xyz-mm",
        nargs=3,
        help=(
            "Optional native GT lower boundary in XYZ millimetres. By default, "
            "voxel index zero is centred at physical zero."
        ),
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--ssim-chunk-depth", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prediction_raw, prediction_metadata = _load_prediction(
        args.prediction, args.prediction_key
    )
    prediction, bbox_min_xyz_mm, voxel_size_mm, source_axis_order = (
        _resolve_prediction_grid(
            prediction_raw,
            prediction_metadata,
            explicit_axis_order=args.prediction_axis_order,
            explicit_bbox_min_xyz_mm=args.bbox_min_xyz_mm,
            explicit_voxel_size_mm=args.voxel_size_mm,
        )
    )
    if args.prediction_domain == "logit":
        prediction = _stable_sigmoid(prediction)

    prediction_case_id = prediction_metadata.get("case_id")
    projection_case_id = _optional_case_id(
        args.projection, fallback_to_stem=False
    )
    ground_truth_case_id = _optional_case_id(
        args.ground_truth, fallback_to_stem=True
    )
    known_case_ids = {
        value
        for value in (
            prediction_case_id,
            projection_case_id,
            ground_truth_case_id,
        )
        if value is not None
    }
    if len(known_case_ids) > 1:
        raise ValueError(
            "Prediction, projection, and ground-truth case IDs disagree: "
            + ", ".join(sorted(known_case_ids))
        )

    metrics = evaluate_case(
        prediction,
        args.ground_truth,
        args.projection,
        bbox_min_xyz_mm=bbox_min_xyz_mm,
        voxel_size_mm=voxel_size_mm,
        ground_truth_origin_xyz_mm=(
            None
            if args.ground_truth_origin_xyz_mm is None
            else _three_floats(args.ground_truth_origin_xyz_mm)
        ),
        prediction_threshold=args.prediction_threshold,
        ssim_window_size=args.ssim_window_size,
        ssim_chunk_depth=args.ssim_chunk_depth,
    )
    metrics["prediction_source_axis_order"] = source_axis_order
    metrics["prediction_file_metadata"] = prediction_metadata
    rendered = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
