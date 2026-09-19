"""Offline graph and geometry evaluation for saved vessel-volume predictions.

The evaluator does not load a checkpoint or require CUDA.  It scans a directory
of AutoCAR-style prediction NPZ files, converts every thresholded volume to a
26-connected centreline graph with EDT radii, and compares that graph with the
same raw vessel-code reference used by the parametric evaluator.

All predictions are resampled onto one shared 0.5-mm lattice.  By default, the
raw vessel code's XYZ+radius polylines are rasterized on that lattice and
provide the ground-truth vessel mask as well as the raw centreline.  A
directory of native voxel ground truths can instead be supplied for the Dice
mask while retaining the raw centreline for centreline Dice, clDice, Chamfer
distance, and derived radius error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluate_npz import (
    _load_center_offset_mm,
    _load_ground_truth,
    _load_prediction,
    _resolve_prediction_grid,
    _scalar_text,
    _stable_sigmoid,
)
from src.geometry.vascular_surface import (
    VascularCenterlineGraph,
    extract_centerline_graph,
)
from src.geometry.voxel_grid import VoxelGrid, resample_binary_volume_nearest
from src.metrics import centerline_radius_errors, masked_dice_3d


_EVALUATION_VOXEL_SIZE_MM = 0.5


_RAW_FRAME_ALIASES = {
    "native": "native",
    "absolute": "native",
    "absolute_world": "native",
    "native_xyz_mm": "native",
    "world": "native",
    "projection_centered": "projection_centered",
    "projection-centred": "projection_centered",
    "projection_centred": "projection_centered",
    "centered": "projection_centered",
    "centred": "projection_centered",
    "projection_centered_xyz_mm": "projection_centered",
}


@dataclass(frozen=True)
class RawVesselCode:
    """Validated raw vessel-code geometry in its stored coordinate frame."""

    path: Path
    case_id: str
    vessel_xyzr_mm: np.ndarray
    branch_exists: np.ndarray
    point_valid: np.ndarray
    coordinate_frame: str
    vessel_key: str

    @property
    def active_point_mask(self) -> np.ndarray:
        return self.branch_exists[:, None] & self.point_valid


def _canonical_case_id(raw: Any) -> str:
    text = _scalar_text(raw)
    if not text:
        raise ValueError("Case ID cannot be empty.")
    if text.isdigit():
        return str(int(text))
    tokens = re.findall(r"\d+", text)
    if tokens:
        return str(int(tokens[-1]))
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", text) is None:
        raise ValueError(
            f"Case ID {text!r} cannot be represented as a safe filename."
        )
    return text.casefold()


def _case_id_from_path(path: Path) -> str:
    if path.parent.name.isdigit():
        return str(int(path.parent.name))
    tokens = re.findall(r"\d+", path.stem)
    if not tokens:
        raise ValueError(
            f"Could not infer a case ID from {path}; store case_id in the NPZ "
            "or place the file below a numeric case directory."
        )
    return str(int(tokens[-1]))


def _embedded_case_id(data: Any, path: Path) -> str | None:
    for key in ("case_id", "source_case_id", "sample_name"):
        if key not in data.files:
            continue
        value = np.asarray(data[key])
        if value.size != 1:
            raise ValueError(f"{path} {key} must contain one scalar value.")
        return _canonical_case_id(value.reshape(()).item())
    return None


def _npz_case_id(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        embedded = _embedded_case_id(data, path)
    return embedded if embedded is not None else _case_id_from_path(path)


def _canonical_split(raw: Any, *, path: Path) -> str:
    value = _scalar_text(raw).casefold()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
        "unspecified": "unspecified",
    }
    if value not in aliases:
        raise ValueError(
            f"{path} dataset_split must identify train, validation, test, "
            "or unspecified; "
            f"got {value!r}."
        )
    return aliases[value]


def _prediction_split(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "dataset_split" in data.files:
            value = np.asarray(data["dataset_split"])
            if value.size != 1:
                raise ValueError(f"{path} dataset_split must be scalar.")
            split = _scalar_text(value.reshape(()).item())
            if not split:
                raise ValueError(f"{path} dataset_split cannot be empty.")
            return _canonical_split(split, path=path)
    parent = path.parent.name.casefold()
    recognized = {
        "train",
        "training",
        "val",
        "valid",
        "validation",
        "test",
        "testing",
    }
    return _canonical_split(
        parent if parent in recognized else "unspecified",
        path=path,
    )


def _discover_npz_by_key(
    root: Path,
    *,
    required_keys: Sequence[str],
    label: str,
    preferred_filename: str | None = None,
) -> dict[str, Path]:
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {source}")
    candidates: dict[str, list[Path]] = {}
    for path in sorted(source.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                if not any(key in data.files for key in required_keys):
                    continue
                case_id = _embedded_case_id(data, path)
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not inspect {label} file {path}.") from error
        resolved_id = case_id if case_id is not None else _case_id_from_path(path)
        candidates.setdefault(resolved_id, []).append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No {label} NPZ files containing any of {list(required_keys)} "
            f"were found under {source}."
        )
    result: dict[str, Path] = {}
    for case_id, paths in candidates.items():
        selected = paths
        if len(paths) > 1 and preferred_filename is not None:
            preferred = [
                path
                for path in paths
                if path.name.casefold() == preferred_filename.casefold()
            ]
            if len(preferred) == 1:
                selected = preferred
        if len(selected) != 1:
            raise ValueError(
                f"Multiple {label} files map to case {case_id}: "
                + ", ".join(str(path) for path in paths)
            )
        result[case_id] = selected[0]
    return result


def _artery_markers_text(value: Any) -> set[str]:
    text = _scalar_text(value).casefold()
    markers = set(
        re.findall(
            r"(?<![a-z0-9])(lca|rca)(?![a-z0-9])",
            text,
        )
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if normalized in {"left", "left_coronary", "left_coronary_artery"}:
        markers.add("lca")
    if normalized in {"right", "right_coronary", "right_coronary_artery"}:
        markers.add("rca")
    return markers


def _artery_markers(path: Path) -> set[str]:
    markers = _artery_markers_text(path)
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "artery_type",
            "vessel_type",
            "anatomy",
            "vessel_side",
            "sample_name",
            "case_id",
            "source_case_id",
        ):
            if key not in data.files:
                continue
            value = np.asarray(data[key])
            if value.size != 1 or value.dtype.kind not in {"U", "S"}:
                continue
            markers.update(_artery_markers_text(value.reshape(()).item()))
    return markers


def _validate_artery_paths(
    paths: Mapping[str, Path], *, artery: str, label: str
) -> dict[str, int]:
    verified = 0
    for case_id, path in paths.items():
        markers = _artery_markers(path)
        if markers and markers != {artery}:
            raise ValueError(
                f"Case {case_id} {label} path conflicts with --artery "
                f"{artery}: {path} contains marker(s) {sorted(markers)}."
            )
        verified += int(markers == {artery})
    return {
        "files": len(paths),
        "verified_from_path_or_metadata": verified,
        "explicit_artery_only": len(paths) - verified,
    }


def _raw_frame(
    requested: str,
    *,
    payload: Any,
    vessel_key: str,
    path: Path,
) -> str:
    normalized = requested.strip().lower().replace("-", "_")
    if normalized != "auto":
        resolved = _RAW_FRAME_ALIASES.get(normalized)
        if resolved is None:
            raise ValueError(f"Unsupported raw vessel coordinate frame: {requested}")
        return resolved
    for key in ("coordinate_frame", "vessel_coordinate_frame"):
        if key not in payload.files:
            continue
        value = np.asarray(payload[key])
        if value.size != 1:
            raise ValueError(f"{path} {key} must be scalar.")
        stored = _scalar_text(value.reshape(()).item()).lower().replace("-", "_")
        resolved = _RAW_FRAME_ALIASES.get(stored)
        if resolved is None:
            raise ValueError(f"{path} declares unsupported {key}={stored!r}.")
        return resolved
    if vessel_key in {"raw_vessel_code_mm", "uniform_arc_vessel_code_mm"}:
        return "native"
    raise ValueError(
        f"Cannot infer the coordinate frame of {path} key {vessel_key!r}; "
        "pass --raw-coordinate-frame explicitly."
    )


def load_raw_vessel_code(
    path: Path,
    *,
    vessel_key: str,
    branch_exists_key: str,
    point_valid_key: str,
    coordinate_frame: str,
    scale_to_mm: float | None,
) -> RawVesselCode:
    """Load the common raw centreline/radius target for one case."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        if vessel_key not in data.files:
            raise KeyError(
                f"{source} lacks raw vessel key {vessel_key!r}; available: "
                f"{sorted(data.files)}"
            )
        vessel = np.asarray(data[vessel_key], dtype=np.float64)
        if vessel.ndim != 3 or vessel.shape[-1] < 4:
            raise ValueError(
                f"{source} {vessel_key} must have shape [M,N,4+], got "
                f"{vessel.shape}. XYZ plus radius are required to compute Dice."
            )
        vessel = vessel[..., :4].copy()
        if scale_to_mm is None:
            if vessel_key.endswith("_mm"):
                scale = 1.0
            elif "input_scale_to_mm" in data.files:
                scale = float(np.asarray(data["input_scale_to_mm"]).reshape(()))
            else:
                raise ValueError(
                    f"Units of {source} {vessel_key!r} are ambiguous; pass "
                    "--raw-scale-to-mm."
                )
        else:
            scale = float(scale_to_mm)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("raw vessel scale-to-mm must be positive and finite.")
        vessel *= scale

        if branch_exists_key in data.files:
            branch_exists = np.asarray(
                data[branch_exists_key], dtype=np.bool_
            ).reshape(-1)
        else:
            branch_exists = np.ones(vessel.shape[0], dtype=np.bool_)
        if branch_exists.shape != (vessel.shape[0],):
            raise ValueError(
                f"{source} {branch_exists_key} must have shape "
                f"({vessel.shape[0]},), got {branch_exists.shape}."
            )

        if point_valid_key in data.files:
            point_valid = np.asarray(data[point_valid_key], dtype=np.bool_)
        else:
            point_valid = np.isfinite(vessel).all(axis=-1) & (vessel[..., 3] > 0)
        if point_valid.shape != vessel.shape[:2]:
            raise ValueError(
                f"{source} {point_valid_key} must have shape {vessel.shape[:2]}, "
                f"got {point_valid.shape}."
            )
        active = branch_exists[:, None] & point_valid
        if not np.any(active):
            raise ValueError(f"{source} contains no active raw vessel points.")
        if not np.isfinite(vessel[active]).all():
            raise ValueError(f"{source} active raw vessel points contain NaN/Inf.")
        if np.any(vessel[..., 3][active] <= 0.0):
            raise ValueError(f"{source} active raw vessel radii must be positive.")
        frame = _raw_frame(
            coordinate_frame,
            payload=data,
            vessel_key=vessel_key,
            path=source,
        )
        embedded = _embedded_case_id(data, source)
    return RawVesselCode(
        path=source,
        case_id=embedded if embedded is not None else _case_id_from_path(source),
        vessel_xyzr_mm=vessel,
        branch_exists=branch_exists,
        point_valid=point_valid,
        coordinate_frame=frame,
        vessel_key=vessel_key,
    )


def _center_offset_from_prediction_or_projection(
    *,
    case_id: str,
    prediction_metadata: Mapping[str, Any],
    projection_paths: Mapping[str, Path] | None,
    raw_frame: str,
) -> tuple[np.ndarray, str]:
    embedded = prediction_metadata.get("projection_center_offset_xyz_mm")
    if embedded is not None:
        offset = np.asarray(embedded, dtype=np.float64)
        source = "prediction_npz"
        if projection_paths is not None and case_id in projection_paths:
            projection_offset = _load_center_offset_mm(projection_paths[case_id])
            if not np.allclose(offset, projection_offset, rtol=0.0, atol=1e-4):
                raise ValueError(
                    f"Case {case_id} prediction/projection centre offsets disagree: "
                    f"{offset.tolist()} vs {projection_offset.tolist()}."
                )
    elif projection_paths is not None and case_id in projection_paths:
        offset = _load_center_offset_mm(projection_paths[case_id])
        source = "projection_npz"
    elif raw_frame == "projection_centered":
        offset = np.zeros(3, dtype=np.float64)
        source = "not_required_raw_already_projection_centered"
    else:
        raise ValueError(
            f"Case {case_id} raw vessel code is in native coordinates, but its "
            "saved prediction contains no projection_center_offset_xyz_mm. "
            "Pass --projection-dir for legacy predictions."
        )
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError(f"Case {case_id} has an invalid projection centre offset.")
    return offset, source


def _active_segments(raw: RawVesselCode):
    active = raw.active_point_mask
    for branch_index in np.flatnonzero(raw.branch_exists):
        valid_indices = np.flatnonzero(active[branch_index])
        if valid_indices.size == 0:
            continue
        run_starts = np.r_[0, np.flatnonzero(np.diff(valid_indices) != 1) + 1]
        run_ends = np.r_[run_starts[1:], valid_indices.size]
        for start, end in zip(run_starts, run_ends):
            run = valid_indices[int(start) : int(end)]
            if run.size == 1:
                index = int(run[0])
                point = raw.vessel_xyzr_mm[branch_index, index]
                yield point, point
                continue
            for first, second in zip(run[:-1], run[1:]):
                yield (
                    raw.vessel_xyzr_mm[branch_index, int(first)],
                    raw.vessel_xyzr_mm[branch_index, int(second)],
                )


def _centered_raw_vessel(
    raw: RawVesselCode, center_offset_xyz_mm: np.ndarray
) -> RawVesselCode:
    if raw.coordinate_frame == "projection_centered":
        return raw
    vessel = raw.vessel_xyzr_mm.copy()
    coordinates = vessel[..., :3]
    coordinates[raw.active_point_mask] -= center_offset_xyz_mm[None, :]
    return RawVesselCode(
        path=raw.path,
        case_id=raw.case_id,
        vessel_xyzr_mm=vessel,
        branch_exists=raw.branch_exists,
        point_valid=raw.point_valid,
        coordinate_frame="projection_centered",
        vessel_key=raw.vessel_key,
    )


def _raw_fov_audit(
    raw_centered: RawVesselCode, evaluation_grid: VoxelGrid
) -> dict[str, Any]:
    values = raw_centered.vessel_xyzr_mm[raw_centered.active_point_mask]
    points = values[:, :3]
    radii = values[:, 3:4]
    lower = np.asarray(evaluation_grid.origin_xyz_mm, dtype=np.float64)
    upper = np.asarray(
        evaluation_grid.upper_bound_xyz_mm, dtype=np.float64
    )
    tolerance = 1e-6
    center_inside = np.all(
        (points >= lower[None, :] - tolerance)
        & (points < upper[None, :] + tolerance),
        axis=1,
    )
    surface_inside = np.all(
        (points - radii >= lower[None, :] - tolerance)
        & (points + radii <= upper[None, :] + tolerance),
        axis=1,
    )
    count = int(len(points))
    center_count = int(center_inside.sum())
    surface_count = int(surface_inside.sum())
    return {
        "active_raw_points": count,
        "raw_centerline_points_inside_evaluation_fov": center_count,
        "raw_centerline_point_inside_evaluation_fov_fraction": (
            center_count / count
        ),
        "raw_radius_points_fully_inside_evaluation_fov": surface_count,
        "raw_radius_point_fully_inside_evaluation_fov_fraction": (
            surface_count / count
        ),
        "fully_contained": surface_count == count,
    }


def rasterize_raw_vessel_mask(
    raw_centered: RawVesselCode,
    *,
    shape_zyx: Sequence[int],
    origin_xyz_mm: Sequence[float],
    spacing_xyz_mm: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize radius-varying centreline capsules on a ZYX voxel grid."""

    shape = np.asarray(tuple(shape_zyx), dtype=np.int64)
    origin = np.asarray(tuple(origin_xyz_mm), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    if shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("shape_zyx must contain three positive dimensions.")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_xyz_mm must contain three finite values.")
    if (
        spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("spacing_xyz_mm must contain three positive values.")
    mask = np.zeros(tuple(int(value) for value in shape), dtype=np.bool_)
    shape_xyz = shape[::-1]
    segments = 0
    for start, end in _active_segments(raw_centered):
        segments += 1
        start_xyz = np.asarray(start[:3], dtype=np.float64)
        end_xyz = np.asarray(end[:3], dtype=np.float64)
        start_radius = float(start[3])
        end_radius = float(end[3])
        endpoints = np.stack((start_xyz, end_xyz))
        endpoint_indices_xyz = (endpoints - origin[None, :]) / spacing[None, :] - 0.5
        expansion = max(start_radius, end_radius) / spacing + 1.0
        lower_xyz = np.floor(
            endpoint_indices_xyz.min(axis=0) - expansion
        ).astype(np.int64)
        upper_xyz = np.ceil(
            endpoint_indices_xyz.max(axis=0) + expansion
        ).astype(np.int64)
        lower_xyz = np.maximum(lower_xyz, 0)
        upper_xyz = np.minimum(upper_xyz, shape_xyz - 1)
        if np.any(lower_xyz > upper_xyz):
            continue
        x = np.arange(lower_xyz[0], upper_xyz[0] + 1, dtype=np.int64)
        y = np.arange(lower_xyz[1], upper_xyz[1] + 1, dtype=np.int64)
        z = np.arange(lower_xyz[2], upper_xyz[2] + 1, dtype=np.int64)
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        world = np.stack(
            (
                origin[0] + (xx.ravel() + 0.5) * spacing[0],
                origin[1] + (yy.ravel() + 0.5) * spacing[1],
                origin[2] + (zz.ravel() + 0.5) * spacing[2],
            ),
            axis=1,
        )
        segment = end_xyz - start_xyz
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            parameter = np.zeros(len(world), dtype=np.float64)
            closest = np.broadcast_to(start_xyz, world.shape)
        else:
            parameter = np.clip(
                ((world - start_xyz) @ segment) / length_squared,
                0.0,
                1.0,
            )
            closest = start_xyz[None, :] + parameter[:, None] * segment[None, :]
        radius = start_radius + parameter * (end_radius - start_radius)
        inside = np.sum((world - closest) ** 2, axis=1) <= radius**2
        target = (
            slice(int(lower_xyz[2]), int(upper_xyz[2]) + 1),
            slice(int(lower_xyz[1]), int(upper_xyz[1]) + 1),
            slice(int(lower_xyz[0]), int(upper_xyz[0]) + 1),
        )
        mask[target] |= inside.reshape(len(z), len(y), len(x))
    return mask, {
        "num_segments_rasterized": segments,
        "foreground_voxels_before_centerline_union": int(mask.sum()),
    }


def rasterize_raw_centerline(
    raw_centered: RawVesselCode,
    *,
    shape_zyx: Sequence[int],
    origin_xyz_mm: Sequence[float],
    spacing_xyz_mm: Sequence[float],
) -> np.ndarray:
    """Voxelize ordered raw polylines with a 26-connected DDA sampling rule."""

    shape = np.asarray(tuple(shape_zyx), dtype=np.int64)
    origin = np.asarray(tuple(origin_xyz_mm), dtype=np.float64)
    spacing = np.asarray(tuple(spacing_xyz_mm), dtype=np.float64)
    shape_xyz = shape[::-1]
    skeleton = np.zeros(tuple(int(value) for value in shape), dtype=np.bool_)
    for start, end in _active_segments(raw_centered):
        start_xyz = np.asarray(start[:3], dtype=np.float64)
        end_xyz = np.asarray(end[:3], dtype=np.float64)
        delta_voxels = (end_xyz - start_xyz) / spacing
        sample_count = max(int(math.ceil(np.max(np.abs(delta_voxels)))), 1) + 1
        parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
        points = start_xyz[None, :] + parameter[:, None] * (
            end_xyz - start_xyz
        )[None, :]
        indices_xyz = np.floor((points - origin[None, :]) / spacing[None, :]).astype(
            np.int64
        )
        valid = np.all(
            (indices_xyz >= 0) & (indices_xyz < shape_xyz[None, :]), axis=1
        )
        indices_xyz = indices_xyz[valid]
        if indices_xyz.size:
            skeleton[
                indices_xyz[:, 2], indices_xyz[:, 1], indices_xyz[:, 0]
            ] = True
    return skeleton


def _graph_skeleton_mask(
    graph: VascularCenterlineGraph, shape_zyx: Sequence[int]
) -> np.ndarray:
    result = np.zeros(tuple(int(value) for value in shape_zyx), dtype=np.bool_)
    if graph.node_index_zyx.size:
        result[tuple(graph.node_index_zyx.T)] = True
    return result


def _raw_point_arrays(raw: RawVesselCode) -> dict[str, np.ndarray]:
    branch_indices, point_indices = np.nonzero(raw.active_point_mask)
    points = raw.vessel_xyzr_mm[
        branch_indices, point_indices, :3
    ].astype(np.float32)
    radii = raw.vessel_xyzr_mm[
        branch_indices, point_indices, 3
    ].astype(np.float32)
    lookup = np.full(raw.active_point_mask.shape, -1, dtype=np.int64)
    lookup[branch_indices, point_indices] = np.arange(len(points), dtype=np.int64)
    edges: list[tuple[int, int]] = []
    for branch_index in np.flatnonzero(raw.branch_exists):
        for point_index in range(raw.point_valid.shape[1] - 1):
            if (
                raw.active_point_mask[branch_index, point_index]
                and raw.active_point_mask[branch_index, point_index + 1]
            ):
                edges.append(
                    (
                        int(lookup[branch_index, point_index]),
                        int(lookup[branch_index, point_index + 1]),
                    )
                )
    return {
        "node_xyz_mm": points,
        "node_radius_mm": radii,
        "node_branch_index": branch_indices.astype(np.int32),
        "node_point_index": point_indices.astype(np.int32),
        "edge_node_indices": np.asarray(edges, dtype=np.int32).reshape(-1, 2),
    }


def _cldice_from_raw_centerline(
    ground_truth_mask: np.ndarray,
    prediction_mask: np.ndarray,
    ground_truth_centerline: np.ndarray,
    prediction_centerline: np.ndarray,
) -> dict[str, float | int]:
    predicted_count = int(prediction_centerline.sum())
    ground_truth_count = int(ground_truth_centerline.sum())
    predicted_in_ground_truth = int(
        np.count_nonzero(prediction_centerline & ground_truth_mask)
    )
    ground_truth_in_prediction = int(
        np.count_nonzero(ground_truth_centerline & prediction_mask)
    )
    precision = (
        predicted_in_ground_truth / predicted_count
        if predicted_count
        else (1.0 if not np.any(ground_truth_mask) else 0.0)
    )
    sensitivity = (
        ground_truth_in_prediction / ground_truth_count
        if ground_truth_count
        else (1.0 if not np.any(prediction_mask) else 0.0)
    )
    denominator = precision + sensitivity
    score = (
        0.0 if denominator == 0.0 else 2.0 * precision * sensitivity / denominator
    )
    return {
        "cldice_3d": float(score),
        "cldice_loss_3d": float(1.0 - score),
        "topology_precision": float(precision),
        "topology_sensitivity": float(sensitivity),
        "prediction_centerline_voxels": predicted_count,
        "ground_truth_centerline_voxels": ground_truth_count,
        "prediction_centerline_in_ground_truth_voxels": (
            predicted_in_ground_truth
        ),
        "ground_truth_centerline_in_prediction_voxels": (
            ground_truth_in_prediction
        ),
    }


def _save_graph_pair(
    path: Path,
    *,
    case_id: str,
    split: str,
    prediction_graph: VascularCenterlineGraph,
    prediction_node_xyz_mm: np.ndarray,
    raw_reference: RawVesselCode,
    comparison_coordinate_frame: str,
    center_offset_xyz_mm: np.ndarray,
    evaluation_grid: VoxelGrid,
    view_indices: Sequence[int] | None,
    expected_view_indices: Sequence[int],
    metrics: Mapping[str, Any],
) -> None:
    raw_arrays = _raw_point_arrays(raw_reference)
    path.parent.mkdir(parents=True, exist_ok=True)

    def optional(name: str) -> np.ndarray:
        value = metrics.get(name)
        return np.asarray(np.nan if value is None else value, dtype=np.float64)

    np.savez_compressed(
        path,
        representation=np.asarray("autocar_voxel_graph_vs_raw_vessel_code"),
        case_id=np.asarray(case_id),
        dataset_split=np.asarray(split),
        source_view_indices=np.asarray(
            [] if view_indices is None else view_indices, dtype=np.int64
        ),
        expected_view_indices=np.asarray(
            expected_view_indices, dtype=np.int64
        ),
        comparison_coordinate_frame=np.asarray(comparison_coordinate_frame),
        # Retained as a concise alias for consumers of this initial schema.
        coordinate_frame=np.asarray(comparison_coordinate_frame),
        evaluation_grid_coordinate_frame=np.asarray(
            "projection_centered_xyz_mm"
        ),
        prediction_node_index_coordinate_frame=np.asarray(
            "projection_centered_evaluation_grid_zyx"
        ),
        prediction_node_xyz_mm_coordinate_frame=np.asarray(
            comparison_coordinate_frame
        ),
        ground_truth_node_xyz_mm_coordinate_frame=np.asarray(
            comparison_coordinate_frame
        ),
        projection_center_offset_xyz_mm=center_offset_xyz_mm.astype(np.float32),
        evaluation_volume_axis_order=np.asarray("zyx"),
        evaluation_shape_zyx=np.asarray(
            evaluation_grid.shape_zyx, dtype=np.int32
        ),
        evaluation_grid_origin_xyz_mm=np.asarray(
            evaluation_grid.origin_xyz_mm, dtype=np.float32
        ),
        evaluation_grid_upper_bound_xyz_mm=np.asarray(
            evaluation_grid.upper_bound_xyz_mm, dtype=np.float32
        ),
        evaluation_voxel_spacing_xyz_mm=np.asarray(
            evaluation_grid.spacing_xyz_mm, dtype=np.float32
        ),
        prediction_node_index_zyx=prediction_graph.node_index_zyx,
        prediction_node_xyz_mm=np.asarray(
            prediction_node_xyz_mm, dtype=np.float32
        ),
        prediction_node_projection_centered_xyz_mm=(
            prediction_graph.node_xyz_mm
        ),
        prediction_node_radius_mm=prediction_graph.node_radius_mm,
        prediction_edge_node_indices=prediction_graph.edge_node_indices,
        prediction_node_degree=prediction_graph.node_degree,
        prediction_node_kind=prediction_graph.node_kind,
        prediction_component_id=prediction_graph.component_id,
        ground_truth_node_xyz_mm=raw_arrays["node_xyz_mm"],
        ground_truth_node_radius_mm=raw_arrays["node_radius_mm"],
        ground_truth_node_branch_index=raw_arrays["node_branch_index"],
        ground_truth_node_point_index=raw_arrays["node_point_index"],
        ground_truth_edge_node_indices=raw_arrays["edge_node_indices"],
        dice_3d=optional("dice_3d"),
        cldice_3d=optional("cldice_3d"),
        centerline_voxel_dice_3d=optional("centerline_voxel_dice_3d"),
        centerline_chamfer_distance_mm=optional(
            "centerline_chamfer_distance_mm"
        ),
        derived_radius_mae_mm=optional("derived_radius_mae_mm"),
    )


def _evaluate_case(
    *,
    prediction_path: Path,
    raw_path: Path,
    projection_paths: Mapping[str, Path] | None,
    ground_truth_path: Path | None,
    prediction_key: str,
    prediction_domain: str,
    prediction_threshold: float,
    expected_view_indices: Sequence[int],
    allow_missing_view_indices: bool,
    raw_vessel_key: str,
    branch_exists_key: str,
    point_valid_key: str,
    raw_coordinate_frame: str,
    raw_scale_to_mm: float | None,
    ground_truth_origin_xyz_mm: Sequence[float] | None,
    evaluation_bbox_min_xyz_mm: Sequence[float],
    evaluation_bbox_max_xyz_mm: Sequence[float],
    allow_clipped_raw_reference: bool,
    graph_path: Path,
    mask_path: Path | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prediction_raw, prediction_metadata = _load_prediction(
        prediction_path, prediction_key
    )
    expected_views = [int(value) for value in expected_view_indices]
    stored_views = prediction_metadata.get("view_indices")
    if stored_views is None:
        if not allow_missing_view_indices:
            raise ValueError(
                f"{prediction_path} does not record view_indices. The strict "
                f"evaluation expects {expected_views}; pass "
                "--allow-missing-view-indices only for an externally audited "
                "third-party prediction."
            )
        view_indices_audit = "missing_explicitly_allowed"
    else:
        stored_views = [int(value) for value in stored_views]
        if stored_views != expected_views:
            raise ValueError(
                f"{prediction_path} used ordered view indices {stored_views}, "
                f"but this evaluation requires {expected_views}."
            )
        view_indices_audit = "matched"
    prediction, bbox_min, source_voxel_size, axis_order = _resolve_prediction_grid(
        prediction_raw,
        prediction_metadata,
        explicit_axis_order=None,
        explicit_bbox_min_xyz_mm=None,
        explicit_voxel_size_mm=None,
    )
    if prediction_domain == "logit":
        prediction = _stable_sigmoid(prediction)
    elif np.min(prediction) < 0.0 or np.max(prediction) > 1.0:
        raise ValueError(
            f"{prediction_path} is declared as probability but contains values "
            "outside [0,1]."
        )
    case_id = _canonical_case_id(
        prediction_metadata.get("case_id", _case_id_from_path(prediction_path))
    )
    raw = load_raw_vessel_code(
        raw_path,
        vessel_key=raw_vessel_key,
        branch_exists_key=branch_exists_key,
        point_valid_key=point_valid_key,
        coordinate_frame=raw_coordinate_frame,
        scale_to_mm=raw_scale_to_mm,
    )
    if raw.case_id != case_id:
        raise ValueError(
            f"Prediction case {case_id} was paired with raw vessel case "
            f"{raw.case_id}: {raw_path}"
        )
    center_offset, center_offset_source = _center_offset_from_prediction_or_projection(
        case_id=case_id,
        prediction_metadata=prediction_metadata,
        projection_paths=projection_paths,
        raw_frame=raw.coordinate_frame,
    )
    raw_centered = _centered_raw_vessel(raw, center_offset)
    source_prediction_mask = np.asarray(
        prediction >= float(prediction_threshold), dtype=np.bool_
    )
    source_grid = VoxelGrid(
        shape_zyx=tuple(int(value) for value in prediction.shape),
        spacing_xyz_mm=(float(source_voxel_size),) * 3,
        origin_xyz_mm=tuple(float(value) for value in bbox_min),
    )
    evaluation_grid = VoxelGrid.from_bounds(
        evaluation_bbox_min_xyz_mm,
        evaluation_bbox_max_xyz_mm,
        (_EVALUATION_VOXEL_SIZE_MM,) * 3,
    )
    if source_grid == evaluation_grid:
        prediction_mask = source_prediction_mask
    else:
        prediction_mask, prediction_valid_fov = resample_binary_volume_nearest(
            source_prediction_mask,
            source_grid,
            evaluation_grid,
        )
        if not np.all(prediction_valid_fov):
            raise RuntimeError(
                f"Case {case_id} canonical 0.5-mm evaluation grid extends "
                "outside the saved prediction field of view. Supply common "
                "--evaluation-bbox-min-xyz-mm/--evaluation-bbox-max-xyz-mm "
                "bounds that every compared method covers."
            )
    raw_fov_audit = _raw_fov_audit(raw_centered, evaluation_grid)
    if not raw_fov_audit["fully_contained"] and not allow_clipped_raw_reference:
        raise ValueError(
            f"Case {case_id} raw vessel reference is not fully contained in "
            "the canonical evaluation grid: "
            f"{raw_fov_audit['raw_centerline_points_inside_evaluation_fov']}"
            f"/{raw_fov_audit['active_raw_points']} centerline points and "
            f"{raw_fov_audit['raw_radius_points_fully_inside_evaluation_fov']}"
            f"/{raw_fov_audit['active_raw_points']} radius samples are fully "
            "inside. Change the common evaluation bounds or explicitly pass "
            "--allow-clipped-raw-reference."
        )
    spacing_xyz = (_EVALUATION_VOXEL_SIZE_MM,) * 3
    prediction_graph = extract_centerline_graph(
        prediction_mask,
        threshold=0.5,
        origin_xyz_mm=evaluation_grid.origin_xyz_mm,
        spacing_xyz_mm=spacing_xyz,
    )
    prediction_centerline = _graph_skeleton_mask(
        prediction_graph, evaluation_grid.shape_zyx
    )
    raw_centerline = rasterize_raw_centerline(
        raw_centered,
        shape_zyx=evaluation_grid.shape_zyx,
        origin_xyz_mm=evaluation_grid.origin_xyz_mm,
        spacing_xyz_mm=spacing_xyz,
    )
    if not np.any(raw_centerline):
        raise ValueError(
            f"Case {case_id} raw centreline has no points inside the prediction "
            "field of view. Check the coordinate frame and centre offset."
        )

    rasterization: dict[str, Any]
    if ground_truth_path is None:
        ground_truth_mask, rasterization = rasterize_raw_vessel_mask(
            raw_centered,
            shape_zyx=evaluation_grid.shape_zyx,
            origin_xyz_mm=evaluation_grid.origin_xyz_mm,
            spacing_xyz_mm=spacing_xyz,
        )
        centerline_added = int(np.count_nonzero(raw_centerline & ~ground_truth_mask))
        ground_truth_mask |= raw_centerline
        rasterization["centerline_voxels_added_to_mask"] = centerline_added
        ground_truth_source = (
            "raw_vessel_code_radius_tube_plus_centerline_union"
        )
        evaluation_valid_fov = None
    else:
        ground_truth_native, native_spacing = _load_ground_truth(ground_truth_path)
        native_origin = (
            -0.5 * native_spacing
            if ground_truth_origin_xyz_mm is None
            else np.asarray(ground_truth_origin_xyz_mm, dtype=np.float64)
        )
        native_grid = VoxelGrid(
            shape_zyx=tuple(int(value) for value in ground_truth_native.shape),
            spacing_xyz_mm=tuple(float(value) for value in native_spacing),
            origin_xyz_mm=tuple(float(value) for value in native_origin),
        )
        ground_truth_mask, valid_fov = resample_binary_volume_nearest(
            ground_truth_native,
            native_grid,
            evaluation_grid,
            target_to_source_offset_xyz_mm=center_offset,
        )
        ground_truth_mask &= valid_fov
        evaluation_valid_fov = valid_fov
        rasterization = {
            "native_ground_truth_shape_zyx": list(ground_truth_native.shape),
            "native_ground_truth_spacing_xyz_mm": native_spacing.tolist(),
            "native_ground_truth_origin_xyz_mm": native_origin.tolist(),
            "valid_fov_voxels": int(valid_fov.sum()),
        }
        ground_truth_source = "native_voxel_npz"

    if evaluation_valid_fov is not None and not np.any(evaluation_valid_fov):
        raise ValueError(
            f"Case {case_id} has no valid 0.5-mm evaluation voxels inside "
            "the ground-truth field of view."
        )
    if evaluation_valid_fov is None:
        evaluated_prediction_mask = prediction_mask
        evaluated_ground_truth_mask = ground_truth_mask
        evaluated_prediction_centerline = prediction_centerline
        evaluated_raw_centerline = raw_centerline
    else:
        evaluated_prediction_mask = prediction_mask & evaluation_valid_fov
        evaluated_ground_truth_mask = ground_truth_mask
        evaluated_prediction_centerline = (
            prediction_centerline & evaluation_valid_fov
        )
        evaluated_raw_centerline = raw_centerline & evaluation_valid_fov
    if not np.any(evaluated_raw_centerline):
        raise ValueError(
            f"Case {case_id} raw centreline has no points inside the valid "
            "0.5-mm evaluation field of view."
        )

    dice = masked_dice_3d(
        ground_truth_mask, prediction_mask, mask=evaluation_valid_fov
    )
    centerline_voxel_dice = masked_dice_3d(
        raw_centerline, prediction_centerline, mask=evaluation_valid_fov
    )
    cldice = _cldice_from_raw_centerline(
        evaluated_ground_truth_mask,
        evaluated_prediction_mask,
        evaluated_raw_centerline,
        evaluated_prediction_centerline,
    )
    if raw.coordinate_frame == "native":
        raw_reference = raw
        prediction_comparison_xyz = (
            prediction_graph.node_xyz_mm.astype(np.float64)
            + center_offset[None, :]
        )
        comparison_coordinate_frame = "native_xyz_mm"
    else:
        raw_reference = raw_centered
        prediction_comparison_xyz = prediction_graph.node_xyz_mm.astype(
            np.float64
        )
        comparison_coordinate_frame = "projection_centered_xyz_mm"
    raw_arrays = _raw_point_arrays(raw_reference)
    graph_errors = centerline_radius_errors(
        prediction_comparison_xyz,
        prediction_graph.node_radius_mm,
        raw_arrays["node_xyz_mm"],
        raw_arrays["node_radius_mm"],
    )
    intersection = int(
        np.count_nonzero(
            evaluated_ground_truth_mask & evaluated_prediction_mask
        )
    )
    metrics: dict[str, Any] = {
        "dice_3d": dice,
        "cldice_3d": cldice["cldice_3d"],
        "cldice_loss_3d": cldice["cldice_loss_3d"],
        "topology_precision": cldice["topology_precision"],
        "topology_sensitivity": cldice["topology_sensitivity"],
        "predicted_centerline_voxels": cldice[
            "prediction_centerline_voxels"
        ],
        "ground_truth_centerline_voxels": cldice[
            "ground_truth_centerline_voxels"
        ],
        "predicted_centerline_in_ground_truth_voxels": cldice[
            "prediction_centerline_in_ground_truth_voxels"
        ],
        "ground_truth_centerline_in_prediction_voxels": cldice[
            "ground_truth_centerline_in_prediction_voxels"
        ],
        "centerline_voxel_dice_3d": centerline_voxel_dice,
        "intersection_voxels": intersection,
        "predicted_foreground_voxels": int(
            evaluated_prediction_mask.sum()
        ),
        "ground_truth_foreground_voxels": int(
            evaluated_ground_truth_mask.sum()
        ),
        "graph_error_valid": graph_errors["valid"],
        "prediction_graph_nodes": graph_errors["prediction_nodes"],
        "ground_truth_graph_nodes": graph_errors["ground_truth_nodes"],
        "centerline_pred_to_gt_mean_error_mm": graph_errors[
            "centerline_pred_to_gt_mean_error_mm"
        ],
        "centerline_gt_to_pred_mean_error_mm": graph_errors[
            "centerline_gt_to_pred_mean_error_mm"
        ],
        "centerline_mean_error_mm": graph_errors["centerline_mean_error_mm"],
        "centerline_chamfer_distance_mm": graph_errors[
            "centerline_chamfer_distance_mm"
        ],
        "derived_radius_pred_to_gt_mae_mm": graph_errors[
            "radius_pred_to_gt_mae_mm"
        ],
        "derived_radius_gt_to_pred_mae_mm": graph_errors[
            "radius_gt_to_pred_mae_mm"
        ],
        "derived_radius_mae_mm": graph_errors["radius_mae_mm"],
    }
    split = _canonical_split(
        prediction_metadata.get(
            "dataset_split", _prediction_split(prediction_path)
        ),
        path=prediction_path,
    )
    _save_graph_pair(
        graph_path,
        case_id=case_id,
        split=split,
        prediction_graph=prediction_graph,
        prediction_node_xyz_mm=prediction_comparison_xyz,
        raw_reference=raw_reference,
        comparison_coordinate_frame=comparison_coordinate_frame,
        center_offset_xyz_mm=center_offset,
        evaluation_grid=evaluation_grid,
        view_indices=stored_views,
        expected_view_indices=expected_views,
        metrics=metrics,
    )
    if mask_path is not None:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mask_path,
            case_id=np.asarray(case_id),
            source_view_indices=np.asarray(
                [] if stored_views is None else stored_views,
                dtype=np.int64,
            ),
            expected_view_indices=np.asarray(
                expected_views, dtype=np.int64
            ),
            ground_truth_mask_zyx=ground_truth_mask.astype(np.uint8),
            prediction_mask_zyx=prediction_mask.astype(np.uint8),
            raw_ground_truth_centerline_zyx=raw_centerline.astype(np.uint8),
            prediction_centerline_zyx=prediction_centerline.astype(np.uint8),
            evaluation_valid_fov_is_full=np.asarray(
                evaluation_valid_fov is None, dtype=np.bool_
            ),
            volume_axis_order=np.asarray("zyx"),
            bbox_min_xyz_mm=np.asarray(
                evaluation_grid.origin_xyz_mm, dtype=np.float32
            ),
            bbox_max_xyz_mm=np.asarray(
                evaluation_grid.upper_bound_xyz_mm, dtype=np.float32
            ),
            voxel_size_mm=np.asarray(
                _EVALUATION_VOXEL_SIZE_MM, dtype=np.float32
            ),
            **(
                {}
                if evaluation_valid_fov is None
                else {
                    "evaluation_valid_fov_zyx": (
                        evaluation_valid_fov.astype(np.uint8)
                    )
                }
            ),
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "case_id": case_id,
        "split": split,
        "prediction_file": str(prediction_path),
        "raw_vessel_file": str(raw_path),
        "ground_truth_volume_file": (
            None if ground_truth_path is None else str(ground_truth_path)
        ),
        "graph_file": str(graph_path),
        "mask_file": None if mask_path is None else str(mask_path),
        "prediction_source_axis_order": axis_order,
        "source_prediction_shape_zyx": list(prediction.shape),
        "source_prediction_voxel_size_mm": float(source_voxel_size),
        "source_prediction_bbox_min_xyz_mm": list(source_grid.origin_xyz_mm),
        "source_prediction_bbox_max_xyz_mm": list(
            source_grid.upper_bound_xyz_mm
        ),
        "evaluation_shape_zyx": list(evaluation_grid.shape_zyx),
        "bbox_min_xyz_mm": list(evaluation_grid.origin_xyz_mm),
        "bbox_max_xyz_mm": list(evaluation_grid.upper_bound_xyz_mm),
        "voxel_size_mm": _EVALUATION_VOXEL_SIZE_MM,
        "prediction_threshold": float(prediction_threshold),
        "view_indices": stored_views,
        "expected_view_indices": expected_views,
        "view_indices_audit": view_indices_audit,
        "projection_center_offset_xyz_mm": center_offset.tolist(),
        "projection_center_offset_source": center_offset_source,
        "raw_vessel_coordinate_frame": raw.coordinate_frame,
        "ground_truth_mask_source": ground_truth_source,
        "raw_centerline_in_ground_truth_mask_fraction": float(
            np.count_nonzero(
                evaluated_raw_centerline & evaluated_ground_truth_mask
            )
            / int(evaluated_raw_centerline.sum())
        ),
        "evaluation_valid_fov_voxels": (
            int(np.prod(evaluation_grid.shape_zyx))
            if evaluation_valid_fov is None
            else int(evaluation_valid_fov.sum())
        ),
        "evaluation_valid_fov_fraction": float(
            1.0
            if evaluation_valid_fov is None
            else evaluation_valid_fov.mean()
        ),
        "rasterization": rasterization,
        "raw_reference_fov_audit": raw_fov_audit,
        "processing_elapsed_ms": elapsed_ms,
        **metrics,
    }


def _mean_standard_error(values: Sequence[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        None
        if len(array) < 2
        else float(array.std(ddof=1) / math.sqrt(len(array))),
    )


def _summary(records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]):
    metric_map = {
        "dice_3d": "dice_3d",
        "cldice_3d": "cldice_3d",
        "cldice_loss_3d": "cldice_loss_3d",
        "centerline_voxel_dice_3d": "centerline_voxel_dice_3d",
        "centerline_mean_error_mm": "centerline_mean_error_mm",
        "centerline_chamfer_distance_mm": "centerline_chamfer_distance_mm",
        "derived_radius_mae_mm": "derived_radius_mae_mm",
        "processing_elapsed_ms": "processing_elapsed_ms",
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "num_cases": len(records),
        "protocol": dict(protocol),
    }
    for output_name, record_name in metric_map.items():
        values = [
            float(record[record_name])
            for record in records
            if record.get(record_name) is not None
        ]
        mean, standard_error = (
            _mean_standard_error(values) if values else (None, None)
        )
        result[f"macro_{output_name}"] = mean
        result[f"macro_{output_name}_standard_error"] = standard_error
        result[f"macro_{output_name}_num_cases"] = len(values)

    intersection = sum(
        int(record["intersection_voxels"]) for record in records
    )
    predicted = sum(
        int(record["predicted_foreground_voxels"])
        for record in records
    )
    ground_truth = sum(
        int(record["ground_truth_foreground_voxels"])
        for record in records
    )
    result["micro_dice_3d"] = (
        1.0
        if predicted + ground_truth == 0
        else 2.0 * intersection / (predicted + ground_truth)
    )
    predicted_centerline = sum(
        int(record["predicted_centerline_voxels"])
        for record in records
    )
    ground_truth_centerline = sum(
        int(record["ground_truth_centerline_voxels"])
        for record in records
    )
    predicted_in_ground_truth = sum(
        int(record["predicted_centerline_in_ground_truth_voxels"])
        for record in records
    )
    ground_truth_in_prediction = sum(
        int(record["ground_truth_centerline_in_prediction_voxels"])
        for record in records
    )
    precision = (
        predicted_in_ground_truth / predicted_centerline
        if predicted_centerline
        else (1.0 if ground_truth_centerline == 0 else 0.0)
    )
    sensitivity = (
        ground_truth_in_prediction / ground_truth_centerline
        if ground_truth_centerline
        else (1.0 if predicted_centerline == 0 else 0.0)
    )
    result["micro_topology_precision"] = precision
    result["micro_topology_sensitivity"] = sensitivity
    result["micro_cldice_3d"] = (
        0.0
        if precision + sensitivity == 0.0
        else 2.0 * precision * sensitivity / (precision + sensitivity)
    )
    split_names = list(dict.fromkeys(str(record["split"]) for record in records))
    result["by_split"] = {
        split: _summary(
            [record for record in records if record["split"] == split],
            protocol,
        )
        for split in split_names
    } if len(split_names) > 1 else {}
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flat_rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def _summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    def row(scope: str, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "scope": scope,
            **{
                key: item
                for key, item in value.items()
                if item is None or isinstance(item, (str, int, float, bool))
            },
        }

    return [row("all", summary)] + [
        row(str(split), split_summary)
        for split, split_summary in summary.get("by_split", {}).items()
    ]


def evaluate_prediction_directory(args: argparse.Namespace) -> dict[str, Any]:
    prediction_paths = _discover_npz_by_key(
        args.prediction_dir,
        required_keys=(args.prediction_key,),
        label="prediction",
    )
    raw_paths = _discover_npz_by_key(
        args.raw_vessel_dir,
        required_keys=(args.raw_vessel_key,),
        label="raw vessel",
        preferred_filename=args.raw_preferred_filename,
    )
    projection_paths = (
        None
        if args.projection_dir is None
        else _discover_npz_by_key(
            args.projection_dir,
            required_keys=(
                "projection_center_offset",
                "projection_center_offset_xyz_mm",
            ),
            label="projection",
        )
    )
    ground_truth_paths = (
        None
        if args.ground_truth_volume_dir is None
        else _discover_npz_by_key(
            args.ground_truth_volume_dir,
            required_keys=("vol",),
            label="ground-truth volume",
        )
    )
    artery_audit = {
        "prediction": _validate_artery_paths(
            prediction_paths, artery=args.artery, label="prediction"
        ),
        "raw_vessel": _validate_artery_paths(
            raw_paths, artery=args.artery, label="raw-vessel"
        ),
    }
    if projection_paths is not None:
        artery_audit["projection"] = _validate_artery_paths(
            projection_paths, artery=args.artery, label="projection"
        )
    if ground_truth_paths is not None:
        artery_audit["ground_truth_volume"] = _validate_artery_paths(
            ground_truth_paths,
            artery=args.artery,
            label="ground-truth volume",
        )
    requested = (
        None
        if not args.case_id
        else {_canonical_case_id(value) for value in args.case_id}
    )
    case_ids = [
        case_id
        for case_id in sorted(
            prediction_paths,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        if requested is None or case_id in requested
    ]
    if requested is not None:
        missing_predictions = requested - set(case_ids)
        if missing_predictions:
            raise ValueError(
                "Requested cases have no predictions: "
                + ", ".join(sorted(missing_predictions))
            )
    missing_raw = [case_id for case_id in case_ids if case_id not in raw_paths]
    if missing_raw:
        raise ValueError(
            "Predictions have no matching raw vessel code for cases: "
            + ", ".join(missing_raw)
        )
    if ground_truth_paths is not None:
        missing_ground_truth = [
            case_id for case_id in case_ids if case_id not in ground_truth_paths
        ]
        if missing_ground_truth:
            raise ValueError(
                "Predictions have no matching ground-truth volume for cases: "
                + ", ".join(missing_ground_truth)
            )
    if not case_ids:
        raise ValueError("No prediction cases were selected.")
    if not 0.0 <= float(args.prediction_threshold) <= 1.0:
        raise ValueError("--prediction-threshold must lie in [0,1].")
    expected_views = tuple(int(value) for value in args.expected_view_indices)
    if (
        len(expected_views) != 2
        or any(value < 0 for value in expected_views)
        or expected_views[0] == expected_views[1]
    ):
        raise ValueError(
            "--expected-view-indices must contain two distinct non-negative "
            "indices in input order."
        )
    canonical_evaluation_grid = VoxelGrid.from_bounds(
        args.evaluation_bbox_min_xyz_mm,
        args.evaluation_bbox_max_xyz_mm,
        (_EVALUATION_VOXEL_SIZE_MM,) * 3,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite "
            "to replace matching evaluation artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "saved_volume_to_raw_vessel_graph_evaluation",
        "artery": args.artery,
        "artery_pairing_audit": artery_audit,
        "prediction_key": args.prediction_key,
        "prediction_domain": args.prediction_domain,
        "prediction_threshold": float(args.prediction_threshold),
        "expected_view_indices": list(args.expected_view_indices),
        "missing_view_indices_policy": (
            "explicitly_allowed"
            if args.allow_missing_view_indices
            else "error"
        ),
        "evaluation_voxel_spacing_xyz_mm": [
            _EVALUATION_VOXEL_SIZE_MM,
            _EVALUATION_VOXEL_SIZE_MM,
            _EVALUATION_VOXEL_SIZE_MM,
        ],
        "evaluation_bbox_min_xyz_mm": list(
            args.evaluation_bbox_min_xyz_mm
        ),
        "evaluation_bbox_max_xyz_mm": list(
            args.evaluation_bbox_max_xyz_mm
        ),
        "evaluation_shape_zyx": list(
            canonical_evaluation_grid.shape_zyx
        ),
        "evaluation_grid_source": "shared_cli_bounds",
        "allow_clipped_raw_reference": bool(
            args.allow_clipped_raw_reference
        ),
        "prediction_graph_derivation": (
            "threshold_lee_skeletonize_26n_edt_radius"
        ),
        "ground_truth_mask_source": (
            "native_voxel_npz"
            if ground_truth_paths is not None
            else "raw_vessel_code_radius_tube_plus_centerline_union"
        ),
        "ground_truth_centerline_source": args.raw_vessel_key,
        "raw_vessel_key": args.raw_vessel_key,
        "raw_branch_exists_key": args.branch_exists_key,
        "raw_point_valid_key": args.point_valid_key,
        "raw_coordinate_frame_requested": args.raw_coordinate_frame,
        "raw_scale_to_mm_override": args.raw_scale_to_mm,
        "ground_truth_origin_xyz_mm": args.ground_truth_origin_xyz_mm,
        "centerline_voxel_dice": "binary_dice_on_common_grid",
        "cldice": (
            "harmonic_mean_of_predicted_skeleton_in_gt_mask_and_raw_gt_"
            "centerline_in_prediction_mask"
        ),
        "metric_roi": (
            "entire_shared_grid_for_raw_tube_gt_or_shared_grid_intersection_"
            "with_native_gt_fov_for_native_voxel_gt"
        ),
        "chamfer": (
            "unhalved_sum_of_bidirectional_mean_nearest_neighbor_l2_"
            "distances_mm"
        ),
        "coordinate_alignment": (
            "per_case_raw_frame_resolution_with_native_xyz_mm_minus_"
            "projection_center_offset_when_required"
        ),
        "radius_error_status": (
            "derived_postprocessing_metric_for_prediction_vs_native_raw_radius"
        ),
    }
    records: list[dict[str, Any]] = []
    for case_id in case_ids:
        split_hint = _prediction_split(prediction_paths[case_id])
        graph_path = output_dir / "graphs" / split_hint / f"{case_id}.npz"
        mask_path = (
            output_dir / "masks" / split_hint / f"{case_id}.npz"
            if args.save_masks
            else None
        )
        record = _evaluate_case(
            prediction_path=prediction_paths[case_id],
            raw_path=raw_paths[case_id],
            projection_paths=projection_paths,
            ground_truth_path=(
                None
                if ground_truth_paths is None
                else ground_truth_paths[case_id]
            ),
            prediction_key=args.prediction_key,
            prediction_domain=args.prediction_domain,
            prediction_threshold=args.prediction_threshold,
            expected_view_indices=args.expected_view_indices,
            allow_missing_view_indices=args.allow_missing_view_indices,
            raw_vessel_key=args.raw_vessel_key,
            branch_exists_key=args.branch_exists_key,
            point_valid_key=args.point_valid_key,
            raw_coordinate_frame=args.raw_coordinate_frame,
            raw_scale_to_mm=args.raw_scale_to_mm,
            ground_truth_origin_xyz_mm=args.ground_truth_origin_xyz_mm,
            evaluation_bbox_min_xyz_mm=args.evaluation_bbox_min_xyz_mm,
            evaluation_bbox_max_xyz_mm=args.evaluation_bbox_max_xyz_mm,
            allow_clipped_raw_reference=args.allow_clipped_raw_reference,
            graph_path=graph_path,
            mask_path=mask_path,
        )
        records.append(record)
        chamfer = record["centerline_chamfer_distance_mm"]
        chamfer_text = (
            "undefined" if chamfer is None else f"{float(chamfer):.6f} mm"
        )
        print(
            f"case={case_id} split={record['split']} "
            f"Dice={record['dice_3d']:.6f} "
            f"clDice={record['cldice_3d']:.6f} "
            f"Chamfer={chamfer_text}",
            flush=True,
        )
    summary = _summary(records, protocol)
    _write_json(output_dir / "per_case_metrics.json", records)
    _write_csv(output_dir / "per_case_metrics.csv", records)
    _write_json(output_dir / "evaluation_summary.json", summary)
    _write_csv(output_dir / "evaluation_summary.csv", _summary_rows(summary))
    _write_csv(
        output_dir / "timing_per_case.csv",
        [
            {
                "case_id": record["case_id"],
                "split": record["split"],
                "processing_elapsed_ms": record["processing_elapsed_ms"],
            }
            for record in records
        ],
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "protocol": protocol,
            "num_cases": len(records),
            "per_case_metrics_json": "per_case_metrics.json",
            "per_case_metrics_csv": "per_case_metrics.csv",
            "summary": "evaluation_summary.json",
            "summary_csv": "evaluation_summary.csv",
            "timing": "timing_per_case.csv",
            "graphs": "graphs/<split>/<case_id>.npz",
            "masks": "masks/<split>/<case_id>.npz" if args.save_masks else None,
            "graph_files": [
                str(Path(record["graph_file"]).relative_to(output_dir))
                for record in records
            ],
            "mask_files": [
                str(Path(record["mask_file"]).relative_to(output_dir))
                for record in records
                if record["mask_file"] is not None
            ],
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artery", choices=("lca", "rca"), required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--raw-vessel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projection-dir", type=Path)
    parser.add_argument("--ground-truth-volume-dir", type=Path)
    parser.add_argument("--prediction-key", default="prediction_volume_zyx")
    parser.add_argument(
        "--prediction-domain",
        choices=("probability", "logit"),
        default="probability",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--expected-view-indices",
        nargs=2,
        type=int,
        default=(0, 1),
        metavar=("VIEW_1", "VIEW_2"),
        help="Required ordered source-view pair recorded in prediction NPZs.",
    )
    parser.add_argument(
        "--allow-missing-view-indices",
        action="store_true",
        help=(
            "Allow third-party prediction NPZs without view_indices; pair "
            "mismatches are never allowed."
        ),
    )
    parser.add_argument(
        "--evaluation-bbox-min-xyz-mm",
        nargs=3,
        type=float,
        default=(-100.0, -100.0, -100.0),
        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
        help="Shared 0.5-mm evaluation-grid lower boundary.",
    )
    parser.add_argument(
        "--evaluation-bbox-max-xyz-mm",
        nargs=3,
        type=float,
        default=(100.0, 100.0, 100.0),
        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
        help="Shared 0.5-mm evaluation-grid exclusive upper boundary.",
    )
    parser.add_argument("--raw-vessel-key", default="raw_vessel_code_mm")
    parser.add_argument("--branch-exists-key", default="branch_exists")
    parser.add_argument("--point-valid-key", default="point_valid_mask")
    parser.add_argument(
        "--raw-coordinate-frame",
        choices=("auto", "native", "projection-centered"),
        default="auto",
    )
    parser.add_argument("--raw-scale-to-mm", type=float)
    parser.add_argument(
        "--raw-preferred-filename",
        default="original.npz",
        help=(
            "When a case has multiple raw targets, select this unique filename."
        ),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--ground-truth-origin-xyz-mm",
        nargs=3,
        type=float,
        help=(
            "Optional common native lower-bound origin for voxel GT. The "
            "default is -0.5*spacing per case."
        ),
    )
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument(
        "--allow-clipped-raw-reference",
        action="store_true",
        help=(
            "Allow raw centreline/radius targets outside the shared grid; "
            "clipping fractions remain recorded."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_prediction_directory(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
