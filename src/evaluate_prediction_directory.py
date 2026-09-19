"""Offline graph and geometry evaluation for saved vessel-volume predictions.

The evaluator does not load a checkpoint or require CUDA.  It accepts one NPZ
file or a directory of AutoCAR, DeepCA, or 3DGR-CAR prediction NPZ files,
normalizes their method-specific grid metadata, converts every thresholded
volume to a 26-connected centreline graph with EDT radii, and compares that
graph with the same raw vessel-code reference used by the parametric evaluator.

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
_PREDICTION_METHODS = ("autocar", "deepca", "3dgrcar")
_RAW_VESSEL_AUTO_KEYS = (
    "raw_vessel_code_mm",
    "uniform_arc_vessel_code_mm",
    "branches_xyzr_resampled",
    "artery",
)
_RAW_VESSEL_SCHEMA_SCALE_TO_MM = {
    "raw_vessel_code_mm": 1.0,
    "uniform_arc_vessel_code_mm": 1.0,
    "branches_xyzr_resampled": 1.0,
    "artery": 1000.0,
}


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
    scale_to_mm: float
    scale_source: str
    branch_exists_source: str
    point_valid_source: str

    @property
    def active_point_mask(self) -> np.ndarray:
        return self.branch_exists[:, None] & self.point_valid


@dataclass(frozen=True)
class NormalizedPrediction:
    """One method-specific prediction represented on a physical ZYX grid."""

    path: Path
    method: str
    method_resolution: str
    volume_key: str
    volume_zyx: np.ndarray
    source_grid: VoxelGrid
    coordinate_frame: str
    coordinate_frame_source: str
    origin_convention: str
    origin_convention_source: str
    metadata: Mapping[str, Any]
    volume_is_binary_mask: bool
    checkpoint: str | None


def _metadata_scalar_value(data: Any, key: str, path: Path) -> Any:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{path} metadata {key!r} must be scalar.")
    return value.reshape(()).item()


def _canonical_prediction_method(raw: Any, *, path: Path) -> str:
    value = _scalar_text(raw).strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "autocar": "autocar",
        "autocad": "autocar",
        "deepca": "deepca",
        "3dgrcar": "3dgrcar",
        "threedgrcar": "3dgrcar",
    }
    if value not in aliases:
        raise ValueError(
            f"{path} declares unsupported prediction_method={value!r}; "
            f"supported methods are {list(_PREDICTION_METHODS)}."
        )
    return aliases[value]


def _detect_prediction_method(data: Any, path: Path) -> str:
    keys = set(data.files)
    matches: list[str] = []
    if {
        "prediction_volume_zyx",
        "bbox_min_xyz_mm",
        "voxel_size_mm",
    }.issubset(keys):
        matches.append("autocar")
    if {"vol", "spacing", "origin"}.issubset(keys):
        matches.append("deepca")
    if {
        "prediction_mask_zyx",
        "ground_truth_spacing_xyz_m",
        "ground_truth_origin_xyz_m",
    }.issubset(keys):
        matches.append("3dgrcar")
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely detect the prediction method for {path}; "
            f"schema matches={matches}, available keys={sorted(keys)}. Pass "
            "--prediction-method and ensure the method's grid metadata is present."
        )
    return matches[0]


def _resolve_prediction_method(
    data: Any, path: Path, requested_method: str
) -> tuple[str, str]:
    embedded = (
        None
        if "prediction_method" not in data.files
        else _canonical_prediction_method(
            _metadata_scalar_value(data, "prediction_method", path), path=path
        )
    )
    requested = None if requested_method == "auto" else requested_method
    if requested is not None and embedded is not None and requested != embedded:
        raise ValueError(
            f"{path} declares prediction_method={embedded!r}, which conflicts "
            f"with --prediction-method={requested!r}."
        )
    if requested is not None:
        return requested, "cli"
    if embedded is not None:
        return embedded, "npz_metadata"
    return _detect_prediction_method(data, path), "schema_detection"


def _validated_volume(data: Any, key: str, path: Path) -> np.ndarray:
    if key not in data.files:
        raise KeyError(
            f"Prediction NPZ {path} has no {key!r} key; available keys: "
            f"{sorted(data.files)}"
        )
    volume = np.asarray(data[key]).squeeze()
    if volume.ndim != 3:
        raise ValueError(
            f"{path} {key!r} must reduce to one 3D volume, got {volume.shape}."
        )
    if not (
        np.issubdtype(volume.dtype, np.number)
        or np.issubdtype(volume.dtype, np.bool_)
    ):
        raise ValueError(f"{path} {key!r} must contain numeric values.")
    if not np.isfinite(volume).all():
        raise ValueError(f"{path} {key!r} contains NaN or infinite values.")
    return volume


def _xyz_vector(
    data: Any,
    key: str,
    path: Path,
    *,
    scale_to_mm: float = 1.0,
    positive: bool = False,
) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"{path} is missing required grid metadata {key!r}.")
    value = np.asarray(data[key], dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{path} {key!r} must contain three finite XYZ values.")
    value = value * float(scale_to_mm)
    if positive and np.any(value <= 0.0):
        raise ValueError(f"{path} {key!r} must contain positive values.")
    return value


def _stored_text(data: Any, path: Path, keys: Sequence[str]) -> tuple[str, str] | None:
    for key in keys:
        if key in data.files:
            value = _scalar_text(_metadata_scalar_value(data, key, path)).strip()
            if not value:
                raise ValueError(f"{path} metadata {key!r} cannot be empty.")
            return value, key
    return None


def _resolve_prediction_frame(
    data: Any,
    path: Path,
    *,
    method: str,
    requested: str,
    schema_default: tuple[str, str] | None = None,
) -> tuple[str, str]:
    explicit = requested.strip().lower().replace("-", "_")
    stored = _stored_text(
        data,
        path,
        ("prediction_coordinate_frame", "volume_coordinate_frame", "coordinate_frame"),
    )
    stored_frame: str | None = None
    if stored is not None:
        stored_frame = _RAW_FRAME_ALIASES.get(
            stored[0].strip().lower().replace("-", "_")
        )
        if stored_frame is None:
            raise ValueError(
                f"{path} declares unsupported {stored[1]}={stored[0]!r}."
            )
    if explicit != "auto":
        resolved = _RAW_FRAME_ALIASES.get(explicit)
        if resolved is None:
            raise ValueError(
                f"Unsupported --prediction-coordinate-frame={requested!r}."
            )
        if stored_frame is not None and resolved != stored_frame:
            raise ValueError(
                f"{path} coordinate frame {stored_frame!r} conflicts with "
                f"--prediction-coordinate-frame={resolved!r}."
            )
        return resolved, "cli"
    if stored_frame is not None:
        return stored_frame, f"npz_metadata:{stored[1]}"
    if schema_default is not None:
        return schema_default
    if method == "autocar":
        return "projection_centered", "autocar_schema_default"
    if method == "deepca":
        return "native", "deepca_exporter_contract"
    raise ValueError(
        f"{path} does not declare the physical coordinate frame of its {method} "
        "volume. Pass --prediction-coordinate-frame native or "
        "--prediction-coordinate-frame projection-centered; this cannot be "
        "safely inferred from an origin alone."
    )


def _resolve_origin_convention(
    data: Any,
    path: Path,
    *,
    method: str,
    requested: str,
    schema_default: tuple[str, str] | None = None,
) -> tuple[str, str]:
    aliases = {
        "lower_bound": "lower_bound",
        "lower_boundary": "lower_bound",
        "voxel_center": "voxel_center",
        "first_voxel_center": "voxel_center",
        "first_voxel_centre": "voxel_center",
    }
    explicit = requested.strip().lower().replace("-", "_")
    stored = _stored_text(
        data,
        path,
        ("prediction_origin_convention", "origin_convention"),
    )
    stored_convention: str | None = None
    if stored is not None:
        stored_convention = aliases.get(
            stored[0].strip().lower().replace("-", "_")
        )
        if stored_convention is None:
            raise ValueError(
                f"{path} declares unsupported {stored[1]}={stored[0]!r}."
            )
    if explicit != "auto":
        resolved = aliases.get(explicit)
        if resolved is None:
            raise ValueError(
                f"Unsupported --prediction-origin-convention={requested!r}."
            )
        if stored_convention is not None and resolved != stored_convention:
            raise ValueError(
                f"{path} origin convention {stored_convention!r} conflicts "
                f"with --prediction-origin-convention={resolved!r}."
            )
        return resolved, "cli"
    if stored_convention is not None:
        return stored_convention, f"npz_metadata:{stored[1]}"
    if schema_default is not None:
        return schema_default
    if method == "autocar":
        return "lower_bound", "autocar_bbox_schema"
    if method == "deepca":
        return "voxel_center", "deepca_exporter_contract"
    raise ValueError(
        f"{path} does not state whether its {method} origin is the lower voxel "
        "boundary or the first voxel centre. Pass "
        "--prediction-origin-convention lower-bound or "
        "--prediction-origin-convention voxel-center."
    )


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


def _safe_scalar_metadata(data: Any, key: str, path: Path) -> Any | None:
    """Read scalar metadata without enabling pickle-backed object arrays."""

    try:
        value = np.asarray(data[key])
    except ValueError as error:
        if "Object arrays cannot be loaded when allow_pickle=False" in str(error):
            return None
        raise
    if value.size != 1:
        raise ValueError(f"{path} {key} must contain one scalar value.")
    if value.dtype.kind == "O":
        return None
    return value.reshape(()).item()


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
    # Stage-2 ImageCAS files commonly store source_case_id as an object array,
    # but also provide the pickle-free sample_name (for example, lca_0001).
    # Prefer safe string metadata and never enable pickle merely to identify a
    # case supplied by an external NPZ.
    for key in ("case_id", "sample_name", "source_case_id"):
        if key not in data.files:
            continue
        value = _safe_scalar_metadata(data, key, path)
        if value is not None:
            return _canonical_case_id(value)
    return None


def _npz_case_id(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        embedded = _embedded_case_id(data, path)
    return embedded if embedded is not None else _case_id_from_path(path)


def _prediction_case_id_from_path(path: Path) -> str:
    """Infer only an isolated numeric case token, never the ``3`` in 3DGR."""

    if path.parent.name.isdigit():
        return str(int(path.parent.name))
    matches = re.findall(r"(?:^|[_-])(\d+)(?=$|[_-])", path.stem)
    if len(matches) == 1:
        return str(int(matches[0]))
    raise ValueError(
        f"{path} contains no unambiguous case_id metadata or isolated numeric "
        "case token. For one prediction NPZ, pass --case-id-override."
    )


def _prediction_case_id(
    data: Any,
    path: Path,
    *,
    case_id_override: str | None,
) -> tuple[str, str]:
    embedded = _embedded_case_id(data, path)
    override = (
        None
        if case_id_override is None
        else _canonical_case_id(case_id_override)
    )
    if embedded is not None and override is not None and embedded != override:
        raise ValueError(
            f"{path} embeds case_id={embedded!r}, which conflicts with "
            f"--case-id-override={override!r}."
        )
    if embedded is not None:
        return embedded, "npz_metadata"
    if override is not None:
        return override, "cli_override"
    return _prediction_case_id_from_path(path), "path"


def _default_prediction_key(method: str) -> str:
    return {
        "autocar": "prediction_volume_zyx",
        "deepca": "vol",
        "3dgrcar": "prediction_mask_zyx",
    }[method]


def _discover_prediction_npzs(
    source_path: Path,
    *,
    requested_method: str,
    prediction_key: str | None,
    case_id_override: str | None,
    artery: str | None = None,
) -> dict[str, Path]:
    source = Path(source_path).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Prediction file must be .npz: {source}")
        paths = [source]
        is_single_file = True
    elif source.is_dir():
        paths = sorted(source.rglob("*.npz"))
        is_single_file = False
    else:
        raise FileNotFoundError(
            f"Prediction file or directory does not exist: {source}"
        )
    if case_id_override is not None and not is_single_file:
        raise ValueError(
            "--case-id-override is only valid when --prediction-dir points "
            "to one NPZ file."
        )
    candidates: dict[str, list[Path]] = {}
    inspection_errors: list[str] = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                if requested_method == "auto":
                    looks_like_prediction = (
                        "prediction_method" in keys
                        or {
                            "prediction_volume_zyx",
                            "bbox_min_xyz_mm",
                            "voxel_size_mm",
                        }.issubset(keys)
                        or {"vol", "spacing", "origin"}.issubset(keys)
                        or {
                            "prediction_mask_zyx",
                            "ground_truth_spacing_xyz_m",
                            "ground_truth_origin_xyz_m",
                        }.issubset(keys)
                    )
                else:
                    expected_key = prediction_key or _default_prediction_key(
                        requested_method
                    )
                    looks_like_prediction = (
                        expected_key in keys or "prediction_method" in keys
                    )
                if not looks_like_prediction:
                    continue
                if artery is not None and not _matches_requested_artery(
                    path, artery=artery
                ):
                    continue
                method, _ = _resolve_prediction_method(
                    data, path, requested_method
                )
                key = prediction_key or _default_prediction_key(method)
                if key not in data.files:
                    if is_single_file:
                        raise KeyError(
                            f"{path} has no {key!r} volume for method {method}; "
                            f"available keys: {sorted(data.files)}"
                        )
                    continue
                case_id, _ = _prediction_case_id(
                    data,
                    path,
                    case_id_override=case_id_override,
                )
        except (OSError, ValueError, KeyError) as error:
            if is_single_file:
                raise
            inspection_errors.append(f"{path}: {error}")
            continue
        candidates.setdefault(case_id, []).append(path)
    if inspection_errors:
        raise ValueError(
            "Malformed or ambiguous prediction NPZ files were found: "
            + " | ".join(inspection_errors[:5])
        )
    if not candidates:
        raise FileNotFoundError(
            f"No {requested_method} prediction NPZ files were found under "
            f"{source}."
        )
    result: dict[str, Path] = {}
    for case_id, case_paths in candidates.items():
        if len(case_paths) != 1:
            raise ValueError(
                f"Multiple prediction files map to case {case_id}: "
                + ", ".join(str(path) for path in case_paths)
            )
        result[case_id] = case_paths[0]
    return result


def _common_prediction_metadata(data: Any, path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    embedded_case = _embedded_case_id(data, path)
    if embedded_case is not None:
        metadata["case_id"] = embedded_case
    for key in ("dataset_split", "evaluation_role"):
        if key in data.files:
            metadata[key] = _scalar_text(
                _metadata_scalar_value(data, key, path)
            )
    if "threshold" in data.files:
        threshold = float(_metadata_scalar_value(data, "threshold", path))
        if not math.isfinite(threshold):
            raise ValueError(f"{path} threshold must be finite.")
        metadata["threshold"] = threshold
    checkpoint = _stored_text(
        data,
        path,
        ("checkpoint", "checkpoint_path", "model_checkpoint"),
    )
    if checkpoint is not None:
        metadata["checkpoint"] = checkpoint[0]
        metadata["checkpoint_metadata_key"] = checkpoint[1]
    if "view_indices" in data.files:
        raw = np.asarray(data["view_indices"])
        try:
            values = raw.astype(np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path} view_indices must contain integers."
            ) from error
        if (
            values.ndim != 1
            or values.size == 0
            or not np.array_equal(raw, values)
        ):
            raise ValueError(
                f"{path} view_indices must be a non-empty 1D integer array."
            )
        metadata["view_indices"] = [int(value) for value in values]
    if "projection_center_offset_xyz_mm" in data.files:
        metadata["projection_center_offset_xyz_mm"] = _xyz_vector(
            data, "projection_center_offset_xyz_mm", path
        ).tolist()
    elif "projection_center_offset_xyz_m" in data.files:
        metadata["projection_center_offset_xyz_mm"] = _xyz_vector(
            data,
            "projection_center_offset_xyz_m",
            path,
            scale_to_mm=1000.0,
        ).tolist()
    return metadata


def _axis_order(data: Any, path: Path, *, method: str, key: str) -> str:
    stored = _stored_text(data, path, ("volume_axis_order", "axis_order"))
    if stored is None:
        if key.endswith("_zyx") or method == "3dgrcar":
            return "zyx"
        raise ValueError(
            f"{path} does not declare axis_order for volume key {key!r}."
        )
    value = stored[0].casefold()
    if value not in {"xyz", "zyx"}:
        raise ValueError(
            f"{path} {stored[1]} must be 'xyz' or 'zyx', got {value!r}."
        )
    return value


def _to_zyx(volume: np.ndarray, axis_order: str) -> np.ndarray:
    result = volume.transpose(2, 1, 0) if axis_order == "xyz" else volume
    return np.ascontiguousarray(result)


def _lower_bound_origin(
    stored_origin_xyz_mm: np.ndarray,
    spacing_xyz_mm: np.ndarray,
    convention: str,
) -> np.ndarray:
    if convention == "lower_bound":
        return stored_origin_xyz_mm
    if convention == "voxel_center":
        return stored_origin_xyz_mm - 0.5 * spacing_xyz_mm
    raise AssertionError(f"Unexpected origin convention {convention!r}.")


def _resolve_3dgrcar_alignment(
    data: Any, path: Path, requested: str
) -> tuple[str, str]:
    aliases = {
        "physical": "physical",
        "same_grid": "same_grid",
        "samegrid": "same_grid",
    }
    stored = _stored_text(
        data,
        path,
        ("ground_truth_alignment", "prediction_grid_alignment"),
    )
    stored_value: str | None = None
    if stored is not None:
        stored_value = aliases.get(
            stored[0].strip().casefold().replace("-", "_")
        )
        if stored_value is None:
            raise ValueError(
                f"{path} declares unsupported {stored[1]}={stored[0]!r}."
            )
    explicit = requested.strip().casefold().replace("-", "_")
    if explicit != "auto":
        resolved = aliases.get(explicit)
        if resolved is None:
            raise ValueError(
                f"Unsupported --3dgrcar-alignment={requested!r}."
            )
        if stored_value is not None and stored_value != resolved:
            raise ValueError(
                f"{path} declares 3DGR-CAR alignment {stored_value!r}, which "
                f"conflicts with --3dgrcar-alignment={resolved!r}."
            )
        return resolved, "cli"
    if stored_value is not None:
        return stored_value, f"npz_metadata:{stored[1]}"
    raise ValueError(
        f"{path} omits 3DGR-CAR ground-truth alignment mode. Its "
        "ground_truth_spacing/origin describe the saved prediction grid only "
        "for same-grid evaluation; in physical mode the saved mask remains on "
        "the centred reconstruction grid. Pass --3dgrcar-alignment physical "
        "or --3dgrcar-alignment same-grid."
    )


def _load_normalized_prediction(
    path: Path,
    *,
    requested_method: str,
    prediction_key: str | None,
    coordinate_frame: str,
    origin_convention: str,
    three_dgrcar_alignment: str,
    case_id_override: str | None,
) -> NormalizedPrediction:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        method, method_resolution = _resolve_prediction_method(
            data, source, requested_method
        )
        key = prediction_key or _default_prediction_key(method)
        raw_volume = _validated_volume(data, key, source)
        if (
            method == "3dgrcar" and key == "prediction_mask_zyx"
        ) or (method == "deepca" and key == "vol"):
            unique = np.unique(raw_volume)
            if not np.all(np.isin(unique, (0, 1))):
                raise ValueError(
                    f"{source} {key} is expected to be a binary mask but has "
                    f"values={unique[:10].tolist()}."
                )
        axis_order = _axis_order(data, source, method=method, key=key)
        volume = _to_zyx(raw_volume, axis_order)
        alignment: str | None = None
        alignment_source: str | None = None
        frame_default: tuple[str, str] | None = None
        convention_default: tuple[str, str] | None = None
        if method == "3dgrcar":
            alignment, alignment_source = _resolve_3dgrcar_alignment(
                data, source, three_dgrcar_alignment
            )
            if alignment == "physical":
                frame_default = (
                    "projection_centered",
                    "3dgrcar_physical_alignment_contract",
                )
            else:
                frame_default = (
                    "native",
                    "3dgrcar_same_grid_alignment_contract",
                )
            convention_default = (
                "voxel_center",
                "3dgrcar_exporter_contract",
            )
        frame, frame_source = _resolve_prediction_frame(
            data,
            source,
            method=method,
            requested=coordinate_frame,
            schema_default=frame_default,
        )
        convention, convention_source = _resolve_origin_convention(
            data,
            source,
            method=method,
            requested=origin_convention,
            schema_default=convention_default,
        )
        if method == "3dgrcar" and frame_default is not None:
            if frame != frame_default[0]:
                raise ValueError(
                    f"{source} 3DGR-CAR {alignment} alignment requires "
                    f"prediction frame {frame_default[0]!r}, not {frame!r}."
                )
            if convention != "voxel_center":
                raise ValueError(
                    f"{source} 3DGR-CAR exporter origins describe the first "
                    "voxel centre; lower-bound interpretation is invalid."
                )
        metadata = _common_prediction_metadata(data, source)
        case_id, case_id_source = _prediction_case_id(
            data,
            source,
            case_id_override=case_id_override,
        )
        metadata["case_id"] = case_id
        metadata["case_id_source"] = case_id_source
        metadata["volume_axis_order"] = axis_order
        if alignment is not None:
            metadata["3dgrcar_alignment"] = alignment
            metadata["3dgrcar_alignment_source"] = alignment_source

        if method == "autocar":
            if "voxel_size_mm" not in data.files:
                raise ValueError(
                    f"{source} is missing AutoCAR voxel_size_mm metadata."
                )
            size = float(
                _metadata_scalar_value(data, "voxel_size_mm", source)
            )
            if not math.isfinite(size) or size <= 0.0:
                raise ValueError(f"{source} voxel_size_mm must be positive.")
            spacing = np.full(3, size, dtype=np.float64)
            stored_origin = _xyz_vector(data, "bbox_min_xyz_mm", source)
            units_source = "npz_mm"
        elif method == "deepca":
            spacing = _xyz_vector(
                data, "spacing", source, positive=True
            )
            stored_origin = _xyz_vector(data, "origin", source)
            units_source = "deepca_exporter_contract_mm"
        else:
            if alignment == "same_grid":
                spacing = _xyz_vector(
                    data,
                    "ground_truth_spacing_xyz_m",
                    source,
                    scale_to_mm=1000.0,
                    positive=True,
                )
                stored_origin = _xyz_vector(
                    data,
                    "ground_truth_origin_xyz_m",
                    source,
                    scale_to_mm=1000.0,
                )
                units_source = "3dgrcar_same_grid_gt_metadata_metres"
            else:
                center_offset = _xyz_vector(
                    data,
                    "projection_center_offset_xyz_m",
                    source,
                    scale_to_mm=1000.0,
                )
                shift_zyx = _xyz_vector(
                    data,
                    "applied_prediction_shift_zyx_voxels",
                    source,
                )
                offset_zyx = center_offset[::-1]
                nonzero_shift = np.abs(shift_zyx) > 1e-12
                nonzero_offset = np.abs(offset_zyx) > 1e-12
                if np.any(nonzero_shift != nonzero_offset):
                    raise ValueError(
                        f"{source} physical-mode projection offset and voxel "
                        "shift disagree on which axes were translated; the "
                        "prediction-grid spacing cannot be recovered safely."
                    )
                usable = nonzero_shift & nonzero_offset
                if not np.any(usable):
                    raise ValueError(
                        f"{source} cannot recover its physical-mode "
                        "reconstruction spacing from projection offset/shift. "
                        "Re-export volume_extent metadata."
                    )
                recovered = offset_zyx[usable] / shift_zyx[usable]
                if np.any(recovered <= 0.0):
                    raise ValueError(
                        f"{source} physical-mode projection offset and voxel "
                        "shift have inconsistent signs; the prediction-grid "
                        "spacing cannot be recovered safely."
                    )
                if not np.allclose(
                    recovered,
                    recovered[0],
                    rtol=5e-5,
                    atol=1e-6,
                ):
                    raise ValueError(
                        f"{source} physical-mode spacing inferred from centre "
                        f"offset/shift is inconsistent: {recovered.tolist()}."
                    )
                spacing = np.full(3, recovered[0], dtype=np.float64)
                extent_xyz = (
                    np.asarray(volume.shape[::-1], dtype=np.float64) - 1.0
                ) * spacing
                stored_origin = -0.5 * extent_xyz
                units_source = (
                    "3dgrcar_physical_grid_derived_from_offset_and_shift"
                )
            for companion in (
                "ground_truth_volume_zyx",
                "ground_truth_mask_zyx",
                "roi_mask_zyx",
            ):
                if companion in data.files and np.asarray(data[companion]).shape != volume.shape:
                    raise ValueError(
                        f"{source} {companion} shape does not match {key}: "
                        f"{np.asarray(data[companion]).shape} versus {volume.shape}."
                    )
            if "applied_prediction_shift_zyx_voxels" in data.files:
                shift = np.asarray(
                    data["applied_prediction_shift_zyx_voxels"],
                    dtype=np.float64,
                )
                if shift.shape != (3,) or not np.isfinite(shift).all():
                    raise ValueError(
                        f"{source} applied_prediction_shift_zyx_voxels must "
                        "contain three finite values."
                    )
                metadata["applied_prediction_shift_zyx_voxels"] = shift.tolist()
            metadata["axis_direction_sign_xyz"] = [1, 1, 1]
            metadata["axis_direction_sign_source"] = (
                "3dgrcar_exporter_default_not_embedded"
            )
        lower_origin = _lower_bound_origin(
            stored_origin, spacing, convention
        )
        metadata["source_prediction_coordinate_frame"] = frame
        metadata["source_prediction_coordinate_frame_source"] = frame_source
        effective_frame = frame
        effective_frame_source = frame_source
        if frame == "native":
            if method == "deepca":
                inferred_center = stored_origin + 0.5 * (
                    np.asarray(volume.shape[::-1], dtype=np.float64) - 1.0
                ) * spacing
                offset_source = "deepca_grid_center_exporter_contract"
            else:
                embedded_offset = metadata.get(
                    "projection_center_offset_xyz_mm"
                )
                if embedded_offset is None:
                    raise ValueError(
                        f"{source} stores a native-frame {method} volume but "
                        "does not provide projection_center_offset_xyz_mm. "
                        "The shared evaluation grid is projection-centred."
                    )
                inferred_center = np.asarray(
                    embedded_offset, dtype=np.float64
                )
                offset_source = "prediction_npz"
            embedded_offset = metadata.get("projection_center_offset_xyz_mm")
            if embedded_offset is not None and not np.allclose(
                inferred_center,
                embedded_offset,
                rtol=0.0,
                atol=1e-4,
            ):
                raise ValueError(
                    f"{source} inferred grid centre "
                    f"{inferred_center.tolist()} disagrees with its stored "
                    "projection centre offset."
                )
            metadata["projection_center_offset_xyz_mm"] = (
                inferred_center.tolist()
            )
            metadata["projection_center_offset_inference"] = offset_source
            lower_origin = lower_origin - inferred_center
            effective_frame = "projection_centered"
            effective_frame_source = (
                f"normalized_from_native_using_{offset_source}"
            )
        source_grid = VoxelGrid(
            shape_zyx=tuple(int(value) for value in volume.shape),
            spacing_xyz_mm=tuple(float(value) for value in spacing),
            origin_xyz_mm=tuple(float(value) for value in lower_origin),
        )
        metadata["stored_origin_xyz_mm"] = stored_origin.tolist()
        metadata["spacing_xyz_mm"] = spacing.tolist()
        metadata["physical_units_source"] = units_source
        if "bbox_max_xyz_mm" in data.files:
            declared_max = _xyz_vector(data, "bbox_max_xyz_mm", source)
            if not np.allclose(
                declared_max,
                source_grid.upper_bound_xyz_mm,
                rtol=0.0,
                atol=max(1e-6, float(np.max(spacing)) * 1e-5),
            ):
                raise ValueError(
                    f"{source} shape/origin/spacing disagree with bbox_max_xyz_mm."
                )
        checkpoint = metadata.get("checkpoint")
    return NormalizedPrediction(
        path=source,
        method=method,
        method_resolution=method_resolution,
        volume_key=key,
        volume_zyx=volume,
        source_grid=source_grid,
        coordinate_frame=effective_frame,
        coordinate_frame_source=effective_frame_source,
        origin_convention=convention,
        origin_convention_source=convention_source,
        metadata=metadata,
        volume_is_binary_mask=(
            (method == "3dgrcar" and key == "prediction_mask_zyx")
            or (method == "deepca" and key == "vol")
        ),
        checkpoint=None if checkpoint is None else str(checkpoint),
    )


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
    artery: str | None = None,
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
                if artery is not None and not _matches_requested_artery(
                    path, artery=artery
                ):
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
    path_markers: set[str] = set()
    for component in reversed(path.parts):
        component_markers = _artery_markers_text(component)
        if len(component_markers) == 1:
            path_markers = component_markers
            break
    metadata_markers: set[str] = set()
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
            value = _safe_scalar_metadata(data, key, path)
            if value is None or not isinstance(value, (str, bytes, np.str_)):
                continue
            metadata_markers.update(_artery_markers_text(value))
    if metadata_markers:
        # Components containing both labels are ignored above because they are
        # commonly shared roots (for example, lca_rca_results), not file-level
        # anatomy. A single nearest marker is specific enough to cross-check.
        metadata_markers.update(path_markers)
        return metadata_markers
    return path_markers


def _matches_requested_artery(path: Path, *, artery: str) -> bool:
    markers = _artery_markers(path)
    if len(markers) > 1:
        raise ValueError(
            f"Cannot assign {path} to one anatomy because it contains "
            f"conflicting artery markers {sorted(markers)}."
        )
    return not markers or artery in markers


def _resolve_raw_vessel_key(payload: Any, requested: str, path: Path) -> str:
    if requested != "auto":
        if requested not in payload.files:
            raise KeyError(
                f"{path} lacks raw vessel key {requested!r}; available: "
                f"{sorted(payload.files)}"
            )
        return requested
    for candidate in _RAW_VESSEL_AUTO_KEYS:
        if candidate in payload.files:
            return candidate
    raise KeyError(
        f"{path} has none of the supported raw vessel keys "
        f"{list(_RAW_VESSEL_AUTO_KEYS)}; available: {sorted(payload.files)}"
    )


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
        value = _safe_scalar_metadata(payload, key, path)
        if value is None:
            raise ValueError(
                f"{path} {key} is pickle-backed object metadata and cannot be "
                "safely inspected. Pass --raw-coordinate-frame explicitly."
            )
        stored = _scalar_text(value).lower().replace("-", "_")
        resolved = _RAW_FRAME_ALIASES.get(stored)
        if resolved is None:
            raise ValueError(f"{path} declares unsupported {key}={stored!r}.")
        return resolved
    if vessel_key in _RAW_VESSEL_AUTO_KEYS:
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
        resolved_vessel_key = _resolve_raw_vessel_key(data, vessel_key, source)
        vessel = np.asarray(data[resolved_vessel_key], dtype=np.float64)
        single_branch_input = vessel.ndim == 2
        if single_branch_input and vessel.shape[-1] >= 4:
            vessel = vessel[None, ...]
        if vessel.ndim != 3 or vessel.shape[-1] < 4:
            raise ValueError(
                f"{source} {resolved_vessel_key} must have shape [M,N,4+] "
                f"or [N,4+], got {vessel.shape}. XYZ plus radius are required "
                "to compute Dice."
            )
        vessel = vessel[..., :4].copy()
        if scale_to_mm is None:
            schema_scale = _RAW_VESSEL_SCHEMA_SCALE_TO_MM.get(
                resolved_vessel_key
            )
            suffix_scale = 1.0 if resolved_vessel_key.endswith("_mm") else None
            expected_scale = (
                schema_scale if schema_scale is not None else suffix_scale
            )
            needs_metadata_scale = (
                expected_scale is None or resolved_vessel_key == "artery"
            )
            metadata_scale = (
                float(np.asarray(data["input_scale_to_mm"]).reshape(()))
                if needs_metadata_scale and "input_scale_to_mm" in data.files
                else None
            )
            if expected_scale is not None:
                scale = expected_scale
                scale_source = (
                    f"verified_schema:{resolved_vessel_key}"
                    if schema_scale is not None
                    else "millimetre_key_suffix"
                )
                # input_scale_to_mm in transformed Stage-2 targets describes
                # the metre-valued `artery` field, so it may legitimately be
                # 1000 beside a separate *_mm array. Cross-check it only when
                # the selected field is artery; fixed-unit mm keys ignore it.
                if (
                    resolved_vessel_key == "artery"
                    and metadata_scale is not None
                    and not math.isclose(
                        metadata_scale, scale, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise ValueError(
                        f"{source} input_scale_to_mm={metadata_scale} conflicts "
                        f"with the verified {resolved_vessel_key!r} scale "
                        f"{scale}."
                    )
            elif metadata_scale is not None:
                scale = metadata_scale
                scale_source = "npz_metadata:input_scale_to_mm"
            else:
                raise ValueError(
                    f"Units of {source} {resolved_vessel_key!r} are ambiguous; "
                    "pass "
                    "--raw-scale-to-mm."
                )
        else:
            scale = float(scale_to_mm)
            scale_source = "cli_override"
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("raw vessel scale-to-mm must be positive and finite.")
        vessel *= scale

        if point_valid_key in data.files:
            point_valid = np.asarray(data[point_valid_key], dtype=np.bool_)
            if single_branch_input and point_valid.shape == vessel.shape[1:2]:
                point_valid = point_valid[None, :]
            point_valid_source = f"npz:{point_valid_key}"
        else:
            point_valid = np.isfinite(vessel).all(axis=-1) & (vessel[..., 3] > 0)
            point_valid_source = "inferred_finite_positive_radius"
        if point_valid.shape != vessel.shape[:2]:
            raise ValueError(
                f"{source} {point_valid_key} must have shape {vessel.shape[:2]}, "
                f"got {point_valid.shape}."
            )
        if branch_exists_key in data.files:
            branch_exists = np.asarray(
                data[branch_exists_key], dtype=np.bool_
            ).reshape(-1)
            branch_exists_source = f"npz:{branch_exists_key}"
        else:
            # Legacy Stage-2 files pad artery to a fixed branch capacity with
            # all-zero rows. Infer branch existence from their valid points so
            # padded slots do not become empty ground-truth branches.
            branch_exists = np.count_nonzero(point_valid, axis=1) >= 2
            branch_exists_source = "inferred_at_least_two_valid_points"
        if branch_exists.shape != (vessel.shape[0],):
            raise ValueError(
                f"{source} {branch_exists_key} must have shape "
                f"({vessel.shape[0]},), got {branch_exists.shape}."
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
            vessel_key=resolved_vessel_key,
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
        vessel_key=resolved_vessel_key,
        scale_to_mm=scale,
        scale_source=scale_source,
        branch_exists_source=branch_exists_source,
        point_valid_source=point_valid_source,
    )


def _center_offset_from_prediction_or_projection(
    *,
    case_id: str,
    prediction_metadata: Mapping[str, Any],
    raw_path: Path,
    projection_paths: Mapping[str, Path] | None,
    raw_frame: str,
    prediction_frame: str,
) -> tuple[np.ndarray, str]:
    embedded = prediction_metadata.get("projection_center_offset_xyz_mm")
    candidates: list[tuple[str, np.ndarray]] = []
    if embedded is not None:
        embedded_source = str(
            prediction_metadata.get(
                "projection_center_offset_inference", "prediction_npz"
            )
        )
        candidates.append(
            (embedded_source, np.asarray(embedded, dtype=np.float64))
        )
    with np.load(raw_path, allow_pickle=False) as raw_payload:
        raw_has_offset = any(
            key in raw_payload.files
            for key in (
                "projection_center_offset",
                "projection_center_offset_xyz_mm",
            )
        )
    if raw_has_offset:
        candidates.append(
            ("raw_vessel_npz", _load_center_offset_mm(raw_path))
        )
    if projection_paths is not None and case_id in projection_paths:
        candidates.append(
            (
                "projection_npz",
                _load_center_offset_mm(projection_paths[case_id]),
            )
        )
    if candidates:
        source, offset = candidates[0]
        for comparison_source, comparison_offset in candidates[1:]:
            if not np.allclose(
                offset, comparison_offset, rtol=0.0, atol=1e-4
            ):
                raise ValueError(
                    f"Case {case_id} centre offsets disagree between "
                    f"{source} ({offset.tolist()}) and {comparison_source} "
                    f"({comparison_offset.tolist()})."
                )
    elif raw_frame == prediction_frame:
        offset = np.zeros(3, dtype=np.float64)
        source = f"not_required_matching_{raw_frame}_frames"
    else:
        raise ValueError(
            f"Case {case_id} raw vessel code is in {raw_frame} coordinates and "
            f"the prediction is in {prediction_frame} coordinates, but the "
            "prediction and raw-vessel NPZ contain no projection centre offset. "
            "Pass --projection-dir only for such legacy files."
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
        scale_to_mm=raw.scale_to_mm,
        scale_source=raw.scale_source,
        branch_exists_source=raw.branch_exists_source,
        point_valid_source=raw.point_valid_source,
    )


def _raw_vessel_in_frame(
    raw: RawVesselCode,
    *,
    target_frame: str,
    center_offset_xyz_mm: np.ndarray,
) -> RawVesselCode:
    if raw.coordinate_frame == target_frame:
        return raw
    if target_frame == "projection_centered":
        return _centered_raw_vessel(raw, center_offset_xyz_mm)
    if raw.coordinate_frame != "projection_centered" or target_frame != "native":
        raise ValueError(
            f"Cannot transform raw vessel frame {raw.coordinate_frame!r} to "
            f"prediction frame {target_frame!r}."
        )
    vessel = raw.vessel_xyzr_mm.copy()
    coordinates = vessel[..., :3]
    coordinates[raw.active_point_mask] += center_offset_xyz_mm[None, :]
    return RawVesselCode(
        path=raw.path,
        case_id=raw.case_id,
        vessel_xyzr_mm=vessel,
        branch_exists=raw.branch_exists,
        point_valid=raw.point_valid,
        coordinate_frame="native",
        vessel_key=raw.vessel_key,
        scale_to_mm=raw.scale_to_mm,
        scale_source=raw.scale_source,
        branch_exists_source=raw.branch_exists_source,
        point_valid_source=raw.point_valid_source,
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
    prediction_method: str,
    prediction_graph: VascularCenterlineGraph,
    prediction_node_xyz_mm: np.ndarray,
    raw_reference: RawVesselCode,
    comparison_coordinate_frame: str,
    evaluation_coordinate_frame: str,
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
        representation=np.asarray(
            f"{prediction_method}_voxel_graph_vs_raw_vessel_code"
        ),
        prediction_method=np.asarray(prediction_method),
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
            f"{evaluation_coordinate_frame}_xyz_mm"
        ),
        prediction_node_index_coordinate_frame=np.asarray(
            f"{evaluation_coordinate_frame}_evaluation_grid_zyx"
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
        prediction_node_evaluation_frame_xyz_mm=prediction_graph.node_xyz_mm,
        **(
            {
                "prediction_node_projection_centered_xyz_mm": (
                    prediction_graph.node_xyz_mm
                )
            }
            if evaluation_coordinate_frame == "projection_centered"
            else {"prediction_node_native_xyz_mm": prediction_graph.node_xyz_mm}
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
    prediction_method: str,
    prediction_key: str | None,
    prediction_domain: str,
    prediction_threshold: float,
    prediction_coordinate_frame: str,
    prediction_origin_convention: str,
    three_dgrcar_alignment: str,
    case_id_override: str | None,
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
    normalized = _load_normalized_prediction(
        prediction_path,
        requested_method=prediction_method,
        prediction_key=prediction_key,
        coordinate_frame=prediction_coordinate_frame,
        origin_convention=prediction_origin_convention,
        three_dgrcar_alignment=three_dgrcar_alignment,
        case_id_override=case_id_override,
    )
    prediction_metadata = normalized.metadata
    prediction = normalized.volume_zyx
    source_grid = normalized.source_grid
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
    if normalized.volume_is_binary_mask:
        if prediction_domain == "logit":
            raise ValueError(
                f"{prediction_path} uses the stored 3DGR-CAR binary mask and "
                "cannot be interpreted as logits."
            )
        source_prediction_mask = np.asarray(prediction, dtype=np.bool_)
        threshold_source = f"stored_binary_mask:{normalized.volume_key}"
    elif prediction_domain == "logit":
        prediction = _stable_sigmoid(prediction)
        source_prediction_mask = np.asarray(
            prediction >= float(prediction_threshold), dtype=np.bool_
        )
        threshold_source = "cli_or_default_after_sigmoid"
    elif np.min(prediction) < 0.0 or np.max(prediction) > 1.0:
        raise ValueError(
            f"{prediction_path} is declared as probability but contains values "
            "outside [0,1]."
        )
    else:
        source_prediction_mask = np.asarray(
            prediction >= float(prediction_threshold), dtype=np.bool_
        )
        threshold_source = "cli_or_default_probability"
    case_id = _canonical_case_id(prediction_metadata["case_id"])
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
        raw_path=raw_path,
        projection_paths=projection_paths,
        raw_frame=raw.coordinate_frame,
        prediction_frame=normalized.coordinate_frame,
    )
    if (
        ground_truth_path is not None
        and normalized.coordinate_frame == "projection_centered"
        and center_offset_source.startswith("not_required_matching_")
    ):
        raise ValueError(
            f"Case {case_id} uses a native voxel ground truth, but no exact "
            "projection centre offset is available to map the centred "
            "evaluation grid into that native volume. Embed the offset in "
            "the prediction/raw-vessel NPZ or pass --projection-dir."
        )
    raw_evaluation = _raw_vessel_in_frame(
        raw,
        target_frame=normalized.coordinate_frame,
        center_offset_xyz_mm=center_offset,
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
    raw_fov_audit = _raw_fov_audit(raw_evaluation, evaluation_grid)
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
        raw_evaluation,
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
            raw_evaluation,
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
            target_to_source_offset_xyz_mm=(
                center_offset
                if normalized.coordinate_frame == "projection_centered"
                else (0.0, 0.0, 0.0)
            ),
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
        prediction_comparison_xyz = prediction_graph.node_xyz_mm.astype(
            np.float64
        )
        if normalized.coordinate_frame == "projection_centered":
            prediction_comparison_xyz = (
                prediction_comparison_xyz + center_offset[None, :]
            )
        comparison_coordinate_frame = "native_xyz_mm"
    else:
        raw_reference = raw
        prediction_comparison_xyz = prediction_graph.node_xyz_mm.astype(
            np.float64
        )
        if normalized.coordinate_frame == "native":
            prediction_comparison_xyz = (
                prediction_comparison_xyz - center_offset[None, :]
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
        prediction_method=normalized.method,
        prediction_graph=prediction_graph,
        prediction_node_xyz_mm=prediction_comparison_xyz,
        raw_reference=raw_reference,
        comparison_coordinate_frame=comparison_coordinate_frame,
        evaluation_coordinate_frame=normalized.coordinate_frame,
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
            prediction_method=np.asarray(normalized.method),
            evaluation_coordinate_frame=np.asarray(
                normalized.coordinate_frame
            ),
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
        "prediction_method": normalized.method,
        "prediction_method_resolution": normalized.method_resolution,
        "prediction_volume_key": normalized.volume_key,
        "prediction_source_axis_order": prediction_metadata[
            "volume_axis_order"
        ],
        "prediction_source_coordinate_frame": prediction_metadata[
            "source_prediction_coordinate_frame"
        ],
        "prediction_source_coordinate_frame_source": prediction_metadata[
            "source_prediction_coordinate_frame_source"
        ],
        "prediction_evaluation_coordinate_frame": (
            normalized.coordinate_frame
        ),
        "prediction_coordinate_frame_normalization": (
            normalized.coordinate_frame_source
        ),
        "prediction_origin_convention": normalized.origin_convention,
        "prediction_origin_convention_source": (
            normalized.origin_convention_source
        ),
        "source_prediction_shape_zyx": list(prediction.shape),
        "source_prediction_spacing_xyz_mm": list(
            source_grid.spacing_xyz_mm
        ),
        "source_prediction_voxel_size_mm": (
            float(source_grid.spacing_xyz_mm[0])
            if np.allclose(
                source_grid.spacing_xyz_mm,
                source_grid.spacing_xyz_mm[0],
                rtol=0.0,
                atol=1e-9,
            )
            else None
        ),
        "source_prediction_bbox_min_xyz_mm": list(source_grid.origin_xyz_mm),
        "source_prediction_bbox_max_xyz_mm": list(
            source_grid.upper_bound_xyz_mm
        ),
        "evaluation_shape_zyx": list(evaluation_grid.shape_zyx),
        "bbox_min_xyz_mm": list(evaluation_grid.origin_xyz_mm),
        "bbox_max_xyz_mm": list(evaluation_grid.upper_bound_xyz_mm),
        "voxel_size_mm": _EVALUATION_VOXEL_SIZE_MM,
        "prediction_threshold": (
            None
            if normalized.volume_is_binary_mask
            else float(prediction_threshold)
        ),
        "prediction_threshold_requested": float(prediction_threshold),
        "prediction_threshold_source": threshold_source,
        "prediction_used_precomputed_binary_mask": bool(
            normalized.volume_is_binary_mask
        ),
        "prediction_mask_generation_threshold": (
            prediction_metadata.get("threshold")
            if normalized.volume_is_binary_mask
            else None
        ),
        "prediction_mask_generation_threshold_audit": (
            "recorded"
            if normalized.volume_is_binary_mask
            and "threshold" in prediction_metadata
            else (
                "missing_from_precomputed_mask"
                if normalized.volume_is_binary_mask
                else "not_applicable"
            )
        ),
        "prediction_checkpoint": normalized.checkpoint,
        "prediction_checkpoint_audit": (
            "recorded" if normalized.checkpoint is not None else "missing"
        ),
        "prediction_case_id_source": prediction_metadata["case_id_source"],
        "prediction_split_source": (
            "npz_metadata"
            if "dataset_split" in prediction_metadata
            else (
                "parent_directory"
                if _prediction_split(prediction_path) != "unspecified"
                else "missing_recorded_as_unspecified"
            )
        ),
        "prediction_format_metadata": {
            key: value
            for key, value in prediction_metadata.items()
            if key
            in {
                "3dgrcar_alignment",
                "3dgrcar_alignment_source",
                "applied_prediction_shift_zyx_voxels",
                "axis_direction_sign_xyz",
                "axis_direction_sign_source",
                "physical_units_source",
                "projection_center_offset_inference",
                "stored_origin_xyz_mm",
            }
        },
        "view_indices": stored_views,
        "expected_view_indices": expected_views,
        "view_indices_audit": view_indices_audit,
        "projection_center_offset_xyz_mm": center_offset.tolist(),
        "projection_center_offset_source": center_offset_source,
        "raw_vessel_key": raw.vessel_key,
        "raw_vessel_scale_to_mm": raw.scale_to_mm,
        "raw_vessel_scale_source": raw.scale_source,
        "raw_vessel_coordinate_frame": raw.coordinate_frame,
        "raw_branch_exists_source": raw.branch_exists_source,
        "raw_point_valid_source": raw.point_valid_source,
        "raw_active_branches": int(np.count_nonzero(raw.branch_exists)),
        "raw_active_points": int(np.count_nonzero(raw.active_point_mask)),
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
    prediction_paths = _discover_prediction_npzs(
        args.prediction_dir,
        requested_method=args.prediction_method,
        prediction_key=args.prediction_key,
        case_id_override=args.case_id_override,
        artery=args.artery,
    )
    raw_discovery_keys = (
        _RAW_VESSEL_AUTO_KEYS
        if args.raw_vessel_key == "auto"
        else (args.raw_vessel_key,)
    )
    raw_paths = _discover_npz_by_key(
        args.raw_vessel_dir,
        required_keys=raw_discovery_keys,
        label="raw vessel",
        preferred_filename=args.raw_preferred_filename,
        artery=args.artery,
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
            artery=args.artery,
        )
    )
    ground_truth_paths = (
        None
        if args.ground_truth_volume_dir is None
        else _discover_npz_by_key(
            args.ground_truth_volume_dir,
            required_keys=("vol",),
            label="ground-truth volume",
            artery=args.artery,
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
        "prediction_method_requested": args.prediction_method,
        "prediction_key": args.prediction_key,
        "prediction_domain": args.prediction_domain,
        "prediction_threshold": float(args.prediction_threshold),
        "prediction_coordinate_frame_requested": (
            args.prediction_coordinate_frame
        ),
        "prediction_origin_convention_requested": (
            args.prediction_origin_convention
        ),
        "three_dgrcar_alignment_requested": args.three_dgrcar_alignment,
        "case_id_override": args.case_id_override,
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
        "ground_truth_centerline_source": (
            "per_case_auto_resolved_raw_vessel_key"
            if args.raw_vessel_key == "auto"
            else args.raw_vessel_key
        ),
        "raw_vessel_key": args.raw_vessel_key,
        "raw_vessel_key_requested": args.raw_vessel_key,
        "raw_vessel_auto_keys": (
            list(_RAW_VESSEL_AUTO_KEYS)
            if args.raw_vessel_key == "auto"
            else None
        ),
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
            "method_specific_prediction_grid_normalization_to_projection_"
            "centered_coordinates_then_per_case_raw_frame_resolution"
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
            prediction_method=args.prediction_method,
            prediction_key=args.prediction_key,
            prediction_domain=args.prediction_domain,
            prediction_threshold=args.prediction_threshold,
            prediction_coordinate_frame=args.prediction_coordinate_frame,
            prediction_origin_convention=args.prediction_origin_convention,
            three_dgrcar_alignment=args.three_dgrcar_alignment,
            case_id_override=args.case_id_override,
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
    protocol["prediction_methods_resolved"] = sorted(
        {str(record["prediction_method"]) for record in records}
    )
    protocol["missing_prediction_provenance"] = {
        "checkpoint_cases": [
            str(record["case_id"])
            for record in records
            if record["prediction_checkpoint"] is None
        ],
        "split_cases": [
            str(record["case_id"])
            for record in records
            if record["split"] == "unspecified"
        ],
        "view_indices_cases": [
            str(record["case_id"])
            for record in records
            if record["view_indices"] is None
        ],
    }
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
    parser.add_argument(
        "--prediction-dir",
        "--prediction-path",
        dest="prediction_dir",
        type=Path,
        required=True,
        help="One prediction NPZ file or a directory containing prediction NPZs.",
    )
    parser.add_argument("--raw-vessel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--projection-dir",
        type=Path,
        help=(
            "Legacy fallback only when neither prediction nor raw-vessel "
            "NPZs embed the projection centre offset."
        ),
    )
    parser.add_argument("--ground-truth-volume-dir", type=Path)
    parser.add_argument(
        "--prediction-method",
        choices=("auto", *_PREDICTION_METHODS),
        default="auto",
        help=(
            "Prediction NPZ schema. 'auto' uses prediction_method metadata or "
            "a unique schema signature and records how it was resolved."
        ),
    )
    parser.add_argument(
        "--prediction-key",
        help=(
            "Override the method-default volume key. Defaults: AutoCAR "
            "prediction_volume_zyx, DeepCA vol, and 3DGR-CAR "
            "prediction_mask_zyx."
        ),
    )
    parser.add_argument(
        "--prediction-domain",
        choices=("probability", "logit"),
        default="probability",
    )
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument(
        "--prediction-coordinate-frame",
        choices=("auto", "native", "projection-centered"),
        default="auto",
        help=(
            "Physical frame of the saved volume before normalization to the "
            "shared projection-centred evaluation frame."
        ),
    )
    parser.add_argument(
        "--prediction-origin-convention",
        choices=("auto", "lower-bound", "voxel-center"),
        default="auto",
        help=(
            "Whether the stored origin is the lower voxel boundary or first "
            "voxel centre. AutoCAR/DeepCA use verified schema contracts."
        ),
    )
    parser.add_argument(
        "--3dgrcar-alignment",
        dest="three_dgrcar_alignment",
        choices=("auto", "physical", "same-grid"),
        default="auto",
        help=(
            "Required for legacy 3DGR-CAR NPZs that omit their alignment mode. "
            "Physical and same-grid files use different saved prediction grids."
        ),
    )
    parser.add_argument(
        "--case-id-override",
        help=(
            "Case ID for one prediction NPZ that lacks case_id metadata. This "
            "is rejected for directory input."
        ),
    )
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
    parser.add_argument(
        "--raw-vessel-key",
        default="auto",
        help=(
            "Raw XYZ+radius array key. The default 'auto' prefers "
            "raw_vessel_code_mm, then uniform_arc_vessel_code_mm, then the "
            "native ImageCAS branches_xyzr_resampled field, then the legacy "
            "Stage-2 artery field. Stage-2 artery values are automatically "
            "converted from metres to millimetres."
        ),
    )
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
