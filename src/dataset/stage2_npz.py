"""Paired Stage-2 projection/voxel dataset.

The projection files were produced by ``vessel_tree_generator`` and contain
views plus camera metadata.  The voxel files contain ImageCAS-style arrays in
XYZ order.  This loader establishes one unambiguous training contract:

* camera/world distances are millimetres;
* image tensors/arrays have shape ``[V, 1, H, W]``;
* ground-truth volumes have canonical array order ``[Z, Y, X]``;
* absent an explicit override, native voxel index ``(0,0,0)`` is centred at
  physical ``(0,0,0)`` and its lower-bound origin is ``-0.5 * spacing``;
* camera matrices project centred XYZ millimetres to ``(column, row)``;
* ``projection_center_offset_xyz_mm`` maps centred prediction coordinates back
  to GT coordinates by addition.

PyTorch is imported only when ``output_type="torch"`` is requested, so schema,
geometry and metric tests can run in a lightweight NumPy environment.
"""

import hashlib
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.dataset.case_ids import VALID_CASE_ID_MODES, normalize_case_id
from src.geometry.projection_geometry import (
    DEFAULT_SOURCE_TO_ISOCENTER_MM,
    ProjectionGeometry,
)


PathLike = Union[str, os.PathLike]
PathSource = Union[PathLike, Sequence[PathLike]]
_VALID_VIEW_MODES = {"all", "fixed", "random_pair"}
_VALID_OUTPUT_TYPES = {"numpy", "torch"}


class Stage2NPZError(ValueError):
    """Raised when Stage-2 inputs violate the dataset contract."""


@dataclass(frozen=True)
class Stage2NPZRecord:
    """One projection sample paired to its ground-truth voxel file."""

    case_id: str
    projection_path: Path
    voxel_path: Path


def _discover_npz_files(source: PathSource, label: str) -> List[Path]:
    if isinstance(source, (str, os.PathLike)):
        sources: Iterable[PathLike] = [source]
    else:
        sources = source

    files: List[Path] = []
    for item in sources:
        path = Path(item).expanduser()
        if path.is_dir():
            files.extend(path.rglob("*.npz"))
        elif path.is_file():
            if path.suffix.lower() != ".npz":
                raise Stage2NPZError(f"{label} file is not an NPZ: {path}")
            files.append(path)
        else:
            raise Stage2NPZError(f"{label} path does not exist: {path}")

    unique_files = sorted({path.resolve() for path in files}, key=str)
    if not unique_files:
        raise Stage2NPZError(f"No NPZ files were found for {label}.")
    return unique_files


def _scalar(data: Any, key: str, path: Path) -> Any:
    if key not in data:
        raise Stage2NPZError(f"Missing required key '{key}' in {path}.")
    value = np.asarray(data[key])
    if value.shape != ():
        raise Stage2NPZError(
            f"Key '{key}' in {path} must be scalar, got shape {value.shape}."
        )
    return value.item()


def _scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _validated_case_id(
    value: Any, path: Path, case_id_mode: str = "literal"
) -> str:
    """Return a non-empty identifier that is also safe as one filename."""

    try:
        case_id = normalize_case_id(value, mode=case_id_mode)
    except (TypeError, UnicodeError, ValueError) as error:
        raise Stage2NPZError(f"Invalid case_id in {path}: {error}") from error
    if (
        not case_id
        or case_id in {".", ".."}
        or "/" in case_id
        or "\\" in case_id
        or "\x00" in case_id
    ):
        raise Stage2NPZError(
            f"case_id in {path} must be a non-empty filename-safe identifier, "
            f"got {case_id!r}."
        )
    return case_id


def _case_id_from_projection(path: Path, case_id_mode: str) -> str:
    try:
        with np.load(path, allow_pickle=False) as data:
            case_id = _validated_case_id(
                _scalar(data, "case_id", path), path, case_id_mode
            )
    except (OSError, ValueError) as error:
        if isinstance(error, Stage2NPZError):
            raise
        raise Stage2NPZError(
            f"Could not read projection NPZ {path}: {error}"
        ) from error
    return case_id


def _case_id_from_voxel(path: Path, case_id_mode: str) -> str:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "case_id" in data:
                case_id = _validated_case_id(
                    _scalar(data, "case_id", path), path, case_id_mode
                )
            else:
                case_id = _validated_case_id(path.stem, path, case_id_mode)
    except (OSError, ValueError) as error:
        if isinstance(error, Stage2NPZError):
            raise
        raise Stage2NPZError(f"Could not read voxel NPZ {path}: {error}") from error
    return case_id


def _require_array(data: Any, key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise Stage2NPZError(f"Missing required key '{key}' in {path}.")
    return np.asarray(data[key])


def _finite_numeric_array(
    data: Any, key: str, path: Path, expected_shape: Optional[Tuple[int, ...]] = None
) -> np.ndarray:
    value = _require_array(data, key, path)
    if not (
        np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise Stage2NPZError(f"Key '{key}' in {path} must be numeric.")
    if expected_shape is not None and value.shape != expected_shape:
        raise Stage2NPZError(
            f"Key '{key}' in {path} must have shape {expected_shape}, "
            f"got {value.shape}."
        )
    if not np.isfinite(value).all():
        raise Stage2NPZError(f"Key '{key}' in {path} contains NaN or Inf.")
    return value


def _optional_text(data: Any, key: str) -> Optional[str]:
    if key not in data:
        return None
    value = np.asarray(data[key])
    if value.shape != ():
        return None
    return _scalar_text(value.item())


def _convert_length_to_mm(
    value: float, units: Optional[str], default_units: str
) -> float:
    units = (units or default_units).strip().lower()
    if units in {"mm", "millimetre", "millimetres", "millimeter", "millimeters"}:
        return float(value)
    if units in {"m", "metre", "metres", "meter", "meters"}:
        return float(value) * 1000.0
    raise Stage2NPZError(f"Unsupported length unit '{units}'.")


class Stage2NPZDataset:
    """Load projection NPZ files paired with voxel ground truth by ``case_id``.

    Args:
        projection_source: Projection NPZ file, directory, or sequence thereof.
        voxel_source: Voxel NPZ file, directory, or sequence thereof.  A voxel
            file's scalar ``case_id`` is used when present; otherwise its stem
            is the case ID (for example, ``1.npz`` is case ``"1"``).
        view_mode: ``"all"``, ``"fixed"`` or ``"random_pair"``.
        fixed_view_indices: Exactly two zero-based view positions for fixed
            mode.  Their supplied order is preserved.
        fixed_view_labels: Optional expected ``anchor_clinical_views`` labels
            at those two positions. When supplied, missing or inconsistent
            per-case metadata is rejected before training/evaluation.
        random_seed: Base seed for deterministic random pair selection.
        output_type: ``"numpy"`` or ``"torch"``.  Torch is imported lazily.
        case_id_mode: ``"literal"`` preserves IDs verbatim;
            ``"imagecas_numeric"`` joins numeric voxel stems to prefixed or
            path-based ImageCAS split/projection identifiers.
        expected_imager_pixel_spacing_mm: Optional detector-spacing invariant
            checked against each projection file when it is loaded.
        gt_origin_xyz_mm: Optional lower physical boundary of GT voxel
            ``(0,0,0)``.  When omitted, the source convention is that index
            ``(0,0,0)`` is centred at physical ``(0,0,0)``, so the lower
            boundary is derived per case as ``-0.5 * gt_spacing_xyz_mm``.
        source_to_isocenter_mm: The generator's fixed value is 750 mm.

    ``set_epoch`` changes random training pairs deterministically without using
    mutable worker-local random-number-generator state.
    """

    def __init__(
        self,
        projection_source: PathSource,
        voxel_source: PathSource,
        *,
        view_mode: str = "all",
        fixed_view_indices: Optional[Sequence[int]] = None,
        fixed_view_labels: Optional[Sequence[str]] = None,
        random_seed: int = 0,
        minimum_pair_angle_deg: float = 0.0,
        output_type: str = "numpy",
        case_ids: Optional[Sequence[str]] = None,
        case_id_mode: str = "literal",
        expected_imager_pixel_spacing_mm: Optional[float] = None,
        gt_origin_xyz_mm: Optional[Sequence[float]] = None,
        source_to_isocenter_mm: float = DEFAULT_SOURCE_TO_ISOCENTER_MM,
    ) -> None:
        if view_mode not in _VALID_VIEW_MODES:
            raise ValueError(
                f"view_mode must be one of {sorted(_VALID_VIEW_MODES)}, "
                f"got {view_mode!r}."
            )
        if output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of {sorted(_VALID_OUTPUT_TYPES)}, "
                f"got {output_type!r}."
            )
        if case_id_mode not in VALID_CASE_ID_MODES:
            raise ValueError(
                f"case_id_mode must be one of {VALID_CASE_ID_MODES}, "
                f"got {case_id_mode!r}."
            )
        if view_mode == "fixed":
            if fixed_view_indices is None or len(fixed_view_indices) != 2:
                raise ValueError(
                    "fixed_view_indices must contain exactly two indices in fixed mode."
                )
            fixed_indices = tuple(int(index) for index in fixed_view_indices)
            if len(set(fixed_indices)) != 2 or min(fixed_indices) < 0:
                raise ValueError(
                    "fixed_view_indices must contain two distinct non-negative indices."
                )
            fixed_labels = None
            if fixed_view_labels is not None:
                fixed_labels = tuple(str(value).strip() for value in fixed_view_labels)
                if len(fixed_labels) != 2 or any(not value for value in fixed_labels):
                    raise ValueError(
                        "fixed_view_labels must contain exactly two non-empty labels."
                    )
        else:
            if fixed_view_indices is not None or fixed_view_labels is not None:
                raise ValueError(
                    "fixed_view_indices/fixed_view_labels are only valid when "
                    "view_mode='fixed'."
                )
            fixed_indices = None
            fixed_labels = None

        gt_origin = None
        if gt_origin_xyz_mm is not None:
            gt_origin = np.asarray(gt_origin_xyz_mm, dtype=np.float32)
            if gt_origin.shape != (3,) or not np.isfinite(gt_origin).all():
                raise ValueError(
                    "gt_origin_xyz_mm must be None or contain three finite "
                    "lower-bound values."
                )

        source_to_isocenter_mm = float(source_to_isocenter_mm)
        if not np.isfinite(source_to_isocenter_mm) or source_to_isocenter_mm <= 0:
            raise ValueError("source_to_isocenter_mm must be finite and positive.")
        minimum_pair_angle_deg = float(minimum_pair_angle_deg)
        if (
            not np.isfinite(minimum_pair_angle_deg)
            or minimum_pair_angle_deg < 0.0
            or minimum_pair_angle_deg > 180.0
        ):
            raise ValueError("minimum_pair_angle_deg must lie in [0,180].")
        expected_pixel_spacing = None
        if expected_imager_pixel_spacing_mm is not None:
            expected_pixel_spacing = float(expected_imager_pixel_spacing_mm)
            if (
                not np.isfinite(expected_pixel_spacing)
                or expected_pixel_spacing <= 0.0
            ):
                raise ValueError(
                    "expected_imager_pixel_spacing_mm must be finite and positive."
                )

        requested: Optional[Tuple[str, ...]] = None
        requested_set: Optional[set[str]] = None
        if case_ids is not None:
            requested = tuple(
                _validated_case_id(
                    case_id, Path("<case_ids>"), case_id_mode
                )
                for case_id in case_ids
            )
            if not requested:
                raise ValueError("case_ids must contain at least one non-empty ID.")
            if len(set(requested)) != len(requested):
                raise ValueError("case_ids must not contain duplicates.")
            requested_set = set(requested)

        projection_files = _discover_npz_files(projection_source, "projections")
        voxel_files = _discover_npz_files(voxel_source, "voxel ground truth")

        voxel_by_case: Dict[str, Path] = {}
        for voxel_path in voxel_files:
            case_id = _case_id_from_voxel(voxel_path, case_id_mode)
            if case_id in voxel_by_case:
                raise Stage2NPZError(
                    f"Duplicate voxel files for case_id {case_id!r}: "
                    f"{voxel_by_case[case_id]} and {voxel_path}."
                )
            voxel_by_case[case_id] = voxel_path

        records: List[Stage2NPZRecord] = []
        missing_cases: List[str] = []
        seen_projection_cases: Dict[str, Path] = {}
        for projection_path in projection_files:
            case_id = _case_id_from_projection(projection_path, case_id_mode)
            if case_id in seen_projection_cases:
                raise Stage2NPZError(
                    f"Duplicate projection files for case_id {case_id!r}: "
                    f"{seen_projection_cases[case_id]} and {projection_path}."
                )
            seen_projection_cases[case_id] = projection_path
            if requested_set is not None and case_id not in requested_set:
                continue
            voxel_path = voxel_by_case.get(case_id)
            if voxel_path is None:
                missing_cases.append(f"{case_id} ({projection_path.name})")
                continue
            records.append(Stage2NPZRecord(case_id, projection_path, voxel_path))

        if missing_cases:
            raise Stage2NPZError(
                "No voxel NPZ was found for projection case(s): "
                + ", ".join(missing_cases)
            )
        if requested is not None:
            record_by_case = {record.case_id: record for record in records}
            missing = [case_id for case_id in requested if case_id not in record_by_case]
            if missing:
                raise Stage2NPZError(
                    "Requested case_ids have no paired inputs: " + ", ".join(missing)
                )
            records = [record_by_case[case_id] for case_id in requested]
        if not records:
            raise Stage2NPZError("No paired projection/voxel cases were found.")

        self.records = tuple(records)
        self.view_mode = view_mode
        self.fixed_view_indices = fixed_indices
        self.fixed_view_labels = fixed_labels
        self.random_seed = int(random_seed)
        self.minimum_pair_angle_deg = minimum_pair_angle_deg
        self.output_type = output_type
        self.case_id_mode = case_id_mode
        self.expected_imager_pixel_spacing_mm = expected_pixel_spacing
        self.gt_origin_xyz_mm = gt_origin
        self.source_to_isocenter_mm = source_to_isocenter_mm
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used for deterministic random-pair selection."""

        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def validate_projection_metadata(self) -> None:
        """Eagerly validate every indexed projection without loading GT volumes."""

        for record in self.records:
            self._load_projection(record.projection_path, record.case_id)

    def _view_indices(
        self,
        case_id: str,
        num_views: int,
        view_directions_world: np.ndarray,
    ) -> np.ndarray:
        if self.view_mode == "all":
            return np.arange(num_views, dtype=np.int64)
        if num_views < 2:
            raise Stage2NPZError(
                f"Case {case_id!r} has {num_views} view(s); two are required."
            )
        if self.view_mode == "fixed":
            indices = np.asarray(self.fixed_view_indices, dtype=np.int64)
            if np.any(indices >= num_views):
                raise Stage2NPZError(
                    f"Fixed view indices {indices.tolist()} are invalid for case "
                    f"{case_id!r} with {num_views} views."
                )
            return indices

        candidates = np.asarray(
            list(itertools.combinations(range(num_views), 2)), dtype=np.int64
        )
        if self.minimum_pair_angle_deg > 0.0:
            direction_pairs = view_directions_world[candidates]
            cosines = np.sum(
                direction_pairs[:, 0] * direction_pairs[:, 1], axis=1
            )
            angles = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
            candidates = candidates[angles >= self.minimum_pair_angle_deg]
        if len(candidates) == 0:
            raise Stage2NPZError(
                f"Case {case_id!r} has no view pair separated by at least "
                f"{self.minimum_pair_angle_deg:g} degrees."
            )

        selection_key = (
            f"stage2-npz-v1\0{self.random_seed}\0{self.epoch}\0{case_id}"
        ).encode("utf-8")
        seed = int.from_bytes(
            hashlib.blake2b(selection_key, digest_size=8).digest(), "little"
        )
        generator = np.random.default_rng(seed)
        return candidates[generator.integers(len(candidates))].copy()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        projection = self._load_projection(record.projection_path, record.case_id)
        voxel = self._load_voxel(record.voxel_path)

        num_views = projection["images"].shape[0]
        geometry: ProjectionGeometry = projection.pop("_geometry")
        selected = self._view_indices(
            record.case_id, num_views, geometry.view_directions_world
        )
        selected_view_labels = None
        if "anchor_clinical_views" in projection:
            selected_view_labels = tuple(
                projection["anchor_clinical_views"][int(view_index)]
                for view_index in selected
            )
        if self.fixed_view_labels is not None:
            if selected_view_labels is None:
                raise Stage2NPZError(
                    f"Case {record.case_id!r} has no anchor_clinical_views "
                    "metadata required by fixed_view_labels."
                )
            if selected_view_labels != self.fixed_view_labels:
                raise Stage2NPZError(
                    f"Case {record.case_id!r} fixed view labels "
                    f"{selected_view_labels!r} do not match expected "
                    f"{self.fixed_view_labels!r}."
                )
        pair_angle_deg = None
        if len(selected) == 2:
            cosine = float(
                np.dot(
                    geometry.view_directions_world[selected[0]],
                    geometry.view_directions_world[selected[1]],
                )
            )
            pair_angle_deg = np.float32(
                np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )

        sample: Dict[str, Any] = {
            "case_id": record.case_id,
            "sample_name": projection["sample_name"],
            "projection_path": str(record.projection_path),
            "voxel_path": str(record.voxel_path),
            "images": np.ascontiguousarray(
                projection["images"][selected, None], dtype=np.float32
            ),
            "view_indices": selected,
            "theta_deg": np.ascontiguousarray(
                geometry.theta_deg[selected], dtype=np.float32
            ),
            "phi_deg": np.ascontiguousarray(
                geometry.phi_deg[selected], dtype=np.float32
            ),
            "world2pix4x4": np.ascontiguousarray(
                geometry.world2pix4x4[selected], dtype=np.float32
            ),
            "camera_source_xyz_mm": np.ascontiguousarray(
                geometry.source_xyz_mm[selected], dtype=np.float32
            ),
            "detector_center_xyz_mm": np.ascontiguousarray(
                geometry.detector_center_xyz_mm[selected], dtype=np.float32
            ),
            "detector_x_xyz": np.ascontiguousarray(
                geometry.detector_x_xyz[selected], dtype=np.float32
            ),
            "detector_y_xyz": np.ascontiguousarray(
                geometry.detector_y_xyz[selected], dtype=np.float32
            ),
            "view_directions_world": np.ascontiguousarray(
                geometry.view_directions_world[selected], dtype=np.float32
            ),
            "projection_center_offset_xyz_mm": projection[
                "projection_center_offset_xyz_mm"
            ],
            "gt_volume_zyx": voxel["gt_volume_zyx"],
            "gt_spacing_xyz_mm": voxel["gt_spacing_xyz_mm"],
            "gt_origin_xyz_mm": (
                -0.5 * voxel["gt_spacing_xyz_mm"]
                if self.gt_origin_xyz_mm is None
                else self.gt_origin_xyz_mm.copy()
            ).astype(np.float32, copy=False),
            "image_dim": np.int64(geometry.image_dim),
            "sid_mm": np.float32(geometry.sid_mm),
            "source_to_isocenter_mm": np.float32(
                geometry.source_to_isocenter_mm
            ),
            "imager_pixel_spacing_mm": np.float32(
                geometry.pixel_spacing_mm
            ),
        }
        if pair_angle_deg is not None:
            sample["pair_angle_deg"] = pair_angle_deg
        if selected_view_labels is not None:
            sample["view_labels"] = selected_view_labels

        for key in ("view_features", "source_view_indices"):
            if key in projection:
                sample[key] = np.ascontiguousarray(projection[key][selected])
        for key in (
            "raw_vessel_code_mm",
            "reconstructed_vessel_code_mm",
            "point_valid_mask",
            "branch_exists",
            "projected_branch_indices",
        ):
            if key in projection:
                sample[key] = projection[key]

        if self.output_type == "torch":
            return _to_torch(sample)
        return sample

    def _load_projection(self, path: Path, expected_case_id: str) -> Dict[str, Any]:
        try:
            with np.load(path, allow_pickle=False) as data:
                case_id = _validated_case_id(
                    _scalar(data, "case_id", path), path, self.case_id_mode
                )
                if case_id != expected_case_id:
                    raise Stage2NPZError(
                        f"case_id in {path} changed from {expected_case_id!r} "
                        f"to {case_id!r} after dataset indexing."
                    )

                images = _finite_numeric_array(data, "images", path)
                if images.ndim != 3:
                    raise Stage2NPZError(
                        f"images in {path} must have shape [V,H,W], got {images.shape}."
                    )
                num_views, height, width = images.shape
                if num_views == 0 or height != width:
                    raise Stage2NPZError(
                        f"images in {path} must contain square non-empty views, "
                        f"got {images.shape}."
                    )
                if images.min() < 0.0 or images.max() > 1.0:
                    raise Stage2NPZError(
                        f"images in {path} must have values within [0,1]."
                    )

                theta = _finite_numeric_array(
                    data, "theta_deg", path, (num_views,)
                ).astype(np.float32)
                phi = _finite_numeric_array(
                    data, "phi_deg", path, (num_views,)
                ).astype(np.float32)
                raw_image_dim = _scalar(data, "image_dim", path)
                if (
                    isinstance(raw_image_dim, (bool, np.bool_))
                    or not np.isfinite(raw_image_dim)
                    or int(raw_image_dim) != raw_image_dim
                ):
                    raise Stage2NPZError(
                        f"image_dim in {path} must be a finite integer."
                    )
                image_dim = int(raw_image_dim)
                if image_dim != height:
                    raise Stage2NPZError(
                        f"image_dim={image_dim} in {path} does not match images "
                        f"with shape {images.shape}."
                    )

                sid = float(_scalar(data, "sid", path))
                sid_mm = _convert_length_to_mm(
                    sid, _optional_text(data, "sid_units"), default_units="m"
                )
                pixel_spacing = float(
                    _scalar(data, "imager_pixel_spacing", path)
                )
                pixel_spacing_mm = _convert_length_to_mm(
                    pixel_spacing,
                    _optional_text(data, "imager_pixel_spacing_units"),
                    default_units="mm",
                )
                if (
                    self.expected_imager_pixel_spacing_mm is not None
                    and not np.isclose(
                        pixel_spacing_mm,
                        self.expected_imager_pixel_spacing_mm,
                        rtol=0.0,
                        atol=1e-5,
                    )
                ):
                    raise Stage2NPZError(
                        f"imager_pixel_spacing in {path} is "
                        f"{pixel_spacing_mm:g} mm, expected "
                        f"{self.expected_imager_pixel_spacing_mm:g} mm."
                    )

                center_offset = _finite_numeric_array(
                    data, "projection_center_offset", path, (3,)
                ).astype(np.float64)
                center_units = _optional_text(
                    data, "projection_center_offset_units"
                )
                if center_units is not None:
                    center_offset_mm = np.array(
                        [
                            _convert_length_to_mm(value, center_units, "m")
                            for value in center_offset
                        ],
                        dtype=np.float32,
                    )
                else:
                    scale_to_mm = (
                        float(_scalar(data, "input_scale_to_mm", path))
                        if "input_scale_to_mm" in data
                        else 1000.0
                    )
                    if not np.isfinite(scale_to_mm) or scale_to_mm <= 0.0:
                        raise Stage2NPZError(
                            f"input_scale_to_mm in {path} must be finite and positive."
                        )
                    center_offset_mm = (center_offset * scale_to_mm).astype(
                        np.float32
                    )

                geometry = ProjectionGeometry.from_angles(
                    theta_deg=theta,
                    phi_deg=phi,
                    image_dim=image_dim,
                    sid_mm=sid_mm,
                    pixel_spacing_mm=pixel_spacing_mm,
                    source_to_isocenter_mm=self.source_to_isocenter_mm,
                )

                if "view_directions_world" in data:
                    recorded_directions = _finite_numeric_array(
                        data,
                        "view_directions_world",
                        path,
                        (num_views, 3),
                    )
                    if not np.allclose(
                        recorded_directions,
                        geometry.view_directions_world,
                        rtol=1e-5,
                        atol=1e-5,
                    ):
                        maximum_error = float(
                            np.max(
                                np.abs(
                                    recorded_directions
                                    - geometry.view_directions_world
                                )
                            )
                        )
                        raise Stage2NPZError(
                            f"view_directions_world in {path} disagrees with "
                            f"theta/phi geometry (max error {maximum_error:.3g})."
                        )

                result: Dict[str, Any] = {
                    "images": np.ascontiguousarray(images, dtype=np.float32),
                    "sample_name": _optional_text(data, "sample_name")
                    or path.stem,
                    "projection_center_offset_xyz_mm": np.ascontiguousarray(
                        center_offset_mm, dtype=np.float32
                    ),
                    "_geometry": geometry,
                }

                if "anchor_clinical_views" in data:
                    raw_labels = np.asarray(data["anchor_clinical_views"])
                    if raw_labels.shape != (num_views,):
                        raise Stage2NPZError(
                            f"anchor_clinical_views in {path} must have shape "
                            f"({num_views},), got {raw_labels.shape}."
                        )
                    labels = tuple(_scalar_text(value) for value in raw_labels)
                    if any(not value for value in labels):
                        raise Stage2NPZError(
                            f"anchor_clinical_views in {path} contains an empty label."
                        )
                    result["anchor_clinical_views"] = labels

                optional_view_arrays = {
                    "view_features": (num_views, None),
                    "view_indices": (num_views,),
                }
                for input_key, expected in optional_view_arrays.items():
                    if input_key not in data:
                        continue
                    value = _finite_numeric_array(data, input_key, path)
                    if expected[-1] is None:
                        valid_shape = value.ndim == 2 and value.shape[0] == num_views
                    else:
                        valid_shape = value.shape == expected
                    if not valid_shape:
                        raise Stage2NPZError(
                            f"{input_key} in {path} has incompatible shape "
                            f"{value.shape} for {num_views} views."
                        )
                    output_key = (
                        "source_view_indices"
                        if input_key == "view_indices"
                        else input_key
                    )
                    result[output_key] = np.ascontiguousarray(value)

                for key in (
                    "raw_vessel_code_mm",
                    "reconstructed_vessel_code_mm",
                    "point_valid_mask",
                    "branch_exists",
                    "projected_branch_indices",
                ):
                    if key in data:
                        result[key] = np.ascontiguousarray(data[key])
                return result
        except Stage2NPZError:
            raise
        except (OSError, ValueError, TypeError) as error:
            raise Stage2NPZError(
                f"Could not load projection NPZ {path}: {error}"
            ) from error

    @staticmethod
    def _load_voxel(path: Path) -> Dict[str, np.ndarray]:
        try:
            with np.load(path, allow_pickle=False) as data:
                volume_xyz = _require_array(data, "vol", path)
                if volume_xyz.ndim != 3 or min(volume_xyz.shape) == 0:
                    raise Stage2NPZError(
                        f"vol in {path} must have non-empty shape [X,Y,Z], "
                        f"got {volume_xyz.shape}."
                    )
                if not (
                    np.issubdtype(volume_xyz.dtype, np.number)
                    or np.issubdtype(volume_xyz.dtype, np.bool_)
                ):
                    raise Stage2NPZError(f"vol in {path} must be numeric or boolean.")
                if not np.isfinite(volume_xyz).all():
                    raise Stage2NPZError(f"vol in {path} contains NaN or Inf.")
                if np.any(volume_xyz < 0):
                    raise Stage2NPZError(
                        f"vol in {path} contains negative segmentation labels."
                    )

                spacing_xyz = _finite_numeric_array(
                    data, "spacing", path, (3,)
                ).astype(np.float32)
                spacing_units = _optional_text(data, "spacing_units")
                if spacing_units is not None:
                    spacing_xyz = np.array(
                        [
                            _convert_length_to_mm(value, spacing_units, "mm")
                            for value in spacing_xyz
                        ],
                        dtype=np.float32,
                    )
                if np.any(spacing_xyz <= 0.0):
                    raise Stage2NPZError(
                        f"spacing in {path} must contain positive XYZ values."
                    )

                # Nonzero labels all represent foreground; keep storage compact.
                volume_zyx = np.ascontiguousarray(
                    np.transpose(volume_xyz != 0, (2, 1, 0)), dtype=np.uint8
                )
                return {
                    "gt_volume_zyx": volume_zyx,
                    "gt_spacing_xyz_mm": np.ascontiguousarray(
                        spacing_xyz, dtype=np.float32
                    ),
                }
        except Stage2NPZError:
            raise
        except (OSError, ValueError, TypeError) as error:
            raise Stage2NPZError(f"Could not load voxel NPZ {path}: {error}") from error


def _to_torch(sample: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "output_type='torch' requires PyTorch; install it or use "
            "output_type='numpy'."
        ) from error

    converted: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, np.ndarray) and (
            np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.bool_)
        ):
            converted[key] = torch.from_numpy(np.ascontiguousarray(value))
        elif isinstance(value, np.generic):
            converted[key] = torch.as_tensor(value.item())
        else:
            converted[key] = value
    return converted
