"""Second-view translational calibration perturbation for AutoCAR evaluation.

The stored Stage-2 artery is already projection-centred by subtracting the
recorded ``projection_center_offset``.  This module reconstructs the same tube
surface, verifies a zero-translation re-render against the stored second view,
then adds a fixed positive XYZ translation to that centred artery.  Camera
angles and model-facing camera matrices are never changed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.geometry.projection_geometry import ProjectionGeometry


TRANSLATION_SIGN_CONVENTION = (
    "translation_xyz_mm moves the projection-centred artery; the equivalent "
    "source-detector/isocentre displacement is its negative"
)


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


def resolve_evaluation_view_translation(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one child-run translation condition."""

    raw = config.get("evaluation_view_translation")
    if raw is None:
        return {
            "enabled": False,
            "translation_xyz_mm": [0.0, 0.0, 0.0],
            "translation_magnitude_mm": 0.0,
            "perturbed_input_position": 1,
            "renderer_num_circle_points": 120,
            "minimum_clean_rerender_dice": 0.98,
            "visibility_warning_threshold": 0.95,
            "fail_below_visibility_threshold": False,
            "sign_convention": TRANSLATION_SIGN_CONVENTION,
        }
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation_view_translation must be a JSON object.")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation_view_translation.enabled must be boolean.")
    vector_raw = raw.get("translation_xyz_mm", raw.get("translation_mm"))
    if not isinstance(vector_raw, (list, tuple)) or len(vector_raw) != 3:
        raise ValueError(
            "evaluation_view_translation.translation_xyz_mm must be [x,y,z]."
        )
    vector = [
        _finite_number(
            value,
            label=f"evaluation_view_translation.translation_xyz_mm[{index}]",
        )
        for index, value in enumerate(vector_raw)
    ]
    magnitude = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
    if enabled and magnitude <= 0.0:
        raise ValueError("An enabled evaluation view translation must be non-zero.")
    if any(value < 0.0 for value in vector):
        raise ValueError(
            "This fixed-positive-direction stress test does not accept negative "
            "translation components."
        )
    position = raw.get("perturbed_input_position", 1)
    if isinstance(position, bool) or int(position) != 1:
        raise ValueError(
            "evaluation_view_translation.perturbed_input_position must be 1."
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
        label="evaluation_view_translation.minimum_clean_rerender_dice",
    )
    visibility_threshold = _finite_number(
        raw.get("visibility_warning_threshold", 0.95),
        label="evaluation_view_translation.visibility_warning_threshold",
    )
    if not 0.0 <= minimum_dice <= 1.0:
        raise ValueError("minimum_clean_rerender_dice must lie in [0,1].")
    if not 0.0 <= visibility_threshold <= 1.0:
        raise ValueError("visibility_warning_threshold must lie in [0,1].")
    fail_visibility = raw.get("fail_below_visibility_threshold", False)
    if not isinstance(fail_visibility, bool):
        raise ValueError("fail_below_visibility_threshold must be boolean.")
    return {
        "enabled": enabled,
        "translation_xyz_mm": vector,
        "translation_magnitude_mm": magnitude,
        "perturbed_input_position": 1,
        "renderer_num_circle_points": int(circle_points),
        "minimum_clean_rerender_dice": minimum_dice,
        "visibility_warning_threshold": visibility_threshold,
        "fail_below_visibility_threshold": fail_visibility,
        "sign_convention": TRANSLATION_SIGN_CONVENTION,
    }


def _estimate_derivatives(centerline: np.ndarray) -> np.ndarray:
    derivatives = np.zeros_like(centerline)
    if len(centerline) < 2:
        return derivatives
    derivatives[0] = centerline[1] - centerline[0]
    derivatives[-1] = centerline[-1] - centerline[-2]
    if len(centerline) > 2:
        derivatives[1:-1] = 0.5 * (centerline[2:] - centerline[:-2])
    return derivatives


def _parallel_transport_surface(
    centerline_xyz_mm: np.ndarray,
    radius_mm: np.ndarray,
    *,
    num_circle_points: int,
) -> np.ndarray:
    """Reproduce the Stage-2 tube/end-cap construction without repo imports."""

    centerline = np.asarray(centerline_xyz_mm, dtype=np.float64)
    radius = np.asarray(radius_mm, dtype=np.float64).reshape(-1)
    derivatives = _estimate_derivatives(centerline)
    derivative_norm = np.linalg.norm(derivatives, axis=1)
    valid_indices = np.flatnonzero(derivative_norm > 1e-12)
    if valid_indices.size == 0:
        raise ValueError("A vessel branch has no non-zero centreline tangent.")
    for index in np.flatnonzero(derivative_norm <= 1e-12):
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        derivatives[index] = derivatives[nearest]
        derivative_norm[index] = derivative_norm[nearest]
    tangents = derivatives / derivative_norm[:, None]

    angles = np.linspace(0.0, 2.0 * np.pi, int(num_circle_points))
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    coordinate_axes = np.eye(3, dtype=np.float64)
    previous_normal: np.ndarray | None = None
    rings: list[np.ndarray] = []
    for index, tangent in enumerate(tangents):
        normal: np.ndarray | None = None
        if previous_normal is not None:
            transported = previous_normal - np.dot(previous_normal, tangent) * tangent
            transported_norm = float(np.linalg.norm(transported))
            if transported_norm > 1e-12:
                normal = transported / transported_norm
        if normal is None:
            for axis_index in np.argsort(np.abs(coordinate_axes @ tangent)):
                candidate = coordinate_axes[int(axis_index)]
                candidate = candidate - np.dot(candidate, tangent) * tangent
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate_norm > 1e-12:
                    normal = candidate / candidate_norm
                    break
        if normal is None:
            raise ValueError("Could not construct a finite vessel tube frame.")
        conormal = np.cross(normal, tangent)
        conormal /= max(float(np.linalg.norm(conormal)), 1e-12)
        normal = np.cross(tangent, conormal)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        previous_normal = normal

        if index == 0:
            ring_radii = np.linspace(0.0, radius[index], 50)[1:]
        elif index == len(centerline) - 1:
            ring_radii = np.flip(np.linspace(0.0, radius[index], 50)[1:])
        else:
            ring_radii = np.asarray([radius[index]])
        for ring_radius in ring_radii:
            rings.append(
                centerline[index][None, :]
                + float(ring_radius) * cosine * normal[None, :]
                + float(ring_radius) * sine * conormal[None, :]
            )
    surface = np.stack(rings).astype(np.float32)
    if not np.isfinite(surface).all():
        raise ValueError("Vessel tube construction produced NaN or infinity.")
    return surface


def _projection_vessel(
    path: Path,
    *,
    num_circle_points: int,
) -> dict[str, Any]:
    """Load the exact artery and centring metadata used for stored projections."""

    with np.load(path, allow_pickle=False) as payload:
        if "artery" not in payload.files:
            raise KeyError(
                f"{path} lacks 'artery'; controlled translation re-rendering "
                "requires the Stage-2 source artery stored in the projection NPZ."
            )
        artery_m = np.asarray(payload["artery"], dtype=np.float32)
        if artery_m.ndim != 3 or artery_m.shape[-1] < 4:
            raise ValueError(f"{path} artery must have shape [M,N,4].")
        projected_indices = (
            np.asarray(payload["projected_branch_indices"], dtype=np.int64).reshape(-1)
            if "projected_branch_indices" in payload.files
            else np.arange(artery_m.shape[0], dtype=np.int64)
        )
        if (
            projected_indices.size == 0
            or np.any(projected_indices < 0)
            or np.any(projected_indices >= artery_m.shape[0])
        ):
            raise ValueError(f"{path} has invalid projected_branch_indices.")
        if "projection_center_offset" not in payload.files:
            raise KeyError(
                f"{path} lacks projection_center_offset required to reproduce "
                "the stored projection centring."
            )
        center_offset_m = np.asarray(
            payload["projection_center_offset"], dtype=np.float32
        ).reshape(3)
        render_mode = (
            str(np.asarray(payload["mask_render_mode"]).reshape(()).item())
            if "mask_render_mode" in payload.files
            else "filled"
        )

    artery_mm = artery_m * np.float32(1000.0)
    center_offset_mm = center_offset_m * np.float32(1000.0)
    surfaces: list[np.ndarray] = []
    centerline_parts: list[np.ndarray] = []
    for branch_index in projected_indices:
        branch = artery_mm[int(branch_index)]
        valid = np.logical_and(
            np.any(np.abs(branch[:, :3]) > 0.0, axis=1),
            branch[:, 3] > 0.0,
        )
        if int(valid.sum()) < 2:
            continue
        centered = branch[valid, :3] - center_offset_mm.reshape(1, 3)
        centerline_parts.append(centered.astype(np.float32))
        surface = _parallel_transport_surface(
            branch[valid, :3],
            np.clip(branch[valid, 3], 1e-2, None),
            num_circle_points=num_circle_points,
        )
        surfaces.append(surface - center_offset_mm.reshape(1, 1, 3))
    if not surfaces:
        raise ValueError(f"{path} has no valid projected vessel branches.")
    return {
        "surface_rings_xyz_mm": surfaces,
        "centerline_xyz_mm": np.concatenate(centerline_parts, axis=0),
        "projected_branch_indices": projected_indices.tolist(),
        "projection_center_offset_xyz_mm": center_offset_mm.tolist(),
        "mask_render_mode": render_mode,
    }


def _render_surface_mask(
    surfaces_xyz_mm: Sequence[np.ndarray],
    *,
    geometry: ProjectionGeometry,
    view_index: int,
    render_mode: str,
) -> np.ndarray:
    """Render a Stage-2-compatible binary mask on the native detector grid."""

    try:
        from skimage import draw, filters, morphology
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "View-translation rendering requires scikit-image. Install the "
            "maintained project requirements before evaluation."
        ) from error

    image_dim = int(geometry.image_dim)
    mask = np.zeros((image_dim, image_dim), dtype=np.bool_)
    if render_mode == "point":
        points = np.concatenate(
            [np.asarray(surface).reshape(-1, 3) for surface in surfaces_xyz_mm],
            axis=0,
        )
        pixels = np.rint(geometry.project_points_xy(points, view_index))
        valid = np.logical_and(
            np.isfinite(pixels).all(axis=1),
            np.logical_and(
                np.all(pixels > 0.0, axis=1),
                np.all(pixels < image_dim, axis=1),
            ),
        )
        pixels = pixels[valid].astype(np.int64)
        if len(pixels):
            mask[pixels[:, 1], pixels[:, 0]] = True
    elif render_mode == "filled":
        for surface in surfaces_xyz_mm:
            rings = np.asarray(surface, dtype=np.float32).reshape(
                -1, np.asarray(surface).shape[-2], 3
            )
            previous_rows: np.ndarray | None = None
            previous_columns: np.ndarray | None = None
            for ring in rings:
                pixels = np.rint(geometry.project_points_xy(ring, view_index))
                valid = np.logical_and(
                    np.isfinite(pixels).all(axis=1),
                    np.logical_and(
                        np.all(pixels > 0.0, axis=1),
                        np.all(pixels < image_dim, axis=1),
                    ),
                )
                pixels = pixels[valid]
                if len(pixels) < 3:
                    continue
                rows = np.clip(pixels[:, 1].astype(np.int32), 0, image_dim - 1)
                columns = np.clip(
                    pixels[:, 0].astype(np.int32), 0, image_dim - 1
                )
                polygon_rows, polygon_columns = draw.polygon(
                    rows, columns, shape=mask.shape
                )
                mask[polygon_rows, polygon_columns] = True
                if previous_rows is not None and previous_columns is not None:
                    ring_length = min(len(rows), len(previous_rows))
                    for point_index in range(0, ring_length, 2):
                        line_rows, line_columns = draw.line(
                            int(previous_rows[point_index]),
                            int(previous_columns[point_index]),
                            int(rows[point_index]),
                            int(columns[point_index]),
                        )
                        inside = np.logical_and.reduce(
                            (
                                line_rows >= 0,
                                line_rows < image_dim,
                                line_columns >= 0,
                                line_columns < image_dim,
                            )
                        )
                        mask[line_rows[inside], line_columns[inside]] = True
                previous_rows = rows
                previous_columns = columns
    else:
        raise ValueError(f"Unsupported Stage-2 mask_render_mode={render_mode!r}.")
    closed = morphology.closing(mask, morphology.disk(2))
    return (filters.gaussian(closed, sigma=0.5) > 0.25).astype(np.float32)


def _binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    first_mask = np.asarray(first) > 0.5
    second_mask = np.asarray(second) > 0.5
    denominator = int(first_mask.sum()) + int(second_mask.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(first_mask, second_mask).sum() / denominator)


def _visible_fraction(
    points_xyz_mm: np.ndarray,
    *,
    geometry: ProjectionGeometry,
    view_index: int,
) -> float:
    pixels = geometry.project_points_xy(points_xyz_mm, view_index)
    valid = np.logical_and(
        np.isfinite(pixels).all(axis=1),
        np.logical_and(
            np.all(pixels > 0.0, axis=1),
            np.all(pixels < geometry.image_dim, axis=1),
        ),
    )
    return float(np.mean(valid)) if valid.size else 0.0


def apply_evaluation_view_translation(
    sample: Mapping[str, Any],
    options: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replace only input position 1 and retain the original camera geometry."""

    result = dict(sample)
    if not bool(options.get("enabled", False)):
        result["view_translation_applied"] = False
        return result, None

    import torch

    images = sample["images"]
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(
            "Translation evaluation expects torch images shaped [V,1,H,W]."
        )
    if images.shape[0] != 2 or images.shape[1] != 1:
        raise ValueError(
            "Translation evaluation requires exactly two single-channel views."
        )
    position = int(options["perturbed_input_position"])
    projection_path = Path(str(sample["projection_path"])).expanduser().resolve()
    vessel = _projection_vessel(
        projection_path,
        num_circle_points=int(options["renderer_num_circle_points"]),
    )
    geometry = ProjectionGeometry.from_angles(
        theta_deg=np.asarray(sample["theta_deg"].detach().cpu(), dtype=np.float32),
        phi_deg=np.asarray(sample["phi_deg"].detach().cpu(), dtype=np.float32),
        image_dim=int(np.asarray(sample["image_dim"].detach().cpu()).item()),
        sid_mm=float(np.asarray(sample["sid_mm"].detach().cpu()).item()),
        pixel_spacing_mm=float(
            np.asarray(sample["imager_pixel_spacing_mm"].detach().cpu()).item()
        ),
        source_to_isocenter_mm=float(
            np.asarray(sample["source_to_isocenter_mm"].detach().cpu()).item()
        ),
    )
    source_view_indices = np.asarray(
        sample["view_indices"].detach().cpu(), dtype=np.int64
    )
    stored = np.asarray(images[position, 0].detach().cpu(), dtype=np.float32)
    surfaces = vessel["surface_rings_xyz_mm"]
    render_mode = str(vessel["mask_render_mode"])
    clean = _render_surface_mask(
        surfaces,
        geometry=geometry,
        view_index=position,
        render_mode=render_mode,
    )
    clean_dice = _binary_dice(clean, stored)
    minimum_dice = float(options["minimum_clean_rerender_dice"])
    if clean_dice < minimum_dice:
        raise ValueError(
            f"{projection_path} zero-translation second-view re-render "
            f"Dice={clean_dice:.6f} is below {minimum_dice:.6f}. The renderer, "
            "branch subset, or centring metadata does not reproduce the stored "
            "input, so the translation would not be controlled."
        )

    translation = np.asarray(options["translation_xyz_mm"], dtype=np.float32)
    translated_surfaces = [
        np.asarray(surface, dtype=np.float32) + translation.reshape(1, 1, 3)
        for surface in surfaces
    ]
    perturbed = _render_surface_mask(
        translated_surfaces,
        geometry=geometry,
        view_index=position,
        render_mode=render_mode,
    )
    translated_images = images.clone()
    translated_images[position, 0] = torch.from_numpy(perturbed).to(
        device=images.device, dtype=images.dtype
    )
    result["images"] = translated_images
    result["view_translation_applied"] = True
    result["view_translation_xyz_mm"] = translation

    translated_centerline = (
        np.asarray(vessel["centerline_xyz_mm"], dtype=np.float32)
        + translation.reshape(1, 3)
    )
    translated_surface_points = np.concatenate(
        [surface.reshape(-1, 3) for surface in translated_surfaces], axis=0
    )
    centerline_visibility = _visible_fraction(
        translated_centerline, geometry=geometry, view_index=position
    )
    surface_visibility = _visible_fraction(
        translated_surface_points, geometry=geometry, view_index=position
    )
    threshold = float(options["visibility_warning_threshold"])
    below_threshold = bool(
        centerline_visibility < threshold or surface_visibility < threshold
    )
    if below_threshold and bool(options["fail_below_visibility_threshold"]):
        raise ValueError(
            f"{projection_path} translated view visibility is below "
            f"threshold={threshold:.6f}: centerline={centerline_visibility:.6f}, "
            f"surface={surface_visibility:.6f}."
        )
    stored_foreground = int((stored > 0.5).sum())
    perturbed_foreground = int((perturbed > 0.5).sum())
    record = {
        "case_id": str(sample["case_id"]),
        "projection_path": str(projection_path),
        "selected_source_view_indices": source_view_indices.tolist(),
        "accurate_input_position": 0,
        "perturbed_input_position": 1,
        "accurate_source_view_index": int(source_view_indices[0]),
        "perturbed_source_view_index": int(source_view_indices[1]),
        "theta_change_deg": 0.0,
        "phi_change_deg": 0.0,
        "nominal_theta_deg": np.asarray(
            sample["theta_deg"].detach().cpu(), dtype=float
        ).tolist(),
        "nominal_phi_deg": np.asarray(
            sample["phi_deg"].detach().cpu(), dtype=float
        ).tolist(),
        "artery_translation_xyz_mm": translation.astype(float).tolist(),
        "equivalent_isocentre_translation_xyz_mm": (
            -translation
        ).astype(float).tolist(),
        "translation_magnitude_mm": float(np.linalg.norm(translation)),
        "clean_rerender_dice_vs_stored": clean_dice,
        "visible_centerline_point_fraction": centerline_visibility,
        "visible_vessel_surface_point_fraction": surface_visibility,
        "visibility_warning_threshold": threshold,
        "below_visibility_warning_threshold": below_threshold,
        "stored_foreground_pixels": stored_foreground,
        "perturbed_foreground_pixels": perturbed_foreground,
        "stored_foreground_pixel_ratio": float(stored_foreground / stored.size),
        "perturbed_foreground_pixel_ratio": float(
            perturbed_foreground / perturbed.size
        ),
        "perturbed_to_stored_foreground_ratio": (
            None
            if stored_foreground == 0
            else float(perturbed_foreground / stored_foreground)
        ),
        "projected_branch_indices": vessel["projected_branch_indices"],
        "projection_center_offset_xyz_mm": vessel[
            "projection_center_offset_xyz_mm"
        ],
        "renderer_num_circle_points": int(options["renderer_num_circle_points"]),
        "mask_render_mode": render_mode,
        "image_features_recomputed": True,
        "image_feature_cache_used": False,
        "sign_convention": TRANSLATION_SIGN_CONVENTION,
    }
    return result, record


def summarize_translation_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {"num_cases": 0}
    summary: dict[str, Any] = {"num_cases": len(records)}
    for key in (
        "clean_rerender_dice_vs_stored",
        "visible_centerline_point_fraction",
        "visible_vessel_surface_point_fraction",
        "stored_foreground_pixel_ratio",
        "perturbed_foreground_pixel_ratio",
        "perturbed_to_stored_foreground_ratio",
    ):
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and not isinstance(record.get(key), bool)
            and math.isfinite(float(record[key]))
        ]
        summary[key] = (
            None
            if not values
            else {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    summary["cases_below_visibility_warning_threshold"] = sum(
        bool(record.get("below_visibility_warning_threshold"))
        for record in records
    )
    return summary


__all__ = [
    "TRANSLATION_SIGN_CONVENTION",
    "apply_evaluation_view_translation",
    "resolve_evaluation_view_translation",
    "summarize_translation_records",
]
