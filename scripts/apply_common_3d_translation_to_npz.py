#!/usr/bin/env python3
"""Apply one camera-consistent 3D translation to two NPZ input masks.

This utility is intended for real two-view masks whose apparent detector
placement is inconsistent with the projection-centred synthetic training data.
It does not centre the two masks independently.  Instead, it fits one world
translation whose projections best explain the requested two detector shifts,
then applies the corresponding integer translation to each binary mask.

Because a real vessel's depth is unavailable, each projected mask is shifted
rigidly by the displacement of the world origin.  This is a documented
small-object approximation to re-rendering the unknown 3D vessel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.projection_geometry import ProjectionGeometry


def _foreground_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(np.asarray(mask) > 0.5)
    if rows.size == 0:
        raise ValueError("Cannot calibrate or translate an empty input mask.")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _bbox_centres_xy(images: np.ndarray) -> np.ndarray:
    centres = []
    for image in images:
        x0, y0, x1, y1 = _foreground_bbox_xyxy(image)
        centres.append((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    return np.asarray(centres, dtype=np.float64)


def _geometry(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    *,
    image_dim: int,
    sid_mm: float,
    pixel_spacing_mm: float,
    source_to_isocenter_mm: float,
) -> ProjectionGeometry:
    return ProjectionGeometry.from_angles(
        theta_deg=np.asarray(theta_deg, dtype=np.float64),
        phi_deg=np.asarray(phi_deg, dtype=np.float64),
        image_dim=int(image_dim),
        sid_mm=float(sid_mm),
        pixel_spacing_mm=float(pixel_spacing_mm),
        source_to_isocenter_mm=float(source_to_isocenter_mm),
    )


def _project_translation_xy(
    geometry: ProjectionGeometry, translation_xyz_mm: np.ndarray
) -> np.ndarray:
    origin = np.zeros((1, 3), dtype=np.float64)
    translated = np.asarray(translation_xyz_mm, dtype=np.float64).reshape(1, 3)
    shifts = []
    for view_index in range(geometry.num_views):
        origin_xy = geometry.project_points_xy(origin, view_index)[0]
        translated_xy = geometry.project_points_xy(translated, view_index)[0]
        shifts.append(translated_xy - origin_xy)
    return np.asarray(shifts, dtype=np.float64)


def _fit_translation_xyz_mm(
    geometry: ProjectionGeometry,
    desired_shifts_xy_px: np.ndarray,
    *,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single translation using a small Gauss-Newton solve."""

    desired = np.asarray(desired_shifts_xy_px, dtype=np.float64)
    if desired.shape != (2, 2):
        raise ValueError(
            "desired_shifts_xy_px must have shape [2,2] in (x,y) order."
        )

    epsilon_mm = 0.01
    zero = np.zeros(3, dtype=np.float64)
    jacobian = np.empty((2, 2, 3), dtype=np.float64)
    for axis in range(3):
        step = zero.copy()
        step[axis] = epsilon_mm
        jacobian[:, :, axis] = (
            _project_translation_xy(geometry, step)
            - _project_translation_xy(geometry, zero)
        ) / epsilon_mm
    translation = np.linalg.lstsq(
        jacobian.reshape(4, 3), desired.reshape(4), rcond=None
    )[0]

    for _ in range(iterations):
        projected = _project_translation_xy(geometry, translation)
        residual = desired - projected
        local_jacobian = np.empty((2, 2, 3), dtype=np.float64)
        for axis in range(3):
            stepped = translation.copy()
            stepped[axis] += epsilon_mm
            local_jacobian[:, :, axis] = (
                _project_translation_xy(geometry, stepped) - projected
            ) / epsilon_mm
        increment = np.linalg.lstsq(
            local_jacobian.reshape(4, 3), residual.reshape(4), rcond=None
        )[0]
        translation += increment
        if float(np.linalg.norm(increment)) < 1e-7:
            break

    projected = _project_translation_xy(geometry, translation)
    return translation, projected


def _shift_binary_mask(
    mask: np.ndarray, *, shift_x: int, shift_y: int
) -> tuple[np.ndarray, int]:
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {binary.shape}.")
    height, width = binary.shape
    output = np.zeros_like(binary)

    source_x0 = max(0, -shift_x)
    source_x1 = min(width, width - shift_x)
    source_y0 = max(0, -shift_y)
    source_y1 = min(height, height - shift_y)
    if source_x1 > source_x0 and source_y1 > source_y0:
        destination_x0 = source_x0 + shift_x
        destination_x1 = source_x1 + shift_x
        destination_y0 = source_y0 + shift_y
        destination_y1 = source_y1 + shift_y
        output[destination_y0:destination_y1, destination_x0:destination_x1] = (
            binary[source_y0:source_y1, source_x0:source_x1]
        )
    clipped_pixels = int(binary.sum(dtype=np.int64) - output.sum(dtype=np.int64))
    return output, clipped_pixels


def _scalar_float(
    arrays: Mapping[str, np.ndarray], key: str, fallback: float
) -> float:
    if key not in arrays:
        return float(fallback)
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}.")
    return float(value.reshape(-1)[0])


def _load_json_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return result


def _json_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def apply_common_translation(
    input_path: Path,
    output_path: Path,
    *,
    target_bbox_centres_xy_px: np.ndarray | None,
    calibration_input_path: Path | None,
    calibration_output_path: Path | None,
    sid_mm: float,
    pixel_spacing_mm: float,
    source_to_isocenter_mm: float,
    reference_case_ids: Sequence[str],
    reference_note: str,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_path}; pass --overwrite to opt in."
        )
    if calibration_output_path is not None:
        if calibration_output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to replace {calibration_output_path}; pass "
                "--overwrite to opt in."
            )
    if (target_bbox_centres_xy_px is None) == (calibration_input_path is None):
        raise ValueError(
            "Provide exactly one of target bbox centres or a calibration JSON."
        )

    with np.load(input_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    for key in ("images", "theta_deg", "phi_deg"):
        if key not in arrays:
            raise KeyError(f"{input_path} is missing required key {key!r}.")
    images = np.asarray(arrays["images"])
    if images.ndim != 3 or images.shape[0] != 2:
        raise ValueError(f"images must have shape [2,H,W], got {images.shape}.")
    if images.shape[1] != images.shape[2]:
        raise ValueError(f"Input masks must be square, got {images.shape[1:]}")
    theta_deg = np.asarray(arrays["theta_deg"], dtype=np.float64).reshape(-1)
    phi_deg = np.asarray(arrays["phi_deg"], dtype=np.float64).reshape(-1)
    if theta_deg.shape != (2,) or phi_deg.shape != (2,):
        raise ValueError("theta_deg and phi_deg must each contain two values.")

    image_dim = int(images.shape[-1])
    effective_sid_mm = _scalar_float(arrays, "sid", sid_mm)
    effective_spacing_mm = _scalar_float(
        arrays, "imager_pixel_spacing", pixel_spacing_mm
    )
    effective_source_to_isocenter_mm = _scalar_float(
        arrays, "source_to_isocenter_mm", source_to_isocenter_mm
    )
    geometry = _geometry(
        theta_deg,
        phi_deg,
        image_dim=image_dim,
        sid_mm=effective_sid_mm,
        pixel_spacing_mm=effective_spacing_mm,
        source_to_isocenter_mm=effective_source_to_isocenter_mm,
    )
    centres_before = _bbox_centres_xy(images)

    source_calibration: dict[str, Any] | None = None
    if calibration_input_path is not None:
        source_calibration = _load_json_object(calibration_input_path)
        stored_theta = np.asarray(
            source_calibration["camera"]["theta_deg"], dtype=np.float64
        )
        stored_phi = np.asarray(
            source_calibration["camera"]["phi_deg"], dtype=np.float64
        )
        if not (
            np.allclose(theta_deg, stored_theta, atol=1e-4, rtol=0.0)
            and np.allclose(phi_deg, stored_phi, atol=1e-4, rtol=0.0)
        ):
            raise ValueError(
                "The calibration camera angles do not match this NPZ. "
                "This safety check prevents applying Case 10 shifts to a "
                "different camera pair."
            )
        translation_xyz_mm = np.asarray(
            source_calibration["fit"]["translation_xyz_mm"],
            dtype=np.float64,
        )
        projected_shifts = _project_translation_xy(
            geometry, translation_xyz_mm
        )
        applied_shifts = np.asarray(
            source_calibration["fit"]["applied_integer_shift_xy_px"],
            dtype=np.int64,
        )
        if applied_shifts.shape != (2, 2):
            raise ValueError(
                "Calibration applied_integer_shift_xy_px must have shape [2,2]."
            )
        desired_shifts = np.asarray(
            source_calibration["fit"]["desired_shift_xy_px"],
            dtype=np.float64,
        )
        target_centres = centres_before + desired_shifts
        fit_mode = "replay_saved_calibration"
    else:
        target_centres = np.asarray(
            target_bbox_centres_xy_px, dtype=np.float64
        ).reshape(2, 2)
        desired_shifts = target_centres - centres_before
        translation_xyz_mm, projected_shifts = _fit_translation_xyz_mm(
            geometry, desired_shifts
        )
        applied_shifts = np.rint(projected_shifts).astype(np.int64)
        fit_mode = "fit_from_target_bbox_centres"

    adjusted = []
    clipped_pixels = []
    for image, (shift_x, shift_y) in zip(images, applied_shifts):
        shifted, clipped = _shift_binary_mask(
            image, shift_x=int(shift_x), shift_y=int(shift_y)
        )
        adjusted.append(shifted.astype(np.float32))
        clipped_pixels.append(clipped)
    adjusted_images = np.stack(adjusted, axis=0)
    centres_after = _bbox_centres_xy(adjusted_images)
    residual = desired_shifts - projected_shifts

    camera_record = {
        "theta_deg": theta_deg.tolist(),
        "phi_deg": phi_deg.tolist(),
        "image_dim": image_dim,
        "sid_mm": effective_sid_mm,
        "pixel_spacing_mm": effective_spacing_mm,
        "source_to_isocenter_mm": effective_source_to_isocenter_mm,
        "pixel_coordinate_order": "xy=(column,row)",
    }
    calibration: dict[str, Any] = {
        "schema_version": 1,
        "operation": "common_3d_translation_of_two_input_masks",
        "fit_mode": fit_mode,
        "camera": camera_record,
        "fit": {
            "foreground_bbox_centres_before_xy_px": centres_before.tolist(),
            "target_foreground_bbox_centres_xy_px": target_centres.tolist(),
            "desired_shift_xy_px": desired_shifts.tolist(),
            "translation_xyz_mm": translation_xyz_mm.tolist(),
            "projected_origin_shift_xy_px": projected_shifts.tolist(),
            "fit_residual_xy_px": residual.tolist(),
            "fit_residual_l2_px": float(np.linalg.norm(residual)),
            "applied_integer_shift_xy_px": applied_shifts.tolist(),
            "foreground_bbox_centres_after_xy_px": centres_after.tolist(),
            "clipped_foreground_pixels": clipped_pixels,
        },
        "reference": {
            "case_ids": [str(case_id) for case_id in reference_case_ids],
            "note": reference_note,
        },
        "convention": {
            "translation": (
                "positive xyz translates the artery in the LAS world "
                "coordinate system used by ProjectionGeometry"
            ),
            "image_shift": "positive x is right; positive y is down",
            "mask_resampling": "integer translation with binary nearest-neighbour semantics",
            "independent_per_view_centering": False,
            "cropping_or_zoom": False,
            "approximation": (
                "the unknown 3D vessel is represented by the projected "
                "displacement of the world origin, applied uniformly to each mask"
            ),
        },
    }
    if source_calibration is not None:
        calibration["replayed_calibration_id_sha256"] = source_calibration.get(
            "calibration_id_sha256"
        )
    calibration_id = _json_digest(calibration)
    calibration["calibration_id_sha256"] = calibration_id

    arrays["images_before_common_3d_translation"] = images.astype(
        np.float32, copy=False
    )
    arrays["images"] = adjusted_images
    arrays["image_dim"] = np.asarray(image_dim, dtype=np.int32)
    arrays["imager_pixel_spacing"] = np.asarray(
        effective_spacing_mm, dtype=np.float32
    )
    arrays["imager_pixel_spacing_units"] = np.asarray("mm")
    arrays["common_3d_translation_applied"] = np.asarray(True)
    arrays["common_3d_translation_xyz_mm"] = translation_xyz_mm.astype(
        np.float32
    )
    arrays["common_3d_translation_projected_shift_xy_px"] = (
        projected_shifts.astype(np.float32)
    )
    arrays["common_3d_translation_applied_integer_shift_xy_px"] = (
        applied_shifts.astype(np.int32)
    )
    arrays["common_3d_translation_desired_shift_xy_px"] = (
        desired_shifts.astype(np.float32)
    )
    arrays["common_3d_translation_fit_residual_xy_px"] = residual.astype(
        np.float32
    )
    arrays["common_3d_translation_bbox_centres_before_xy_px"] = (
        centres_before.astype(np.float32)
    )
    arrays["common_3d_translation_bbox_centres_after_xy_px"] = (
        centres_after.astype(np.float32)
    )
    arrays["common_3d_translation_target_bbox_centres_xy_px"] = (
        target_centres.astype(np.float32)
    )
    arrays["common_3d_translation_clipped_foreground_pixels"] = np.asarray(
        clipped_pixels, dtype=np.int64
    )
    arrays["common_3d_translation_calibration_id_sha256"] = np.asarray(
        calibration_id
    )
    arrays["common_3d_translation_calibration_json"] = np.asarray(
        json.dumps(calibration, sort_keys=True)
    )

    metadata: dict[str, Any] = {}
    if "conversion_metadata_json" in arrays:
        raw = np.asarray(arrays["conversion_metadata_json"])
        if raw.shape == ():
            metadata = json.loads(str(raw.item()))
    metadata["common_3d_translation_alignment"] = calibration
    arrays["conversion_metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    if calibration_output_path is not None:
        calibration_output_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_output_path.write_text(
            json.dumps(calibration, indent=2, sort_keys=True) + "\n"
        )
    return output_path.resolve(), calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--target-bbox-centres-xy",
        type=float,
        nargs=4,
        metavar=("VIEW1_X", "VIEW1_Y", "VIEW2_X", "VIEW2_Y"),
        help=(
            "Target foreground bounding-box centres in 256-pixel model-input "
            "coordinates. One 3D translation is fitted; views are not centred "
            "independently."
        ),
    )
    mode.add_argument(
        "--calibration-json",
        type=Path,
        help="Replay a previously saved calibration exactly.",
    )
    parser.add_argument("--calibration-output-json", type=Path)
    parser.add_argument("--sid-mm", type=float, default=900.0)
    parser.add_argument("--pixel-spacing-mm", type=float, default=0.65)
    parser.add_argument(
        "--source-to-isocenter-mm", type=float, default=750.0
    )
    parser.add_argument("--reference-case-ids", nargs="*", default=())
    parser.add_argument("--reference-note", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    target = None
    if args.target_bbox_centres_xy is not None:
        target = np.asarray(args.target_bbox_centres_xy, dtype=np.float64).reshape(
            2, 2
        )
    output_path, calibration = apply_common_translation(
        args.input_npz,
        args.output_npz,
        target_bbox_centres_xy_px=target,
        calibration_input_path=args.calibration_json,
        calibration_output_path=args.calibration_output_json,
        sid_mm=float(args.sid_mm),
        pixel_spacing_mm=float(args.pixel_spacing_mm),
        source_to_isocenter_mm=float(args.source_to_isocenter_mm),
        reference_case_ids=tuple(args.reference_case_ids),
        reference_note=str(args.reference_note),
        overwrite=bool(args.overwrite),
    )
    print(output_path)
    print(
        json.dumps(
            {
                "calibration_id_sha256": calibration[
                    "calibration_id_sha256"
                ],
                "translation_xyz_mm": calibration["fit"][
                    "translation_xyz_mm"
                ],
                "applied_integer_shift_xy_px": calibration["fit"][
                    "applied_integer_shift_xy_px"
                ],
                "fit_residual_l2_px": calibration["fit"][
                    "fit_residual_l2_px"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
