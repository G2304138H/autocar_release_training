#!/usr/bin/env python3
"""Build a target-free LCA prediction NPZ from two segmented AutoCAR views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


SWAP_AXIS = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
AUTOCAR_TO_LAS = np.asarray(
    ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    dtype=np.float64,
)


def _wrap_180(value: np.ndarray) -> np.ndarray:
    return (value + 180.0) % 360.0 - 180.0


def _load_binary_mask(path: Path, threshold: int) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D mask at {path}, got {image.shape}.")
    return (image > threshold).astype(np.uint8)


def _resize_binary(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(mask * np.uint8(255))
    resized = image.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 127).astype(np.float32)


def _pose_directions(
    poses: np.ndarray, pose_convention: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if poses.shape != (2, 4, 4):
        raise ValueError(f"Expected exactly two 4x4 poses, got {poses.shape}.")
    rotations = poses[:, :3, :3].astype(np.float64)
    if pose_convention == "camera_to_world":
        raw_forward = rotations[:, :, 2]
    else:
        raw_forward = rotations[:, 2, :]
    norms = np.linalg.norm(raw_forward, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-8):
        raise ValueError("A camera pose has a degenerate forward direction.")
    raw_forward = raw_forward / norms

    # The released AutoCAR pose basis maps to the current LAS LCA projector as
    # d_LAS = -M S^T q = (-q_y, q_z, q_x).  This includes the projector's
    # centre-to-detector polarity convention.
    model_direction = np.stack(
        (-raw_forward[:, 1], raw_forward[:, 2], raw_forward[:, 0]), axis=1
    )
    theta = _wrap_180(
        np.rad2deg(np.arctan2(model_direction[:, 1], model_direction[:, 0]))
    )
    phi = np.rad2deg(
        np.arccos(np.clip(model_direction[:, 2], -1.0, 1.0))
    )

    clinical_primary = np.rad2deg(
        np.arctan2(-raw_forward[:, 1], raw_forward[:, 2])
    )
    clinical_secondary = np.rad2deg(
        np.arctan2(
            raw_forward[:, 0],
            np.hypot(raw_forward[:, 1], raw_forward[:, 2]),
        )
    )

    # AutoCAR anatomical/X-ray spherical angles retained only for auditing.
    autocar_direction = np.stack(
        (raw_forward[:, 0], raw_forward[:, 2], -raw_forward[:, 1]), axis=1
    )
    autocar_theta = np.rad2deg(
        np.arctan2(autocar_direction[:, 1], autocar_direction[:, 0])
    )
    autocar_phi = np.rad2deg(
        np.arccos(np.clip(autocar_direction[:, 2], -1.0, 1.0))
    )
    return (
        raw_forward.astype(np.float32),
        model_direction.astype(np.float32),
        theta.astype(np.float32),
        phi.astype(np.float32),
        clinical_primary.astype(np.float32),
        clinical_secondary.astype(np.float32),
        autocar_theta.astype(np.float32),
        autocar_phi.astype(np.float32),
    )


def _detector_plane_metadata(
    poses: np.ndarray,
    pose_convention: str,
    clinical_primary_deg: np.ndarray,
    clinical_secondary_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotations = poses[:, :3, :3].astype(np.float64)
    if pose_convention == "world_to_camera":
        rotations = np.transpose(rotations, (0, 2, 1))

    # Proper patient-frame change for the complete camera-to-world rotation.
    # Unlike the two-angle model-direction conversion, this does not reverse
    # the optical axis and therefore preserves detector-plane handedness.
    raw_to_las = AUTOCAR_TO_LAS @ SWAP_AXIS.T
    rotations_las = np.einsum("ij,vjk->vik", raw_to_las, rotations)

    rolls = []
    for rotation, primary_deg, secondary_deg in zip(
        rotations, clinical_primary_deg, clinical_secondary_deg
    ):
        primary = np.deg2rad(float(primary_deg))
        secondary_internal = -np.deg2rad(float(secondary_deg))
        rx = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, np.cos(primary), -np.sin(primary)),
                (0.0, np.sin(primary), np.cos(primary)),
            )
        )
        rz = np.asarray(
            (
                (
                    np.cos(secondary_internal),
                    -np.sin(secondary_internal),
                    0.0,
                ),
                (
                    np.sin(secondary_internal),
                    np.cos(secondary_internal),
                    0.0,
                ),
                (0.0, 0.0, 1.0),
            )
        )
        expected_zero_roll = SWAP_AXIS @ (rx @ rz) @ SWAP_AXIS.T
        residual = expected_zero_roll.T @ rotation
        rolls.append(np.rad2deg(np.arctan2(residual[1, 0], residual[0, 0])))

    # The released viewer rotates its local plane by +90 degrees before the
    # pose: image columns point along pose column 1 and rows-down along column 0.
    column_axis_las = rotations_las[:, :, 1]
    row_down_axis_las = rotations_las[:, :, 0]
    return (
        np.asarray(rolls, dtype=np.float32),
        rotations_las.astype(np.float32),
        column_axis_las.astype(np.float32),
        row_down_axis_las.astype(np.float32),
    )


def build(args: argparse.Namespace) -> Path:
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must lie in [0, 255].")
    if args.image_size < 1:
        raise ValueError("--image-size must be positive.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace {args.output}; pass --overwrite to opt in."
        )

    with np.load(args.sample_npz, allow_pickle=False) as sample:
        if args.pose_key not in sample.files:
            raise KeyError(
                f"{args.sample_npz} does not contain pose key {args.pose_key!r}."
            )
        poses = np.asarray(sample[args.pose_key], dtype=np.float32)
        copied = {
            key: np.asarray(sample[key])
            for key in ("case_id", "frame_id_triplet", "source_stem")
            if key in sample.files
        }

    native_masks = np.stack(
        [
            _load_binary_mask(args.view0_mask, args.threshold),
            _load_binary_mask(args.view1_mask, args.threshold),
        ],
        axis=0,
    )
    model_masks = np.stack(
        [_resize_binary(mask, args.image_size) for mask in native_masks], axis=0
    )

    (
        raw_forward,
        model_direction,
        theta,
        phi,
        clinical_primary,
        clinical_secondary,
        autocar_theta,
        autocar_phi,
    ) = _pose_directions(poses, args.pose_convention)
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    view_features = np.stack(
        (
            np.sin(theta_rad),
            np.cos(theta_rad),
            np.sin(phi_rad),
            np.cos(phi_rad),
        ),
        axis=1,
    ).astype(np.float32)
    (
        detector_roll,
        camera_rotation_las,
        detector_column_axis_las,
        detector_row_down_axis_las,
    ) = _detector_plane_metadata(
        poses,
        args.pose_convention,
        clinical_primary,
        clinical_secondary,
    )

    metadata = {
        "purpose": "target-free prediction-only inference",
        "artery": "lca",
        "stored_view_order": ["AutoCAR view 0", "AutoCAR view 1"],
        "mask_threshold_uint8": int(args.threshold),
        "model_image_size": [int(args.image_size), int(args.image_size)],
        "model_resize_mode": "nearest",
        "pose_source_key": args.pose_key,
        "pose_convention": args.pose_convention,
        "angle_conversion": (
            "q=normalized camera +z in released pose basis; "
            "d_las=(-q_y,q_z,q_x); theta=atan2(d_y,d_x); phi=acos(d_z)"
        ),
        "projection_center_offset_available": False,
        "detector_plane_orientation": (
            "preserved from the complete released 3x3 pose; no image flip "
            "has been applied"
        ),
        "warning": (
            "theta/phi retain the central-ray direction but not detector "
            "in-plane roll or missing X-ray intrinsics"
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "images": model_masks,
        "source_masks_native": native_masks,
        "source_view_indices": np.asarray([0, 1], dtype=np.int64),
        "source_mask_filenames": np.asarray(
            [args.view0_mask.name, args.view1_mask.name]
        ),
        "theta_deg": theta,
        "phi_deg": phi,
        "view_features": view_features,
        "view_directions_world": model_direction,
        "model_view_directions_world": model_direction,
        "autocar_raw_pose_forward_directions": raw_forward,
        "autocar_camera_to_world": poses,
        "autocar_camera_centers_raw": poses[:, :3, 3].astype(np.float32),
        "camera_to_world_rotation_las": camera_rotation_las,
        "detector_in_plane_roll_deg": detector_roll,
        "detector_column_axis_las": detector_column_axis_las,
        "detector_row_down_axis_las": detector_row_down_axis_las,
        "clinical_primary_deg": clinical_primary,
        "clinical_secondary_deg": clinical_secondary,
        "autocar_xray_theta_deg": autocar_theta,
        "autocar_xray_phi_deg": autocar_phi,
        "view_indices": np.asarray([0, 1], dtype=np.int64),
        "vessel_type": np.asarray("lca"),
        "angle_units": np.asarray("degrees"),
        "view_feature_encoding": np.asarray(
            "[sin(theta), cos(theta), sin(phi), cos(phi)]"
        ),
        "angle_convention": np.asarray(
            "LCA LAS projector: theta=wrap180(90-primary), phi=90-secondary"
        ),
        "input_foreground_definition": np.asarray(
            "binary vessel mask: vessel=1, background=0"
        ),
        "external_images_flipped_upside_down": np.asarray(False),
        "has_projection_center_offset": np.asarray(False),
        "conversion_metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
    }
    if args.official_autocar_case_id is not None:
        if "case_id" in copied:
            arrays["source_case_id"] = copied.pop("case_id")
        arrays["case_id"] = np.asarray(
            args.official_autocar_case_id, dtype=np.int32
        )
        arrays["official_autocar_case_id"] = np.asarray(
            args.official_autocar_case_id, dtype=np.int32
        )
        metadata["official_autocar_case_id"] = int(
            args.official_autocar_case_id
        )
        metadata["source_case_id"] = (
            int(arrays["source_case_id"].reshape(()).item())
            if "source_case_id" in arrays
            else None
        )
        arrays["conversion_metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
    arrays.update(copied)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    return args.output.resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sample-npz", type=Path, required=True)
    result.add_argument("--view0-mask", type=Path, required=True)
    result.add_argument("--view1-mask", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pose-key", default="cameras_world2cam")
    result.add_argument(
        "--pose-convention",
        choices=("camera_to_world", "world_to_camera"),
        default="camera_to_world",
    )
    result.add_argument("--image-size", type=int, default=256)
    result.add_argument("--threshold", type=int, default=127)
    result.add_argument("--official-autocar-case-id", type=int)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    print(build(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
