#!/usr/bin/env python3
"""Canonicalize real AutoCAR LCA masks for prediction-only inference.

The input NPZ is expected to contain the two masks and camera metadata written
by ``build_real_lca_prediction_npz.py``.  Each native mask is first rotated by
its recorded detector in-plane roll and is then rotated by 180 degrees, as
required by the current LCA projector image convention.  The checkpoint camera
angles are copied unchanged because they already include the projector's
source/detector polarity convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _rotate_binary(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    source = Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    rotated = source.rotate(
        float(angle_deg),
        resample=Image.Resampling.NEAREST,
        expand=False,
        fillcolor=0,
    )
    return (np.asarray(rotated, dtype=np.uint8) > 127).astype(np.uint8)


def _resize_binary(mask: np.ndarray, size: int) -> np.ndarray:
    source = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
    resized = source.resize((size, size), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 127).astype(np.float32)


def _wrap_rotation(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _dilate_binary_disk(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if radius_px <= 0:
        return binary
    height, width = binary.shape
    padded = np.pad(binary, radius_px, mode="constant")
    dilated = np.zeros_like(binary)
    for delta_y in range(-radius_px, radius_px + 1):
        for delta_x in range(-radius_px, radius_px + 1):
            if delta_x * delta_x + delta_y * delta_y > radius_px * radius_px:
                continue
            y0 = radius_px + delta_y
            x0 = radius_px + delta_x
            dilated = np.maximum(
                dilated, padded[y0 : y0 + height, x0 : x0 + width]
            )
    return dilated


def canonicalize(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    dilation_radius_model_px: int = 0,
    dilation_before_rotation: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_path}; pass --overwrite to opt in."
        )
    if dilation_radius_model_px < 0:
        raise ValueError("dilation_radius_model_px cannot be negative.")

    with np.load(input_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    required = {
        "images",
        "source_masks_native",
        "detector_in_plane_roll_deg",
        "theta_deg",
        "phi_deg",
        "view_features",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"{input_path} is missing required keys: {missing}")

    raw_native = (np.asarray(arrays["source_masks_native"]) > 0).astype(
        np.uint8
    )
    rolls = np.asarray(
        arrays["detector_in_plane_roll_deg"], dtype=np.float64
    ).reshape(-1)
    if raw_native.ndim != 3 or raw_native.shape[0] != 2:
        raise ValueError(
            f"source_masks_native must have shape [2,H,W], got {raw_native.shape}."
        )
    if rolls.shape != (2,) or not np.isfinite(rolls).all():
        raise ValueError(
            "detector_in_plane_roll_deg must contain two finite values."
        )

    images = np.asarray(arrays["images"])
    if images.ndim != 3 or images.shape[0] != 2 or images.shape[1] != images.shape[2]:
        raise ValueError(f"images must have shape [2,S,S], got {images.shape}.")
    image_size = int(images.shape[-1])

    native_scale = float(raw_native.shape[-1]) / float(image_size)
    dilation_radius_native_px = int(
        round(float(dilation_radius_model_px) * native_scale)
    )
    source_native_thickened = np.stack(
        [
            _dilate_binary_disk(mask, dilation_radius_native_px)
            for mask in raw_native
        ],
        axis=0,
    )

    detector_corrected_undilated = np.stack(
        [_rotate_binary(mask, roll) for mask, roll in zip(raw_native, rolls)],
        axis=0,
    )
    final_native_undilated = np.stack(
        [
            _rotate_binary(mask, 180.0)
            for mask in detector_corrected_undilated
        ],
        axis=0,
    )
    final_images_undilated = np.stack(
        [_resize_binary(mask, image_size) for mask in final_native_undilated],
        axis=0,
    )

    if dilation_before_rotation:
        detector_corrected = np.stack(
            [
                _rotate_binary(mask, roll)
                for mask, roll in zip(source_native_thickened, rolls)
            ],
            axis=0,
        )
        final_native = np.stack(
            [_rotate_binary(mask, 180.0) for mask in detector_corrected],
            axis=0,
        )
        final_images = np.stack(
            [_resize_binary(mask, image_size) for mask in final_native], axis=0
        )
        dilation_stage = "native source mask before orientation correction"
    else:
        detector_corrected = detector_corrected_undilated
        final_images = np.stack(
            [
                _dilate_binary_disk(mask, dilation_radius_model_px).astype(
                    np.float32
                )
                for mask in final_images_undilated
            ],
            axis=0,
        )
        final_native = np.stack(
            [
                _dilate_binary_disk(mask, dilation_radius_native_px)
                for mask in final_native_undilated
            ],
            axis=0,
        )
        dilation_stage = "oriented model input after orientation correction"

    total_rotation = _wrap_rotation(rolls + 180.0).astype(np.float32)
    arrays["source_masks_native_raw"] = raw_native.astype(np.uint8, copy=False)
    arrays["source_masks_native_after_thickening"] = (
        source_native_thickened.astype(np.uint8, copy=False)
    )
    arrays["source_masks_native_detector_plane_corrected_before_thickening"] = (
        detector_corrected_undilated.astype(np.uint8, copy=False)
    )
    arrays["source_masks_native_detector_plane_corrected"] = (
        detector_corrected.astype(np.uint8, copy=False)
    )
    arrays["source_masks_native_model_input_before_thickening"] = (
        final_native_undilated.astype(np.uint8, copy=False)
    )
    arrays["source_masks_native_model_input"] = final_native.astype(
        np.uint8, copy=False
    )
    arrays["images_original_before_thickening"] = images.astype(
        np.float32, copy=False
    )
    arrays["images_before_detector_plane_correction"] = np.stack(
        [
            _resize_binary(mask, image_size)
            for mask in (
                source_native_thickened
                if dilation_before_rotation
                else raw_native
            )
        ],
        axis=0,
    )
    arrays[
        "images_detector_plane_corrected_before_thickening"
    ] = np.stack(
        [
            _resize_binary(mask, image_size)
            for mask in detector_corrected_undilated
        ],
        axis=0,
    )
    arrays["images_detector_plane_corrected"] = np.stack(
        [_resize_binary(mask, image_size) for mask in detector_corrected], axis=0
    )
    arrays["images_before_thickening"] = final_images_undilated.astype(
        np.float32, copy=False
    )
    arrays["images"] = final_images
    arrays["detector_plane_correction_deg_pil_ccw"] = rolls.astype(np.float32)
    arrays["additional_model_input_rotation_deg_pil_ccw"] = np.full(
        2, 180.0, dtype=np.float32
    )
    arrays["total_applied_rotation_deg_pil_ccw"] = total_rotation
    arrays["model_input_orientation"] = np.asarray(
        "recorded detector roll corrected, then rotated 180 degrees"
    )
    arrays["input_mask_dilation_radius_model_px"] = np.asarray(
        dilation_radius_model_px, dtype=np.int32
    )
    arrays["input_mask_dilation_radius_native_px"] = np.asarray(
        dilation_radius_native_px, dtype=np.int32
    )
    arrays["input_mask_dilation_footprint"] = np.asarray("binary disk")
    arrays["input_mask_dilation_stage"] = np.asarray(dilation_stage)
    arrays["external_images_flipped_upside_down"] = np.asarray(False)

    metadata: dict[str, object] = {}
    if "conversion_metadata_json" in arrays:
        raw_metadata = np.asarray(arrays["conversion_metadata_json"])
        if raw_metadata.shape == ():
            metadata = json.loads(str(raw_metadata.item()))
    metadata.update(
        {
            "model_input_canonicalization": (
                "PIL rotate(+detector_in_plane_roll_deg), then PIL rotate(180); "
                "nearest-neighbour, expand=False, fill=0"
            ),
            "checkpoint_angles_changed_by_image_rotation": False,
            "theta_phi_policy": (
                "retain theta_deg/phi_deg because they already use the current "
                "LCA projector source-detector polarity convention"
            ),
            "total_applied_rotation_deg_pil_ccw": total_rotation.tolist(),
            "input_mask_thickening": {
                "operation": "binary disk dilation",
                "stage": dilation_stage,
                "model_radius_px": int(dilation_radius_model_px),
                "native_radius_px": int(dilation_radius_native_px),
                "physical_calibration": (
                    "none; sensitivity input because the released real data "
                    "does not provide sufficient detector calibration"
                ),
            },
        }
    )
    arrays["conversion_metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument(
        "--dilation-radius-model-px",
        type=int,
        default=0,
        help=(
            "Disk radius added to the final vessel masks at model resolution. "
            "For example, 2 increases the radius by approximately two pixels."
        ),
    )
    parser.add_argument(
        "--dilation-before-rotation",
        action="store_true",
        help=(
            "Dilate the native segmented mask before detector-roll correction "
            "and the final 180-degree rotation."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        canonicalize(
            args.input_npz,
            args.output_npz,
            overwrite=bool(args.overwrite),
            dilation_radius_model_px=int(args.dilation_radius_model_px),
            dilation_before_rotation=bool(args.dilation_before_rotation),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
