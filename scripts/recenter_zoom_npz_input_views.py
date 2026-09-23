#!/usr/bin/env python3
"""Recenter and zoom binary input masks in a two-view inference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _foreground_bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(np.asarray(mask) > 0.5)
    if len(rows) == 0:
        raise ValueError("Cannot recenter an empty input mask.")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _recenter_zoom_binary(
    mask: np.ndarray, zoom_factor: float
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError(f"Expected a square 2D mask, got {binary.shape}.")
    size = int(binary.shape[0])
    bbox = _foreground_bbox_xyxy(binary)
    centre_x = 0.5 * (bbox[0] + bbox[2])
    centre_y = 0.5 * (bbox[1] + bbox[3])
    crop_size = max(1, min(size, int(round(size / zoom_factor))))
    left = int(round(centre_x - 0.5 * (crop_size - 1)))
    top = int(round(centre_y - 0.5 * (crop_size - 1)))
    crop_box = (left, top, left + crop_size, top + crop_size)

    source = Image.fromarray(binary * np.uint8(255))
    cropped = source.crop(crop_box)
    resized = cropped.resize((size, size), resample=Image.Resampling.NEAREST)
    output = (np.asarray(resized, dtype=np.uint8) > 127).astype(np.uint8)
    return output, crop_box, float(size) / float(crop_size)


def recenter_zoom(
    input_path: Path,
    output_path: Path,
    *,
    reference_pixel_spacing_mm: float,
    target_pixel_spacing_mm: float,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_path}; pass --overwrite to opt in."
        )
    for label, value in (
        ("reference_pixel_spacing_mm", reference_pixel_spacing_mm),
        ("target_pixel_spacing_mm", target_pixel_spacing_mm),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
    zoom_factor = reference_pixel_spacing_mm / target_pixel_spacing_mm
    if zoom_factor < 1.0:
        raise ValueError(
            "This utility currently requires reference spacing >= target "
            "spacing so that the operation is a centred crop/zoom."
        )

    with np.load(input_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    if "images" not in arrays:
        raise KeyError(f"{input_path} is missing required key 'images'.")
    images = np.asarray(arrays["images"])
    if images.ndim != 3 or images.shape[0] != 2:
        raise ValueError(f"images must have shape [2,H,W], got {images.shape}.")

    original_bboxes = []
    crop_boxes = []
    effective_zooms = []
    adjusted_images = []
    for image in images:
        original_bboxes.append(_foreground_bbox_xyxy(image))
        adjusted, crop_box, effective_zoom = _recenter_zoom_binary(
            image, zoom_factor
        )
        adjusted_images.append(adjusted.astype(np.float32))
        crop_boxes.append(crop_box)
        effective_zooms.append(effective_zoom)
    adjusted_stack = np.stack(adjusted_images, axis=0)

    arrays["images_before_recenter_zoom"] = images.astype(
        np.float32, copy=False
    )
    arrays["images"] = adjusted_stack
    arrays["image_dim"] = np.asarray(images.shape[-1], dtype=np.int32)
    if "imager_pixel_spacing" in arrays:
        arrays["imager_pixel_spacing_before_recenter_zoom"] = np.asarray(
            arrays["imager_pixel_spacing"]
        )
    arrays["imager_pixel_spacing"] = np.asarray(
        target_pixel_spacing_mm, dtype=np.float32
    )
    arrays["imager_pixel_spacing_units"] = np.asarray("mm")

    native_crop_boxes = None
    native_effective_zooms = None
    native_key = "source_masks_native_model_input"
    if native_key in arrays:
        native = np.asarray(arrays[native_key])
        if native.ndim == 3 and native.shape[0] == 2:
            native_adjusted = []
            native_crop_boxes_list = []
            native_effective_zooms_list = []
            for mask in native:
                adjusted, crop_box, effective_zoom = _recenter_zoom_binary(
                    mask, zoom_factor
                )
                native_adjusted.append(adjusted)
                native_crop_boxes_list.append(crop_box)
                native_effective_zooms_list.append(effective_zoom)
            arrays[
                "source_masks_native_model_input_before_recenter_zoom"
            ] = native.astype(np.uint8, copy=False)
            arrays[native_key] = np.stack(native_adjusted, axis=0).astype(
                np.uint8
            )
            native_crop_boxes = native_crop_boxes_list
            native_effective_zooms = native_effective_zooms_list

    adjusted_bboxes = [
        _foreground_bbox_xyxy(image) for image in adjusted_stack
    ]
    arrays["input_recenter_zoom_applied"] = np.asarray(True)
    arrays["input_recenter_method"] = np.asarray(
        "foreground bounding-box midpoint moved to image centre"
    )
    arrays["input_zoom_nominal_factor"] = np.asarray(
        zoom_factor, dtype=np.float32
    )
    arrays["input_zoom_effective_factors"] = np.asarray(
        effective_zooms, dtype=np.float32
    )
    arrays["input_zoom_reference_pixel_spacing_mm"] = np.asarray(
        reference_pixel_spacing_mm, dtype=np.float32
    )
    arrays["input_zoom_target_pixel_spacing_mm"] = np.asarray(
        target_pixel_spacing_mm, dtype=np.float32
    )
    arrays["input_zoom_crop_boxes_xyxy"] = np.asarray(
        crop_boxes, dtype=np.int32
    )
    arrays["input_foreground_bboxes_before_xyxy"] = np.asarray(
        original_bboxes, dtype=np.int32
    )
    arrays["input_foreground_bboxes_after_xyxy"] = np.asarray(
        adjusted_bboxes, dtype=np.int32
    )
    if native_crop_boxes is not None:
        arrays["native_input_zoom_crop_boxes_xyxy"] = np.asarray(
            native_crop_boxes, dtype=np.int32
        )
        arrays["native_input_zoom_effective_factors"] = np.asarray(
            native_effective_zooms, dtype=np.float32
        )

    metadata: dict[str, object] = {}
    if "conversion_metadata_json" in arrays:
        raw = np.asarray(arrays["conversion_metadata_json"])
        if raw.shape == ():
            metadata = json.loads(str(raw.item()))
    metadata["input_recenter_zoom"] = {
        "applied": True,
        "centering_method": "foreground bounding-box midpoint",
        "reference_pixel_spacing_mm": float(reference_pixel_spacing_mm),
        "target_pixel_spacing_mm": float(target_pixel_spacing_mm),
        "nominal_zoom_factor": float(zoom_factor),
        "effective_zoom_factors": effective_zooms,
        "crop_boxes_xyxy": [list(box) for box in crop_boxes],
        "foreground_bboxes_before_xyxy": [
            list(box) for box in original_bboxes
        ],
        "foreground_bboxes_after_xyxy": [
            list(box) for box in adjusted_bboxes
        ],
        "images_resampled": True,
        "view_directions_changed": False,
        "interpretation": (
            "sensitivity preprocessing that enlarges evidence by the spacing "
            "ratio while retaining 0.65-mm projection geometry"
        ),
    }
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
        "--reference-pixel-spacing-mm", type=float, default=0.75
    )
    parser.add_argument(
        "--target-pixel-spacing-mm", type=float, default=0.65
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        recenter_zoom(
            args.input_npz,
            args.output_npz,
            reference_pixel_spacing_mm=float(
                args.reference_pixel_spacing_mm
            ),
            target_pixel_spacing_mm=float(args.target_pixel_spacing_mm),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
