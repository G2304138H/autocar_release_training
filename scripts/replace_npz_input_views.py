#!/usr/bin/env python3
"""Replace a two-view inference NPZ's final masks with edited PNG masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _load_thresholded(path: Path, size: tuple[int, int], threshold: int) -> tuple[np.ndarray, np.ndarray]:
    grayscale = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if grayscale.shape != size:
        raise ValueError(
            f"{path} has shape {grayscale.shape}; expected {size}. "
            "Resize explicitly before replacing model inputs."
        )
    binary = (grayscale > threshold).astype(np.float32)
    return grayscale, binary


def _resize_binary(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    source = Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    resized = source.resize(size_wh, resample=Image.Resampling.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 127).astype(np.uint8)


def replace_views(
    base_npz: Path,
    view_1_png: Path,
    view_2_png: Path,
    output_npz: Path,
    *,
    threshold: int,
    overwrite: bool,
) -> tuple[Path, list[Path]]:
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must lie in [0,255].")
    if output_npz.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_npz}; pass --overwrite to opt in."
        )
    with np.load(base_npz, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    if "images" not in arrays:
        raise KeyError(f"{base_npz} does not contain an 'images' array.")
    previous = np.asarray(arrays["images"])
    if previous.ndim == 4 and previous.shape[1] == 1:
        previous = previous[:, 0]
    if previous.ndim != 3 or previous.shape[0] != 2:
        raise ValueError(f"Expected images with shape [2,H,W], got {previous.shape}.")
    expected_hw = (int(previous.shape[1]), int(previous.shape[2]))

    grayscale_views: list[np.ndarray] = []
    binary_views: list[np.ndarray] = []
    for path in (view_1_png, view_2_png):
        grayscale, binary = _load_thresholded(path, expected_hw, threshold)
        grayscale_views.append(grayscale)
        binary_views.append(binary)
    replacement = np.stack(binary_views, axis=0).astype(np.float32)

    arrays["images_before_manual_edit"] = previous.astype(np.float32, copy=False)
    arrays["manual_edit_source_grayscale_uint8"] = np.stack(
        grayscale_views, axis=0
    )
    arrays["images"] = replacement
    arrays["manual_edit_applied"] = np.asarray(True)
    arrays["manual_edit_threshold_uint8"] = np.asarray(threshold, dtype=np.uint8)
    arrays["manual_edit_source_filenames"] = np.asarray(
        [view_1_png.name, view_2_png.name]
    )
    arrays["manual_edit_operation"] = np.asarray(
        "user removed side branches; grayscale converted to vessel=1 with value > threshold"
    )

    if "source_masks_native_model_input" in arrays:
        old_native = np.asarray(arrays["source_masks_native_model_input"])
        if old_native.ndim == 3 and old_native.shape[0] == 2:
            arrays["source_masks_native_model_input_before_manual_edit"] = old_native
            native_height, native_width = int(old_native.shape[1]), int(old_native.shape[2])
            arrays["source_masks_native_model_input"] = np.stack(
                [
                    _resize_binary(mask, (native_width, native_height))
                    for mask in replacement
                ],
                axis=0,
            )

    before_counts = np.sum(previous > 0.5, axis=(1, 2)).astype(int)
    after_counts = np.sum(replacement > 0.5, axis=(1, 2)).astype(int)
    metadata: dict[str, object] = {}
    if "conversion_metadata_json" in arrays:
        raw = np.asarray(arrays["conversion_metadata_json"])
        if raw.shape == ():
            metadata = json.loads(str(raw.item()))
    metadata["manual_mask_edit"] = {
        "applied": True,
        "operation": "user removed side branches",
        "threshold_rule": f"grayscale > {threshold} becomes vessel=1; otherwise 0",
        "source_files_in_view_order": [view_1_png.name, view_2_png.name],
        "foreground_pixels_before": before_counts.tolist(),
        "foreground_pixels_after": after_counts.tolist(),
        "camera_metadata_changed": False,
    }
    arrays["conversion_metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True)
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    thresholded_outputs: list[Path] = []
    for view_number, mask in enumerate(replacement, start=1):
        output = output_npz.with_name(
            f"{output_npz.stem}_view_{view_number}_thresholded.png"
        )
        Image.fromarray((mask > 0.5).astype(np.uint8) * 255).save(
            output, format="PNG", optimize=True
        )
        thresholded_outputs.append(output.resolve())
    return output_npz.resolve(), thresholded_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-npz", type=Path, required=True)
    parser.add_argument("--view-1-png", type=Path, required=True)
    parser.add_argument("--view-2-png", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_npz, thresholded = replace_views(
        args.base_npz,
        args.view_1_png,
        args.view_2_png,
        args.output_npz,
        threshold=int(args.threshold),
        overwrite=bool(args.overwrite),
    )
    print(output_npz)
    for output in thresholded:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
