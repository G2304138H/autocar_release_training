#!/usr/bin/env python3
"""Swap the model-facing direction assignment of a two-view inference NPZ.

The image order and image-preprocessing metadata are deliberately unchanged.
Only arrays that define the geometry presented to the reconstruction model are
reversed.  Original values are retained under ``view_direction_swap_original_*``
keys so the stress-test file remains auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_GEOMETRY_KEYS = (
    "theta_deg",
    "phi_deg",
    "view_features",
    "view_directions_world",
    "model_view_directions_world",
    "world2pix4x4",
    "world2pix3x4",
)


def swap_view_directions(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace {output_path}; pass --overwrite to opt in."
        )

    with np.load(input_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    for required in ("images", "theta_deg", "phi_deg"):
        if required not in arrays:
            raise KeyError(f"{input_path} is missing required key {required!r}.")
    images = np.asarray(arrays["images"])
    if images.ndim < 1 or images.shape[0] != 2:
        raise ValueError(
            f"images must contain exactly two ordered views, got {images.shape}."
        )

    swapped_keys: list[str] = []
    for key in MODEL_GEOMETRY_KEYS:
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.ndim < 1 or value.shape[0] != 2:
            raise ValueError(
                f"{key} must have two rows to swap, got {value.shape}."
            )
        arrays[f"view_direction_swap_original_{key}"] = value.copy()
        arrays[key] = np.ascontiguousarray(value[::-1])
        swapped_keys.append(key)

    arrays["view_direction_swap_applied"] = np.asarray(True)
    arrays["view_direction_assignment_source_positions"] = np.asarray(
        [1, 0], dtype=np.int64
    )
    arrays["view_direction_swap_scope"] = np.asarray(
        "model geometry only; images, image order, detector corrections, and "
        "source-view indices unchanged"
    )
    arrays["view_direction_swap_keys"] = np.asarray(swapped_keys)

    if "clinical_primary_deg" in arrays:
        arrays["assigned_clinical_primary_deg"] = np.ascontiguousarray(
            np.asarray(arrays["clinical_primary_deg"])[::-1]
        )
    if "clinical_secondary_deg" in arrays:
        arrays["assigned_clinical_secondary_deg"] = np.ascontiguousarray(
            np.asarray(arrays["clinical_secondary_deg"])[::-1]
        )

    metadata: dict[str, object] = {}
    if "conversion_metadata_json" in arrays:
        raw = np.asarray(arrays["conversion_metadata_json"])
        if raw.shape == ():
            metadata = json.loads(str(raw.item()))
    metadata["view_direction_swap_stress_test"] = {
        "applied": True,
        "image_positions_unchanged": True,
        "direction_source_position_by_image_position": [1, 0],
        "swapped_model_geometry_keys": swapped_keys,
        "detector_correction_metadata_swapped": False,
        "source_view_indices_swapped": False,
        "purpose": (
            "test deliberately incorrect cross-assignment of the two camera "
            "directions while holding the processed masks fixed"
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        swap_view_directions(
            args.input_npz,
            args.output_npz,
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
