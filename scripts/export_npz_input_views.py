#!/usr/bin/env python3
"""Export a two-view NPZ ``images`` array as separate binary PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def export_views(input_npz: Path, output_dir: Path, prefix: str) -> list[Path]:
    with np.load(input_npz, allow_pickle=False) as payload:
        if "images" not in payload.files:
            raise KeyError(f"{input_npz} does not contain an 'images' array.")
        images = np.asarray(payload["images"])
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[0] != 2:
        raise ValueError(f"Expected images with shape [2,H,W], got {images.shape}.")
    if not np.isfinite(images).all():
        raise ValueError("images contains NaN or Inf.")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, image in enumerate(images, start=1):
        binary = (np.asarray(image) > 0.5).astype(np.uint8) * np.uint8(255)
        output = output_dir / f"{prefix}_view_{index}.png"
        Image.fromarray(binary).save(output, format="PNG", optimize=True)
        outputs.append(output.resolve())
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    for output in export_views(args.input_npz, args.output_dir, args.prefix):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
