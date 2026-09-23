#!/usr/bin/env python3
"""Render labeled LCA masks for clinical view-direction inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Helvetica.ttc" if not bold else "HelveticaNeue.ttc"
    return ImageFont.truetype(f"/System/Library/Fonts/{name}", size=size)


def _centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    draw.text((xy[0] - width / 2, xy[1]), text, font=font, fill=fill)


def _clinical_label(primary: float, secondary: float) -> str:
    primary_name = "LAO" if primary >= 0.0 else "RAO"
    secondary_name = "CRA" if secondary >= 0.0 else "CAU"
    return (
        f"{primary_name} {abs(primary):.2f}°  |  "
        f"{secondary_name} {abs(secondary):.2f}°"
    )


def render(input_npz: Path, output: Path) -> Path:
    with np.load(input_npz, allow_pickle=False) as payload:
        mask_key = (
            "source_masks_native"
            if "source_masks_native" in payload.files
            else "images"
        )
        masks = np.asarray(payload[mask_key])
        primary = np.asarray(payload["clinical_primary_deg"], dtype=np.float64)
        secondary = np.asarray(
            payload["clinical_secondary_deg"], dtype=np.float64
        )
        theta = np.asarray(payload["theta_deg"], dtype=np.float64)
        phi = np.asarray(payload["phi_deg"], dtype=np.float64)
        detector_roll = np.asarray(
            payload["detector_in_plane_roll_deg"], dtype=np.float64
        )
        official_case = int(
            np.asarray(payload["official_autocar_case_id"]).reshape(()).item()
        )
        source_case = int(
            np.asarray(payload["source_case_id"]).reshape(()).item()
        )
    if masks.shape[0] != 2 or masks.ndim != 3:
        raise ValueError(f"Expected two [H,W] masks, got {masks.shape}.")

    panel = 512
    margin = 64
    gap = 48
    title_height = 82
    column_header_height = 190
    row_header_height = 54
    footer_height = 104
    width = margin * 2 + panel * 2 + gap
    height = (
        margin
        + title_height
        + column_header_height
        + row_header_height
        + panel
        + footer_height
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    dark = (23, 29, 36)
    muted = (83, 94, 108)
    border = (190, 198, 207)

    _centered_text(
        draw,
        (width / 2, margin),
        (
            f"Official AutoCAR Case {official_case}: LCA Clinical View Check "
            f"(local source_case_id={source_case})"
        ),
        _font(30, bold=True),
        dark,
    )

    header_y = margin + title_height
    for view_index in range(2):
        x0 = margin + view_index * (panel + gap)
        cx = x0 + panel / 2
        lines = (
            (f"View {view_index}", 29, True, dark),
            (
                _clinical_label(primary[view_index], secondary[view_index]),
                26,
                True,
                dark,
            ),
            (
                f"primary α={primary[view_index]:+.2f}°, "
                f"secondary β={secondary[view_index]:+.2f}°",
                20,
                False,
                muted,
            ),
            (
                f"LCA projector θ={theta[view_index]:.2f}°, "
                f"φ={phi[view_index]:.2f}°",
                20,
                False,
                muted,
            ),
            (
                f"detector in-plane roll={detector_roll[view_index]:+.2f}°",
                20,
                False,
                muted,
            ),
        )
        y = header_y
        for text, size, bold, color in lines:
            _centered_text(draw, (cx, y), text, _font(size, bold=bold), color)
            y += 37 if size >= 26 else 29

    first_label_y = header_y + column_header_height
    _centered_text(
        draw,
        (width / 2, first_label_y),
        "Segmentation in the detector-plane orientation recorded by the full pose",
        _font(23, bold=True),
        dark,
    )
    first_image_y = first_label_y + row_header_height

    for view_index in range(2):
        x0 = margin + view_index * (panel + gap)
        source = (np.asarray(masks[view_index]) > 0.5).astype(np.uint8) * 255
        image = Image.fromarray(source).resize(
            (panel, panel), resample=Image.Resampling.NEAREST
        )
        canvas.paste(image.convert("RGB"), (x0, first_image_y))
        draw.rectangle(
            (x0, first_image_y, x0 + panel - 1, first_image_y + panel - 1),
            outline=border,
            width=2,
        )

    footer_y = first_image_y + panel + 28
    _centered_text(
        draw,
        (width / 2, footer_y),
        (
            "No vertical flip is applied. Compare CT projections using the recorded detector-plane roll."
        ),
        _font(19),
        muted,
    )
    _centered_text(
        draw,
        (width / 2, footer_y + 28),
        "Clinical primary/secondary and LCA θ/φ specify the ray; roll specifies image-plane orientation.",
        _font(19),
        muted,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.input_npz, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
