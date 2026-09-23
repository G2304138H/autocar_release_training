#!/usr/bin/env python3
"""Prepare selected real AutoCAR LCA cases in the current model orientation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat


SWAP_AXIS = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/HelveticaNeue.ttc"
        if bold
        else "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _camera_metadata(poses: np.ndarray) -> list[dict[str, float]]:
    if poses.shape != (2, 4, 4):
        raise ValueError(f"Expected two 4x4 poses, got {poses.shape}.")
    rotations = poses[:, :3, :3].astype(np.float64)
    forward = rotations[:, :, 2]
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    primary = np.rad2deg(np.arctan2(-forward[:, 1], forward[:, 2]))
    secondary = np.rad2deg(
        np.arctan2(
            forward[:, 0],
            np.hypot(forward[:, 1], forward[:, 2]),
        )
    )
    theta = (90.0 - primary + 180.0) % 360.0 - 180.0
    phi = 90.0 - secondary

    rolls: list[float] = []
    for rotation, primary_deg, secondary_deg in zip(
        rotations, primary, secondary
    ):
        alpha = np.deg2rad(float(primary_deg))
        beta_internal = -np.deg2rad(float(secondary_deg))
        rx = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, np.cos(alpha), -np.sin(alpha)),
                (0.0, np.sin(alpha), np.cos(alpha)),
            )
        )
        rz = np.asarray(
            (
                (np.cos(beta_internal), -np.sin(beta_internal), 0.0),
                (np.sin(beta_internal), np.cos(beta_internal), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        expected_zero_roll = SWAP_AXIS @ (rx @ rz) @ SWAP_AXIS.T
        residual = expected_zero_roll.T @ rotation
        rolls.append(
            float(np.rad2deg(np.arctan2(residual[1, 0], residual[0, 0])))
        )

    separation = float(
        np.rad2deg(
            np.arccos(np.clip(np.dot(forward[0], forward[1]), -1.0, 1.0))
        )
    )
    result: list[dict[str, float]] = []
    for view_index in range(2):
        result.append(
            {
                "view_number": view_index + 1,
                "clinical_primary_deg": float(primary[view_index]),
                "clinical_secondary_deg": float(secondary[view_index]),
                "model_theta_deg": float(theta[view_index]),
                "model_phi_deg": float(phi[view_index]),
                "detector_roll_correction_deg_pil_ccw": rolls[view_index],
                "additional_rotation_deg_pil_ccw": 180.0,
                "total_rotation_deg_pil_ccw": float(
                    (rolls[view_index] + 180.0 + 180.0) % 360.0 - 180.0
                ),
                "view_pair_separation_deg": separation,
            }
        )
    return result


def _border_fill(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    border = max(2, min(width, height) // 32)
    strips = (
        rgb.crop((0, 0, width, border)),
        rgb.crop((0, height - border, width, height)),
        rgb.crop((0, 0, border, height)),
        rgb.crop((width - border, 0, width, height)),
    )
    values = []
    for channel in range(3):
        channel_means = [ImageStat.Stat(item).mean[channel] for item in strips]
        values.append(int(round(float(np.median(channel_means)))))
    return tuple(values)  # type: ignore[return-value]


def _transform(
    source: Image.Image, roll_deg: float
) -> tuple[Image.Image, Image.Image]:
    rgb = source.convert("RGB")
    corrected = rgb.rotate(
        float(roll_deg),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=_border_fill(rgb),
    )
    final = corrected.transpose(Image.Transpose.ROTATE_180)
    return corrected, final


def _clinical_label(primary: float, secondary: float) -> str:
    primary_name = "LAO" if primary >= 0.0 else "RAO"
    secondary_name = "CRA" if secondary >= 0.0 else "CAU"
    return (
        f"{primary_name} {abs(primary):.2f} deg, "
        f"{secondary_name} {abs(secondary):.2f} deg"
    )


def _comparison_panel(
    case_id: int,
    originals: list[Image.Image],
    corrected: list[Image.Image],
    final: list[Image.Image],
    metadata: list[dict[str, float]],
    output: Path,
) -> None:
    panel = 512
    gap = 28
    margin = 36
    title_height = 66
    column_header = 106
    row_header = 42
    rows = (
        ("Representative original video frame", originals),
        ("Detector-plane roll cancelled", corrected),
        ("Final model-oriented image: additional 180 deg", final),
    )
    width = margin * 2 + panel * 2 + gap
    height = (
        margin
        + title_height
        + column_header
        + len(rows) * (row_header + panel)
        + margin
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, margin),
        f"AutoCAR real LCA Case {case_id:02d}",
        font=_font(30, bold=True),
        fill=(20, 25, 31),
    )
    header_y = margin + title_height
    for index, item in enumerate(metadata):
        x = margin + index * (panel + gap)
        lines = (
            f"View {index + 1}",
            _clinical_label(
                item["clinical_primary_deg"], item["clinical_secondary_deg"]
            ),
            (
                f"model theta={item['model_theta_deg']:.2f}, "
                f"phi={item['model_phi_deg']:.2f}; "
                f"roll={item['detector_roll_correction_deg_pil_ccw']:+.2f} deg"
            ),
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (x, header_y + line_index * 29),
                line,
                font=_font(21 if line_index else 24, bold=line_index < 2),
                fill=(28, 34, 42),
            )

    y = header_y + column_header
    for label, images in rows:
        draw.text(
            (margin, y + 7),
            label,
            font=_font(22, bold=True),
            fill=(35, 43, 52),
        )
        y += row_header
        for index, image in enumerate(images):
            fitted = image.convert("RGB").resize(
                (panel, panel), resample=Image.Resampling.LANCZOS
            )
            x = margin + index * (panel + gap)
            canvas.paste(fitted, (x, y))
        y += panel
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def prepare_case(
    case_id: int,
    source_root: Path,
    frame_root: Path,
    output_root: Path,
) -> Path:
    case_name = f"case_{case_id:02d}"
    pose_source = source_root / f"{case_name}_camera_poses.json"
    poses = np.asarray(json.loads(pose_source.read_text()), dtype=np.float64)
    camera_metadata = _camera_metadata(poses)
    case_output = output_root / case_name
    case_output.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pose_source, case_output / pose_source.name)
    originals: list[Image.Image] = []
    corrected_images: list[Image.Image] = []
    final_images: list[Image.Image] = []
    for view_number, item in enumerate(camera_metadata, start=1):
        video_source = source_root / f"{case_name}_view_{view_number}.mp4"
        shutil.copy2(video_source, case_output / video_source.name)
        frame_source = frame_root / f"{video_source.name}.png"
        if not frame_source.is_file():
            raise FileNotFoundError(
                f"Missing representative frame {frame_source}. Generate it "
                "with qlmanage before running this script."
            )
        original = Image.open(frame_source).convert("RGB")
        detector_corrected, final = _transform(
            original, item["detector_roll_correction_deg_pil_ccw"]
        )
        original.save(
            case_output / f"{case_name}_view_{view_number}_original.png",
            format="PNG",
            optimize=True,
        )
        detector_corrected.save(
            case_output
            / f"{case_name}_view_{view_number}_detector_roll_corrected.png",
            format="PNG",
            optimize=True,
        )
        final.save(
            case_output
            / f"{case_name}_view_{view_number}_final_rot180.png",
            format="PNG",
            optimize=True,
        )
        originals.append(original.copy())
        corrected_images.append(detector_corrected)
        final_images.append(final)

    separation = camera_metadata[0]["view_pair_separation_deg"]
    manifest = {
        "case_id": case_id,
        "view_pair_separation_deg": separation,
        "frame_source": (
            "macOS Quick Look representative 512x512 frame generated directly "
            "from each original MP4"
        ),
        "transform_order": [
            "PIL rotate(+detector roll) with bicubic interpolation, expand=False",
            "exact 180-degree image rotation",
        ],
        "views": camera_metadata,
    }
    (case_output / "processing_metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _comparison_panel(
        case_id,
        originals,
        corrected_images,
        final_images,
        camera_metadata,
        case_output / f"{case_name}_orientation_comparison.png",
    )
    return case_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for case_id in args.cases:
        print(
            prepare_case(
                int(case_id),
                args.source_root,
                args.frame_root,
                args.output_root,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
