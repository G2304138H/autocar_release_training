#!/usr/bin/env python3
"""Evaluate one variable-view AutoCAR checkpoint with the first N input views.

Each view count runs through the same Stage-2 evaluator and case split. The
resolved JSON configurations and outputs are kept in separate directories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_imagecas_npz import DATASETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artery", choices=sorted(DATASETS))
    parser.add_argument("--projection-source", type=Path)
    parser.add_argument("--voxel-source", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--case-id-mode", choices=("literal", "imagecas_numeric"))
    parser.add_argument("--expected-pixel-spacing-mm", type=float)
    parser.add_argument("--fallback-pixel-spacing-mm", type=float)
    parser.add_argument("--fallback-sid-mm", type=float)
    parser.add_argument("--source-to-isocenter-mm", type=float)
    parser.add_argument(
        "--view-counts", nargs="+", type=int, default=(1, 2, 4),
        help="Evaluate these counts using source indices [0, ..., N-1].",
    )
    parser.add_argument("--split", choices=("val", "test", "val_test"), default="val_test")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--prediction-only", action="store_true",
        help="Export prediction NPZs without Dice, SSIM, or centerline metrics.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("32", "16-mixed"), default="32")
    parser.add_argument("--output-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--save-paper-masks", action="store_true")
    parser.add_argument("--save-centerline-graphs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _source_path(value: Path | None, default: Path | None, label: str) -> Path:
    selected = value if value is not None else default
    if selected is None:
        raise ValueError(f"--{label} is required without --artery.")
    path = selected.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def evaluation_configs(args: argparse.Namespace) -> list[tuple[Path, dict]]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    defaults = DATASETS[args.artery] if args.artery else {}
    projection_source = _source_path(
        args.projection_source, defaults.get("projection_source"), "projection-source"
    )
    voxel_source = _source_path(
        args.voxel_source, defaults.get("voxel_source"), "voxel-source"
    )
    split_json = _source_path(args.split_json, defaults.get("split_json"), "split-json")
    if not split_json.is_file():
        raise ValueError(f"--split-json must be a file: {split_json}")
    counts = tuple(args.view_counts)
    if not counts or len(set(counts)) != len(counts) or any(
        count < 1 or count > 7 for count in counts
    ):
        raise ValueError("--view-counts must contain distinct integers from 1 to 7.")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive.")
    output_root = args.output_root.expanduser().resolve()
    common = {
        "checkpoint_path": str(checkpoint),
        "projection_source": str(projection_source),
        "voxel_source": str(voxel_source),
        "split_json_path": str(split_json),
        "evaluation_mode": "prediction" if args.prediction_only else "paper_metric",
        "eval_split": args.split,
        "num_eval_cases": args.max_cases if args.max_cases is not None else "all",
        "eval_case_ids": None,
        "eval_view_selection": "fixed",
        "evaluation_view_labels": None,
        "evaluation_view_directions": {
            "accurate": True,
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
        },
        "case_id_mode": args.case_id_mode or (
            "imagecas_numeric" if args.artery else "literal"
        ),
        "expected_imager_pixel_spacing_mm": (
            args.expected_pixel_spacing_mm
            if args.expected_pixel_spacing_mm is not None
            else defaults.get("pixel_spacing_mm")
        ),
        "fallback_imager_pixel_spacing_mm": (
            args.fallback_pixel_spacing_mm
            if args.fallback_pixel_spacing_mm is not None
            else defaults.get("fallback_pixel_spacing_mm")
        ),
        "fallback_sid_mm": (
            args.fallback_sid_mm
            if args.fallback_sid_mm is not None
            else defaults.get("fallback_sid_mm")
        ),
        "source_to_isocenter_mm": (
            args.source_to_isocenter_mm
            if args.source_to_isocenter_mm is not None
            else defaults.get("source_to_isocenter_mm", 750.0)
        ),
        "save_prediction_npz_files": True,
        "device": args.device,
        "precision": args.precision,
        "output_dtype": args.output_dtype,
        "paper_metric_volume_threshold": 0.5,
        "paper_metric_ssim_window_size": 7,
        "paper_metric_save_masks": args.save_paper_masks,
        "paper_metric_save_centerline_graphs": args.save_centerline_graphs,
        "ssim_chunk_depth": 8,
        "ground_truth_origin_xyz_mm": None,
        "overwrite": args.overwrite,
    }
    plans = []
    for count in counts:
        config = {
            **common,
            "eval_output_dir": str(output_root / f"views_{count}"),
            "eval_num_views": count,
            "evaluation_view_indices": list(range(count)),
        }
        plans.append((output_root / "configs" / f"views_{count}.json", config))
    return plans


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = evaluation_configs(args)
    for config_path, config in plans:
        result_dir = Path(config["eval_output_dir"])
        if result_dir.exists() and any(result_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Evaluation output is not empty: {result_dir}. Use --overwrite to replace artifacts."
            )
    for config_path, config in plans:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        command = [sys.executable, "-u", "-m", "src.eval_npz", "--config", str(config_path)]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            log_path = config_path.parent.parent / "logs" / f"{config_path.stem}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Evaluator console log: {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nCommand: " + " ".join(command) + "\n")
                log.flush()
                with subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                ) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
