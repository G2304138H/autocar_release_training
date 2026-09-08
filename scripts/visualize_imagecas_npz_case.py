#!/usr/bin/env python3
"""Render one val/test ImageCAS case with a trained LCA or RCA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_imagecas_npz as shared
from src.dataset.case_splits import load_case_splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artery", choices=("lca", "rca"), required=True)
    parser.add_argument(
        "--case-number",
        type=int,
        required=True,
        help="Positive physical ImageCAS case number from the val or test split.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs",
        help=(
            "Hydra logging root. The artery task name is read from its training "
            "configuration before the latest completed run is selected."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit trained artery .ckpt; otherwise select the latest run's best.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Select a particular Hydra training run and its unique best checkpoint.",
    )
    parser.add_argument("--projection-source", type=Path)
    parser.add_argument("--voxel-source", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Optional output override. By default writes below the selected "
            "training experiment directory."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("32", "16-mixed"), default="32"
    )
    parser.add_argument(
        "--output-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=24,
        help="Number of rotating 3D overlay frames; use 0 to skip the GIF.",
    )
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--visualization-max-points", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate everything and write the config without inference.",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _input_path(
    args: argparse.Namespace,
    dataset: Mapping[str, Any],
    name: str,
) -> Path:
    override = getattr(args, name)
    return _resolved(override if override is not None else dataset[name])


def _evaluation_config(
    args: argparse.Namespace,
    *,
    checkpoint: Path,
    checkpoint_source: str,
    experiment_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = shared.DATASETS[args.artery]
    labels = dataset["evaluation_view_labels"]
    return {
        "artery_type": args.artery,
        "training_config": str(dataset["training_config"]),
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_choice": "best",
        "checkpoint_selection_source": checkpoint_source,
        "projection_source": str(_input_path(args, dataset, "projection_source")),
        "voxel_source": str(_input_path(args, dataset, "voxel_source")),
        "split_json_path": str(_input_path(args, dataset, "split_json")),
        "eval_output_dir": str(output_dir),
        "evaluation_mode": "visualisation",
        "eval_split": "val_test",
        "num_eval_cases": "all",
        "eval_case_ids": [str(args.case_number)],
        "eval_num_views": 2,
        "eval_view_selection": "fixed",
        "evaluation_view_indices": list(dataset["evaluation_view_indices"]),
        "evaluation_view_labels": None if labels is None else list(labels),
        "evaluation_view_directions": {
            "accurate": True,
            "theta_change_deg": 0.0,
            "phi_change_deg": 0.0,
        },
        "case_id_mode": "imagecas_numeric",
        "expected_imager_pixel_spacing_mm": dataset[
            "expected_imager_pixel_spacing_mm"
        ],
        "fallback_imager_pixel_spacing_mm": dataset[
            "fallback_imager_pixel_spacing_mm"
        ],
        "fallback_sid_mm": dataset["fallback_sid_mm"],
        "source_to_isocenter_mm": dataset["source_to_isocenter_mm"],
        "save_prediction_npz_files": True,
        "max_visualizations": 1,
        "device": args.device,
        "precision": args.precision,
        "output_dtype": args.output_dtype,
        "prediction_threshold": 0.5,
        "ssim_window_size": 7,
        "ssim_chunk_depth": 8,
        "paper_metric_save_masks": False,
        "ground_truth_origin_xyz_mm": None,
        "visualization_gif_frames": args.gif_frames,
        "visualization_gif_fps": args.gif_fps,
        "visualization_max_points": args.visualization_max_points,
        "overwrite": args.overwrite,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_config_inputs(
    config: Mapping[str, Any],
    *,
    case_number: int,
) -> None:
    for key in ("projection_source", "voxel_source"):
        path = Path(config[key])
        if not path.exists():
            raise FileNotFoundError(f"{key} does not exist: {path}")
    split_json = Path(config["split_json_path"])
    if not split_json.is_file():
        raise FileNotFoundError(f"split_json_path does not exist: {split_json}")
    splits = load_case_splits(split_json, case_id_mode="imagecas_numeric")
    case_id = str(case_number)
    memberships = [name for name in ("val", "test") if case_id in splits[name]]
    if not memberships:
        raise ValueError(
            f"Case {case_id} is not present in the validation or test split in "
            f"{split_json}."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.case_number < 1:
        raise ValueError("--case-number must be positive.")
    if args.gif_frames < 0:
        raise ValueError("--gif-frames cannot be negative.")
    if args.gif_fps < 1:
        raise ValueError("--gif-fps must be positive.")
    if args.visualization_max_points < 1:
        raise ValueError("--visualization-max-points must be positive.")

    checkpoint, experiment_dir, checkpoint_source = (
        shared.resolve_trained_checkpoint(
            artery=args.artery,
            log_dir=args.log_dir,
            checkpoint=args.checkpoint,
            experiment_dir=args.experiment_dir,
        )
    )
    output_dir = (
        _resolved(args.output_dir)
        if args.output_dir is not None
        else experiment_dir
        / "evaluation"
        / checkpoint.stem
        / f"case_{args.case_number}"
    )
    shared._assert_output_available(output_dir, overwrite=args.overwrite)
    config_path = (
        experiment_dir
        / "evaluation_configs"
        / f"visualisation_{checkpoint.stem}_case_{args.case_number}.json"
    )
    config = _evaluation_config(
        args,
        checkpoint=checkpoint,
        checkpoint_source=checkpoint_source,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
    )
    _validate_config_inputs(config, case_number=args.case_number)
    _write_json(config_path, config)

    command = [sys.executable, "-m", "src.eval_npz", "--config", str(config_path)]
    print(
        f"{args.artery.upper()} case {args.case_number}: checkpoint={checkpoint} "
        f"({checkpoint_source})",
        flush=True,
    )
    print(f"Output: {output_dir}", flush=True)
    print(f"Launching: {shlex.join(command)}", flush=True)
    if args.dry_run:
        print(f"Dry run complete. Resolved config: {config_path}")
        return 0

    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(ROOT)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    record = json.loads(
        (output_dir / "evaluation_record.json").read_text(encoding="utf-8")
    )
    split = record["case_split_by_case"][str(args.case_number)]
    visualization_dir = (
        output_dir
        / "visualization"
        / str(args.case_number)
        / split
        / "final"
    )
    if not visualization_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation completed without the expected visualization: "
            f"{visualization_dir}"
        )
    print(f"Completed case visualization: {visualization_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
