#!/usr/bin/env python3
"""Evaluate LCA/RCA AutoCAR robustness to inaccurate input view directions."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_imagecas_npz as shared


DEFAULT_AXIS_DEGREES = (2.0, 5.0, 10.0, 15.0)
DEFAULT_COMBINED_DEGREES = (5.0, 10.0)
DEFAULT_VISUALIZATION_CONDITIONS = (
    (-10.0, 0.0),
    (10.0, 0.0),
    (0.0, -10.0),
    (0.0, 10.0),
    (-10.0, -10.0),
    (-10.0, 10.0),
    (10.0, -10.0),
    (10.0, 10.0),
)


@dataclass(frozen=True)
class RobustnessPlan:
    artery: str
    checkpoint: Path
    checkpoint_source: str
    experiment_dir: Path
    config_path: Path
    output_dir: Path
    config: Mapping[str, Any]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artery", choices=("both", "lca", "rca"), default="both"
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional common output root; defaults to each training run.",
    )
    for artery in ("lca", "rca"):
        parser.add_argument(f"--{artery}-checkpoint", type=Path)
        parser.add_argument(f"--{artery}-experiment-dir", type=Path)
        parser.add_argument(f"--{artery}-projection-source", type=Path)
        parser.add_argument(f"--{artery}-voxel-source", type=Path)
        parser.add_argument(f"--{artery}-split-json", type=Path)
        parser.add_argument(
            f"--{artery}-accurate-baseline-summary",
            type=Path,
            help=(
                f"Override the {artery.upper()} accurate paper-metric "
                "performance_summary.json."
            ),
        )
    parser.add_argument(
        "--axis-degrees",
        type=float,
        nargs="+",
        default=list(DEFAULT_AXIS_DEGREES),
        help="Positive magnitudes for signed theta-only and phi-only errors.",
    )
    parser.add_argument(
        "--combined-degrees",
        type=float,
        nargs="+",
        default=list(DEFAULT_COMBINED_DEGREES),
        help="Positive magnitudes for all signed theta/phi combinations.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Run every condition without monitor bundles.",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=2,
        help="Cases rendered for each representative condition (default: 2).",
    )
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--visualization-max-points", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("32", "16-mixed"), default="32")
    parser.add_argument(
        "--output-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--save-paper-mask-files",
        action="store_true",
        help="Retain every condition's 128^3 comparison masks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and write sweep configurations without inference.",
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _input_path(args: argparse.Namespace, artery: str, name: str) -> Path:
    override = getattr(args, f"{artery}_{name}")
    source = override if override is not None else shared.DATASETS[artery][name]
    return _resolved(Path(source))


def _config_for_artery(
    args: argparse.Namespace,
    artery: str,
    *,
    checkpoint: Path,
    experiment_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = shared.DATASETS[artery]
    labels = dataset["evaluation_view_labels"]
    baseline_override = getattr(args, f"{artery}_accurate_baseline_summary")
    planned_conditions: set[tuple[float, float]] = set()
    for degrees in args.axis_degrees:
        planned_conditions.update(
            {
                (-degrees, 0.0),
                (degrees, 0.0),
                (0.0, -degrees),
                (0.0, degrees),
            }
        )
    for degrees in args.combined_degrees:
        planned_conditions.update(
            {
                (-degrees, -degrees),
                (-degrees, degrees),
                (degrees, -degrees),
                (degrees, degrees),
            }
        )
    return {
        "artery_type": artery,
        "training_config": str(dataset["training_config"]),
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint),
        "checkpoint_choice": "best",
        "projection_source": str(_input_path(args, artery, "projection_source")),
        "voxel_source": str(_input_path(args, artery, "voxel_source")),
        "split_json_path": str(_input_path(args, artery, "split_json")),
        "eval_output_dir": str(output_dir),
        "evaluation_mode": "inaccurate_view_direction",
        "eval_split": "val_test",
        "num_eval_cases": args.max_cases if args.max_cases is not None else "all",
        "eval_case_ids": None,
        "eval_num_views": 2,
        "eval_view_selection": "fixed",
        "evaluation_view_indices": list(dataset["evaluation_view_indices"]),
        "evaluation_view_labels": None if labels is None else list(labels),
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
        "max_visualizations": args.max_visualizations,
        "visualization_gif_frames": args.gif_frames,
        "visualization_gif_fps": args.gif_fps,
        "visualization_max_points": args.visualization_max_points,
        "device": args.device,
        "precision": args.precision,
        "output_dtype": args.output_dtype,
        "paper_metric_volume_threshold": 0.5,
        "paper_metric_ssim_window_size": 7,
        "ssim_chunk_depth": 8,
        "ground_truth_origin_xyz_mm": None,
        "overwrite": args.overwrite,
        "view_direction_robustness": {
            "axis_degrees": list(args.axis_degrees),
            "combined_degrees": list(args.combined_degrees),
            "visualization_conditions_deg": (
                []
                if args.no_visualizations
                else [
                    list(pair)
                    for pair in DEFAULT_VISUALIZATION_CONDITIONS
                    if pair in planned_conditions
                ]
            ),
            "accurate_baseline_summary": (
                "auto"
                if baseline_override is None
                else str(_resolved(baseline_override))
            ),
            "require_accurate_baseline": True,
            "record_paper_metrics": True,
            "save_mask_npz_files": args.save_paper_mask_files,
            "output_dir": None,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _build_plans(
    args: argparse.Namespace,
    *,
    output_root: Path | None,
) -> list[RobustnessPlan]:
    arteries = ("lca", "rca") if args.artery == "both" else (args.artery,)
    plans: list[RobustnessPlan] = []
    for artery in arteries:
        checkpoint, experiment_dir, checkpoint_source = (
            shared.resolve_trained_checkpoint(
                artery=artery,
                log_dir=args.log_dir,
                checkpoint=getattr(args, f"{artery}_checkpoint"),
                experiment_dir=getattr(args, f"{artery}_experiment_dir"),
            )
        )
        output_dir = (
            output_root / artery
            if output_root is not None
            else experiment_dir
            / "evaluation_inaccurate_view_direction_robustness"
            / checkpoint.stem
        )
        config_path = (
            output_root / "configs" / f"{artery}_inaccurate_view_direction.json"
            if output_root is not None
            else experiment_dir
            / "evaluation_configs"
            / f"inaccurate_view_direction_{checkpoint.stem}.json"
        )
        config = _config_for_artery(
            args,
            artery,
            checkpoint=checkpoint,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
        )
        shared._validate_input_paths(config, artery)
        plans.append(
            RobustnessPlan(
                artery=artery,
                checkpoint=checkpoint,
                checkpoint_source=checkpoint_source,
                experiment_dir=experiment_dir,
                config_path=config_path,
                output_dir=output_dir,
                config=config,
            )
        )
    return plans


def _command(plan: RobustnessPlan) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.eval_npz",
        "--config",
        str(plan.config_path),
    ]


def _plan_record(plans: Sequence[RobustnessPlan]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluation_mode": "inaccurate_view_direction",
        "eval_split": "val_test",
        "aggregation_policy": "report_lca_and_rca_separately",
        "runs": [
            {
                "artery": plan.artery,
                "checkpoint": str(plan.checkpoint),
                "checkpoint_source": plan.checkpoint_source,
                "experiment_dir": str(plan.experiment_dir),
                "config": str(plan.config_path),
                "output_dir": str(plan.output_dir),
                "command": _command(plan),
            }
            for plan in plans
        ],
    }


def _combined_summary(plans: Sequence[RobustnessPlan]) -> dict[str, Any]:
    arteries: dict[str, Any] = {}
    for plan in plans:
        summary_path = plan.output_dir / "view_direction_robustness_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"{plan.artery.upper()} sweep did not write {summary_path}."
            )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        arteries[plan.artery] = {
            "checkpoint": str(plan.checkpoint),
            "checkpoint_source": plan.checkpoint_source,
            "output_dir": str(plan.output_dir),
            "summary_file": str(summary_path),
            "robustness": payload,
        }
    return {
        "schema_version": 1,
        "evaluation_mode": "inaccurate_view_direction",
        "eval_split": "val_test",
        "aggregation_policy": (
            "LCA and RCA are reported separately; no cross-artery pooled statistic"
        ),
        "arteries": arteries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive.")
    if any(value <= 0.0 for value in args.axis_degrees):
        raise ValueError("--axis-degrees values must be positive.")
    if any(value <= 0.0 for value in args.combined_degrees):
        raise ValueError("--combined-degrees values must be positive.")
    if args.max_visualizations < 0:
        raise ValueError("--max-visualizations cannot be negative.")
    if args.gif_frames < 0 or args.gif_fps < 1:
        raise ValueError("--gif-frames must be non-negative and --gif-fps positive.")
    if args.visualization_max_points < 1:
        raise ValueError("--visualization-max-points must be positive.")

    args.log_dir = _resolved(args.log_dir)
    output_root = _resolved(args.output_root) if args.output_root else None
    plans = _build_plans(args, output_root=output_root)
    for plan in plans:
        _write_json(plan.config_path, plan.config)
    plan_record = _plan_record(plans)
    plan_paths = (
        [output_root / "evaluation_plan.json"]
        if output_root is not None
        else [
            plan.experiment_dir
            / "evaluation_configs"
            / f"inaccurate_view_direction_{plan.checkpoint.stem}_plan.json"
            for plan in plans
        ]
    )
    for path in plan_paths:
        _write_json(path, plan_record)
    for plan in plans:
        print(
            f"{plan.artery.upper()}: checkpoint={plan.checkpoint} "
            f"({plan.checkpoint_source})",
            flush=True,
        )
        print(f"Launching: {shlex.join(_command(plan))}", flush=True)
    if args.dry_run:
        print(
            "Dry run complete. Resolved configs: "
            + ", ".join(str(plan.config_path) for plan in plans)
        )
        return 0

    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(ROOT)
    for plan in plans:
        subprocess.run(
            _command(plan), cwd=ROOT, env=environment, check=True
        )

    combined = _combined_summary(plans)
    summary_name = (
        "combined_view_direction_robustness_summary.json"
        if len(plans) > 1
        else "view_direction_robustness_launcher_summary.json"
    )
    summary_paths = (
        [output_root / summary_name]
        if output_root is not None
        else [plan.output_dir / summary_name for plan in plans]
    )
    for path in summary_paths:
        _write_json(path, combined)
    print(
        "Completed inaccurate-view-direction evaluation: "
        + ", ".join(str(path) for path in summary_paths)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
