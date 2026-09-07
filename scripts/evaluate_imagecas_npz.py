#!/usr/bin/env python3
"""Run val+test paper-metric evaluation for trained LCA and RCA models.

The launcher resolves the best checkpoint from each maintained Hydra training
run, writes one fully resolved JSON configuration per artery, and invokes the
config-driven evaluator in a separate process for each model.  Separate
processes keep CUDA/spconv state isolated between the LCA and RCA runs.
"""

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


# These values intentionally match configs/data/stage2_npz_{lca,rca}.yaml and
# scripts/train_imagecas_npz.py.  The fallbacks are required only for older
# projection files that omit detector spacing and SID metadata.
DATASETS: Mapping[str, Mapping[str, Any]] = {
    "lca": {
        "training_config": ROOT / "configs" / "experiment" / "stage2_npz_lca.yaml",
        "projection_source": Path(
            "/dataset/reny0012/vessel_code_stage_2_lca_paired/anchors"
        ),
        "voxel_source": Path("/dataset/reny0012/imagecas_voxel/lca"),
        "split_json": Path(
            "/export/home2/reny0012/result/vessel_trees/"
            "multi_branch_experiments/"
            "vggt_branch_prefix_lca_grouped_visibility_1to7_clinical_views_"
            "refiner_5mm/split.json"
        ),
        "expected_imager_pixel_spacing_mm": 0.65,
        "fallback_imager_pixel_spacing_mm": 0.65,
        "fallback_sid_mm": 900.0,
        "source_to_isocenter_mm": 750.0,
        "evaluation_view_indices": (0, 6),
        "evaluation_view_labels": ("RAO 25, CAU 35", "LAO 5, CRA 40"),
    },
    "rca": {
        "training_config": ROOT / "configs" / "experiment" / "stage2_npz_rca.yaml",
        "projection_source": Path(
            "/dataset/reny0012/imagecas_autocar_6/"
            "stage_2_imagecas_all_branch"
        ),
        "voxel_source": Path("/dataset/reny0012/imagecas_voxel/rca"),
        "split_json": Path(
            "/export/home2/reny0012/result/vessel_trees/"
            "experiments_for_paper/"
            "exp6_rca_bspline_parallel_no_cross_vggt_refiner/split.json"
        ),
        "expected_imager_pixel_spacing_mm": 0.55,
        "fallback_imager_pixel_spacing_mm": 0.55,
        "fallback_sid_mm": 900.0,
        "source_to_isocenter_mm": 750.0,
        "evaluation_view_indices": (0, 6),
        "evaluation_view_labels": None,
    },
}


@dataclass(frozen=True)
class EvaluationPlan:
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
        "--artery",
        choices=("both", "lca", "rca"),
        default="both",
        help="Evaluate both trained models by default, or select one.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs",
        help=(
            "Hydra logging root. Automatic lookup reads each training config's "
            "task_name and searches <log-dir>/<task_name>/runs."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional common root for separate lca/ and rca/ results. By "
            "default, each result is saved below its own training run directory."
        ),
    )
    for artery in ("lca", "rca"):
        parser.add_argument(
            f"--{artery}-checkpoint",
            type=Path,
            help=f"Explicit trained {artery.upper()} .ckpt (overrides discovery).",
        )
        parser.add_argument(
            f"--{artery}-experiment-dir",
            type=Path,
            help=(
                f"Hydra run directory for {artery.upper()}; its unique best "
                "non-last checkpoint is selected."
            ),
        )
        parser.add_argument(
            f"--{artery}-projection-source",
            type=Path,
            help=f"Override the {artery.upper()} projection NPZ source.",
        )
        parser.add_argument(
            f"--{artery}-voxel-source",
            type=Path,
            help=f"Override the {artery.upper()} ground-truth voxel NPZ source.",
        )
        parser.add_argument(
            f"--{artery}-split-json",
            type=Path,
            help=f"Override the {artery.upper()} case split manifest.",
        )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device accepted by src.eval_npz (default: auto).",
    )
    parser.add_argument(
        "--precision",
        choices=("32", "16-mixed"),
        default="32",
        help="Model-forward precision; 32 is the tested spconv setting.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Dtype of saved full-resolution probability volumes.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional combined val+test case limit for a smoke run.",
    )
    parser.add_argument(
        "--no-paper-mask-files",
        action="store_true",
        help="Compute paper metrics without retaining the 128^3 comparison masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow matching artifacts in non-empty artery output directories "
            "to be replaced."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs/checkpoints and write configs without launching "
            "evaluation."
        ),
    )
    return parser


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _checkpoint_directory_candidates(experiment_dir: Path) -> tuple[Path, ...]:
    directories = [experiment_dir / "checkpoints", experiment_dir]
    return tuple(dict.fromkeys(_resolved(path) for path in directories))


def _training_task_name(artery: str) -> str:
    """Read the Hydra task name from the maintained artery training config."""

    config_path = Path(DATASETS[artery]["training_config"])
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration does not exist for {artery.upper()}: "
            f"{config_path}"
        )
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("task_name:"):
            continue
        value = stripped.split(":", 1)[1].split("#", 1)[0].strip()
        task_name = value.strip("'\"")
        if task_name and not any(character.isspace() for character in task_name):
            return task_name
        break
    raise ValueError(f"No valid task_name was found in {config_path}.")


def _best_checkpoint_in_run(experiment_dir: Path) -> Path:
    """Return a run's best checkpoint without mistaking last.ckpt for best."""

    experiment_dir = _resolved(experiment_dir)
    directories = _checkpoint_directory_candidates(experiment_dir)
    best_aliases = [directory / "best.ckpt" for directory in directories]
    existing_aliases = [path for path in best_aliases if path.is_file()]
    if len(existing_aliases) == 1:
        return existing_aliases[0].resolve()
    if len(existing_aliases) > 1:
        raise ValueError(
            f"Multiple best.ckpt aliases were found under {experiment_dir}; "
            "pass an explicit artery checkpoint."
        )

    candidates = sorted(
        {
            path.resolve()
            for directory in directories
            if directory.is_dir()
            for path in directory.glob("*.ckpt")
            if path.name != "last.ckpt"
        },
        key=str,
    )
    if len(candidates) == 1:
        return candidates[0]
    searched = ", ".join(str(directory / "*.ckpt") for directory in directories)
    if not candidates:
        raise FileNotFoundError(
            "No best training checkpoint was found. last.ckpt is intentionally "
            f"not treated as best. Searched: {searched}"
        )
    raise ValueError(
        f"Found {len(candidates)} non-last checkpoints under {experiment_dir}; "
        "pass the intended checkpoint explicitly. Candidates: "
        + ", ".join(str(path) for path in candidates)
    )


def _latest_trained_checkpoint(log_dir: Path, artery: str) -> tuple[Path, Path]:
    task_name = _training_task_name(artery)
    runs_dir = _resolved(log_dir) / task_name / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(
            f"Training run directory does not exist for {artery.upper()}: {runs_dir}. "
            f"Pass --{artery}-checkpoint or --{artery}-experiment-dir."
        )
    runs = sorted(
        (path for path in runs_dir.iterdir() if path.is_dir()),
        # Hydra names these directories YYYY-MM-DD_HH-MM-SS. Directory mtimes
        # are unsuitable because writing a later evaluation below an old run
        # would otherwise make that training run appear newest.
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in runs:
        try:
            return _best_checkpoint_in_run(run_dir), run_dir.resolve()
        except FileNotFoundError:
            # A newer run may have started but not reached its first validation.
            continue
        except ValueError as error:
            raise ValueError(
                f"The newest checkpoint-bearing {artery.upper()} run is ambiguous: "
                f"{error}"
            ) from error
    raise FileNotFoundError(
        f"No completed {artery.upper()} run with a best checkpoint was found in "
        f"{runs_dir}. Pass --{artery}-checkpoint explicitly."
    )


def _infer_experiment_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def resolve_trained_checkpoint(
    *,
    artery: str,
    log_dir: Path,
    checkpoint: Path | None = None,
    experiment_dir: Path | None = None,
) -> tuple[Path, Path, str]:
    """Resolve an explicit checkpoint or the configured task's latest best run."""

    if artery not in DATASETS:
        raise ValueError(f"Unsupported artery: {artery!r}.")
    if checkpoint is not None:
        resolved_checkpoint = _resolved(checkpoint)
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(
                f"Explicit {artery.upper()} checkpoint does not exist: "
                f"{resolved_checkpoint}"
            )
        if resolved_checkpoint.suffix != ".ckpt":
            raise ValueError(
                f"Explicit {artery.upper()} checkpoint must end in .ckpt: "
                f"{resolved_checkpoint}"
            )
        resolved_experiment_dir = (
            _resolved(experiment_dir)
            if experiment_dir is not None
            else _infer_experiment_dir(resolved_checkpoint)
        )
        if not resolved_experiment_dir.is_dir():
            raise NotADirectoryError(
                f"{artery.upper()} experiment directory does not exist: "
                f"{resolved_experiment_dir}"
            )
        return resolved_checkpoint, resolved_experiment_dir, "explicit"
    if experiment_dir is not None:
        resolved_experiment_dir = _resolved(experiment_dir)
        return (
            _best_checkpoint_in_run(resolved_experiment_dir),
            resolved_experiment_dir,
            "experiment_best",
        )
    resolved_checkpoint, resolved_experiment_dir = _latest_trained_checkpoint(
        log_dir, artery
    )
    return resolved_checkpoint, resolved_experiment_dir, "latest_run_best"


def _resolve_checkpoint(
    args: argparse.Namespace,
    artery: str,
) -> tuple[Path, Path, str]:
    return resolve_trained_checkpoint(
        artery=artery,
        log_dir=args.log_dir,
        checkpoint=getattr(args, f"{artery}_checkpoint"),
        experiment_dir=getattr(args, f"{artery}_experiment_dir"),
    )


def _input_path(args: argparse.Namespace, artery: str, name: str) -> Path:
    override = getattr(args, f"{artery}_{name}")
    return _resolved(override if override is not None else DATASETS[artery][name])


def _validate_input_paths(config: Mapping[str, Any], artery: str) -> None:
    for key in ("projection_source", "voxel_source"):
        path = Path(config[key])
        if not path.exists():
            raise FileNotFoundError(
                f"{artery.upper()} {key} does not exist: {path}"
            )
    split_json = Path(config["split_json_path"])
    if not split_json.is_file():
        raise FileNotFoundError(
            f"{artery.upper()} split_json_path does not exist: {split_json}"
        )


def _evaluation_config(
    args: argparse.Namespace,
    artery: str,
    *,
    checkpoint: Path,
    experiment_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = DATASETS[artery]
    labels = dataset["evaluation_view_labels"]
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
        "evaluation_mode": "paper_metric",
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
        "max_visualizations": 0,
        "device": args.device,
        "precision": args.precision,
        "output_dtype": args.output_dtype,
        "paper_metric_volume_threshold": 0.5,
        "paper_metric_ssim_window_size": 7,
        "paper_metric_save_masks": not args.no_paper_mask_files,
        "ssim_chunk_depth": 8,
        "ground_truth_origin_xyz_mm": None,
        "overwrite": args.overwrite,
    }


def _evaluation_command(plan: EvaluationPlan) -> list[str]:
    return [sys.executable, "-m", "src.eval_npz", "--config", str(plan.config_path)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _assert_output_available(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Evaluation output is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Evaluation output directory is not empty: {output_dir}. "
            "Pass --overwrite to reuse it."
        )


def _build_plans(
    args: argparse.Namespace,
    *,
    output_root: Path | None,
) -> list[EvaluationPlan]:
    arteries = ("lca", "rca") if args.artery == "both" else (args.artery,)
    plans: list[EvaluationPlan] = []
    for artery in arteries:
        checkpoint, experiment_dir, checkpoint_source = _resolve_checkpoint(
            args, artery
        )
        output_dir = (
            output_root / artery
            if output_root is not None
            else experiment_dir / "evaluation_paper_metric" / checkpoint.stem
        )
        _assert_output_available(output_dir, overwrite=args.overwrite)
        config_path = (
            output_root / "configs" / f"{artery}_paper_metric.json"
            if output_root is not None
            else experiment_dir
            / "evaluation_configs"
            / f"paper_metric_{checkpoint.stem}.json"
        )
        config = _evaluation_config(
            args,
            artery,
            checkpoint=checkpoint,
            experiment_dir=experiment_dir,
            output_dir=output_dir,
        )
        _validate_input_paths(config, artery)
        plans.append(
            EvaluationPlan(
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


def _plan_record(plans: Sequence[EvaluationPlan]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluation_mode": "paper_metric",
        "eval_split": "val_test",
        "aggregation_policy": "report_lca_and_rca_separately",
        "runs": [
            {
                "artery": plan.artery,
                "training_config": str(DATASETS[plan.artery]["training_config"]),
                "checkpoint": str(plan.checkpoint),
                "checkpoint_source": plan.checkpoint_source,
                "experiment_dir": str(plan.experiment_dir),
                "config": str(plan.config_path),
                "output_dir": str(plan.output_dir),
                "command": _evaluation_command(plan),
            }
            for plan in plans
        ],
    }


def _plan_record_paths(
    plans: Sequence[EvaluationPlan],
    *,
    output_root: Path | None,
) -> tuple[Path, ...]:
    if output_root is not None:
        return (output_root / "evaluation_plan.json",)
    return tuple(
        plan.experiment_dir
        / "evaluation_configs"
        / f"paper_metric_{plan.checkpoint.stem}_plan.json"
        for plan in plans
    )


def _combined_summary(plans: Sequence[EvaluationPlan]) -> dict[str, Any]:
    arteries: dict[str, Any] = {}
    for plan in plans:
        paper_path = plan.output_dir / "metrics" / "paper_metric_summary.json"
        performance_path = plan.output_dir / "performance_summary.json"
        if not paper_path.is_file() or not performance_path.is_file():
            raise FileNotFoundError(
                f"{plan.artery.upper()} evaluation completed without its expected "
                f"summary files under {plan.output_dir}."
            )
        arteries[plan.artery] = {
            "checkpoint": str(plan.checkpoint),
            "checkpoint_source": plan.checkpoint_source,
            "output_dir": str(plan.output_dir),
            "performance_summary_file": str(performance_path),
            "paper_metric_summary_file": str(paper_path),
            "paper_metric": json.loads(paper_path.read_text(encoding="utf-8")),
        }
    return {
        "schema_version": 1,
        "evaluation_mode": "paper_metric",
        "eval_split": "val_test",
        "aggregation_policy": (
            "LCA and RCA are reported separately; no cross-artery pooled statistic"
        ),
        "arteries": arteries,
    }


def _summary_paths(
    plans: Sequence[EvaluationPlan],
    *,
    output_root: Path | None,
) -> tuple[Path, ...]:
    filename = (
        "combined_paper_metric_summary.json"
        if len(plans) > 1
        else "paper_metric_launcher_summary.json"
    )
    if output_root is not None:
        return (output_root / filename,)
    return tuple(plan.output_dir / filename for plan in plans)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be positive.")
    log_dir = _resolved(args.log_dir)
    output_root = (
        _resolved(args.output_root)
        if args.output_root is not None
        else None
    )
    if output_root is not None and output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"--output-root is not a directory: {output_root}")

    plans = _build_plans(args, output_root=output_root)
    for plan in plans:
        _write_json(plan.config_path, plan.config)
    plan_record = _plan_record(plans)
    for path in _plan_record_paths(plans, output_root=output_root):
        _write_json(path, plan_record)

    for plan in plans:
        print(
            f"{plan.artery.upper()}: checkpoint={plan.checkpoint} "
            f"({plan.checkpoint_source})",
            flush=True,
        )
        print(f"Launching: {shlex.join(_evaluation_command(plan))}", flush=True)
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
            _evaluation_command(plan),
            cwd=ROOT,
            env=environment,
            check=True,
        )

    summary = _combined_summary(plans)
    summary_paths = _summary_paths(plans, output_root=output_root)
    for path in summary_paths:
        _write_json(path, summary)
    print(
        "Completed paper-metric evaluation: "
        + ", ".join(str(path) for path in summary_paths)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
