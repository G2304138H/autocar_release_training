#!/usr/bin/env python3
"""Preflight and launch the maintained AutoCAR ImageCAS NPZ training runs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.case_splits import (
    apply_training_case_exclusions,
    load_case_splits,
)
from src.dataset.stage2_npz import Stage2NPZDataset


DATASETS = {
    "lca": {
        "experiment": "stage2_npz_lca",
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
        "pixel_spacing_mm": 0.65,
        "fallback_pixel_spacing_mm": 0.65,
        "fallback_sid_mm": 900.0,
        "source_to_isocenter_mm": 750.0,
        "view_indices": (0, 6),
        "view_labels": ("RAO 25, CAU 35", "LAO 5, CRA 40"),
        "excluded_train_case_ids": ("288", "421"),
    },
    "rca": {
        "experiment": "stage2_npz_rca",
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
        "pixel_spacing_mm": 0.55,
        "fallback_pixel_spacing_mm": 0.55,
        "fallback_sid_mm": 900.0,
        "source_to_isocenter_mm": 750.0,
        "view_indices": (0, 6),
        "view_labels": None,
        "excluded_train_case_ids": (
            "0288",
            "0421",
            "0909",
            "0108",
            "0207",
            "0324",
        ),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate physical-case pairing, then launch one LCA or RCA "
            "AutoCAR voxel-supervised training run."
        )
    )
    parser.add_argument("--artery", choices=sorted(DATASETS), required=True)
    parser.add_argument("--projection-source", type=Path)
    parser.add_argument("--voxel-source", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--expected-pixel-spacing-mm", type=float)
    parser.add_argument("--fallback-pixel-spacing-mm", type=float)
    parser.add_argument("--fallback-sid-mm", type=float)
    parser.add_argument("--source-to-isocenter-mm", type=float)
    parser.add_argument(
        "--evaluation-view-indices", type=int, nargs=2, metavar=("FIRST", "SECOND")
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument(
        "--numerical-debug",
        action="store_true",
        help=(
            "Check every gradient and model tensor for NaN/Inf, retain the "
            "contributing case/view context, and abort before an unsafe "
            "optimizer update. This is intentionally slower than normal training."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, splits, pairing, geometry metadata, and sample loading.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Start Hydra immediately (not recommended for a first run).",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides after --.",
    )
    return parser


def _resolved_settings(args: argparse.Namespace) -> dict[str, object]:
    defaults = DATASETS[args.artery]
    return {
        "experiment": defaults["experiment"],
        "projection_source": args.projection_source
        or defaults["projection_source"],
        "voxel_source": args.voxel_source or defaults["voxel_source"],
        "split_json": args.split_json or defaults["split_json"],
        "pixel_spacing_mm": args.expected_pixel_spacing_mm
        if args.expected_pixel_spacing_mm is not None
        else defaults["pixel_spacing_mm"],
        "fallback_pixel_spacing_mm": args.fallback_pixel_spacing_mm
        if args.fallback_pixel_spacing_mm is not None
        else defaults["fallback_pixel_spacing_mm"],
        "fallback_sid_mm": args.fallback_sid_mm
        if args.fallback_sid_mm is not None
        else defaults["fallback_sid_mm"],
        "source_to_isocenter_mm": args.source_to_isocenter_mm
        if args.source_to_isocenter_mm is not None
        else defaults["source_to_isocenter_mm"],
        "view_indices": tuple(args.evaluation_view_indices)
        if args.evaluation_view_indices is not None
        else defaults["view_indices"],
        "view_labels": defaults["view_labels"],
        "excluded_train_case_ids": defaults["excluded_train_case_ids"],
    }


def _preflight(settings: dict[str, object]) -> None:
    projection_source = Path(settings["projection_source"])
    voxel_source = Path(settings["voxel_source"])
    split_json = Path(settings["split_json"])
    for label, path in (
        ("projection source", projection_source),
        ("voxel source", voxel_source),
        ("split JSON", split_json),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    raw_splits = load_case_splits(split_json, case_id_mode="imagecas_numeric")
    splits, removed, absent = apply_training_case_exclusions(
        raw_splits,
        settings["excluded_train_case_ids"],
        case_id_mode="imagecas_numeric",
    )
    all_case_ids = tuple(
        case_id
        for split_name in ("train", "val", "test")
        for case_id in splits[split_name]
    )
    dataset = Stage2NPZDataset(
        projection_source=projection_source,
        voxel_source=voxel_source,
        case_ids=all_case_ids,
        case_id_mode="imagecas_numeric",
        expected_imager_pixel_spacing_mm=float(settings["pixel_spacing_mm"]),
        fallback_imager_pixel_spacing_mm=settings["fallback_pixel_spacing_mm"],
        fallback_sid_mm=settings["fallback_sid_mm"],
        source_to_isocenter_mm=float(settings["source_to_isocenter_mm"]),
        view_mode="fixed",
        fixed_view_indices=settings["view_indices"],
        fixed_view_labels=settings["view_labels"],
        output_type="numpy",
    )
    dataset.validate_projection_metadata()
    index_by_case = {
        record.case_id: index for index, record in enumerate(dataset.records)
    }
    print(
        "Split counts after training exclusions: "
        + ", ".join(f"{name}={len(splits[name])}" for name in splits)
    )
    print(
        "Applied training exclusions: "
        + (", ".join(removed) if removed else "none")
    )
    print(
        "Requested exclusions absent from split manifest: "
        + (", ".join(absent) if absent else "none")
    )
    for split_name in ("train", "val", "test"):
        case_id = splits[split_name][0]
        sample = dataset[index_by_case[case_id]]
        labels = sample.get("view_labels", "not supplied")
        print(
            f"{split_name}: case={case_id}, images={sample['images'].shape}, "
            f"GT={sample['gt_volume_zyx'].shape}, "
            f"image_dim={int(sample['image_dim'])} "
            f"({sample['image_dim_source']}), "
            f"pixel_spacing={float(sample['imager_pixel_spacing_mm']):g} mm "
            f"({sample['imager_pixel_spacing_source']}), "
            f"SID={float(sample['sid_mm']):g} mm ({sample['sid_source']}), "
            f"views={sample['view_indices'].tolist()}, labels={labels}, "
            f"pair_angle={float(sample['pair_angle_deg']):.2f} deg"
        )
    print(
        "Preflight passed: every split case has one projection/voxel pair and "
        "all projection metadata matches the declared detector spacing."
    )


def _hydra_command(
    args: argparse.Namespace, settings: dict[str, object]
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train",
        f"experiment={settings['experiment']}",
        f"data.projection_source={settings['projection_source']}",
        f"data.voxel_source={settings['voxel_source']}",
        f"data.split_json={settings['split_json']}",
        "data.case_id_mode=imagecas_numeric",
        "data.expected_imager_pixel_spacing_mm="
        f"{float(settings['pixel_spacing_mm']):g}",
        "data.source_to_isocenter_mm="
        f"{float(settings['source_to_isocenter_mm']):g}",
        "data.evaluation_view_indices="
        f"[{int(settings['view_indices'][0])},{int(settings['view_indices'][1])}]",
        f"data.num_workers={args.num_workers}",
        f"trainer.max_epochs={args.max_epochs}",
        "data.excluded_train_case_ids=["
        + ",".join(
            f'"{case_id}"' for case_id in settings["excluded_train_case_ids"]
        )
        + "]",
    ]
    if settings["fallback_pixel_spacing_mm"] is not None:
        command.append(
            "data.fallback_imager_pixel_spacing_mm="
            f"{float(settings['fallback_pixel_spacing_mm']):g}"
        )
    if settings["fallback_sid_mm"] is not None:
        command.append(
            f"data.fallback_sid_mm={float(settings['fallback_sid_mm']):g}"
        )
    if args.checkpoint is not None:
        command.append(f"ckpt_path={args.checkpoint}")
    if args.log_dir is not None:
        command.append(f"paths.log_dir={args.log_dir}")
    if args.numerical_debug:
        command.append("model.numerical_debug=true")
    extra = list(args.overrides)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    return command


def main() -> None:
    args = _parser().parse_args()
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.preflight_only and args.skip_preflight:
        raise ValueError("--preflight-only and --skip-preflight are incompatible.")
    settings = _resolved_settings(args)
    if not args.skip_preflight:
        _preflight(settings)
    if args.preflight_only:
        return

    command = _hydra_command(args, settings)
    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(ROOT)
    print("Launching:", shlex.join(command))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
