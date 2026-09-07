"""Export dense NPZ predictions from a trained Stage-2 AutoCAR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.dataset.stage2_npz import Stage2NPZDataset
from src.dataset.case_splits import load_case_splits
from src.evaluate_npz import evaluate_case
from src.modules.autocar_voxel_pl import AutoCARVoxelLit
from src.modules.sparse_utils import rasterize_sparse_channel


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise numeric per-case metrics with mean, SD, and standard error."""

    summary: dict[str, Any] = {"case_count": len(results), "metrics": {}}
    metric_names = sorted(
        {
            name
            for result in results
            for name in ("masked_dice_3d", "ssim_3d", "masked_ssim_3d")
            if isinstance(result.get(name), (int, float))
        }
    )
    for name in metric_names:
        values = np.asarray(
            [result[name] for result in results if result.get(name) is not None],
            dtype=np.float64,
        )
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary["metrics"][name] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": standard_deviation,
            "standard_error": standard_deviation / math.sqrt(len(values)),
        }
    return summary


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if device.type == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if device.index >= device_count:
            raise RuntimeError(
                f"CUDA device {device.index} was requested, but only "
                f"{device_count} device(s) are visible."
            )
    return device


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    if not 0.0 <= float(args.prediction_threshold) <= 1.0:
        raise ValueError("--prediction-threshold must lie in [0,1].")
    model = AutoCARVoxelLit.load_from_checkpoint(
        args.checkpoint, map_location="cpu"
    )
    if model.sparse_backend == "spconv" and device.type != "cuda":
        raise RuntimeError(
            "This checkpoint uses the spconv backend, which requires an NVIDIA "
            "CUDA device for prediction. Pass --device cuda on the Linux GPU "
            "environment, or use a checkpoint trained with a CPU-capable backend."
        )
    if model.recon_net.expected_view_count != len(args.view_indices):
        raise ValueError(
            "The checkpoint expects "
            f"{model.recon_net.expected_view_count} views, but --view-indices "
            f"contains {len(args.view_indices)}."
        )
    model.to(device)
    model.freeze()
    projection = model.recon_net.ray_casting

    if args.split_json is not None and args.case_id:
        raise ValueError("Use either --split-json or --case-id, not both.")
    if args.split_json is not None:
        split_path = args.split_json.resolve()
        splits = load_case_splits(split_path)
        selected_case_ids = list(splits[args.split])
        case_selection: dict[str, Any] = {
            "kind": "split_manifest",
            "split": args.split,
            "path": str(split_path),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "case_ids": selected_case_ids,
        }
    elif args.case_id:
        selected_case_ids = list(args.case_id)
        case_selection = {
            "kind": "explicit_case_ids",
            "case_ids": selected_case_ids,
        }
    else:
        if args.evaluate:
            raise ValueError(
                "--evaluate requires --split-json (recommended) or at least "
                "one explicit --case-id; unrestricted aggregate evaluation "
                "could mix train, validation, and test cases."
            )
        selected_case_ids = None
        case_selection = {"kind": "all_discovered_export_only"}

    dataset = Stage2NPZDataset(
        args.projections,
        args.voxels,
        view_mode="fixed",
        fixed_view_indices=args.view_indices,
        fixed_view_labels=(
            None if args.skip_anchor_label_check else args.expected_view_labels
        ),
        output_type="torch",
        case_ids=selected_case_ids,
    )
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if args.output_dtype == "float16" else np.float32
    use_mixed_precision = device.type == "cuda" and args.precision == "16-mixed"
    case_reports: list[dict[str, Any]] = []

    with torch.inference_mode():
        for sample in dataset:
            masks = sample["images"][None, :, 0].to(
                device=device, dtype=torch.float32
            )
            matrices = sample["world2pix4x4"][None].to(
                device=device, dtype=torch.float32
            )
            precision_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_mixed_precision
                else nullcontext()
            )
            with precision_context:
                prediction, _ = model.recon_net(masks, matrices)
            shape_xyz = projection.spatial_shape_xyz
            dense_tensor = rasterize_sparse_channel(
                prediction,
                backend=model.sparse_backend,
                spatial_shape_xyz=shape_xyz,
                batch_size=1,
                output_device="cpu",
            )[0]
            if dense_tensor.dtype == torch.bfloat16:
                dense_tensor = dense_tensor.float()
            dense = dense_tensor.detach().numpy()

            case_id = str(sample["case_id"])
            pair_angle_deg = float(sample["pair_angle_deg"])
            view_labels = list(sample.get("view_labels", ()))
            output_path = output_directory / f"{case_id}.npz"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Refusing to replace existing prediction {output_path}; "
                    "pass --overwrite to opt in."
                )
            np.savez_compressed(
                output_path,
                prediction_volume_zyx=dense.astype(dtype, copy=False),
                case_id=np.asarray(case_id),
                view_indices=np.asarray(sample["view_indices"], dtype=np.int64),
                pair_angle_deg=np.asarray(pair_angle_deg, dtype=np.float32),
                view_labels=np.asarray(view_labels),
                volume_axis_order=np.asarray("zyx"),
                bbox_min_xyz_mm=projection.bbox_min.detach().cpu().numpy(),
                bbox_max_xyz_mm=projection.bbox_max.detach().cpu().numpy(),
                voxel_size_mm=np.asarray(
                    projection.voxel_size, dtype=np.float32
                ),
            )
            report: dict[str, Any] = {
                "case_id": case_id,
                "prediction": str(output_path),
                "view_indices": [int(value) for value in sample["view_indices"]],
                "pair_angle_deg": pair_angle_deg,
                "view_labels": view_labels,
            }
            if args.evaluate:
                metrics = evaluate_case(
                    dense,
                    Path(sample["voxel_path"]),
                    Path(sample["projection_path"]),
                    bbox_min_xyz_mm=tuple(
                        float(value)
                        for value in projection.bbox_min.detach().cpu()
                    ),
                    voxel_size_mm=projection.voxel_size,
                    prediction_threshold=args.prediction_threshold,
                    ssim_window_size=args.ssim_window_size,
                    ssim_chunk_depth=args.ssim_chunk_depth,
                )
                report.update(metrics)
                metrics_path = output_directory / f"{case_id}.metrics.json"
                metrics_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            case_reports.append(report)

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "inference_precision": "16-mixed" if use_mixed_precision else "32",
        "output_dtype": args.output_dtype,
        "sparse_backend": model.sparse_backend,
        "case_selection": case_selection,
        "projection_protocol": {
            "expected_view_count": model.recon_net.expected_view_count,
            "candidate_mode": projection.candidate_mode,
            "distance_sampling": projection.distance_sampling,
            "max_pixel_distance": projection.max_pixel_distance,
            "support_views": projection.support_views,
            "fusion": projection.fusion,
            "include_distance_feature": projection.include_distance_feature,
            "voxel_chunk_size": projection.voxel_chunk_size,
            "projection_pixel_order": projection.projection_pixel_order,
            "bbox_min_xyz_mm": [
                float(value) for value in projection.bbox_min.detach().cpu()
            ],
            "bbox_max_xyz_mm": [
                float(value) for value in projection.bbox_max.detach().cpu()
            ],
            "voxel_size_mm": projection.voxel_size,
        },
        "cases": case_reports,
    }
    if args.evaluate:
        manifest["aggregate"] = _aggregate_metrics(case_reports)
    manifest_path = output_directory / "summary.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--projections", type=Path, required=True)
    parser.add_argument("--voxels", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--split-json",
        type=Path,
        help="Case-level split manifest; --split defaults to the test set.",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test"
    )
    parser.add_argument("--view-indices", nargs=2, type=int, default=(0, 6))
    parser.add_argument(
        "--expected-view-labels",
        nargs=2,
        default=("RAO 25, CAU 35", "LAO 5, CRA 40"),
        metavar=("VIEW_0_LABEL", "VIEW_1_LABEL"),
        help="Expected anchor_clinical_views labels at --view-indices.",
    )
    parser.add_argument(
        "--skip-anchor-label-check",
        action="store_true",
        help="Allow inputs without the maintained seven-slot clinical-anchor schema.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("16-mixed", "32"), default="16-mixed"
    )
    parser.add_argument(
        "--output-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--ssim-window-size", type=int, default=7)
    parser.add_argument("--ssim-chunk-depth", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = export_predictions(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
