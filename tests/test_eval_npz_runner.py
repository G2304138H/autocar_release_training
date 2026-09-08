from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import src.eval_npz as eval_npz
from src.eval_npz import (
    EvaluationOptions,
    _paper_metric_case,
    _paper_metric_summary,
    _processing_timing_summary,
    _save_prediction_npz,
    _selected_split_cases,
    build_parser,
    resolve_evaluation_options,
    run_evaluation,
)


def test_val_test_selection_preserves_split_order_and_labels():
    case_ids, labels = _selected_split_cases(
        {
            "train": ("1",),
            "val": ("2", "3"),
            "test": ("4", "5"),
        },
        eval_split="val_test",
        requested_case_ids=(),
        limit=3,
    )

    assert case_ids == ("2", "3", "4")
    assert labels == {"2": "validation", "3": "validation", "4": "test"}


def test_explicit_case_selection_cannot_leave_requested_split():
    with pytest.raises(ValueError, match="outside the selected evaluation split"):
        _selected_split_cases(
            {"train": ("1",), "val": ("2",), "test": ("3",)},
            eval_split="test",
            requested_case_ids=("2",),
            limit=None,
        )


def test_config_resolves_relative_paths_and_visualization_alias(tmp_path):
    (tmp_path / "checkpoint.ckpt").touch()
    (tmp_path / "projections").mkdir()
    (tmp_path / "voxels").mkdir()
    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "checkpoint_path": "checkpoint.ckpt",
                "projection_source": "projections",
                "voxel_source": "voxels",
                "split_json_path": "splits.json",
                "eval_output_dir": "result",
                "evaluation_mode": "visualization",
                "eval_split": "val_test",
                "evaluation_view_indices": [0, 6],
                "evaluation_view_labels": None,
                "expected_imager_pixel_spacing_mm": 0.55,
                "fallback_imager_pixel_spacing_mm": 0.55,
                "fallback_sid_mm": 900.0,
                "max_visualizations": "all",
            }
        ),
        encoding="utf-8",
    )

    options = resolve_evaluation_options(config_path)

    assert options.evaluation_mode == "visualisation"
    assert options.case_ids == ("2", "3")
    assert options.case_splits == {"2": "validation", "3": "test"}
    assert options.max_visualizations == 2
    assert options.view_labels is None
    assert options.expected_imager_pixel_spacing_mm == 0.55
    assert options.fallback_imager_pixel_spacing_mm == 0.55
    assert options.fallback_sid_mm == 900.0
    assert options.output_dir == (tmp_path / "result").resolve()


def test_prediction_npz_uses_stable_volume_and_evaluation_fields(tmp_path):
    path = tmp_path / "prediction.npz"
    sample = {
        "case_id": "17",
        "view_indices": torch.tensor([0, 6]),
        "pair_angle_deg": torch.tensor(62.5),
        "view_labels": ("first", "second"),
    }
    protocol = {
        "bbox_min_xyz_mm": [-1.0, -2.0, -3.0],
        "bbox_max_xyz_mm": [1.0, 2.0, 3.0],
        "voxel_size_mm": 0.5,
    }

    _save_prediction_npz(
        path,
        np.ones((2, 3, 4), dtype=np.float32),
        sample,
        dataset_split="test",
        protocol=protocol,
        output_dtype="float16",
    )

    with np.load(path, allow_pickle=False) as payload:
        assert payload["prediction_volume_zyx"].shape == (2, 3, 4)
        assert payload["prediction_volume_zyx"].dtype == np.float16
        assert payload["volume_axis_order"].item() == "zyx"
        assert payload["dataset_split"].item() == "test"
        assert payload["evaluation_role"].item() == "final"
        np.testing.assert_array_equal(payload["view_indices"], [0, 6])


def test_paper_summary_reports_macro_standard_error_and_micro_dice():
    reports = [
        {
            "paper_mask_dice_3d": 0.5,
            "paper_mask_ssim_3d": 0.75,
            "paper_mask_masked_ssim_3d": 0.4,
            "paper_mask_intersection_voxels": 2,
            "paper_mask_predicted_foreground_voxels": 4,
            "paper_mask_ground_truth_foreground_voxels": 4,
        },
        {
            "paper_mask_dice_3d": 1.0,
            "paper_mask_ssim_3d": 0.25,
            "paper_mask_masked_ssim_3d": None,
            "paper_mask_intersection_voxels": 3,
            "paper_mask_predicted_foreground_voxels": 3,
            "paper_mask_ground_truth_foreground_voxels": 3,
        },
    ]

    summary = _paper_metric_summary(reports, protocol={"voxel_size_mm": 0.5})

    assert summary["micro_dice_3d"] == pytest.approx(10.0 / 14.0)
    assert summary["macro_dice_3d"] == pytest.approx(0.75)
    assert summary["macro_dice_3d_standard_error"] == pytest.approx(0.25)
    assert summary["macro_masked_ssim_3d_num_cases"] == 1


def test_processing_timing_summary_reports_combined_and_split_averages():
    rows = [
        {
            "case_id": "2",
            "split": "validation",
            "inference_elapsed_ms": 10.0,
            "processing_elapsed_ms": 30.0,
        },
        {
            "case_id": "3",
            "split": "test",
            "inference_elapsed_ms": 14.0,
            "processing_elapsed_ms": 50.0,
        },
    ]

    summary = _processing_timing_summary(rows, warmup_performed=True)

    assert summary["mean_inference_elapsed_ms"] == 12.0
    assert summary["mean_processing_elapsed_ms"] == 40.0
    assert summary["processing_elapsed_ms_standard_error"] == 10.0
    assert summary["warmup_policy"].startswith("one_untimed")
    assert summary["by_split"]["validation"]["mean_processing_elapsed_ms"] == 30.0
    assert summary["by_split"]["test"]["mean_processing_elapsed_ms"] == 50.0


def test_paper_metric_case_uses_endpoint_aligned_native_fov(monkeypatch):
    monkeypatch.setattr(eval_npz, "_PAPER_METRIC_SHAPE_ZYX", (9, 9, 9))
    volume = np.zeros((9, 9, 9), dtype=np.uint8)
    volume[3:6, 3:6, 3:6] = 1
    sample = {
        "gt_volume_zyx": volume,
        "gt_spacing_xyz_mm": np.ones(3, dtype=np.float32),
        "gt_origin_xyz_mm": np.full(3, -0.5, dtype=np.float32),
        "projection_center_offset_xyz_mm": np.zeros(3, dtype=np.float32),
    }

    result = _paper_metric_case(
        volume.astype(np.float32),
        sample,
        bbox_min_xyz_mm=(-0.5, -0.5, -0.5),
        voxel_size_mm=1.0,
        prediction_threshold=0.5,
        ssim_window_size=7,
        ssim_chunk_depth=2,
    )

    assert result["metrics"]["paper_mask_dice_3d"] == 1.0
    assert result["metrics"]["paper_mask_ssim_3d"] == 1.0
    np.testing.assert_array_equal(result["predicted_mask_zyx"], volume)


def test_cli_matches_parametric_evaluation_override_names():
    args = build_parser().parse_args(
        [
            "--config",
            "eval.json",
            "--case_id",
            "2",
            "--case_id",
            "3",
            "--split",
            "val_test",
            "--max_cases",
            "2",
            "--metrics_only",
        ]
    )

    assert args.case_ids == ["2", "3"]
    assert args.split == "val_test"
    assert args.max_cases == 2
    assert args.metrics_only is True


def test_runner_writes_prediction_metrics_and_audit_manifests(
    monkeypatch,
    tmp_path,
):
    dataset_kwargs = {}

    class FakeModel:
        sparse_backend = "raw"
        recon_net = SimpleNamespace(expected_view_count=2)

        def to(self, device):
            return self

        def freeze(self):
            return None

    class FakeLoader:
        @staticmethod
        def load_from_checkpoint(*args, **kwargs):
            return FakeModel()

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            dataset_kwargs.update(kwargs)
            self.samples = [
                {
                    "case_id": "2",
                    "view_indices": np.asarray([0, 6]),
                    "view_labels": ("first", "second"),
                    "pair_angle_deg": 60.0,
                    "voxel_path": "ground_truth.npz",
                    "projection_path": "projection.npz",
                }
            ]

        def __len__(self):
            return len(self.samples)

        def __iter__(self):
            return iter(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    protocol = {
        "expected_view_count": 2,
        "candidate_mode": "voxel_grid",
        "distance_sampling": "nearest",
        "max_pixel_distance": 0.5,
        "support_views": 2,
        "fusion": "concat",
        "include_distance_feature": True,
        "voxel_chunk_size": 16,
        "projection_pixel_order": "xy",
        "bbox_min_xyz_mm": [-0.5, -0.5, -0.5],
        "bbox_max_xyz_mm": [8.5, 8.5, 8.5],
        "voxel_size_mm": 1.0,
    }
    metrics = {
        "masked_dice_3d": 1.0,
        "ssim_3d": 1.0,
        "masked_ssim_3d": 1.0,
        "intersection_voxels": 1,
        "prediction_foreground_voxels": 1,
        "ground_truth_foreground_voxels": 1,
        "valid_fov_voxels": 729,
    }
    monkeypatch.setattr(eval_npz, "AutoCARVoxelLit", FakeLoader)
    monkeypatch.setattr(eval_npz, "Stage2NPZDataset", FakeDataset)
    monkeypatch.setattr(eval_npz, "_prediction_protocol", lambda model: protocol)
    monkeypatch.setattr(
        eval_npz,
        "_forward_dense",
        lambda *args, **kwargs: (
            np.ones((9, 9, 9), dtype=np.float32),
            12.5,
        ),
    )
    monkeypatch.setattr(eval_npz, "evaluate_case", lambda *args, **kwargs: metrics)
    monkeypatch.setattr(
        eval_npz,
        "_save_metric_histogram",
        lambda reports, path: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_bytes(b"png"),
        ),
    )

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    options = EvaluationOptions(
        config_path=tmp_path / "eval.json",
        checkpoint=checkpoint,
        checkpoint_choice="explicit",
        projection_source=tmp_path,
        voxel_source=tmp_path,
        split_json=split_path,
        output_dir=output_dir,
        evaluation_mode="metric",
        eval_split="val",
        case_ids=("2",),
        case_splits={"2": "validation"},
        case_id_mode="literal",
        expected_imager_pixel_spacing_mm=0.55,
        fallback_imager_pixel_spacing_mm=0.55,
        fallback_sid_mm=900.0,
        view_indices=(0, 6),
        view_labels=("first", "second"),
        device="cpu",
        precision="32",
        output_dtype="float16",
        prediction_threshold=0.5,
        paper_metric_save_masks=True,
        ssim_window_size=7,
        ssim_chunk_depth=2,
        ground_truth_origin_xyz_mm=None,
        source_to_isocenter_mm=750.0,
        max_visualizations=0,
        visualization_gif_frames=0,
        visualization_gif_fps=6,
        visualization_max_points=100,
        overwrite=False,
        resolved_config={},
    )

    summary = run_evaluation(options)

    prediction_path = output_dir / "predictions" / "final" / "validation" / "2.npz"
    with np.load(prediction_path, allow_pickle=False) as payload:
        assert payload["prediction_volume_zyx"].dtype == np.float16
        assert payload["dataset_split"].item() == "validation"
    manifest = json.loads(
        (output_dir / "predictions" / "manifest.json").read_text()
    )
    assert manifest["num_files"] == 1
    assert dataset_kwargs["expected_imager_pixel_spacing_mm"] == 0.55
    assert dataset_kwargs["fallback_imager_pixel_spacing_mm"] == 0.55
    assert dataset_kwargs["fallback_sid_mm"] == 900.0
    assert summary["timing"]["mean_inference_elapsed_ms"] == 12.5
    assert summary["timing"]["mean_processing_elapsed_ms"] > 0.0
    assert summary["roles"]["final"]["masked_dice_3d"] == 1.0
    assert (output_dir / "metrics" / "flat_per_case_metrics.csv").is_file()
    assert (output_dir / "timings" / "processing" / "per_case.csv").is_file()
    timing_summary = json.loads(
        (output_dir / "timings" / "processing" / "summary.json").read_text()
    )
    assert timing_summary["mean_processing_elapsed_ms"] > 0.0
    assert (output_dir / "evaluation_record.json").is_file()
