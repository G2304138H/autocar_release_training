from __future__ import annotations

import json

import numpy as np

from src.evaluate_npz import evaluate_case, main


def _case_files(tmp_path):
    projection_path = tmp_path / "lca_0001.npz"
    ground_truth_path = tmp_path / "1.npz"
    volume_xyz = np.zeros((9, 9, 9), dtype=np.uint8)
    volume_xyz[3:6, 3:6, 3:6] = 1
    np.savez(
        projection_path,
        projection_center_offset=np.zeros(3, dtype=np.float32),
        input_scale_to_mm=np.float32(1000.0),
    )
    np.savez(
        ground_truth_path,
        vol=volume_xyz,
        spacing=np.ones(3, dtype=np.float32),
    )
    return projection_path, ground_truth_path, volume_xyz.transpose(2, 1, 0)


def test_evaluate_case_identity(tmp_path):
    projection, ground_truth, prediction = _case_files(tmp_path)
    metrics = evaluate_case(
        prediction.astype(np.float32),
        ground_truth,
        projection,
        bbox_min_xyz_mm=(-0.5, -0.5, -0.5),
        voxel_size_mm=1.0,
        ssim_window_size=7,
        ssim_chunk_depth=2,
    )
    assert metrics["masked_dice_3d"] == 1.0
    assert metrics["ssim_3d"] == 1.0
    assert metrics["masked_ssim_3d"] == 1.0
    assert metrics["ground_truth_origin_xyz_mm"] == [-0.5, -0.5, -0.5]


def test_evaluate_case_honours_explicit_gt_lower_bound_origin(tmp_path):
    projection, ground_truth, prediction = _case_files(tmp_path)
    metrics = evaluate_case(
        prediction.astype(np.float32),
        ground_truth,
        projection,
        bbox_min_xyz_mm=(10.0, 20.0, 30.0),
        voxel_size_mm=1.0,
        ground_truth_origin_xyz_mm=(10.0, 20.0, 30.0),
        ssim_window_size=7,
        ssim_chunk_depth=2,
    )

    assert metrics["masked_dice_3d"] == 1.0
    assert metrics["ground_truth_origin_xyz_mm"] == [10.0, 20.0, 30.0]


def test_evaluate_case_honours_explicit_length_units(tmp_path):
    projection_path = tmp_path / "projection_units.npz"
    ground_truth_path = tmp_path / "ground_truth_units.npz"
    volume_xyz = np.zeros((9, 9, 9), dtype=np.uint8)
    volume_xyz[3:6, 3:6, 3:6] = 1
    np.savez(
        projection_path,
        projection_center_offset=np.zeros(3, dtype=np.float32),
        projection_center_offset_units=np.asarray("mm"),
        # Explicit units take precedence over this deliberately incompatible
        # fallback scale.
        input_scale_to_mm=np.float32(123.0),
    )
    np.savez(
        ground_truth_path,
        vol=volume_xyz,
        spacing=np.full(3, 0.001, dtype=np.float32),
        spacing_units=np.asarray("m"),
    )

    metrics = evaluate_case(
        volume_xyz.transpose(2, 1, 0).astype(np.float32),
        ground_truth_path,
        projection_path,
        bbox_min_xyz_mm=(-0.5, -0.5, -0.5),
        voxel_size_mm=1.0,
        ssim_window_size=7,
        ssim_chunk_depth=2,
    )

    assert metrics["masked_dice_3d"] == 1.0
    np.testing.assert_allclose(
        metrics["ground_truth_spacing_xyz_mm"], [1.0, 1.0, 1.0]
    )


def test_cli_writes_audited_json(tmp_path, capsys):
    projection, ground_truth, prediction = _case_files(tmp_path)
    prediction_path = tmp_path / "prediction.npy"
    output_path = tmp_path / "metrics.json"
    np.save(prediction_path, prediction.astype(np.float32))
    exit_code = main(
        [
            "--prediction",
            str(prediction_path),
            "--projection",
            str(projection),
            "--ground-truth",
            str(ground_truth),
            "--output",
            str(output_path),
            "--bbox-min-xyz-mm",
            "-0.5",
            "-0.5",
            "-0.5",
            "--prediction-axis-order",
            "zyx",
            "--voxel-size-mm",
            "1",
            "--ssim-window-size",
            "7",
            "--ssim-chunk-depth",
            "2",
        ]
    )
    assert exit_code == 0
    assert json.loads(output_path.read_text())["masked_dice_3d"] == 1.0
    assert '"ssim_3d": 1.0' in capsys.readouterr().out


def test_cli_uses_and_audits_exported_prediction_metadata(tmp_path):
    projection, ground_truth, prediction = _case_files(tmp_path)
    prediction_path = tmp_path / "prediction.npz"
    output_path = tmp_path / "metrics.json"
    np.savez_compressed(
        prediction_path,
        prediction_volume_zyx=prediction.astype(np.float32),
        volume_axis_order=np.asarray("zyx"),
        bbox_min_xyz_mm=np.asarray([-0.5, -0.5, -0.5]),
        bbox_max_xyz_mm=np.asarray([8.5, 8.5, 8.5]),
        voxel_size_mm=np.asarray(1.0),
        case_id=np.asarray("1"),
        view_indices=np.asarray([0, 6]),
    )

    assert (
        main(
            [
                "--prediction",
                str(prediction_path),
                "--projection",
                str(projection),
                "--ground-truth",
                str(ground_truth),
                "--output",
                str(output_path),
                "--ssim-window-size",
                "7",
                "--ssim-chunk-depth",
                "2",
            ]
        )
        == 0
    )

    report = json.loads(output_path.read_text())
    assert report["masked_dice_3d"] == 1.0
    assert report["prediction_source_axis_order"] == "zyx"
    assert report["prediction_file_metadata"]["view_indices"] == [0, 6]


def test_cli_rejects_bare_array_without_explicit_grid(tmp_path):
    projection, ground_truth, prediction = _case_files(tmp_path)
    prediction_path = tmp_path / "prediction.npy"
    np.save(prediction_path, prediction.astype(np.float32))

    with np.testing.assert_raises_regex(ValueError, "axis order is missing"):
        main(
            [
                "--prediction",
                str(prediction_path),
                "--projection",
                str(projection),
                "--ground-truth",
                str(ground_truth),
            ]
        )
