from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from src.metrics.centerline import centerline_radius_errors


def test_identical_graphs_have_zero_centerline_and_radius_error():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    radii = np.asarray([1.0, 1.5, 2.0], dtype=np.float32)

    result = centerline_radius_errors(points, radii, points, radii)

    assert result["valid"] is True
    assert result["centerline_mean_error_mm"] == pytest.approx(0.0)
    assert result["centerline_chamfer_distance_mm"] == pytest.approx(0.0)
    assert result["radius_mae_mm"] == pytest.approx(0.0)


def test_symmetric_errors_use_bidirectional_nearest_spatial_correspondence():
    prediction_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    ground_truth_xyz = np.asarray(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32
    )
    prediction_radius = np.asarray([1.0, 2.0], dtype=np.float32)
    ground_truth_radius = np.asarray([1.25, 2.5], dtype=np.float32)

    result = centerline_radius_errors(
        prediction_xyz,
        prediction_radius,
        ground_truth_xyz,
        ground_truth_radius,
    )

    assert result["centerline_pred_to_gt_mean_error_mm"] == pytest.approx(1.0)
    assert result["centerline_gt_to_pred_mean_error_mm"] == pytest.approx(1.0)
    assert result["centerline_mean_error_mm"] == pytest.approx(1.0)
    assert result["centerline_chamfer_distance_mm"] == pytest.approx(2.0)
    assert result["radius_pred_to_gt_mae_mm"] == pytest.approx(0.375)
    assert result["radius_gt_to_pred_mae_mm"] == pytest.approx(0.375)
    assert result["radius_mae_mm"] == pytest.approx(0.375)


def test_chamfer_distance_matches_unhalved_set_to_set_equation():
    prediction_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    ground_truth_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    result = centerline_radius_errors(
        prediction_xyz,
        np.ones(2, dtype=np.float32),
        ground_truth_xyz,
        np.ones(3, dtype=np.float32),
    )

    # pred->GT: mean([0, 1]) = 0.5; GT->pred: mean([0, 1, 2]) = 1.0.
    assert result["centerline_pred_to_gt_mean_error_mm"] == pytest.approx(0.5)
    assert result["centerline_gt_to_pred_mean_error_mm"] == pytest.approx(1.0)
    assert result["centerline_chamfer_distance_mm"] == pytest.approx(1.5)
    assert result["centerline_mean_error_mm"] == pytest.approx(0.75)


def test_empty_graph_errors_are_undefined_and_auditable():
    empty_points = np.empty((0, 3), dtype=np.float32)
    empty_radii = np.empty((0,), dtype=np.float32)
    ground_truth_points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    ground_truth_radii = np.asarray([1.0], dtype=np.float32)

    result = centerline_radius_errors(
        empty_points,
        empty_radii,
        ground_truth_points,
        ground_truth_radii,
    )

    assert result["valid"] is False
    assert result["prediction_nodes"] == 0
    assert result["ground_truth_nodes"] == 1
    assert result["centerline_mean_error_mm"] is None
    assert result["centerline_chamfer_distance_mm"] is None
    assert result["radius_mae_mm"] is None
