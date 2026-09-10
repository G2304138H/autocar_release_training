import numpy as np
import pytest

from src.metrics import (
    compute_volume_metrics,
    hard_cldice_3d,
    masked_dice_3d,
    masked_ssim_3d,
    structural_similarity_3d,
)


def test_hard_cldice_matches_paper_equation_and_reports_loss():
    truth = np.zeros((5, 5, 5), dtype=bool)
    prediction = np.zeros_like(truth)
    truth_skeleton = np.zeros_like(truth)
    prediction_skeleton = np.zeros_like(truth)
    prediction_skeleton[1, 1, 1:4] = True
    truth_skeleton[2, 2, 0:4] = True
    prediction |= prediction_skeleton
    prediction[2, 2, 0:2] = True
    truth |= truth_skeleton
    truth[1, 1, 1:3] = True

    result = hard_cldice_3d(
        truth,
        prediction,
        ground_truth_skeleton=truth_skeleton,
        prediction_skeleton=prediction_skeleton,
    )

    assert result["topology_precision"] == pytest.approx(2.0 / 3.0)
    assert result["topology_sensitivity"] == pytest.approx(1.0 / 2.0)
    assert result["cldice_3d"] == pytest.approx(4.0 / 7.0)
    assert result["cldice_loss_3d"] == pytest.approx(3.0 / 7.0)


def test_hard_cldice_handles_empty_and_identical_volumes():
    empty = np.zeros((5, 5, 5), dtype=bool)
    vessel = empty.copy()
    vessel[1:4, 2, 2] = True

    assert hard_cldice_3d(empty, empty)["cldice_3d"] == 1.0
    assert hard_cldice_3d(vessel, empty)["cldice_3d"] == 0.0
    assert hard_cldice_3d(vessel, vessel)["cldice_loss_3d"] == 0.0


def test_dice_identity_disjoint_and_exact_empty_cases():
    empty = np.zeros((4, 4, 4), dtype=np.float32)
    one = empty.copy()
    one[1, 1, 1] = 1.0
    other = empty.copy()
    other[2, 2, 2] = 1.0

    assert masked_dice_3d(one, one) == 1.0
    assert masked_dice_3d(one, other) == 0.0
    assert masked_dice_3d(empty, empty) == 1.0
    assert masked_dice_3d(empty, one) == 0.0


def test_dice_two_thirds_overlap_and_symmetry():
    truth = np.zeros((3, 3, 3), dtype=bool)
    prediction = np.zeros_like(truth)
    truth[1, 1, 1] = True
    prediction[1, 1, 1] = True
    prediction[1, 1, 2] = True

    expected = 2.0 / 3.0
    assert masked_dice_3d(truth, prediction) == expected
    assert masked_dice_3d(prediction, truth) == expected


def test_dice_ignores_false_positive_outside_roi():
    truth = np.zeros((5, 5, 5), dtype=bool)
    prediction = np.zeros_like(truth)
    roi = np.zeros_like(truth)
    roi[1:4, 1:4, 1:4] = True
    truth[2, 2, 2] = True
    prediction[2, 2, 2] = True
    prediction[0, 0, 0] = True

    assert masked_dice_3d(truth, prediction, mask=roi) == 1.0
    assert masked_dice_3d(truth, prediction) == 2.0 / 3.0


def test_dice_rejects_empty_or_wrong_shape_mask():
    volume = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="no evaluation voxels"):
        masked_dice_3d(volume, volume, mask=np.zeros_like(volume, dtype=bool))
    with pytest.raises(ValueError, match="does not match"):
        masked_dice_3d(volume, volume, mask=np.ones((2, 2, 2)))


def test_ssim_identity_and_empty_volumes_are_exact_and_finite():
    rng = np.random.default_rng(42)
    volume = rng.random((9, 10, 11), dtype=np.float32)
    empty = np.zeros_like(volume)

    assert structural_similarity_3d(volume, volume, window_size=3) == pytest.approx(
        1.0, abs=1e-12
    )
    empty_score = structural_similarity_3d(empty, empty, window_size=3)
    assert np.isfinite(empty_score)
    assert empty_score == pytest.approx(1.0, abs=1e-12)


def test_ssim_matches_independent_single_window_formula():
    truth = np.linspace(0.0, 1.0, 27, dtype=np.float64).reshape(3, 3, 3)
    prediction = np.square(truth)
    truth_flat = truth.ravel()
    prediction_flat = prediction.ravel()
    mean_truth = float(np.mean(truth_flat))
    mean_prediction = float(np.mean(prediction_flat))
    variance_truth = float(np.var(truth_flat, ddof=1))
    variance_prediction = float(np.var(prediction_flat, ddof=1))
    covariance = float(
        np.sum(
            (truth_flat - mean_truth) * (prediction_flat - mean_prediction)
        )
        / (truth_flat.size - 1)
    )
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    expected = (
        (2 * mean_truth * mean_prediction + c1)
        * (2 * covariance + c2)
        / (
            (mean_truth ** 2 + mean_prediction ** 2 + c1)
            * (variance_truth + variance_prediction + c2)
        )
    )

    score, score_map = structural_similarity_3d(
        truth, prediction, window_size=3, return_map=True
    )

    assert score_map.shape == (1, 1, 1)
    assert score == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert float(score_map[0, 0, 0]) == pytest.approx(expected, rel=1e-6)


def test_ssim_is_symmetric_and_a_shift_reduces_it():
    truth = np.zeros((11, 11, 11), dtype=np.float32)
    truth[4:7, 4:7, 4:7] = 1.0
    shifted = np.zeros_like(truth)
    shifted[4:7, 4:7, 5:8] = 1.0

    forward = structural_similarity_3d(truth, shifted, window_size=3)
    reverse = structural_similarity_3d(shifted, truth, window_size=3)

    assert forward == pytest.approx(reverse, abs=1e-12)
    assert forward < structural_similarity_3d(truth, truth, window_size=3)


def test_ssim_roi_fully_excludes_outside_difference():
    truth = np.zeros((11, 11, 11), dtype=np.float32)
    prediction = truth.copy()
    prediction[0, 0, 0] = 1.0
    roi = np.zeros_like(truth, dtype=bool)
    roi[3:8, 3:8, 3:8] = True

    score = structural_similarity_3d(
        truth, prediction, window_size=3, roi_mask=roi
    )

    assert score == pytest.approx(1.0, abs=1e-12)


def test_ssim_center_mask_and_roi_can_be_combined():
    truth = np.zeros((9, 9, 9), dtype=np.float32)
    prediction = truth.copy()
    truth[4, 4, 4] = 1.0
    prediction[4, 4, 4] = 0.5
    roi = np.ones_like(truth, dtype=bool)
    center_mask = np.zeros_like(truth, dtype=bool)
    center_mask[4, 4, 4] = True

    direct = structural_similarity_3d(
        truth,
        prediction,
        window_size=3,
        roi_mask=roi,
        mask=center_mask,
    )
    wrapped = masked_ssim_3d(
        truth,
        prediction,
        window_size=3,
        roi_mask=roi,
        mask=center_mask,
    )

    assert direct == pytest.approx(wrapped)
    assert direct < 1.0


def test_ssim_chunking_does_not_change_score_or_map():
    rng = np.random.default_rng(123)
    truth = rng.random((10, 11, 12), dtype=np.float32)
    prediction = rng.random((10, 11, 12), dtype=np.float32)

    full_score, full_map = structural_similarity_3d(
        truth,
        prediction,
        window_size=3,
        chunk_depth=None,
        return_map=True,
    )
    chunk_score, chunk_map = structural_similarity_3d(
        truth,
        prediction,
        window_size=3,
        chunk_depth=2,
        return_map=True,
    )

    assert chunk_score == pytest.approx(full_score, abs=1e-12)
    np.testing.assert_allclose(chunk_map, full_map, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("window_size", [1, 2, 4, 3.5])
def test_ssim_rejects_invalid_window_size(window_size):
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="odd integer"):
        structural_similarity_3d(volume, volume, window_size=window_size)


def test_ssim_rejects_oversized_window_and_out_of_range_values():
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds volume"):
        structural_similarity_3d(volume, volume, window_size=7)
    invalid = volume.copy()
    invalid[0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"must lie in \[0, 1.0\]"):
        structural_similarity_3d(volume, invalid, window_size=3)


def test_ssim_rejects_roi_without_a_complete_window():
    volume = np.zeros((7, 7, 7), dtype=np.float32)
    roi = np.zeros_like(volume, dtype=bool)
    roi[3, 3, 3] = True

    with pytest.raises(ValueError, match="No complete SSIM windows"):
        structural_similarity_3d(volume, volume, window_size=3, roi_mask=roi)


def test_compute_volume_metrics_reports_primary_values_and_counts():
    truth = np.zeros((7, 7, 7), dtype=np.float32)
    prediction = np.zeros_like(truth)
    truth[3, 3, 3] = 1.0
    prediction[3, 3, 3] = 1.0
    roi = np.ones_like(truth, dtype=bool)

    metrics = compute_volume_metrics(
        truth,
        prediction,
        valid_fov_mask=roi,
        ssim_window_size=3,
        ssim_chunk_depth=1,
    )

    assert metrics["masked_dice_3d"] == 1.0
    assert metrics["ssim_3d"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["intersection_voxels"] == 1
    assert metrics["prediction_foreground_voxels"] == 1
    assert metrics["ground_truth_foreground_voxels"] == 1
    assert metrics["valid_fov_voxels"] == truth.size


def test_global_ssim_background_baseline_is_not_a_vessel_metric():
    truth = np.zeros((31, 31, 31), dtype=np.float32)
    truth[15, 15, 15] = 1.0
    empty_prediction = np.zeros_like(truth)

    dice = masked_dice_3d(truth, empty_prediction)
    global_ssim = structural_similarity_3d(
        truth, empty_prediction, window_size=7, chunk_depth=4
    )
    vessel_ssim = masked_ssim_3d(
        truth,
        empty_prediction,
        mask=truth.astype(bool),
        window_size=7,
        chunk_depth=4,
    )

    assert dice == 0.0
    assert global_ssim > 0.95
    assert vessel_ssim < 0.3
    assert global_ssim - vessel_ssim > 0.5
