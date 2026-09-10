"""Metrics for physically aligned three-dimensional vascular volumes.

The Dice and SSIM implementations are NumPy-only so validation can run without
a CUDA environment. Hard clDice additionally uses scikit-image morphological
thinning. Dice is evaluated only inside an optional valid-FOV mask. For SSIM,
an ROI is stricter than a center-selection mask: only windows fully contained
in the ROI are eligible, so values outside the physical field of view cannot
leak into the result.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np


def _volume(volume: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(volume)
    if result.ndim != 3:
        raise ValueError(f"{name} must be a 3D ZYX array, got shape {result.shape}.")
    if not np.issubdtype(result.dtype, np.bool_) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _matching_volumes(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth = _volume(ground_truth, "ground_truth")
    prediction = _volume(prediction, "prediction")
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            "ground_truth and prediction must have equal shapes, got "
            f"{ground_truth.shape} and {prediction.shape}."
        )
    return ground_truth, prediction


def _mask(mask: Optional[np.ndarray], shape: Tuple[int, ...], name: str) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=np.bool_)
    result = np.asarray(mask, dtype=np.bool_)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} does not match {shape}.")
    return result


def masked_dice_3d(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    ground_truth_threshold: float = 0.5,
    prediction_threshold: float = 0.5,
) -> float:
    """Compute binary Dice inside ``mask``.

    Both inputs may be binary arrays or continuous values.  Values greater
    than or equal to their threshold are foreground.  Empty/empty is exactly
    1.0; one-empty is exactly 0.0.  An empty evaluation mask is rejected
    because it contains no evidence for either outcome.
    """

    ground_truth, prediction = _matching_volumes(ground_truth, prediction)
    roi = _mask(mask, ground_truth.shape, "mask")
    if not np.any(roi):
        raise ValueError("mask contains no evaluation voxels.")
    if not math.isfinite(float(ground_truth_threshold)) or not math.isfinite(
        float(prediction_threshold)
    ):
        raise ValueError("Dice thresholds must be finite.")

    truth_foreground = (ground_truth >= float(ground_truth_threshold)) & roi
    prediction_foreground = (prediction >= float(prediction_threshold)) & roi
    truth_count = int(np.count_nonzero(truth_foreground))
    prediction_count = int(np.count_nonzero(prediction_foreground))
    denominator = truth_count + prediction_count
    if denominator == 0:
        return 1.0
    intersection = int(
        np.count_nonzero(truth_foreground & prediction_foreground)
    )
    return float(2.0 * intersection / denominator)


def _morphological_skeleton_3d(mask: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Hard 3D clDice requires scikit-image for morphological thinning. "
            "Install the maintained environment requirements before evaluation."
        ) from error
    padded = np.pad(np.asarray(mask, dtype=np.bool_), 1, mode="constant")
    skeleton = skeletonize(padded, method="lee")
    return np.asarray(skeleton[1:-1, 1:-1, 1:-1], dtype=np.bool_)


def hard_cldice_3d(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    ground_truth_skeleton: Optional[np.ndarray] = None,
    prediction_skeleton: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int]]:
    """Compute the hard morphological 3D clDice score and its loss.

    This implements the definition from the AutoCAR paper: topology precision
    is the fraction of the predicted skeleton inside the ground-truth mask,
    topology sensitivity is the fraction of the ground-truth skeleton inside
    the predicted mask, and clDice is their harmonic mean. Skeletons can be
    supplied by graph extraction so evaluation does not thin either volume
    twice. Empty/empty is scored as one and either one-empty case as zero.
    """

    ground_truth, prediction = _matching_volumes(ground_truth, prediction)
    truth_mask = np.asarray(ground_truth >= 0.5, dtype=np.bool_)
    prediction_mask = np.asarray(prediction >= 0.5, dtype=np.bool_)
    truth_skeleton = (
        _morphological_skeleton_3d(truth_mask)
        if ground_truth_skeleton is None
        else np.asarray(ground_truth_skeleton, dtype=np.bool_)
    )
    predicted_skeleton = (
        _morphological_skeleton_3d(prediction_mask)
        if prediction_skeleton is None
        else np.asarray(prediction_skeleton, dtype=np.bool_)
    )
    for name, skeleton, mask in (
        ("ground_truth_skeleton", truth_skeleton, truth_mask),
        ("prediction_skeleton", predicted_skeleton, prediction_mask),
    ):
        if skeleton.shape != truth_mask.shape:
            raise ValueError(
                f"{name} shape {skeleton.shape} does not match {truth_mask.shape}."
            )
        if np.any(skeleton & ~mask):
            raise ValueError(f"{name} contains voxels outside its vessel mask.")

    predicted_count = int(np.count_nonzero(predicted_skeleton))
    truth_count = int(np.count_nonzero(truth_skeleton))
    predicted_in_truth = int(
        np.count_nonzero(predicted_skeleton & truth_mask)
    )
    truth_in_prediction = int(
        np.count_nonzero(truth_skeleton & prediction_mask)
    )

    if predicted_count:
        topology_precision = predicted_in_truth / predicted_count
    else:
        topology_precision = 1.0 if not np.any(truth_mask) else 0.0
    if truth_count:
        topology_sensitivity = truth_in_prediction / truth_count
    else:
        topology_sensitivity = 1.0 if not np.any(prediction_mask) else 0.0
    denominator = topology_precision + topology_sensitivity
    score = (
        0.0
        if denominator == 0.0
        else 2.0 * topology_precision * topology_sensitivity / denominator
    )
    return {
        "cldice_3d": float(score),
        "cldice_loss_3d": float(1.0 - score),
        "topology_precision": float(topology_precision),
        "topology_sensitivity": float(topology_sensitivity),
        "prediction_centerline_voxels": predicted_count,
        "ground_truth_centerline_voxels": truth_count,
        "prediction_centerline_in_ground_truth_voxels": predicted_in_truth,
        "ground_truth_centerline_in_prediction_voxels": truth_in_prediction,
    }


def _box_sum_valid(volume: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float64)
    integral = np.pad(values, ((1, 0), (1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    width = int(window_size)
    return (
        integral[width:, width:, width:]
        - integral[:-width, width:, width:]
        - integral[width:, :-width, width:]
        - integral[width:, width:, :-width]
        + integral[:-width, :-width, width:]
        + integral[:-width, width:, :-width]
        + integral[width:, :-width, :-width]
        - integral[:-width, :-width, :-width]
    )


def _box_mean_valid(volume: np.ndarray, window_size: int) -> np.ndarray:
    return _box_sum_valid(volume, window_size) / float(window_size ** 3)


def _ssim_map_chunk(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    window_size: int,
    data_range: float,
) -> np.ndarray:
    # Promote only the current depth slab.  A 400^3 float64 copy is roughly
    # 512 MiB, so converting both complete inputs before chunking would defeat
    # the purpose of the bounded-memory implementation.
    truth = np.clip(
        np.asarray(ground_truth, dtype=np.float64), 0.0, data_range
    )
    predicted = np.clip(
        np.asarray(prediction, dtype=np.float64), 0.0, data_range
    )
    mean_truth = _box_mean_valid(truth, window_size)
    mean_prediction = _box_mean_valid(predicted, window_size)
    sample_scale = float(window_size ** 3) / float(window_size ** 3 - 1)

    variance_truth = sample_scale * (
        _box_mean_valid(truth * truth, window_size)
        - mean_truth * mean_truth
    )
    variance_prediction = sample_scale * (
        _box_mean_valid(predicted * predicted, window_size)
        - mean_prediction * mean_prediction
    )
    covariance = sample_scale * (
        _box_mean_valid(truth * predicted, window_size)
        - mean_truth * mean_prediction
    )
    # Cancellation can produce tiny negative variances for constant windows.
    variance_truth = np.maximum(variance_truth, 0.0)
    variance_prediction = np.maximum(variance_prediction, 0.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mean_truth * mean_prediction + c1) * (
        2.0 * covariance + c2
    )
    denominator = (
        mean_truth * mean_truth + mean_prediction * mean_prediction + c1
    ) * (variance_truth + variance_prediction + c2)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator != 0.0,
    )


def structural_similarity_3d(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
    roi_mask: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    chunk_depth: Optional[int] = 8,
    return_map: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Compute uniform-window 3D SSIM for continuous volumes.

    The implementation uses the standard ``K1=0.01`` and ``K2=0.03``
    constants with sample covariance.  Inputs must lie in ``[0, data_range]``.

    ``roi_mask`` defines valid physical data: only windows entirely contained
    in this ROI contribute.  ``mask`` is an optional secondary selector on
    valid window centers (for example, vessel-containing centers).  This
    distinction prevents prediction values outside the valid source FOV from
    influencing the score.  ``chunk_depth`` bounds peak memory without
    changing the calculation; ``None`` evaluates all output planes together.

    When ``return_map`` is true, the unmasked valid-window SSIM map is returned
    with shape ``shape - window_size + 1`` in each dimension.
    """

    ground_truth, prediction = _matching_volumes(ground_truth, prediction)
    if not math.isfinite(float(data_range)) or float(data_range) <= 0.0:
        raise ValueError(f"data_range must be positive and finite, got {data_range}.")
    if isinstance(window_size, bool) or int(window_size) != window_size:
        raise ValueError("window_size must be an odd integer >= 3.")
    window_size = int(window_size)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3.")
    if any(size < window_size for size in ground_truth.shape):
        raise ValueError(
            f"window_size {window_size} exceeds volume shape {ground_truth.shape}."
        )
    if chunk_depth is None:
        chunk_depth = ground_truth.shape[0] - window_size + 1
    if isinstance(chunk_depth, bool) or int(chunk_depth) != chunk_depth:
        raise ValueError("chunk_depth must be a positive integer or None.")
    chunk_depth = int(chunk_depth)
    if chunk_depth <= 0:
        raise ValueError("chunk_depth must be a positive integer or None.")

    data_range = float(data_range)
    tolerance = max(1e-12, data_range * 1e-7)
    for name, values in (("ground_truth", ground_truth), ("prediction", prediction)):
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum < -tolerance or maximum > data_range + tolerance:
            raise ValueError(
                f"{name} values must lie in [0, {data_range}], got "
                f"range [{minimum}, {maximum}]."
            )

    roi = None if roi_mask is None else _mask(roi_mask, ground_truth.shape, "roi_mask")
    center_mask = None if mask is None else _mask(mask, ground_truth.shape, "mask")
    if roi is not None and not np.any(roi):
        raise ValueError("roi_mask contains no valid voxels.")
    if center_mask is not None and not np.any(center_mask):
        raise ValueError("mask contains no selected voxels.")

    output_shape = tuple(size - window_size + 1 for size in ground_truth.shape)
    ssim_map = np.empty(output_shape, dtype=np.float32) if return_map else None
    radius = window_size // 2
    total = 0.0
    selected_count = 0
    window_voxels = window_size ** 3

    for output_start in range(0, output_shape[0], chunk_depth):
        output_stop = min(output_start + chunk_depth, output_shape[0])
        input_stop = output_stop + window_size - 1
        chunk_map = _ssim_map_chunk(
            ground_truth[output_start:input_stop],
            prediction[output_start:input_stop],
            window_size=window_size,
            data_range=data_range,
        )
        if ssim_map is not None:
            ssim_map[output_start:output_stop] = chunk_map.astype(
                np.float32, copy=False
            )

        selected = np.ones(chunk_map.shape, dtype=np.bool_)
        if roi is not None:
            roi_count = _box_sum_valid(
                roi[output_start:input_stop], window_size
            )
            selected &= roi_count == window_voxels
        if center_mask is not None:
            selected &= center_mask[
                output_start + radius : output_stop + radius,
                radius : ground_truth.shape[1] - radius,
                radius : ground_truth.shape[2] - radius,
            ]
        count = int(np.count_nonzero(selected))
        if count:
            total += float(np.sum(chunk_map[selected], dtype=np.float64))
            selected_count += count

    if selected_count == 0:
        raise ValueError(
            "No complete SSIM windows are selected by roi_mask and mask."
        )
    score = float(total / selected_count)
    if ssim_map is None:
        return score
    return score, ssim_map


def masked_ssim_3d(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    data_range: float = 1.0,
    window_size: int = 7,
    chunk_depth: Optional[int] = 8,
) -> float:
    """Convenience wrapper selecting SSIM window centers with ``mask``."""

    return float(
        structural_similarity_3d(
            ground_truth,
            prediction,
            data_range=data_range,
            window_size=window_size,
            roi_mask=roi_mask,
            mask=mask,
            chunk_depth=chunk_depth,
        )
    )


def compute_volume_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    valid_fov_mask: Optional[np.ndarray] = None,
    ground_truth_threshold: float = 0.5,
    prediction_threshold: float = 0.5,
    ssim_window_size: int = 7,
    ssim_chunk_depth: Optional[int] = 8,
) -> Dict[str, Union[float, int]]:
    """Return the two primary reconstruction metrics and audit counts."""

    ground_truth, prediction = _matching_volumes(ground_truth, prediction)
    roi = _mask(valid_fov_mask, ground_truth.shape, "valid_fov_mask")
    if not np.any(roi):
        raise ValueError("valid_fov_mask contains no evaluation voxels.")
    truth_binary = (ground_truth >= float(ground_truth_threshold)) & roi
    prediction_binary = (prediction >= float(prediction_threshold)) & roi
    intersection = int(np.count_nonzero(truth_binary & prediction_binary))
    return {
        "masked_dice_3d": masked_dice_3d(
            ground_truth,
            prediction,
            mask=roi,
            ground_truth_threshold=ground_truth_threshold,
            prediction_threshold=prediction_threshold,
        ),
        "ssim_3d": float(
            structural_similarity_3d(
                ground_truth,
                prediction,
                data_range=1.0,
                window_size=ssim_window_size,
                roi_mask=roi,
                chunk_depth=ssim_chunk_depth,
            )
        ),
        "intersection_voxels": intersection,
        "prediction_foreground_voxels": int(
            np.count_nonzero(prediction_binary)
        ),
        "ground_truth_foreground_voxels": int(np.count_nonzero(truth_binary)),
        "valid_fov_voxels": int(np.count_nonzero(roi)),
    }
