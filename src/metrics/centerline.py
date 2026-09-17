"""Physical errors between independently extracted vascular centerline graphs."""

from __future__ import annotations

from typing import Any

import numpy as np


def _points(values: np.ndarray, *, label: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{label} must have shape [N,3], got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} must contain only finite values.")
    return points


def _radii(values: np.ndarray, *, count: int, label: str) -> np.ndarray:
    radii = np.asarray(values, dtype=np.float64)
    if radii.shape != (count,):
        raise ValueError(f"{label} must have shape ({count},), got {radii.shape}.")
    if not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ValueError(f"{label} must contain finite non-negative values.")
    return radii


def centerline_radius_errors(
    prediction_xyz_mm: np.ndarray,
    prediction_radius_mm: np.ndarray,
    ground_truth_xyz_mm: np.ndarray,
    ground_truth_radius_mm: np.ndarray,
) -> dict[str, Any]:
    """Compare two unpaired centerline-radius point sets in physical space.

    Each source point is paired with its nearest point in the other graph. The
    primary centerline error is the arithmetic mean of the prediction-to-GT
    and GT-to-prediction mean distances (symmetric mean Chamfer distance). The
    primary radius MAE uses the same spatial correspondences and symmetric
    averaging. Directional values are retained to expose missing branches and
    spurious branches separately.

    Errors are undefined when either graph is empty. In that case the function
    returns ``None`` for every error and records both node counts so aggregate
    reports cannot silently treat an empty prediction as a perfect result.
    """

    prediction_xyz = _points(prediction_xyz_mm, label="prediction_xyz_mm")
    ground_truth_xyz = _points(ground_truth_xyz_mm, label="ground_truth_xyz_mm")
    prediction_radius = _radii(
        prediction_radius_mm,
        count=len(prediction_xyz),
        label="prediction_radius_mm",
    )
    ground_truth_radius = _radii(
        ground_truth_radius_mm,
        count=len(ground_truth_xyz),
        label="ground_truth_radius_mm",
    )
    result: dict[str, Any] = {
        "valid": bool(len(prediction_xyz) and len(ground_truth_xyz)),
        "prediction_nodes": int(len(prediction_xyz)),
        "ground_truth_nodes": int(len(ground_truth_xyz)),
        "centerline_pred_to_gt_mean_error_mm": None,
        "centerline_gt_to_pred_mean_error_mm": None,
        "centerline_mean_error_mm": None,
        "radius_pred_to_gt_mae_mm": None,
        "radius_gt_to_pred_mae_mm": None,
        "radius_mae_mm": None,
    }
    if not result["valid"]:
        return result

    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Centerline and radius errors require scipy. Install the maintained "
            "environment requirements before running evaluation."
        ) from error

    pred_to_gt_distance, pred_to_gt_index = cKDTree(ground_truth_xyz).query(
        prediction_xyz,
        k=1,
        workers=-1,
    )
    gt_to_pred_distance, gt_to_pred_index = cKDTree(prediction_xyz).query(
        ground_truth_xyz,
        k=1,
        workers=-1,
    )
    pred_to_gt_distance = np.asarray(pred_to_gt_distance, dtype=np.float64)
    gt_to_pred_distance = np.asarray(gt_to_pred_distance, dtype=np.float64)
    pred_to_gt_index = np.asarray(pred_to_gt_index, dtype=np.int64)
    gt_to_pred_index = np.asarray(gt_to_pred_index, dtype=np.int64)

    centerline_pred_to_gt = float(np.mean(pred_to_gt_distance))
    centerline_gt_to_pred = float(np.mean(gt_to_pred_distance))
    radius_pred_to_gt = float(
        np.mean(np.abs(prediction_radius - ground_truth_radius[pred_to_gt_index]))
    )
    radius_gt_to_pred = float(
        np.mean(np.abs(ground_truth_radius - prediction_radius[gt_to_pred_index]))
    )
    result.update(
        {
            "centerline_pred_to_gt_mean_error_mm": centerline_pred_to_gt,
            "centerline_gt_to_pred_mean_error_mm": centerline_gt_to_pred,
            "centerline_mean_error_mm": 0.5
            * (centerline_pred_to_gt + centerline_gt_to_pred),
            "radius_pred_to_gt_mae_mm": radius_pred_to_gt,
            "radius_gt_to_pred_mae_mm": radius_gt_to_pred,
            "radius_mae_mm": 0.5 * (radius_pred_to_gt + radius_gt_to_pred),
        }
    )
    return result


__all__ = ["centerline_radius_errors"]
