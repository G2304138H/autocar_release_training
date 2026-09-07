"""Tests for the released-code replacement sparse backward projection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.modules.ray_casting import SparseBackwardProjection, SparseProjection


def _front_facing_projection() -> "torch.Tensor":
    """Camera at (0, 0, -3) looking along +z with unit focal length."""

    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ]
    )


def _inputs(view_features=(2.0, 4.0)):
    view_count = len(view_features)
    sdf = torch.zeros((1, view_count, 1, 1), dtype=torch.float32)
    features = torch.tensor(view_features, dtype=torch.float32).reshape(
        1, view_count, 1, 1, 1
    )
    projections = _front_facing_projection().reshape(1, 1, 4, 4).repeat(
        1, view_count, 1, 1
    )
    masks = torch.ones_like(sdf)
    return sdf, features, projections, masks


def test_camera_ray_recovers_camera_center_and_axis():
    projection = _front_facing_projection()
    origin, direction = SparseBackwardProjection._camera_rays(
        projection, torch.tensor([[0.0, 0.0]])
    )
    torch.testing.assert_close(origin, torch.tensor([[0.0, 0.0, -3.0]]))
    torch.testing.assert_close(direction, torch.tensor([[0.0, 0.0, 1.0]]))


def test_two_view_mean_fusion_and_world_coordinate_convention():
    module = SparseBackwardProjection(
        [-1, -1, -1],
        [1, 1, 1],
        [1.0],
        backend="raw",
        fusion="mean",
    )
    result, world = module(*_inputs()[:3], active_masks=_inputs()[3])

    assert isinstance(result, SparseProjection)
    assert result.coordinates.tolist() == [[0, 1, 1, 0], [0, 1, 1, 1]]
    torch.testing.assert_close(result.features, torch.tensor([[3.0], [3.0]]))
    torch.testing.assert_close(
        world,
        torch.tensor([[0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]),
    )


def test_concat_fusion_preserves_view_order():
    module = SparseBackwardProjection(
        [-1, -1, -1],
        [1, 1, 1],
        [1.0],
        backend="raw",
        fusion="concat",
    )
    sdf, features, projections, masks = _inputs()
    result, _ = module(sdf, features, projections, active_masks=masks)
    torch.testing.assert_close(
        result.features, torch.tensor([[2.0, 4.0], [2.0, 4.0]])
    )


def test_empty_masks_produce_valid_empty_projection():
    module = SparseBackwardProjection(
        [-1, -1, -1], [1, 1, 1], [1.0], backend="raw"
    )
    sdf, features, projections, masks = _inputs()
    result, world = module(
        sdf, features, projections, active_masks=torch.zeros_like(masks)
    )
    assert result.coordinates.shape == (0, 4)
    assert result.features.shape == (0, 1)
    assert world.shape == (0, 3)


def test_feature_reductions_remain_differentiable():
    module = SparseBackwardProjection(
        [-1, -1, -1], [1, 1, 1], [1.0], backend="raw"
    )
    sdf, features, projections, masks = _inputs()
    features.requires_grad_(True)
    result, _ = module(sdf, features, projections, active_masks=masks)
    result.features.sum().backward()
    assert features.grad is not None
    assert torch.all(features.grad > 0)


def test_support_count_cannot_exceed_input_views():
    module = SparseBackwardProjection(
        [-1, -1, -1],
        [1, 1, 1],
        [1.0],
        backend="raw",
        support_views=3,
    )
    sdf, features, projections, masks = _inputs()
    with pytest.raises(ValueError, match="exceeds view_count"):
        module(sdf, features, projections, active_masks=masks)

