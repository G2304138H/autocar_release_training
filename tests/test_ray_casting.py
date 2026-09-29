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


@pytest.mark.parametrize("view_features", [(2.0,), (2.0, 4.0), (2.0,) * 7])
def test_adaptive_support_keeps_fixed_width_features(view_features):
    module = SparseBackwardProjection(
        [-1, -1, -1], [1, 1, 1], [1.0],
        backend="raw", fusion="mean", support_views=2,
        adaptive_support_views=True, candidate_mode="voxel_grid",
    )
    sdf, features, projections, _ = _inputs(view_features)
    result, _ = module(sdf, features, projections)
    reference_sdf, reference_features, reference_projections, _ = _inputs(
        (sum(view_features) / len(view_features),)
    )
    reference, _ = module(reference_sdf, reference_features, reference_projections)
    assert result.features.shape[0] > 0
    assert result.features.shape[1] == 1
    torch.testing.assert_close(result.coordinates, reference.coordinates)
    torch.testing.assert_close(result.features, reference.features)


@pytest.mark.parametrize("candidate_mode", ["ray", "voxel_grid"])
def test_all_view_support_intersects_seven_views_and_accepts_one(candidate_mode):
    module = SparseBackwardProjection(
        [-1, -1, -1], [1, 1, 1], [1.0], backend="raw",
        fusion="mean", support_views="all", candidate_mode=candidate_mode,
    )
    sdf, features, projections, _ = _inputs((2.0,))
    single, _ = module(sdf, features, projections)
    assert single.features.shape[0] > 0

    sdf, features, projections, _ = _inputs((2.0,) * 7)
    complete, _ = module(sdf, features, projections)
    torch.testing.assert_close(complete.coordinates, single.coordinates)
    torch.testing.assert_close(complete.features, single.features)

    # Six supporting views cannot keep voxels when the seventh rejects them.
    sdf[:, -1] = 1.0
    intersection, _ = module(sdf, features, projections)
    assert intersection.features.shape[0] == 0
    module.support_views = 2
    pairwise_union, _ = module(sdf, features, projections)
    assert pairwise_union.features.shape[0] > 0
