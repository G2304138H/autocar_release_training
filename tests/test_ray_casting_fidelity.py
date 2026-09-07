"""Paper-fidelity contracts for sparse backward projection.

These tests deliberately cover the details that are ambiguous or absent in
the released repository: pixel-axis ordering, the EDT active band and feature
channel, two-view intersection fusion, and numerical edge cases in ray
discretisation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.geometry.projection_geometry import ProjectionGeometry
from src.modules.ray_casting import SparseBackwardProjection, SparseProjection


def _front_facing_projection() -> "torch.Tensor":
    """Camera at (0, 0, -3) looking along +z with unit focal length."""

    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )


def test_asymmetric_xy_ray_matches_stage2_projection_geometry():
    """The Stage-2 matrix's first two outputs are column then row."""

    geometry = ProjectionGeometry.from_angles(
        theta_deg=np.array([37.0]),
        phi_deg=np.array([21.0]),
        image_dim=128,
        sid_mm=900.0,
        pixel_spacing_mm=0.65,
    )
    pixel_xy = np.array([[70.25, 58.5]], dtype=np.float32)
    expected_origins, expected_directions = geometry.pixels_xy_to_rays(
        pixel_xy, 0
    )

    origins, directions = SparseBackwardProjection._camera_rays(
        torch.from_numpy(geometry.world2pix4x4[0]),
        torch.from_numpy(pixel_xy),
    )

    torch.testing.assert_close(
        origins,
        torch.from_numpy(expected_origins),
        rtol=1e-5,
        atol=1e-4,
    )
    torch.testing.assert_close(
        directions,
        torch.from_numpy(expected_directions),
        rtol=1e-5,
        atol=1e-5,
    )


def test_camera_geometry_is_not_downcast_by_autocast():
    if not hasattr(torch, "autocast"):
        pytest.skip("generic torch.autocast is unavailable in this PyTorch build")
    projection = _front_facing_projection()
    pixel_xy = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        origins, directions = SparseBackwardProjection._camera_rays(
            projection, pixel_xy
        )

    assert origins.dtype == torch.float32
    assert directions.dtype == torch.float32


def test_xy_and_rc_pixel_order_produce_the_same_sparse_ray():
    """Swapping matrix rows and declaring RC must preserve the geometry."""

    geometry = ProjectionGeometry.from_angles(
        theta_deg=np.array([17.0]),
        phi_deg=np.array([-23.0]),
        image_dim=128,
        sid_mm=900.0,
        pixel_spacing_mm=0.65,
    )
    projection_xy = torch.from_numpy(geometry.world2pix4x4[0])
    projection_rc = projection_xy[[1, 0, 2, 3]]
    # Match ``distance_maps_from_masks``: out-of-band values are truncated at
    # epsilon before continuous Eq. 7 sampling.
    distance = torch.full((128, 128), 1.0, dtype=torch.float32)
    distance[58, 70] = 0.0
    features = torch.zeros((1, 128, 128), dtype=torch.float32)
    features[0, 58, 70] = 3.0

    common = {
        "bbox_min": [-20.0, -20.0, -20.0],
        "bbox_max": [20.0, 20.0, 20.0],
        "LODs": [2.0],
        "max_pixel_distance": 1.0,
        "support_views": 1,
        "backend": "raw",
    }
    xy_module = SparseBackwardProjection(
        **common, projection_pixel_order="xy"
    )
    rc_module = SparseBackwardProjection(
        **common, projection_pixel_order="rc"
    )

    xy_coordinates, xy_features = xy_module._project_view(
        distance, features, projection_xy, active_mask=None
    )
    rc_coordinates, rc_features = rc_module._project_view(
        distance, features, projection_rc, active_mask=None
    )

    assert xy_coordinates.shape[0] > 0
    torch.testing.assert_close(xy_coordinates, rc_coordinates)
    torch.testing.assert_close(xy_features, rc_features)


def test_edt_band_is_strict_and_not_replaced_by_foreground_only_mask():
    """Equation 5 activates background pixels with EDT strictly below epsilon."""

    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [5.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = torch.tensor([[0.0, 0.75, 1.0]], dtype=torch.float32)
    features = torch.tensor([[[10.0, 20.0, 99.0]]], dtype=torch.float32)

    _, projected_features = module._project_view(
        distance,
        features,
        _front_facing_projection(),
        active_mask=None,
    )

    assert projected_features.numel() > 0
    assert torch.any(projected_features == 20.0)
    assert not torch.any(projected_features == 99.0)


def test_mask_to_distance_map_is_euclidean_and_truncated():
    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=2.1,
        support_views=1,
        backend="raw",
    )
    mask = torch.zeros((5, 5), dtype=torch.float32)
    mask[2, 2] = 1.0

    distance = module.distance_maps_from_masks(mask)

    torch.testing.assert_close(distance[2, 2], torch.tensor(0.0))
    torch.testing.assert_close(distance[2, 3], torch.tensor(1.0))
    torch.testing.assert_close(
        distance[3, 3], torch.tensor(2.0**0.5), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(distance[2, 4], torch.tensor(2.0))
    torch.testing.assert_close(distance[0, 0], torch.tensor(2.1))


def test_paper_concat_prepends_distance_for_each_view():
    """Equations 6 and 9 produce [EDT_A, f_A, EDT_B, f_B]."""

    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion="concat",
        include_distance_feature=True,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = torch.tensor([[[[0.25]], [[0.75]]]], dtype=torch.float32)
    features = torch.tensor([[[[[2.0]]], [[[4.0]]]]], dtype=torch.float32)
    projections = _front_facing_projection().reshape(1, 1, 4, 4).repeat(
        1, 2, 1, 1
    )

    result, _ = module(distance, features, projections)

    assert isinstance(result, SparseProjection)
    assert module.output_channels(input_channels=1, view_count=2) == 4
    torch.testing.assert_close(
        result.features,
        torch.tensor(
            [[0.25, 2.0, 0.75, 4.0], [0.25, 2.0, 0.75, 4.0]]
        ),
    )


def test_empty_paper_concat_has_the_configured_feature_width():
    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion="concat",
        include_distance_feature=True,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = torch.ones((1, 2, 1, 1), dtype=torch.float32)
    features = torch.ones((1, 2, 1, 1, 1), dtype=torch.float32)
    projections = _front_facing_projection().reshape(1, 1, 4, 4).repeat(
        1, 2, 1, 1
    )

    result, world = module(distance, features, projections)

    assert isinstance(result, SparseProjection)
    assert result.features.shape == (0, 4)
    assert result.coordinates.shape == (0, 4)
    assert world.shape == (0, 3)


def test_batch_indices_and_world_coordinates_remain_aligned():
    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        support_views=2,
        fusion="mean",
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = torch.zeros((2, 2, 1, 1), dtype=torch.float32)
    features = torch.tensor([2.0, 4.0, 6.0, 8.0]).reshape(2, 2, 1, 1, 1)
    projections = _front_facing_projection().reshape(1, 1, 4, 4).repeat(
        2, 2, 1, 1
    )

    result, world = module(distance, features, projections)

    assert isinstance(result, SparseProjection)
    assert result.coordinates[:, 0].tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(
        result.features,
        torch.tensor([[3.0], [3.0], [7.0], [7.0]]),
    )
    torch.testing.assert_close(world[:2], world[2:])
    torch.testing.assert_close(world, result.world_coordinates)


def test_short_ray_segment_samples_inside_instead_of_at_far_boundary():
    """A segment shorter than half a voxel must still activate its voxel."""

    module = SparseBackwardProjection(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0],
        support_views=1,
        backend="raw",
    )
    coordinates, features = module._sample_ray_chunk(
        origins=torch.tensor([[0.8, 0.5, 0.5]], dtype=torch.float32),
        directions=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        near=torch.tensor([0.0], dtype=torch.float32),
        far=torch.tensor([0.2], dtype=torch.float32),
        ray_features=torch.tensor([[7.0]], dtype=torch.float32),
    )

    assert coordinates.tolist() == [[0, 0, 0]]
    torch.testing.assert_close(features, torch.tensor([[7.0]]))


def test_geometry_and_world_coordinates_stay_float32_with_half_features():
    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        support_views=2,
        fusion="mean",
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = torch.zeros((1, 2, 1, 1), dtype=torch.float32)
    features = torch.ones((1, 2, 1, 1, 1), dtype=torch.float16)
    projections = _front_facing_projection().reshape(1, 1, 4, 4).repeat(
        1, 2, 1, 1
    )

    result, world = module(distance, features, projections)

    assert result.features.dtype == torch.float16
    assert result.world_coordinates.dtype == torch.float32
    assert world.dtype == torch.float32


def test_half_precision_duplicate_reduction_does_not_overflow():
    """A global fp16 cumulative sum must not corrupt otherwise small means."""

    module = SparseBackwardProjection(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0],
        support_views=1,
        backend="raw",
    )
    sample_count = 70_000
    coordinates = torch.zeros((sample_count, 3), dtype=torch.long)
    features = torch.ones((sample_count, 1), dtype=torch.float16)

    reduced_coordinates, reduced_features = module._deduplicate_view(
        coordinates, features
    )

    assert reduced_coordinates.tolist() == [[0, 0, 0]]
    assert torch.isfinite(reduced_features).all()
    torch.testing.assert_close(
        reduced_features.float(), torch.ones((1, 1), dtype=torch.float32)
    )
