"""Focused tests for Methods Eq. 7 voxel-centre reprojection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.modules.ray_casting import SparseBackwardProjection


def _front_projection(focal_x: float = 4.0, focal_y: float = 4.0):
    """Camera at (0, 0, -3) looking along +z."""

    return torch.tensor(
        [
            [focal_x, 0.0, 0.0, 0.0],
            [0.0, focal_y, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
        dtype=torch.float32,
    )


def _ramp_feature_map():
    rows = torch.arange(3, dtype=torch.float32)[:, None] * 10.0
    columns = torch.arange(3, dtype=torch.float32)[None, :]
    return (rows + columns)[None]


def _axis_ray_projection_inputs():
    distance = torch.zeros((3, 3), dtype=torch.float32)
    feature = _ramp_feature_map()
    active_mask = torch.zeros_like(distance)
    active_mask[0, 0] = 1.0
    return distance, feature, active_mask


def test_view_features_come_from_voxel_centre_reprojection():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance, feature, active_mask = _axis_ray_projection_inputs()

    coordinates, values = module._project_view(
        distance,
        feature,
        _front_projection(),
        active_mask=active_mask,
    )

    # The initiating ray uses pixel (0, 0), whose feature is zero. Its two
    # quantised voxel centres instead project to (0.8, 0.8) and (4/7, 4/7).
    assert coordinates.tolist() == [[0, 0, 0], [0, 0, 1]]
    torch.testing.assert_close(
        values[:, 0],
        torch.tensor([8.8, 44.0 / 7.0]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.all(values[:, 0] != feature[0, 0, 0])


def test_voxel_centre_grid_sampling_backpropagates_to_feature_map():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance, feature, active_mask = _axis_ray_projection_inputs()
    feature.requires_grad_(True)

    _, values = module._project_view(
        distance,
        feature,
        _front_projection(),
        active_mask=active_mask,
    )
    values.sum().backward()

    assert feature.grad is not None
    # Both voxel centres lie between four pixels, so a non-initiating pixel
    # must receive gradient. Initiating-ray feature copying cannot satisfy it.
    assert feature.grad[0, 1, 1] > 0


def test_xy_and_rc_reprojection_sample_the_same_image_location():
    common = {
        "bbox_min": [0.0, 0.0, -1.0],
        "bbox_max": [1.0, 1.0, 0.0],
        "LODs": [1.0],
        "max_pixel_distance": 1.0,
        "support_views": 1,
        "backend": "raw",
    }
    module_xy = SparseBackwardProjection(
        **common, projection_pixel_order="xy"
    )
    module_rc = SparseBackwardProjection(
        **common, projection_pixel_order="rc"
    )
    projection_xy = _front_projection(focal_x=4.0, focal_y=2.0)
    projection_rc = projection_xy[[1, 0, 2, 3]]
    coordinates = torch.tensor([[0, 0, 0]], dtype=torch.long)
    distance = torch.zeros((3, 3), dtype=torch.float32)
    feature = _ramp_feature_map()

    xy_coordinates, xy_values = module_xy._reproject_voxel_features(
        coordinates, distance, feature, projection_xy
    )
    rc_coordinates, rc_values = module_rc._reproject_voxel_features(
        coordinates, distance, feature, projection_rc
    )

    torch.testing.assert_close(xy_coordinates, rc_coordinates)
    torch.testing.assert_close(xy_values, rc_values)
    # Centre (0.5,0.5,-0.5) projects to XY (0.8,0.4), giving 4.8
    # on the asymmetric row*10+column feature map.
    torch.testing.assert_close(xy_values, torch.tensor([[4.8]]))


def test_reprojection_filters_outside_and_distance_threshold_voxels():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [3.0, 1.0, 0.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        projection_pixel_order="xy",
        include_distance_feature=True,
    )
    coordinates = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.long
    )
    distance = torch.ones((2, 2), dtype=torch.float32)
    distance[0, 0] = 0.0
    feature = torch.tensor([[[0.0, 1.0], [10.0, 11.0]]])

    kept_coordinates, values = module._reproject_voxel_features(
        coordinates,
        distance,
        feature,
        _front_projection(focal_x=2.0, focal_y=2.0),
    )

    # The first centre projects to (0.4,0.4): nearest EDT samples pixel (0,0)
    # while the learned feature is bilinear (4.4). Index 1 rounds to inactive
    # column 1, and index 2 lies beyond the two-column detector.
    assert kept_coordinates.tolist() == [[0, 0, 0]]
    torch.testing.assert_close(values, torch.tensor([[0.0, 4.4]]))

    threshold_coordinates, threshold_values = (
        module._reproject_voxel_features(
            coordinates[:1],
            torch.ones_like(distance),
            feature,
            _front_projection(focal_x=2.0, focal_y=2.0),
        )
    )
    assert threshold_coordinates.shape == (0, 3)
    assert threshold_values.shape == (0, 2)


def test_bilinear_distance_mode_retains_a_non_capped_edt_halo():
    nearest = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        distance_sampling="nearest",
    )
    bilinear = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        backend="raw",
        distance_sampling="bilinear",
    )
    mask = torch.zeros((5, 5), dtype=torch.float32)
    mask[2, 2] = 1.0

    nearest_distance = nearest.distance_maps_from_masks(mask)
    bilinear_distance = bilinear.distance_maps_from_masks(mask)

    torch.testing.assert_close(nearest_distance[2, 4], torch.tensor(1.0))
    torch.testing.assert_close(bilinear_distance[2, 4], torch.tensor(2.0))


def test_nearest_distance_uses_generator_ties_to_even_rounding():
    module = SparseBackwardProjection(
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0],
        support_views=1,
        backend="raw",
        distance_sampling="nearest",
    )
    distance = torch.tensor([[10.0, 20.0, 30.0]])
    pixels_xy = torch.tensor([[0.5, 0.0], [1.5, 0.0]])

    sampled = module._sample_distance(distance, pixels_xy)

    # torch.round and the generator's numpy.rint both map ties to even.
    torch.testing.assert_close(sampled, torch.tensor([10.0, 30.0]))


def _truncated_corner_distance(active_column: int):
    distance = torch.ones((2, 2), dtype=torch.float32)
    distance[0, active_column] = 0.0
    return distance


def test_voxel_grid_matches_direct_exhaustive_two_view_projection():
    """Chunked enumeration must equal an unchunked direct Eq. 7 reference."""

    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [4.0, 2.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion="concat",
        include_distance_feature=True,
        candidate_mode="voxel_grid",
        voxel_chunk_size=3,
        backend="raw",
        projection_pixel_order="xy",
    )
    distances = torch.stack(
        [_truncated_corner_distance(0), _truncated_corner_distance(0)]
    )
    feature0 = torch.tensor([[[0.0, 1.0], [10.0, 11.0]]])
    feature1 = feature0 + 100.0
    feature_maps = torch.stack([feature0, feature1])
    projections = torch.stack(
        [
            _front_projection(focal_x=1.0, focal_y=1.0),
            _front_projection(focal_x=2.0, focal_y=1.0),
        ]
    )

    size_x, size_y, size_z = module.spatial_shape_xyz
    exhaustive_coordinates = module._coordinates_from_linear_range(
        0, size_x * size_y * size_z, distances.device
    )
    expected_coordinates, expected0, _ = (
        module._reproject_voxel_features_indexed(
            exhaustive_coordinates,
            distances[0],
            feature_maps[0],
            projections[0],
        )
    )
    expected_coordinates, expected1, keep1 = (
        module._reproject_voxel_features_indexed(
            expected_coordinates,
            distances[1],
            feature_maps[1],
            projections[1],
        )
    )
    expected_features = torch.cat([expected0[keep1], expected1], dim=1)

    # Two batches also verify batch-column construction and that view blocks
    # survive filtering without being shifted onto neighbouring coordinates.
    batched_features = torch.stack([feature_maps, feature_maps + 1000.0])
    result, _ = module(
        distances[None].repeat(2, 1, 1, 1),
        batched_features,
        projections[None].repeat(2, 1, 1, 1),
    )

    count = expected_coordinates.shape[0]
    assert count > 0
    assert result.coordinates[:, 0].tolist() == [0] * count + [1] * count
    torch.testing.assert_close(
        result.coordinates[:count, 1:].long(), expected_coordinates
    )
    torch.testing.assert_close(
        result.coordinates[count:, 1:].long(), expected_coordinates
    )
    torch.testing.assert_close(result.features[:count], expected_features)
    # EDT channels are unchanged while each encoder block gains 1000.
    expected_second = expected_features.clone()
    expected_second[:, 1] += 1000.0
    expected_second[:, 3] += 1000.0
    torch.testing.assert_close(result.features[count:], expected_second)


def test_all_view_grid_samples_encoder_only_after_intersection(monkeypatch):
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [4.0, 1.0, 0.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion="concat",
        candidate_mode="voxel_grid",
        backend="raw",
        projection_pixel_order="xy",
    )
    distances = torch.tensor(
        [[[0.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
    )
    features = torch.arange(4, dtype=torch.float32).reshape(2, 1, 1, 2)
    projections = _front_projection(focal_x=1.0, focal_y=1.0)[
        None
    ].repeat(2, 1, 1)
    coordinates = module._coordinates_from_linear_range(
        0, 4, distances.device
    )

    first_coordinates, _, _, _ = module._gate_voxel_coordinates(
        coordinates, distances[0], projections[0]
    )
    final_coordinates, _, _, _ = module._gate_voxel_coordinates(
        first_coordinates, distances[1], projections[1]
    )
    assert first_coordinates.shape[0] > final_coordinates.shape[0] > 0

    sampled_sizes = []
    original_bilinear_sample = module._bilinear_sample

    def recording_bilinear_sample(image, pixels_xy):
        sampled_sizes.append(pixels_xy.shape[0])
        return original_bilinear_sample(image, pixels_xy)

    monkeypatch.setattr(
        module, "_bilinear_sample", recording_bilinear_sample
    )
    kept_coordinates, _ = module._project_grid_chunk_all_views(
        coordinates, distances, features, projections
    )

    torch.testing.assert_close(kept_coordinates, final_coordinates)
    assert sampled_sizes == [final_coordinates.shape[0]] * 2


@pytest.mark.parametrize("fusion", ["concat", "mean"])
def test_all_view_grid_matches_sequential_outputs_and_gradients(fusion):
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [4.0, 1.0, 0.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion=fusion,
        include_distance_feature=True,
        candidate_mode="voxel_grid",
        backend="raw",
        projection_pixel_order="xy",
    )
    distances = torch.tensor(
        [[[0.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32
    )
    projections = _front_projection(focal_x=1.0, focal_y=1.0)[
        None
    ].repeat(2, 1, 1)
    coordinates = module._coordinates_from_linear_range(
        0, 4, distances.device
    )
    optimized_features = torch.tensor(
        [
            [[[0.0, 2.0]], [[1.0, 3.0]]],
            [[[4.0, 6.0]], [[5.0, 7.0]]],
        ],
        requires_grad=True,
    )
    reference_features = optimized_features.detach().clone().requires_grad_(
        True
    )

    optimized_coordinates, optimized_values = (
        module._project_grid_chunk_all_views(
            coordinates,
            distances,
            optimized_features,
            projections,
        )
    )

    reference_coordinates = coordinates
    reference_blocks = []
    for view_index in range(2):
        reference_coordinates, view_values, kept_indices = (
            module._reproject_voxel_features_indexed(
                reference_coordinates,
                distances[view_index],
                reference_features[view_index],
                projections[view_index],
            )
        )
        reference_blocks = [
            block[kept_indices] for block in reference_blocks
        ]
        reference_blocks.append(view_values)
    if fusion == "concat":
        reference_values = torch.cat(reference_blocks, dim=1)
    else:
        reference_values = torch.stack(reference_blocks, dim=0).mean(dim=0)

    torch.testing.assert_close(
        optimized_coordinates, reference_coordinates
    )
    torch.testing.assert_close(optimized_values, reference_values)

    weights = torch.arange(
        1,
        optimized_values.numel() + 1,
        dtype=optimized_values.dtype,
    ).reshape_as(optimized_values)
    (optimized_values * weights).sum().backward()
    (reference_values * weights).sum().backward()
    torch.testing.assert_close(
        optimized_features.grad, reference_features.grad
    )


def test_voxel_grid_recovers_cell_missed_by_centreline_ray_sampling():
    common = {
        "bbox_min": [0.0, 0.0, -1.0],
        "bbox_max": [2.0, 2.0, 1.0],
        "LODs": [1.0],
        "max_pixel_distance": 1.0,
        "support_views": 1,
        "fusion": "mean",
        "backend": "raw",
        "projection_pixel_order": "xy",
    }
    ray_module = SparseBackwardProjection(**common, candidate_mode="ray")
    grid_module = SparseBackwardProjection(
        **common, candidate_mode="voxel_grid", voxel_chunk_size=3
    )
    distance = _truncated_corner_distance(0)[None, None]
    features = torch.ones((1, 1, 1, 2, 2), dtype=torch.float32)
    projection = _front_projection(focal_x=1.0, focal_y=1.0)[
        None, None
    ]

    ray_result, _ = ray_module(distance, features, projection)
    grid_result, _ = grid_module(distance, features, projection)

    ray_coordinates = {tuple(row) for row in ray_result.coordinates.tolist()}
    grid_coordinates = {
        tuple(row) for row in grid_result.coordinates.tolist()
    }
    assert len(ray_coordinates) == 2
    assert len(grid_coordinates) == 5
    assert (0, 1, 1, 1) in grid_coordinates
    assert (0, 1, 1, 1) not in ray_coordinates


def test_voxel_grid_features_remain_differentiable():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [2.0, 2.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        candidate_mode="voxel_grid",
        voxel_chunk_size=3,
        backend="raw",
        projection_pixel_order="xy",
    )
    distance = _truncated_corner_distance(0)[None, None]
    features = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 2, 2)
    features.requires_grad_(True)
    projection = _front_projection(focal_x=1.0, focal_y=1.0)[
        None, None
    ]

    result, _ = module(distance, features, projection)
    result.features.sum().backward()

    assert result.coordinates.shape[0] == 5
    assert features.grad is not None
    assert features.grad[0, 0, 0, 1, 1] > 0


def test_voxel_grid_empty_intersection_has_configured_concat_width():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [2.0, 2.0, 1.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=2,
        fusion="concat",
        include_distance_feature=True,
        candidate_mode="voxel_grid",
        voxel_chunk_size=3,
        backend="raw",
        projection_pixel_order="xy",
    )
    distances = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    features = torch.ones((1, 2, 1, 2, 2), dtype=torch.float32)
    projections = _front_projection()[None, None].repeat(1, 2, 1, 1)

    result, world = module(distances, features, projections)

    assert result.coordinates.shape == (0, 4)
    assert result.features.shape == (0, 4)
    assert world.shape == (0, 3)


def test_voxel_grid_partial_support_mean_fuses_by_coordinate():
    module = SparseBackwardProjection(
        [0.0, 0.0, -1.0],
        [4.0, 1.0, 0.0],
        [1.0],
        max_pixel_distance=1.0,
        support_views=1,
        fusion="mean",
        candidate_mode="voxel_grid",
        voxel_chunk_size=2,
        backend="raw",
        projection_pixel_order="xy",
    )
    distances = torch.tensor(
        [[[[0.0, 1.0]], [[1.0, 0.0]]]], dtype=torch.float32
    )
    features = torch.tensor([2.0, 6.0], dtype=torch.float32).reshape(
        1, 2, 1, 1, 1
    ).repeat(1, 1, 1, 1, 2)
    projections = _front_projection(focal_x=1.0, focal_y=1.0)[
        None, None
    ].repeat(1, 2, 1, 1)

    result, _ = module(distances, features, projections)

    assert result.coordinates[:, 1:].tolist() == [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
    ]
    torch.testing.assert_close(
        result.features[:, 0], torch.tensor([2.0, 6.0, 6.0, 6.0])
    )
