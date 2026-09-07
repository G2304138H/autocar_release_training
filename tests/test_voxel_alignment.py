import numpy as np
import pytest

from src.geometry.voxel_grid import VoxelGrid, resample_binary_volume_nearest


def test_centered_grid_uses_half_open_voxel_centers():
    grid = VoxelGrid.centered(
        shape_zyx=(2, 4, 6),
        spacing_xyz_mm=(1.0, 2.0, 3.0),
    )

    assert grid.shape_xyz == (6, 4, 2)
    assert grid.origin_xyz_mm == (-3.0, -4.0, -3.0)
    assert grid.upper_bound_xyz_mm == (3.0, 4.0, 3.0)
    np.testing.assert_allclose(
        grid.index_zyx_to_world_xyz(np.array([0, 0, 0])),
        (-2.5, -3.0, -1.5),
    )
    np.testing.assert_allclose(
        grid.index_zyx_to_world_xyz(np.array([1, 3, 5])),
        (2.5, 3.0, 1.5),
    )


def test_from_bounds_produces_exact_autocar_shape():
    grid = VoxelGrid.from_bounds(
        (-100.0, -100.0, -100.0),
        (100.0, 100.0, 100.0),
        (0.5, 0.5, 0.5),
    )

    assert grid.shape_zyx == (400, 400, 400)
    np.testing.assert_allclose(
        grid.index_zyx_to_world_xyz(np.array([0, 0, 0])),
        (-99.75, -99.75, -99.75),
    )
    np.testing.assert_allclose(
        grid.index_zyx_to_world_xyz(np.array([399, 399, 399])),
        (99.75, 99.75, 99.75),
    )


def test_index_world_round_trip_preserves_zyx_xyz_order():
    grid = VoxelGrid(
        shape_zyx=(7, 6, 5),
        spacing_xyz_mm=(0.5, 2.0, 3.5),
        origin_xyz_mm=(-1.0, 10.0, 100.0),
    )
    indices_zyx = np.array([[0, 0, 0], [6, 5, 4], [2, 1, 3]])

    world_xyz = grid.index_zyx_to_world_xyz(indices_zyx)
    recovered_zyx = grid.world_xyz_to_continuous_index_zyx(world_xyz)

    np.testing.assert_allclose(recovered_zyx, indices_zyx)


def test_identity_resampling_preserves_impulse_and_full_fov():
    grid = VoxelGrid((5, 6, 7), (1.0, 1.0, 1.0))
    source = np.zeros(grid.shape_zyx, dtype=np.uint8)
    source[1, 2, 3] = 1

    aligned, valid_fov = resample_binary_volume_nearest(source, grid, grid)

    np.testing.assert_array_equal(aligned, source.astype(bool))
    assert valid_fov.all()


def test_anisotropic_spacing_is_respected():
    grid = VoxelGrid(
        shape_zyx=(3, 4, 5),
        spacing_xyz_mm=(0.4, 1.5, 3.0),
        origin_xyz_mm=(7.0, -4.0, 20.0),
    )
    source = np.zeros(grid.shape_zyx, dtype=bool)
    source[2, 1, 4] = True

    aligned, valid_fov = resample_binary_volume_nearest(source, grid, grid)

    np.testing.assert_array_equal(aligned, source)
    assert valid_fov.all()
    np.testing.assert_allclose(
        grid.index_zyx_to_world_xyz(np.array([2, 1, 4])),
        (8.8, -1.75, 27.5),
    )


def test_positive_center_offset_maps_target_to_higher_source_coordinate():
    source_grid = VoxelGrid((5, 5, 5), (1.0, 1.0, 1.0))
    target_grid = VoxelGrid((5, 5, 5), (1.0, 1.0, 1.0))
    source = np.zeros(source_grid.shape_zyx, dtype=bool)
    source[2, 2, 3] = True

    aligned, _ = resample_binary_volume_nearest(
        source,
        source_grid,
        target_grid,
        target_to_source_offset_xyz_mm=(2.0, 0.0, 0.0),
    )

    expected = np.zeros(target_grid.shape_zyx, dtype=bool)
    # source x=3.5 mm equals target x=1.5 mm + 2 mm offset.
    expected[2, 2, 1] = True
    np.testing.assert_array_equal(aligned, expected)


def test_resampling_returns_half_open_valid_fov_and_zero_padding():
    source_grid = VoxelGrid((3, 3, 3), (1.0, 1.0, 1.0))
    target_grid = VoxelGrid(
        (5, 5, 5),
        (1.0, 1.0, 1.0),
        origin_xyz_mm=(-1.0, -1.0, -1.0),
    )
    source = np.ones(source_grid.shape_zyx, dtype=bool)

    aligned, valid_fov = resample_binary_volume_nearest(
        source, source_grid, target_grid
    )

    expected = np.zeros(target_grid.shape_zyx, dtype=bool)
    expected[1:4, 1:4, 1:4] = True
    np.testing.assert_array_equal(valid_fov, expected)
    np.testing.assert_array_equal(aligned, expected)


def test_invalid_grid_and_source_shape_are_rejected():
    with pytest.raises(ValueError, match="positive integers"):
        VoxelGrid((3, 0, 3), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="integer multiples"):
        VoxelGrid.from_bounds((0, 0, 0), (1, 1, 1), (0.3, 0.3, 0.3))

    grid = VoxelGrid((3, 3, 3), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="does not match"):
        resample_binary_volume_nearest(np.zeros((2, 3, 3)), grid, grid)

