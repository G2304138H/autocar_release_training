from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.modules.ray_casting import SparseProjection
from src.modules.sparse_utils import rasterize_sparse_channel


def test_raw_projection_rasterizes_xyz_to_zyx():
    projection = SparseProjection(
        features=torch.tensor([[0.25], [0.75]]),
        coordinates=torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]]),
        world_coordinates=torch.empty((2, 3)),
        spatial_shape_xyz=(5, 6, 7),
        voxel_size=1.0,
        batch_size=1,
    )
    dense = rasterize_sparse_channel(
        projection,
        backend="raw",
        spatial_shape_xyz=projection.spatial_shape_xyz,
        batch_size=1,
        sigmoid=False,
    )
    assert dense.shape == (1, 7, 6, 5)
    assert dense[0, 3, 2, 1] == pytest.approx(0.25)
    assert dense[0, 6, 5, 4] == pytest.approx(0.75)
    assert torch.count_nonzero(dense) == 2


def test_out_of_grid_coordinate_is_rejected():
    projection = SparseProjection(
        features=torch.ones((1, 1)),
        coordinates=torch.tensor([[0, 2, 0, 0]]),
        world_coordinates=torch.empty((1, 3)),
        spatial_shape_xyz=(2, 2, 2),
        voxel_size=1.0,
        batch_size=1,
    )
    with pytest.raises(ValueError, match="out-of-grid"):
        rasterize_sparse_channel(
            projection,
            backend="raw",
            spatial_shape_xyz=(2, 2, 2),
            batch_size=1,
            sigmoid=False,
        )

