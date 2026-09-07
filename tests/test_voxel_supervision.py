from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.modules.voxel_supervision import sample_sparse_volume_targets


def test_out_of_fov_sparse_coordinates_are_marked_unknown_not_background():
    volume = torch.zeros((1, 2, 2, 2), dtype=torch.uint8)
    volume[0, 0, 0, 0] = 1
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]], dtype=torch.long
    )

    targets, valid = sample_sparse_volume_targets(
        coordinates,
        reference_features=torch.zeros((3, 1)),
        bbox_min_xyz_mm=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size_mm=1.0,
        volumes_zyx=volume,
        spacing_xyz_mm=torch.tensor([[1.0, 1.0, 1.0]]),
        origin_xyz_mm=torch.tensor([[0.0, 0.0, 0.0]]),
        target_to_source_offset_xyz_mm=torch.tensor([[0.0, 0.0, 0.0]]),
    )

    assert targets.tolist() == [1.0, 0.0, 0.0]
    assert valid.tolist() == [True, True, False]


def test_positive_center_offset_maps_target_back_to_native_frame():
    volume = torch.zeros((1, 1, 1, 3), dtype=torch.uint8)
    volume[0, 0, 0, 2] = 1

    targets, valid = sample_sparse_volume_targets(
        torch.tensor([[0, 0, 0, 0]]),
        reference_features=torch.zeros((1, 1)),
        bbox_min_xyz_mm=torch.tensor([0.0, 0.0, 0.0]),
        voxel_size_mm=1.0,
        volumes_zyx=volume,
        spacing_xyz_mm=torch.tensor([[1.0, 1.0, 1.0]]),
        origin_xyz_mm=torch.tensor([[0.0, 0.0, 0.0]]),
        target_to_source_offset_xyz_mm=torch.tensor([[2.0, 0.0, 0.0]]),
    )

    assert valid.item()
    assert targets.item() == 1.0
