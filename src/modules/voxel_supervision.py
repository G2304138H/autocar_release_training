"""Torch helpers for physically aligned sparse voxel supervision."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_sparse_volume_targets(
    coordinates_bxyz: Tensor,
    *,
    reference_features: Tensor,
    bbox_min_xyz_mm: Tensor,
    voxel_size_mm: float,
    volumes_zyx: Tensor,
    spacing_xyz_mm: Tensor,
    origin_xyz_mm: Tensor,
    target_to_source_offset_xyz_mm: Tensor,
) -> tuple[Tensor, Tensor]:
    """Sample native volumes at sparse target centres and flag valid FOV.

    Coordinates are integer ``[batch,x,y,z]`` indices in a target grid whose
    lower boundary is ``bbox_min_xyz_mm``. The positive offset maps target
    coordinates back into the native source frame.
    """

    if coordinates_bxyz.ndim != 2 or coordinates_bxyz.shape[1] != 4:
        raise ValueError("coordinates_bxyz must have shape [N,4].")
    if volumes_zyx.ndim != 4:
        raise ValueError("volumes_zyx must have shape [B,Z,Y,X].")
    batch_size = volumes_zyx.shape[0]
    for name, value in (
        ("spacing_xyz_mm", spacing_xyz_mm),
        ("origin_xyz_mm", origin_xyz_mm),
        ("target_to_source_offset_xyz_mm", target_to_source_offset_xyz_mm),
    ):
        if value.shape != (batch_size, 3):
            raise ValueError(f"{name} must have shape [B,3].")

    device = reference_features.device
    coordinates = coordinates_bxyz.to(device=device, dtype=torch.long)
    bbox_min = bbox_min_xyz_mm.to(device=device, dtype=torch.float32)
    world_xyz = bbox_min + (
        coordinates[:, 1:].to(torch.float32) + 0.5
    ) * float(voxel_size_mm)
    volumes = volumes_zyx.to(device=device)
    spacing = spacing_xyz_mm.to(device=device, dtype=torch.float32)
    origins = origin_xyz_mm.to(device=device, dtype=torch.float32)
    offsets = target_to_source_offset_xyz_mm.to(
        device=device, dtype=torch.float32
    )
    targets = reference_features.new_zeros((coordinates.shape[0],))
    valid_targets = torch.zeros(
        (coordinates.shape[0],), dtype=torch.bool, device=device
    )

    for batch_index in range(batch_size):
        selected = coordinates[:, 0] == batch_index
        if not torch.any(selected):
            continue
        native_world = world_xyz[selected] + offsets[batch_index]
        indices_xyz = torch.floor(
            (native_world - origins[batch_index]) / spacing[batch_index]
        ).to(torch.long)
        shape_xyz = torch.tensor(
            list(reversed(volumes[batch_index].shape)),
            device=device,
            dtype=torch.long,
        )
        valid = torch.all(
            (indices_xyz >= 0) & (indices_xyz < shape_xyz), dim=1
        )
        selected_rows = torch.nonzero(selected, as_tuple=False)[:, 0]
        valid_rows = selected_rows[valid]
        valid_xyz = indices_xyz[valid]
        valid_targets[valid_rows] = True
        targets[valid_rows] = volumes[
            batch_index,
            valid_xyz[:, 2],
            valid_xyz[:, 1],
            valid_xyz[:, 0],
        ].to(dtype=targets.dtype)
    return targets, valid_targets
