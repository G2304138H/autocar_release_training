"""Small compatibility helpers for supported sparse tensor backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.modules.ray_casting import SparseProjection


def sparse_features(sparse_tensor: Any) -> Tensor:
    """Return the feature matrix from raw, MinkowskiEngine, or spconv tensors."""

    if isinstance(sparse_tensor, SparseProjection):
        return sparse_tensor.features
    if hasattr(sparse_tensor, "features"):
        return sparse_tensor.features
    if hasattr(sparse_tensor, "F"):
        return sparse_tensor.F
    raise TypeError(f"Unsupported sparse tensor type: {type(sparse_tensor)!r}.")


def sparse_coordinates_bxyz(sparse_tensor: Any, backend: str) -> Tensor:
    """Return integer coordinates in the canonical ``[batch,x,y,z]`` order."""

    backend = backend.lower()
    if isinstance(sparse_tensor, SparseProjection):
        return sparse_tensor.coordinates.to(dtype=torch.long)
    if backend == "minkowski":
        return sparse_tensor.C.to(dtype=torch.long)
    if backend == "spconv":
        # spconv stores [batch,z,y,x].
        return sparse_tensor.indices[:, [0, 3, 2, 1]].to(dtype=torch.long)
    raise ValueError(f"Unknown sparse backend {backend!r}.")


def rasterize_sparse_channel(
    sparse_tensor: Any,
    *,
    backend: str,
    spatial_shape_xyz: tuple[int, int, int],
    batch_size: int,
    channel: int = 0,
    sigmoid: bool = True,
    output_device: torch.device | str = "cpu",
) -> Tensor:
    """Rasterize one sparse output channel to a dense ``[B,Z,Y,X]`` volume.

    Missing sparse coordinates represent background and are filled with zero.
    The destination defaults to CPU to avoid an unnecessary full ``400^3`` GPU
    allocation during validation.
    """

    features = sparse_features(sparse_tensor)
    coordinates = sparse_coordinates_bxyz(sparse_tensor, backend)
    if channel < 0 or channel >= features.shape[1]:
        raise IndexError(
            f"channel {channel} is outside feature width {features.shape[1]}."
        )
    values = features[:, channel]
    if sigmoid:
        values = torch.sigmoid(values)

    size_x, size_y, size_z = (int(value) for value in spatial_shape_xyz)
    device = torch.device(output_device)
    dense = torch.zeros(
        (int(batch_size), size_z, size_y, size_x),
        dtype=values.dtype,
        device=device,
    )
    if coordinates.numel() == 0:
        return dense

    coordinates = coordinates.to(device=device)
    values = values.to(device=device)
    valid = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < batch_size)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < size_x)
        & (coordinates[:, 2] >= 0)
        & (coordinates[:, 2] < size_y)
        & (coordinates[:, 3] >= 0)
        & (coordinates[:, 3] < size_z)
    )
    if not torch.all(valid):
        bad_count = int((~valid).sum().item())
        raise ValueError(f"Sparse output contains {bad_count} out-of-grid coordinates.")
    dense[
        coordinates[:, 0],
        coordinates[:, 3],
        coordinates[:, 2],
        coordinates[:, 1],
    ] = values
    return dense

