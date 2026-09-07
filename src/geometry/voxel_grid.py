"""Physical voxel-grid conventions and nearest-neighbour alignment.

All arrays in this module use ``(z, y, x)`` axis order.  Physical vectors use
``(x, y, z)`` order and millimetres.  A grid's ``origin_xyz_mm`` is its lower
physical boundary, not the center of its first voxel.  Consequently, voxel
``(z, y, x)`` has physical center

``origin_xyz_mm + ((x, y, z) + 0.5) * spacing_xyz_mm``.

This half-open convention makes an AutoCAR box such as ``[-100, 100)`` mm at
0.5 mm spacing contain exactly 400 voxels on each axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


Array3 = Tuple[float, float, float]
Shape3 = Tuple[int, int, int]


def _finite_xyz(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite XYZ values, got {result}.")
    return result


def _shape_zyx(values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(tuple(values))
    if raw.shape != (3,):
        raise ValueError(f"shape_zyx must contain three values, got {raw}.")
    try:
        shape = raw.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"shape_zyx must contain integers, got {raw}.") from error
    if np.any(shape <= 0) or not np.array_equal(raw, shape):
        raise ValueError(f"shape_zyx must contain positive integers, got {raw}.")
    return shape


@dataclass(frozen=True)
class VoxelGrid:
    """Describe a regular voxel grid in physical space.

    Parameters
    ----------
    shape_zyx:
        Array shape in canonical ZYX order.
    spacing_xyz_mm:
        Positive voxel spacing in physical XYZ order, in millimetres.
    origin_xyz_mm:
        Lower boundary of voxel ``(0, 0, 0)`` in physical XYZ coordinates.
        The first voxel center is half a spacing above this boundary.
    """

    shape_zyx: Shape3
    spacing_xyz_mm: Array3
    origin_xyz_mm: Array3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        shape = _shape_zyx(self.shape_zyx)
        spacing = _finite_xyz(self.spacing_xyz_mm, "spacing_xyz_mm")
        origin = _finite_xyz(self.origin_xyz_mm, "origin_xyz_mm")
        if np.any(spacing <= 0.0):
            raise ValueError(
                f"spacing_xyz_mm must be strictly positive, got {spacing}."
            )
        object.__setattr__(self, "shape_zyx", tuple(int(v) for v in shape))
        object.__setattr__(
            self, "spacing_xyz_mm", tuple(float(v) for v in spacing)
        )
        object.__setattr__(self, "origin_xyz_mm", tuple(float(v) for v in origin))

    @classmethod
    def centered(
        cls,
        shape_zyx: Sequence[int],
        spacing_xyz_mm: Sequence[float],
        center_xyz_mm: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "VoxelGrid":
        """Create a grid whose half-open physical box is centered at ``center``."""

        shape = _shape_zyx(shape_zyx)
        spacing = _finite_xyz(spacing_xyz_mm, "spacing_xyz_mm")
        center = _finite_xyz(center_xyz_mm, "center_xyz_mm")
        if np.any(spacing <= 0.0):
            raise ValueError("spacing_xyz_mm must be strictly positive.")
        extent_xyz = shape[::-1] * spacing
        origin_xyz = center - 0.5 * extent_xyz
        return cls(
            shape_zyx=tuple(int(v) for v in shape),
            spacing_xyz_mm=tuple(float(v) for v in spacing),
            origin_xyz_mm=tuple(float(v) for v in origin_xyz),
        )

    @classmethod
    def from_bounds(
        cls,
        minimum_xyz_mm: Sequence[float],
        maximum_xyz_mm: Sequence[float],
        spacing_xyz_mm: Sequence[float],
        *,
        tolerance: float = 1e-6,
    ) -> "VoxelGrid":
        """Create a half-open grid ``[minimum, maximum)``.

        Each physical extent must be an integer multiple of its spacing.  For
        example, bounds ``(-100, -100, -100)`` to ``(100, 100, 100)`` with
        spacing ``(0.5, 0.5, 0.5)`` produce shape ``(400, 400, 400)``.
        """

        minimum = _finite_xyz(minimum_xyz_mm, "minimum_xyz_mm")
        maximum = _finite_xyz(maximum_xyz_mm, "maximum_xyz_mm")
        spacing = _finite_xyz(spacing_xyz_mm, "spacing_xyz_mm")
        if np.any(spacing <= 0.0):
            raise ValueError("spacing_xyz_mm must be strictly positive.")
        extent = maximum - minimum
        if np.any(extent <= 0.0):
            raise ValueError("maximum_xyz_mm must exceed minimum_xyz_mm.")
        shape_xyz_float = extent / spacing
        shape_xyz = np.rint(shape_xyz_float).astype(np.int64)
        if not np.allclose(
            shape_xyz_float, shape_xyz, rtol=0.0, atol=float(tolerance)
        ):
            raise ValueError(
                "Grid extents must be integer multiples of spacing; got "
                f"extent/spacing={shape_xyz_float}."
            )
        return cls(
            shape_zyx=tuple(int(v) for v in shape_xyz[::-1]),
            spacing_xyz_mm=tuple(float(v) for v in spacing),
            origin_xyz_mm=tuple(float(v) for v in minimum),
        )

    @property
    def shape_xyz(self) -> Shape3:
        """Grid shape in physical XYZ axis order."""

        return tuple(reversed(self.shape_zyx))

    @property
    def upper_bound_xyz_mm(self) -> Array3:
        """Exclusive upper physical boundary in XYZ millimetres."""

        upper = np.asarray(self.origin_xyz_mm) + np.asarray(
            self.shape_xyz
        ) * np.asarray(self.spacing_xyz_mm)
        return tuple(float(v) for v in upper)

    @property
    def center_xyz_mm(self) -> Array3:
        """Physical center of the grid's half-open bounding box."""

        center = 0.5 * (
            np.asarray(self.origin_xyz_mm) + np.asarray(self.upper_bound_xyz_mm)
        )
        return tuple(float(v) for v in center)

    def axis_centers_xyz_mm(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return one-dimensional center coordinates for X, Y, and Z axes."""

        origin = np.asarray(self.origin_xyz_mm)
        spacing = np.asarray(self.spacing_xyz_mm)
        return tuple(
            origin[axis]
            + (np.arange(self.shape_xyz[axis], dtype=np.float64) + 0.5)
            * spacing[axis]
            for axis in range(3)
        )

    def index_zyx_to_world_xyz(self, index_zyx: np.ndarray) -> np.ndarray:
        """Convert one or more ZYX voxel indices to XYZ center coordinates."""

        index = np.asarray(index_zyx, dtype=np.float64)
        if index.shape == () or index.shape[-1] != 3 or not np.isfinite(index).all():
            raise ValueError(
                "index_zyx must have shape (..., 3) and contain finite values."
            )
        index_xyz = index[..., ::-1]
        return np.asarray(self.origin_xyz_mm) + (
            index_xyz + 0.5
        ) * np.asarray(self.spacing_xyz_mm)

    def world_xyz_to_continuous_index_zyx(
        self, world_xyz_mm: np.ndarray
    ) -> np.ndarray:
        """Map XYZ positions to continuous ZYX center-index coordinates."""

        world = np.asarray(world_xyz_mm, dtype=np.float64)
        if world.shape == () or world.shape[-1] != 3 or not np.isfinite(world).all():
            raise ValueError(
                "world_xyz_mm must have shape (..., 3) and contain finite values."
            )
        index_xyz = (
            (world - np.asarray(self.origin_xyz_mm))
            / np.asarray(self.spacing_xyz_mm)
            - 0.5
        )
        return index_xyz[..., ::-1]


def resample_binary_volume_nearest(
    source_volume_zyx: np.ndarray,
    source_grid: VoxelGrid,
    target_grid: VoxelGrid,
    *,
    target_to_source_offset_xyz_mm: Sequence[float] = (0.0, 0.0, 0.0),
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a binary source volume onto a target grid.

    Parameters
    ----------
    target_to_source_offset_xyz_mm:
        Offset added to a target-frame coordinate before sampling the source.
        AutoCAR centers its reconstruction as ``q = p - center_offset``;
        therefore aligning native ground truth requires
        ``p = q + center_offset`` and this argument must be the stored positive
        projection-center offset.

    Returns
    -------
    resampled_zyx, valid_fov_zyx:
        A boolean target-grid volume and a boolean mask identifying target
        centers that lie inside the source's half-open physical field of view.
        Values outside that field of view are always false.
    """

    source = np.asarray(source_volume_zyx)
    if source.ndim != 3:
        raise ValueError(f"source_volume_zyx must be 3D, got shape {source.shape}.")
    if tuple(source.shape) != source_grid.shape_zyx:
        raise ValueError(
            "source volume shape does not match source_grid: "
            f"{source.shape} versus {source_grid.shape_zyx}."
        )
    if not np.issubdtype(source.dtype, np.bool_) and not np.isfinite(source).all():
        raise ValueError("source_volume_zyx contains non-finite values.")
    if not np.isfinite(float(threshold)):
        raise ValueError(f"threshold must be finite, got {threshold}.")

    offset = _finite_xyz(
        target_to_source_offset_xyz_mm, "target_to_source_offset_xyz_mm"
    )
    target_axes_xyz = target_grid.axis_centers_xyz_mm()
    source_origin = np.asarray(source_grid.origin_xyz_mm)
    source_spacing = np.asarray(source_grid.spacing_xyz_mm)
    source_shape_xyz = np.asarray(source_grid.shape_xyz)

    source_indices_xyz = []
    valid_axes_xyz = []
    for axis, target_centers in enumerate(target_axes_xyz):
        source_coordinates = target_centers + offset[axis]
        # With lower-bound origins, containing-cell lookup is equivalent to
        # nearest-neighbour lookup around each voxel center.
        indices = np.floor(
            (source_coordinates - source_origin[axis]) / source_spacing[axis]
        ).astype(np.int64)
        valid = (indices >= 0) & (indices < source_shape_xyz[axis])
        source_indices_xyz.append(
            np.clip(indices, 0, source_shape_xyz[axis] - 1)
        )
        valid_axes_xyz.append(valid)

    x_indices, y_indices, z_indices = source_indices_xyz
    valid_x, valid_y, valid_z = valid_axes_xyz
    source_binary = source if source.dtype == np.bool_ else source >= float(threshold)
    sampled = source_binary[np.ix_(z_indices, y_indices, x_indices)].astype(
        np.bool_, copy=False
    )
    valid_fov = (
        valid_z[:, None, None]
        & valid_y[None, :, None]
        & valid_x[None, None, :]
    )
    sampled = sampled & valid_fov
    return np.ascontiguousarray(sampled), np.ascontiguousarray(valid_fov)

