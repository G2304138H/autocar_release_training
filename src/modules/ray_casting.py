"""Sparse backward projection from vessel pixels to a 3-D voxel grid.

The upstream release imports this module but did not include it.  This
implementation follows the paper's sparse visual-hull construction by
streaming bounded voxel centres, reprojecting them into each view, and retaining
only the coordinates with the configured multi-view support. A legacy
ray-quantisation candidate mode remains available for released-code
compatibility. A dependency-free truncated Euclidean distance transform
supplies the active pixel band from Methods Eq. 5 and can be concatenated with
encoder features as specified by Eqs. 6 and 9.

Projection uses the invertible 4x4 convention shared by the legacy camera and
the Stage-2 geometry adapter. Each voxel centre is projected into its source
view. The discrete EDT field uses nearest-pixel sampling by default, matching
Eq. 5, while learned encoder features use bilinear interpolation consistent
with the supplementary dense ``grid_sample`` description. Discrete voxel
selection is non-differentiable, while encoder sampling and feature reductions
remain differentiable.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from math import ceil, sqrt
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseProjection:
    """Backend-neutral sparse tensor produced by backward projection.

    ``coordinates`` use ``[batch, x, y, z]`` ordering.  Backends which use a
    different convention are converted only at the boundary in
    :meth:`as_backend`.
    """

    features: Tensor
    coordinates: Tensor
    world_coordinates: Tensor
    spatial_shape_xyz: tuple[int, int, int]
    voxel_size: float
    batch_size: int

    def as_backend(self, backend: str):
        """Convert this projection to MinkowskiEngine, spconv, or leave raw."""

        backend = backend.lower()
        if backend == "raw":
            return self
        if backend == "minkowski":
            try:
                import MinkowskiEngine as ME
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "MinkowskiEngine was requested but is not installed."
                ) from exc
            return ME.SparseTensor(
                features=self.features,
                coordinates=self.coordinates.to(dtype=torch.int32),
            )
        if backend == "spconv":
            try:
                import spconv.pytorch as spconv
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("spconv was requested but is not installed.") from exc

            # spconv spatial indices are [batch, z, y, x].
            indices_bzyx = self.coordinates[:, [0, 3, 2, 1]].to(
                dtype=torch.int32
            )
            spatial_shape_zyx = list(reversed(self.spatial_shape_xyz))
            return spconv.SparseConvTensor(
                features=self.features,
                indices=indices_bzyx,
                spatial_shape=spatial_shape_zyx,
                batch_size=self.batch_size,
            )
        raise ValueError(
            f"Unknown sparse backend {backend!r}; expected raw, minkowski, or spconv."
        )


class SparseBackwardProjection(nn.Module):
    """Back-project multi-view image features into a sparse 3-D visual hull.

    Parameters
    ----------
    bbox_min, bbox_max:
        Three-element world-space bounds in the same units as the projection
        matrices (millimetres for the NPZ training path).
    LODs:
        Voxel-size sequence retained for compatibility with the incomplete
        release. Only the final value determines sampling and output grid size;
        progressive coarse-to-fine refinement cannot be recovered from the
        paper or released code and is intentionally not claimed here.
    max_pixel_distance:
        Reprojected locations whose distance-transform value is strictly below
        this threshold are active, matching Methods Eqs. 5, 7 and 8. In legacy
        ray mode these pixels also initiate candidate rays. ``active_masks``
        is an optional ray-mode restriction for ablations.
    support_views:
        Minimum number of distinct views which must back-project to a voxel.
    fusion:
        ``"mean"`` preserves the encoder channel count. ``"concat"`` follows
        the paper literally and requires every input view to support a voxel.
    include_distance_feature:
        Prefix each per-view encoder feature with its truncated Euclidean
        distance-to-vessel value. Together with ``fusion="concat"`` this is
        the feature construction described by Methods Eqs. 6 and 9.
    backend:
        ``"auto"`` selects MinkowskiEngine when available, then spconv, then
        the backend-neutral :class:`SparseProjection` representation.
    projection_pixel_order:
        Ordering of the first two projected matrix coordinates.  The Stage-2
        NPZ geometry uses ``"xy"`` (column, row); the legacy camera class can
        be selected explicitly with ``"rc"`` when required.
    candidate_mode:
        ``"voxel_grid"`` streams every voxel centre through Eq. 7 and is the
        scientifically faithful paper path. ``"ray"`` retains the released
        API's ray-sampling approximation for legacy checkpoints and tests.
    voxel_chunk_size:
        Maximum number of grid coordinates materialised at once by
        ``candidate_mode="voxel_grid"``. The complete grid is never allocated.
    distance_sampling:
        ``"nearest"`` treats Eq. 5 as the paper's literal discrete active-pixel
        set and is the default. ``"bilinear"`` is an explicit interpolation
        experiment; its EDT retains an additional square-root-two pixel halo
        so out-of-band values are not incorrectly capped at epsilon.
    """

    _VALID_FUSIONS = {"mean", "concat"}
    _VALID_BACKENDS = {"auto", "raw", "minkowski", "spconv"}
    _VALID_PIXEL_ORDERS = {"xy", "rc"}
    _VALID_CANDIDATE_MODES = {"ray", "voxel_grid"}
    _VALID_DISTANCE_SAMPLING = {"nearest", "bilinear"}

    def __init__(
        self,
        bbox_min: Sequence[float],
        bbox_max: Sequence[float],
        LODs: Sequence[float],
        *,
        max_pixel_distance: float = 0.5,
        support_views: int = 2,
        fusion: str = "mean",
        backend: str = "auto",
        ray_chunk_size: int = 4096,
        pixel_center_offset: float = 0.0,
        projection_pixel_order: str = "rc",
        include_distance_feature: bool = False,
        candidate_mode: str = "ray",
        voxel_chunk_size: int = 262_144,
        distance_sampling: str = "nearest",
    ) -> None:
        super().__init__()
        bbox_min_tensor = torch.as_tensor(bbox_min, dtype=torch.float32)
        bbox_max_tensor = torch.as_tensor(bbox_max, dtype=torch.float32)
        lods_tensor = torch.as_tensor(LODs, dtype=torch.float32)
        if bbox_min_tensor.shape != (3,) or bbox_max_tensor.shape != (3,):
            raise ValueError("bbox_min and bbox_max must each contain x, y, z.")
        if not torch.all(bbox_max_tensor > bbox_min_tensor):
            raise ValueError("Every bbox_max component must exceed bbox_min.")
        if lods_tensor.ndim != 1 or lods_tensor.numel() == 0:
            raise ValueError("LODs must contain at least one voxel size.")
        if not torch.all(lods_tensor > 0):
            raise ValueError("All LOD voxel sizes must be positive.")
        if not torch.isfinite(torch.tensor(float(max_pixel_distance))):
            raise ValueError("max_pixel_distance must be finite.")
        if float(max_pixel_distance) <= 0:
            raise ValueError("max_pixel_distance must be positive.")
        if support_views < 1:
            raise ValueError("support_views must be at least one.")
        if fusion not in self._VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(self._VALID_FUSIONS)}.")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(self._VALID_BACKENDS)}.")
        if ray_chunk_size < 1:
            raise ValueError("ray_chunk_size must be positive.")
        if candidate_mode not in self._VALID_CANDIDATE_MODES:
            raise ValueError(
                "candidate_mode must be one of "
                f"{sorted(self._VALID_CANDIDATE_MODES)}."
            )
        if voxel_chunk_size < 1:
            raise ValueError("voxel_chunk_size must be positive.")
        if distance_sampling not in self._VALID_DISTANCE_SAMPLING:
            raise ValueError(
                "distance_sampling must be one of "
                f"{sorted(self._VALID_DISTANCE_SAMPLING)}."
            )
        if projection_pixel_order not in self._VALID_PIXEL_ORDERS:
            raise ValueError(
                "projection_pixel_order must be one of "
                f"{sorted(self._VALID_PIXEL_ORDERS)}."
            )

        self.register_buffer("bbox_min", bbox_min_tensor, persistent=False)
        self.register_buffer("bbox_max", bbox_max_tensor, persistent=False)
        self.register_buffer("lods", lods_tensor, persistent=False)
        self._voxel_size = float(lods_tensor[-1].item())
        extent = (bbox_max_tensor - bbox_min_tensor).tolist()
        self._spatial_shape_xyz = tuple(
            int(ceil(length / self._voxel_size)) for length in extent
        )
        self.max_pixel_distance = float(max_pixel_distance)
        self.support_views = int(support_views)
        self.fusion = fusion
        self.backend = backend
        self.ray_chunk_size = int(ray_chunk_size)
        self.pixel_center_offset = float(pixel_center_offset)
        self.projection_pixel_order = projection_pixel_order
        self.include_distance_feature = bool(include_distance_feature)
        self.candidate_mode = candidate_mode
        self.voxel_chunk_size = int(voxel_chunk_size)
        self.distance_sampling = distance_sampling

    @property
    def voxel_size(self) -> float:
        """Finest requested voxel size."""

        return self._voxel_size

    @property
    def spatial_shape_xyz(self) -> tuple[int, int, int]:
        return self._spatial_shape_xyz

    def output_channels(self, input_channels: int, view_count: int) -> int:
        """Return the feature width produced by the configured fusion."""

        per_view_channels = int(input_channels) + int(
            self.include_distance_feature
        )
        if self.fusion == "mean":
            return per_view_channels
        return per_view_channels * int(view_count)

    def _truncated_distance_transform(self, mask: Tensor) -> Tensor:
        """Return exact pixel-centre EDT values within the required halo.

        Only distances that can satisfy ``distance < max_pixel_distance`` are
        needed. Enumerating integer offsets in that small disk avoids a SciPy
        or custom CUDA dependency and gives identical CPU/GPU active bands.
        Nearest sampling caps at epsilon. Optional bilinear distance sampling
        retains an additional square-root-two halo, because any interpolation
        cell can draw on a diagonal neighbour beyond epsilon.
        """

        if mask.ndim != 2:
            raise ValueError(f"mask must be [H,W], got {tuple(mask.shape)}.")
        foreground = (torch.isfinite(mask) & (mask > 0.5)).detach()
        distance_limit = self.max_pixel_distance
        if self.distance_sampling == "bilinear":
            distance_limit += sqrt(2.0)
        distance = torch.full(
            mask.shape,
            distance_limit,
            dtype=torch.float32,
            device=mask.device,
        )
        distance[foreground] = 0.0
        if not torch.any(foreground):
            return distance

        height, width = mask.shape
        radius = int(ceil(distance_limit))
        offsets = [
            (dy * dy + dx * dx, dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if 0 < dy * dy + dx * dx < distance_limit**2
        ]
        offsets.sort(key=lambda item: item[0])
        for squared_distance, dy, dx in offsets:
            row_source_start = max(0, -dy)
            row_source_stop = min(height, height - dy)
            column_source_start = max(0, -dx)
            column_source_stop = min(width, width - dx)
            if (
                row_source_start >= row_source_stop
                or column_source_start >= column_source_stop
            ):
                continue
            candidate = torch.zeros_like(foreground)
            candidate[
                row_source_start + dy : row_source_stop + dy,
                column_source_start + dx : column_source_stop + dx,
            ] = foreground[
                row_source_start:row_source_stop,
                column_source_start:column_source_stop,
            ]
            value = float(squared_distance) ** 0.5
            update = candidate & (distance > value)
            distance[update] = value
        return distance

    def distance_maps_from_masks(self, masks: Tensor) -> Tensor:
        """Compute truncated EDT maps for ``[H,W]`` or ``[B,V,H,W]`` masks."""

        if masks.ndim == 2:
            return self._truncated_distance_transform(masks)
        if masks.ndim != 4:
            raise ValueError(
                f"masks must be [H,W] or [B,V,H,W], got {tuple(masks.shape)}."
            )
        batch_size, view_count = masks.shape[:2]
        maps = [
            self._truncated_distance_transform(masks[batch_index, view_index])
            for batch_index in range(batch_size)
            for view_index in range(view_count)
        ]
        return torch.stack(maps, dim=0).reshape_as(masks)

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if find_spec("MinkowskiEngine") is not None:
            return "minkowski"
        if find_spec("spconv") is not None:
            return "spconv"
        return "raw"

    @staticmethod
    def _camera_rays(projection: Tensor, projected_pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Return rays for pixels ordered like the projection's first rows."""

        if projection.shape != (4, 4):
            raise ValueError(f"Projection matrix must be 4x4, got {projection.shape}.")
        # Matrix inversion and matmul are autocast-eligible; explicitly disable
        # AMP so ill-conditioned cone-beam matrices never become fp16/bfloat16.
        if hasattr(torch, "autocast"):
            autocast_disabled = torch.autocast(
                device_type=projection.device.type, enabled=False
            )
        elif projection.is_cuda:  # pragma: no cover - legacy PyTorch only
            autocast_disabled = torch.cuda.amp.autocast(enabled=False)
        else:  # pragma: no cover - legacy PyTorch only
            autocast_disabled = nullcontext()
        with autocast_disabled:
            projection_float = projection.to(dtype=torch.float32)
            inverse = torch.linalg.inv(projection_float)
            count = projected_pixels.shape[0]
            one = torch.ones(
                (count, 1),
                dtype=torch.float32,
                device=projection.device,
            )
            projected_at_unit_depth = torch.cat(
                [projected_pixels.to(torch.float32), one, one], dim=1
            )
            world_at_unit_depth = projected_at_unit_depth @ inverse.T
            world_at_unit_depth = (
                world_at_unit_depth[:, :3] / world_at_unit_depth[:, 3:]
            )

            projected_origin = torch.tensor(
                [0.0, 0.0, 1.0, 0.0],
                dtype=torch.float32,
                device=projection.device,
            )
            world_origin_h = inverse @ projected_origin
            world_origin = world_origin_h[:3] / world_origin_h[3]
            directions = world_at_unit_depth - world_origin[None, :]
            directions = directions / torch.linalg.vector_norm(
                directions, dim=1, keepdim=True
            ).clamp_min(torch.finfo(directions.dtype).eps)
        return world_origin.expand_as(directions), directions

    def _intersect_box(self, origins: Tensor, directions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Intersect rays with the axis-aligned reconstruction box."""

        bbox_min = self.bbox_min.to(device=origins.device, dtype=origins.dtype)
        bbox_max = self.bbox_max.to(device=origins.device, dtype=origins.dtype)
        epsilon = torch.finfo(directions.dtype).eps * 16
        parallel = directions.abs() <= epsilon
        safe_directions = torch.where(parallel, torch.ones_like(directions), directions)
        t0 = (bbox_min - origins) / safe_directions
        t1 = (bbox_max - origins) / safe_directions
        axis_near = torch.minimum(t0, t1)
        axis_far = torch.maximum(t0, t1)

        inside_parallel = (origins >= bbox_min) & (origins <= bbox_max)
        negative_inf = torch.full_like(axis_near, -torch.inf)
        positive_inf = torch.full_like(axis_far, torch.inf)
        axis_near = torch.where(parallel & inside_parallel, negative_inf, axis_near)
        axis_far = torch.where(parallel & inside_parallel, positive_inf, axis_far)
        invalid_parallel = torch.any(parallel & ~inside_parallel, dim=1)

        near = torch.amax(axis_near, dim=1).clamp_min(0.0)
        far = torch.amin(axis_far, dim=1)
        valid = (~invalid_parallel) & torch.isfinite(near) & torch.isfinite(far) & (far >= near)
        return near, far, valid

    def _sample_ray_chunk(
        self,
        origins: Tensor,
        directions: Tensor,
        near: Tensor,
        far: Tensor,
        ray_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample one ray chunk and return integer XYZ coordinates and features."""

        voxel_size = self.voxel_size
        segment_lengths = (far - near).clamp_min(0.0)
        sample_counts = torch.ceil(segment_lengths / voxel_size).to(torch.long)
        sample_counts = sample_counts.clamp_min(1)
        ray_ids = torch.repeat_interleave(
            torch.arange(origins.shape[0], device=origins.device), sample_counts
        )
        starts = torch.cumsum(sample_counts, dim=0) - sample_counts
        offsets = torch.arange(ray_ids.numel(), device=origins.device) - torch.repeat_interleave(
            starts, sample_counts
        )
        # Place samples at the centres of equal sub-segments. Clamping a fixed
        # half step to ``far`` puts short rays on an exclusive box boundary and
        # can silently drop the only sample.
        step = segment_lengths / sample_counts.to(segment_lengths.dtype)
        distances = near[ray_ids] + (
            offsets.to(near.dtype) + 0.5
        ) * step[ray_ids]
        points = origins[ray_ids] + directions[ray_ids] * distances[:, None]

        bbox_min = self.bbox_min.to(device=points.device, dtype=points.dtype)
        coords_xyz = torch.floor((points - bbox_min) / voxel_size).to(torch.long)
        shape = torch.tensor(
            self.spatial_shape_xyz, device=coords_xyz.device, dtype=coords_xyz.dtype
        )
        valid = torch.all((coords_xyz >= 0) & (coords_xyz < shape), dim=1)
        return coords_xyz[valid], ray_features[ray_ids[valid]]

    @staticmethod
    def _segment_sum(sorted_values: Tensor, counts: Tensor) -> Tensor:
        """Sum consecutive segments without CUDA atomic scatter operations."""

        accumulate_dtype = (
            torch.float32
            if sorted_values.dtype in (torch.float16, torch.bfloat16)
            else sorted_values.dtype
        )
        values = sorted_values.to(accumulate_dtype)
        if hasattr(torch, "segment_reduce"):
            return torch.segment_reduce(
                values, reduce="sum", lengths=counts
            )
        cumulative = torch.cumsum(
            values, dim=0
        )
        ends = torch.cumsum(counts, dim=0) - 1
        result = cumulative[ends]
        if result.shape[0] > 1:
            result = torch.cat(
                [result[:1], result[1:] - cumulative[ends[:-1]]], dim=0
            )
        return result

    def _deduplicate_view(
        self, coords_xyz: Tensor, features: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Average arbitrary values attached to duplicate view voxels.

        This compatibility helper remains useful in focused reduction tests.
        The projection path itself deduplicates coordinates first and obtains
        their values by voxel-centre reprojection, rather than by averaging the
        features of the rays that happened to generate each voxel.
        """

        if coords_xyz.numel() == 0:
            return coords_xyz, features
        _, size_y, size_z = self.spatial_shape_xyz
        keys = (coords_xyz[:, 0] * size_y + coords_xyz[:, 1]) * size_z + coords_xyz[:, 2]
        order = torch.argsort(keys, stable=True)
        sorted_keys = keys[order]
        sorted_features = features[order]
        unique_keys, counts = torch.unique_consecutive(
            sorted_keys, return_counts=True
        )
        reduced = self._segment_sum(sorted_features, counts)
        reduced = (
            reduced / counts.to(reduced.dtype)[:, None]
        ).to(features.dtype)
        x = torch.div(unique_keys, size_y * size_z, rounding_mode="floor")
        remainder = unique_keys.remainder(size_y * size_z)
        y = torch.div(remainder, size_z, rounding_mode="floor")
        z = remainder.remainder(size_z)
        return torch.stack([x, y, z], dim=1), reduced

    def _deduplicate_coordinates(self, coords_xyz: Tensor) -> Tensor:
        """Return unique XYZ voxel coordinates in deterministic key order."""

        if coords_xyz.numel() == 0:
            return coords_xyz
        _, size_y, size_z = self.spatial_shape_xyz
        keys = (
            (coords_xyz[:, 0] * size_y + coords_xyz[:, 1]) * size_z
            + coords_xyz[:, 2]
        )
        unique_keys = torch.unique(keys, sorted=True)
        x = torch.div(unique_keys, size_y * size_z, rounding_mode="floor")
        remainder = unique_keys.remainder(size_y * size_z)
        y = torch.div(remainder, size_z, rounding_mode="floor")
        z = remainder.remainder(size_z)
        return torch.stack([x, y, z], dim=1)

    @staticmethod
    def _bilinear_sample(image: Tensor, pixels_xy: Tensor) -> Tensor:
        """Sample ``[C,H,W]`` at continuous pixel-centre XY coordinates.

        ``grid_sample(..., align_corners=False)`` is the PyTorch-style
        interpolation used by the paper's dense backward-projection baseline.
        Integer coordinates denote pixel centres. The caller performs the
        distance convention's explicit bounds check first; border padding
        defines encoder interpolation at the outer half-pixel, including the
        useful singleton-axis case.
        """

        if image.ndim != 3:
            raise ValueError(f"image must be [C,H,W], got {tuple(image.shape)}.")
        if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
            raise ValueError(
                "pixels_xy must have shape [N,2], got "
                f"{tuple(pixels_xy.shape)}."
            )
        if pixels_xy.shape[0] == 0:
            return image.new_empty((0, image.shape[0]))

        _, height, width = image.shape
        # CPU grid_sample does not implement fp16/bfloat16. Accumulating those
        # inputs in fp32 is differentiable and also avoids interpolation noise
        # under mixed-precision training.
        sample_dtype = (
            torch.float32
            if image.dtype in (torch.float16, torch.bfloat16)
            else image.dtype
        )
        image_for_sampling = image.to(dtype=sample_dtype)
        pixels = pixels_xy.to(device=image.device, dtype=sample_dtype)
        grid_x = 2.0 * (pixels[:, 0] + 0.5) / width - 1.0
        grid_y = 2.0 * (pixels[:, 1] + 0.5) / height - 1.0
        grid = torch.stack([grid_x, grid_y], dim=1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            image_for_sampling[None],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        # [1,C,N,1] -> [N,C]
        return sampled[0, :, :, 0].T.to(dtype=image.dtype)

    def _sample_distance(self, distance_map: Tensor, pixels_xy: Tensor) -> Tensor:
        """Sample the discrete EDT field using the declared convention."""

        if self.distance_sampling == "bilinear":
            return self._bilinear_sample(distance_map[None], pixels_xy)[:, 0]
        # torch.round and numpy.rint both use round-half-to-even. The explicit
        # pixel-cell bounds check in the caller guarantees valid indices.
        pixels_rounded = torch.round(pixels_xy).to(torch.long)
        return distance_map[
            pixels_rounded[:, 1], pixels_rounded[:, 0]
        ]

    def _gate_voxel_coordinates(
        self,
        coords_xyz: Tensor,
        distance_map: Tensor,
        projection: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply the geometric and EDT parts of Eq. 7 to voxel centres.

        Returns the retained coordinates, their continuous detector ``(x,y)``
        locations, sampled EDT values, and indices relative to ``coords_xyz``.
        Keeping this inexpensive selection separate from encoder sampling lets
        the all-view path establish the complete visual-hull intersection
        before materialising learned feature vectors.
        """

        source_indices = torch.arange(
            coords_xyz.shape[0], device=coords_xyz.device, dtype=torch.long
        )
        if coords_xyz.numel() == 0:
            return (
                coords_xyz,
                torch.empty(
                    (0, 2),
                    device=coords_xyz.device,
                    dtype=torch.float32,
                ),
                distance_map.new_empty((0,)),
                source_indices,
            )

        bbox_min = self.bbox_min.to(
            device=coords_xyz.device, dtype=torch.float32
        )
        centres_world = (
            bbox_min
            + (coords_xyz.to(torch.float32) + 0.5) * self.voxel_size
        )
        one = torch.ones(
            (centres_world.shape[0], 1),
            device=centres_world.device,
            dtype=torch.float32,
        )
        homogeneous = torch.cat([centres_world, one], dim=1)

        if hasattr(torch, "autocast"):
            autocast_disabled = torch.autocast(
                device_type=projection.device.type, enabled=False
            )
        elif projection.is_cuda:  # pragma: no cover - legacy PyTorch only
            autocast_disabled = torch.cuda.amp.autocast(enabled=False)
        else:  # pragma: no cover - legacy PyTorch only
            autocast_disabled = nullcontext()
        with autocast_disabled:
            projected_h = homogeneous @ projection.to(torch.float32).T
            denominator = projected_h[:, 3]
            denominator_valid = torch.isfinite(denominator) & (
                denominator.abs() > torch.finfo(torch.float32).eps * 16
            )
            safe_denominator = torch.where(
                denominator_valid, denominator, torch.ones_like(denominator)
            )
            projected_pair = projected_h[:, :2] / safe_denominator[:, None]

        pixels_xy = (
            projected_pair
            if self.projection_pixel_order == "xy"
            else projected_pair[:, [1, 0]]
        )
        height, width = distance_map.shape
        valid = denominator_valid & torch.isfinite(pixels_xy).all(dim=1)
        if self.distance_sampling == "nearest":
            # Mirror generator order exactly: numpy.rint first, image-index
            # bounds second. This also preserves its ties-to-even edge cases.
            rounded_xy = torch.round(pixels_xy)
            valid &= (rounded_xy[:, 0] >= 0) & (
                rounded_xy[:, 0] < width
            )
            valid &= (rounded_xy[:, 1] >= 0) & (
                rounded_xy[:, 1] < height
            )
        else:
            # Continuous grid sampling represents the exterior half-cell of
            # each edge pixel, including a meaningful singleton-axis domain.
            valid &= (pixels_xy[:, 0] >= -0.5) & (
                pixels_xy[:, 0] < width - 0.5
            )
            valid &= (pixels_xy[:, 1] >= -0.5) & (
                pixels_xy[:, 1] < height - 0.5
            )
        coords_xyz = coords_xyz[valid]
        pixels_xy = pixels_xy[valid]
        source_indices = source_indices[valid]
        if coords_xyz.numel() == 0:
            return (
                coords_xyz,
                pixels_xy,
                distance_map.new_empty((0,)),
                source_indices,
            )

        sampled_distance = self._sample_distance(distance_map, pixels_xy)
        active = torch.isfinite(sampled_distance) & (
            sampled_distance < self.max_pixel_distance
        )
        return (
            coords_xyz[active],
            pixels_xy[active],
            sampled_distance[active],
            source_indices[active],
        )

    def _reproject_voxel_features_indexed(
        self,
        coords_xyz: Tensor,
        distance_map: Tensor,
        feature_map: Tensor,
        projection: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Eq. 7 and return kept indices relative to ``coords_xyz``.

        Voxel centres are projected with ``world2pix4x4``. Encoder features are
        bilinear; EDT values use ``distance_sampling`` (nearest by default). A
        projected centre is discarded when its homogeneous projection is
        invalid, lies outside the detector's pixel-cell extent, or has sampled
        EDT greater than or equal to ``max_pixel_distance``.

        Sampling the encoder map is intentionally delayed until after the
        inexpensive EDT test. On a 400-cubed candidate grid this avoids
        materialising 12-channel features for almost every inactive voxel.
        """

        output_channels = feature_map.shape[0] + int(
            self.include_distance_feature
        )
        coords_xyz, pixels_xy, sampled_distance, source_indices = (
            self._gate_voxel_coordinates(
                coords_xyz, distance_map, projection
            )
        )
        if coords_xyz.numel() == 0:
            return (
                coords_xyz,
                feature_map.new_empty((0, output_channels)),
                source_indices,
            )

        sampled_features = self._bilinear_sample(feature_map, pixels_xy)
        if self.include_distance_feature:
            sampled_features = torch.cat(
                [
                    sampled_distance.to(sampled_features.dtype)[:, None],
                    sampled_features,
                ],
                dim=1,
            )
        return coords_xyz, sampled_features, source_indices

    def _reproject_voxel_features(
        self,
        coords_xyz: Tensor,
        distance_map: Tensor,
        feature_map: Tensor,
        projection: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply Methods Eq. 7 to unique voxel coordinates from one view."""

        kept_coordinates, sampled_features, _ = (
            self._reproject_voxel_features_indexed(
                coords_xyz, distance_map, feature_map, projection
            )
        )
        return kept_coordinates, sampled_features

    def _project_view(
        self,
        distance_map: Tensor,
        feature_map: Tensor,
        projection: Tensor,
        active_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        active = torch.isfinite(distance_map) & (
            distance_map < self.max_pixel_distance
        )
        if active_mask is not None:
            if active_mask.shape != distance_map.shape:
                raise ValueError("active mask override must match the input mask.")
            active &= torch.isfinite(active_mask) & (active_mask > 0.5)
        pixels_rc = torch.nonzero(active, as_tuple=False)
        if pixels_rc.numel() == 0:
            return (
                torch.empty(
                    (0, 3), dtype=torch.long, device=distance_map.device
                ),
                feature_map.new_empty(
                    (
                        0,
                        feature_map.shape[0]
                        + int(self.include_distance_feature),
                    )
                ),
            )

        projected_pixels = (
            pixels_rc[:, [1, 0]]
            if self.projection_pixel_order == "xy"
            else pixels_rc
        )
        pixels_float = (
            projected_pixels.to(dtype=projection.dtype)
            + self.pixel_center_offset
        )
        origins, directions = self._camera_rays(projection, pixels_float)
        near, far, valid = self._intersect_box(origins, directions)
        origins, directions = origins[valid], directions[valid]
        near, far = near[valid], far[valid]
        if origins.numel() == 0:
            return (
                torch.empty(
                    (0, 3), dtype=torch.long, device=distance_map.device
                ),
                feature_map.new_empty(
                    (
                        0,
                        feature_map.shape[0]
                        + int(self.include_distance_feature),
                    )
                ),
            )

        coordinate_chunks: list[Tensor] = []
        for start in range(0, origins.shape[0], self.ray_chunk_size):
            stop = min(start + self.ray_chunk_size, origins.shape[0])
            coordinates, _ = self._sample_ray_chunk(
                origins[start:stop],
                directions[start:stop],
                near[start:stop],
                far[start:stop],
                feature_map.new_empty((stop - start, 0)),
            )
            coordinate_chunks.append(self._deduplicate_coordinates(coordinates))
        coordinates = self._deduplicate_coordinates(
            torch.cat(coordinate_chunks, dim=0)
        )
        if coordinates.numel() == 0:
            return (
                coordinates,
                feature_map.new_empty(
                    (
                        0,
                        feature_map.shape[0]
                        + int(self.include_distance_feature),
                    )
                ),
            )

        # Reproject in bounded chunks: sparse visual hulls can contain hundreds
        # of thousands of voxels at 0.5 mm even for two views.
        reprojection_chunk_size = max(self.ray_chunk_size * 16, 65_536)
        reprojected_coordinates: list[Tensor] = []
        reprojected_features: list[Tensor] = []
        for start in range(0, coordinates.shape[0], reprojection_chunk_size):
            stop = min(start + reprojection_chunk_size, coordinates.shape[0])
            chunk_coordinates, chunk_features = (
                self._reproject_voxel_features(
                    coordinates[start:stop],
                    distance_map,
                    feature_map,
                    projection,
                )
            )
            reprojected_coordinates.append(chunk_coordinates)
            reprojected_features.append(chunk_features)
        return (
            torch.cat(reprojected_coordinates, dim=0),
            torch.cat(reprojected_features, dim=0),
        )

    def _coordinates_from_linear_range(
        self, start: int, stop: int, device: torch.device
    ) -> Tensor:
        """Decode a bounded linear range into XYZ grid coordinates."""

        _, size_y, size_z = self.spatial_shape_xyz
        linear = torch.arange(
            start, stop, dtype=torch.long, device=device
        )
        x = torch.div(linear, size_y * size_z, rounding_mode="floor")
        remainder = linear.remainder(size_y * size_z)
        y = torch.div(remainder, size_z, rounding_mode="floor")
        z = remainder.remainder(size_z)
        return torch.stack([x, y, z], dim=1)

    def _project_grid_chunk_all_views(
        self,
        coordinates: Tensor,
        distance_maps: Tensor,
        feature_maps: Tensor,
        projections: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Intersect a chunk in every view before sampling encoder features."""

        view_count = distance_maps.shape[0]
        pixel_blocks: list[Tensor] = []
        distance_blocks: list[Tensor] = []
        for view_index in range(view_count):
            coordinates, view_pixels, view_distances, kept_indices = (
                self._gate_voxel_coordinates(
                    coordinates,
                    distance_maps[view_index],
                    projections[view_index],
                )
            )
            pixel_blocks = [
                block[kept_indices] for block in pixel_blocks
            ]
            distance_blocks = [
                block[kept_indices] for block in distance_blocks
            ]
            pixel_blocks.append(view_pixels)
            distance_blocks.append(view_distances)
            if coordinates.numel() == 0:
                output_channels = self.output_channels(
                    feature_maps.shape[1], view_count
                )
                return (
                    coordinates,
                    feature_maps.new_empty((0, output_channels)),
                )

        feature_blocks: list[Tensor] = []
        for view_index in range(view_count):
            view_features = self._bilinear_sample(
                feature_maps[view_index], pixel_blocks[view_index]
            )
            if self.include_distance_feature:
                view_features = torch.cat(
                    [
                        distance_blocks[view_index].to(
                            view_features.dtype
                        )[:, None],
                        view_features,
                    ],
                    dim=1,
                )
            feature_blocks.append(view_features)

        if self.fusion == "concat":
            fused = torch.cat(feature_blocks, dim=1)
        else:
            # Views share a dtype and channel count. Accumulate low-precision
            # features in fp32, consistent with duplicate/view reductions.
            accumulation_dtype = (
                torch.float32
                if feature_blocks[0].dtype
                in (torch.float16, torch.bfloat16)
                else feature_blocks[0].dtype
            )
            fused = torch.stack(
                [block.to(accumulation_dtype) for block in feature_blocks],
                dim=0,
            ).mean(dim=0)
            fused = fused.to(feature_blocks[0].dtype)
        return coordinates, fused

    def _project_grid_chunk_partial_support(
        self,
        coordinates: Tensor,
        distance_maps: Tensor,
        feature_maps: Tensor,
        projections: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fuse a chunk when only a subset of views must support a voxel."""

        if self.fusion != "mean":
            raise ValueError(
                "Partial view support is defined only for mean fusion."
            )
        channel_count = feature_maps.shape[1] + int(
            self.include_distance_feature
        )
        feature_dtype = feature_maps.dtype
        accumulation_dtype = (
            torch.float32
            if feature_dtype in (torch.float16, torch.bfloat16)
            else feature_dtype
        )
        feature_sum = torch.zeros(
            (coordinates.shape[0], channel_count),
            device=feature_maps.device,
            dtype=accumulation_dtype,
        )
        support_count = torch.zeros(
            coordinates.shape[0],
            device=coordinates.device,
            dtype=torch.long,
        )
        for view_index in range(distance_maps.shape[0]):
            _, view_features, kept_indices = (
                self._reproject_voxel_features_indexed(
                    coordinates,
                    distance_maps[view_index],
                    feature_maps[view_index],
                    projections[view_index],
                )
            )
            feature_sum = feature_sum.index_add(
                0, kept_indices, view_features.to(accumulation_dtype)
            )
            support_count = support_count.index_add(
                0,
                kept_indices,
                torch.ones_like(kept_indices, dtype=support_count.dtype),
            )

        keep = support_count >= self.support_views
        kept_coordinates = coordinates[keep]
        if kept_coordinates.numel() == 0:
            return (
                kept_coordinates,
                feature_maps.new_empty((0, channel_count)),
            )
        fused = feature_sum[keep] / support_count[keep, None].to(
            accumulation_dtype
        )
        return kept_coordinates, fused.to(feature_dtype)

    def _project_voxel_grid(
        self,
        distance_maps: Tensor,
        feature_maps: Tensor,
        projections: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Stream the complete grid and apply Eqs. 7--9 exactly.

        No tensor proportional to the full dense grid is created. When every
        view is required (the paper's two-view intersection), each chunk is
        filtered sequentially so later feature maps see only candidates that
        survived all preceding views.
        """

        size_x, size_y, size_z = self.spatial_shape_xyz
        voxel_count = size_x * size_y * size_z
        all_coordinates: list[Tensor] = []
        all_features: list[Tensor] = []
        require_all_views = self.support_views == distance_maps.shape[0]
        for start in range(0, voxel_count, self.voxel_chunk_size):
            stop = min(start + self.voxel_chunk_size, voxel_count)
            coordinates = self._coordinates_from_linear_range(
                start, stop, distance_maps.device
            )
            if require_all_views:
                kept_coordinates, fused = (
                    self._project_grid_chunk_all_views(
                        coordinates,
                        distance_maps,
                        feature_maps,
                        projections,
                    )
                )
            else:
                kept_coordinates, fused = (
                    self._project_grid_chunk_partial_support(
                        coordinates,
                        distance_maps,
                        feature_maps,
                        projections,
                    )
                )
            all_coordinates.append(kept_coordinates)
            all_features.append(fused)

        return (
            torch.cat(all_coordinates, dim=0),
            torch.cat(all_features, dim=0),
        )

    def _fuse_views(
        self, view_results: list[tuple[Tensor, Tensor]], channels: int
    ) -> tuple[Tensor, Tensor]:
        """Fuse unique per-view voxels and filter by independent view support."""

        nonempty = [(coords, values) for coords, values in view_results if coords.numel()]
        if not nonempty:
            device = view_results[0][0].device
            dtype = view_results[0][1].dtype
            output_channels = (
                channels
                if self.fusion == "mean"
                else channels * len(view_results)
            )
            return (
                torch.empty((0, 3), dtype=torch.long, device=device),
                torch.empty(
                    (0, output_channels), dtype=dtype, device=device
                ),
            )
        if self.support_views > len(view_results):
            raise ValueError(
                f"support_views={self.support_views} exceeds the {len(view_results)} inputs."
            )

        _, size_y, size_z = self.spatial_shape_xyz
        keys_by_view = [
            (coords[:, 0] * size_y + coords[:, 1]) * size_z + coords[:, 2]
            for coords, _ in view_results
        ]
        all_keys = torch.cat(keys_by_view, dim=0)
        unique_keys, inverse, support_counts = torch.unique(
            all_keys, sorted=True, return_inverse=True, return_counts=True
        )
        keep = support_counts >= self.support_views
        kept_keys = unique_keys[keep]
        union_to_kept = torch.full(
            (unique_keys.numel(),), -1, dtype=torch.long, device=all_keys.device
        )
        union_to_kept[keep] = torch.arange(kept_keys.numel(), device=all_keys.device)

        if self.fusion == "mean":
            all_features = torch.cat([values for _, values in view_results], dim=0)
            kept_ids = union_to_kept[inverse]
            supported_rows = kept_ids >= 0
            kept_ids = kept_ids[supported_rows]
            supported_features = all_features[supported_rows]
            order = torch.argsort(kept_ids, stable=True)
            fused = self._segment_sum(
                supported_features[order], support_counts[keep]
            )
            fused = (
                fused / support_counts[keep].to(fused.dtype)[:, None]
            ).to(all_features.dtype)
        else:
            if self.support_views != len(view_results):
                raise ValueError(
                    "concat fusion requires support_views to equal the number of views."
                )
            feature_blocks: list[Tensor] = []
            offset = 0
            for _, values in view_results:
                count = values.shape[0]
                kept_ids = union_to_kept[inverse[offset : offset + count]]
                block = values.new_zeros((kept_keys.numel(), channels))
                valid = kept_ids >= 0
                block[kept_ids[valid]] = values[valid]
                feature_blocks.append(block)
                offset += count
            fused = torch.cat(feature_blocks, dim=1)

        x = torch.div(kept_keys, size_y * size_z, rounding_mode="floor")
        remainder = kept_keys.remainder(size_y * size_z)
        y = torch.div(remainder, size_z, rounding_mode="floor")
        z = remainder.remainder(size_z)
        coordinates = torch.stack([x, y, z], dim=1)
        return coordinates, fused

    def forward(
        self,
        distance_maps: Tensor,
        features: Tensor,
        world2pix4x4: Tensor,
        active_masks: Tensor | None = None,
        *,
        return_raw: bool = False,
    ):
        """Construct a sparse volume.

        Inputs have shapes ``[B,V,H,W]``, ``[B,V,C,H,W]`` and
        ``[B,V,4,4]``.  The return value remains compatible with the released
        model: ``(sparse_volume, world_coordinates)``.
        """

        if distance_maps.ndim != 4:
            raise ValueError(
                "distance_maps must be [B,V,H,W], got "
                f"{distance_maps.shape}."
            )
        if features.ndim != 5:
            raise ValueError(f"features must be [B,V,C,H,W], got {features.shape}.")
        batch_size, view_count, height, width = distance_maps.shape
        if features.shape[:2] != (batch_size, view_count) or features.shape[-2:] != (
            height,
            width,
        ):
            raise ValueError("distance_maps and features have incompatible shapes.")
        if world2pix4x4.shape != (batch_size, view_count, 4, 4):
            raise ValueError(
                "world2pix4x4 must have shape [B,V,4,4], got "
                f"{tuple(world2pix4x4.shape)}."
            )
        if active_masks is not None and active_masks.shape != distance_maps.shape:
            raise ValueError(
                "active_masks must have the same shape as distance_maps."
            )
        if self.candidate_mode == "voxel_grid" and active_masks is not None:
            raise ValueError(
                "active_masks is a legacy ray-mode override and cannot be "
                "used with candidate_mode='voxel_grid'."
            )
        if self.support_views > view_count:
            raise ValueError(
                f"support_views={self.support_views} exceeds view_count={view_count}."
            )
        if self.fusion == "concat" and self.support_views != view_count:
            raise ValueError("concat fusion requires all views to support every retained voxel.")

        all_coordinates: list[Tensor] = []
        all_features: list[Tensor] = []
        world_coordinates: list[Tensor] = []
        channels = features.shape[2] + int(self.include_distance_feature)
        for batch_index in range(batch_size):
            projections = world2pix4x4[batch_index].to(
                device=distance_maps.device, dtype=torch.float32
            )
            if self.candidate_mode == "voxel_grid":
                coords_xyz, fused = self._project_voxel_grid(
                    distance_maps[batch_index],
                    features[batch_index],
                    projections,
                )
            else:
                view_results = []
                for view_index in range(view_count):
                    view_results.append(
                        self._project_view(
                            distance_maps[batch_index, view_index],
                            features[batch_index, view_index],
                            projections[view_index],
                            None
                            if active_masks is None
                            else active_masks[batch_index, view_index],
                        )
                    )
                coords_xyz, fused = self._fuse_views(
                    view_results, channels
                )
            batch_column = torch.full(
                (coords_xyz.shape[0], 1),
                batch_index,
                dtype=torch.long,
                device=coords_xyz.device,
            )
            all_coordinates.append(torch.cat([batch_column, coords_xyz], dim=1))
            all_features.append(fused)
            bbox_min = self.bbox_min.to(device=fused.device, dtype=torch.float32)
            world_coordinates.append(
                bbox_min
                + (coords_xyz.to(torch.float32) + 0.5) * self.voxel_size
            )

        coordinates = torch.cat(all_coordinates, dim=0).to(torch.int32)
        fused_features = torch.cat(all_features, dim=0)
        world = torch.cat(world_coordinates, dim=0)
        projection = SparseProjection(
            features=fused_features,
            coordinates=coordinates,
            world_coordinates=world,
            spatial_shape_xyz=self.spatial_shape_xyz,
            voxel_size=self.voxel_size,
            batch_size=batch_size,
        )
        sparse_volume = projection if return_raw else projection.as_backend(self._resolve_backend())
        return sparse_volume, world
