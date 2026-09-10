"""Extract and render a vascular graph/surface from a predicted voxel volume.

AutoCAR predicts occupancy on a regular grid rather than an ordered vascular
tree. This module provides derived centerline/radius and visualization
post-processing:

* threshold the probability volume;
* skeletonize the foreground into a 26-connected voxel graph;
* estimate a physical radius at every graph node with a Euclidean distance
  transform; and
* extract the predicted occupancy surface with marching cubes and colour it by
  the nearest graph-node radius.

The graph and surface remain explicitly labelled as derived post-processing;
they are not direct model outputs. Paper-metric evaluation uses paired aligned
graphs for hard clDice, while triangle surfaces remain visualization-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class VascularCenterlineGraph:
    """Morphologically thinned voxel graph with physical node radii."""

    source_shape_zyx: tuple[int, int, int]
    threshold: float
    foreground_voxels: int
    node_index_zyx: np.ndarray
    node_xyz_mm: np.ndarray
    node_radius_mm: np.ndarray
    edge_node_indices: np.ndarray
    node_degree: np.ndarray
    node_kind: np.ndarray
    component_id: np.ndarray

    @property
    def component_count(self) -> int:
        return (
            int(np.max(self.component_id)) + 1
            if self.component_id.size
            else 0
        )


@dataclass(frozen=True)
class VascularGraphSurface(VascularCenterlineGraph):
    """Voxel-derived graph and triangle surface in physical XYZ millimetres."""

    surface_vertices_xyz_mm: np.ndarray
    surface_faces: np.ndarray
    surface_vertex_radius_mm: np.ndarray


def _finite_xyz(values: Sequence[float], *, label: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain three finite XYZ values.")
    return result


def _postprocessing_dependencies():
    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
        from skimage.measure import marching_cubes
        from skimage.morphology import skeletonize
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Centerline graph and surface post-processing requires scipy and "
            "scikit-image. "
            "Install the maintained environment requirements before running "
            "evaluation."
        ) from error
    return ndimage, cKDTree, marching_cubes, skeletonize


def _empty_centerline_result(
    shape_zyx: tuple[int, int, int],
    *,
    threshold: float,
) -> VascularCenterlineGraph:
    return VascularCenterlineGraph(
        source_shape_zyx=shape_zyx,
        threshold=threshold,
        foreground_voxels=0,
        node_index_zyx=np.empty((0, 3), dtype=np.int32),
        node_xyz_mm=np.empty((0, 3), dtype=np.float32),
        node_radius_mm=np.empty((0,), dtype=np.float32),
        edge_node_indices=np.empty((0, 2), dtype=np.int32),
        node_degree=np.empty((0,), dtype=np.int16),
        node_kind=np.empty((0,), dtype=np.int8),
        component_id=np.empty((0,), dtype=np.int32),
    )


def _empty_result(
    shape_zyx: tuple[int, int, int],
    *,
    threshold: float,
) -> VascularGraphSurface:
    graph = _empty_centerline_result(shape_zyx, threshold=threshold)
    return VascularGraphSurface(
        source_shape_zyx=graph.source_shape_zyx,
        threshold=graph.threshold,
        foreground_voxels=graph.foreground_voxels,
        node_index_zyx=graph.node_index_zyx,
        node_xyz_mm=graph.node_xyz_mm,
        node_radius_mm=graph.node_radius_mm,
        edge_node_indices=graph.edge_node_indices,
        node_degree=graph.node_degree,
        node_kind=graph.node_kind,
        component_id=graph.component_id,
        surface_vertices_xyz_mm=np.empty((0, 3), dtype=np.float32),
        surface_faces=np.empty((0, 3), dtype=np.int32),
        surface_vertex_radius_mm=np.empty((0,), dtype=np.float32),
    )


def _overlap_slices(
    shape: Sequence[int], offset: Sequence[int]
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source: list[slice] = []
    target: list[slice] = []
    for size, delta in zip(shape, offset):
        if delta >= 0:
            source.append(slice(0, size - delta))
            target.append(slice(delta, size))
        else:
            source.append(slice(-delta, size))
            target.append(slice(0, size + delta))
    return tuple(source), tuple(target)


def _skeleton_edges(
    skeleton_zyx: np.ndarray,
    node_index_zyx: np.ndarray,
) -> np.ndarray:
    """Return unique 26-neighbour edges between skeleton voxels."""

    if node_index_zyx.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    node_ids = np.full(skeleton_zyx.shape, -1, dtype=np.int32)
    node_ids[tuple(node_index_zyx.T)] = np.arange(
        len(node_index_zyx), dtype=np.int32
    )
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
        and (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
    ]
    parts: list[np.ndarray] = []
    for offset in offsets:
        source_slice, target_slice = _overlap_slices(
            skeleton_zyx.shape, offset
        )
        connected = (
            skeleton_zyx[source_slice] & skeleton_zyx[target_slice]
        )
        if not np.any(connected):
            continue
        source_ids = node_ids[source_slice][connected]
        target_ids = node_ids[target_slice][connected]
        parts.append(np.column_stack((source_ids, target_ids)).astype(np.int32))
    return (
        np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, 2), dtype=np.int32)
    )


def _tight_padded_foreground(mask_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied_axes = (
        np.flatnonzero(np.any(mask_zyx, axis=(1, 2))),
        np.flatnonzero(np.any(mask_zyx, axis=(0, 2))),
        np.flatnonzero(np.any(mask_zyx, axis=(0, 1))),
    )
    crop_min_zyx = np.asarray(
        [indices[0] for indices in occupied_axes], dtype=np.int64
    )
    crop_max_zyx = np.asarray(
        [indices[-1] + 1 for indices in occupied_axes], dtype=np.int64
    )
    crop_slices = tuple(
        slice(int(minimum), int(maximum))
        for minimum, maximum in zip(crop_min_zyx, crop_max_zyx)
    )
    padded = np.pad(
        mask_zyx[crop_slices], 1, mode="constant", constant_values=False
    )
    return padded, crop_min_zyx


def extract_centerline_graph(
    prediction_zyx: np.ndarray,
    *,
    threshold: float,
    origin_xyz_mm: Sequence[float],
    spacing_xyz_mm: Sequence[float],
) -> VascularCenterlineGraph:
    """Morphologically thin a volume and estimate graph-node radii with EDT."""

    volume = np.asarray(prediction_zyx)
    if volume.ndim != 3 or min(volume.shape, default=0) <= 0:
        raise ValueError(
            f"prediction_zyx must be a non-empty 3D array, got {volume.shape}."
        )
    is_numeric = np.issubdtype(volume.dtype, np.number) or np.issubdtype(
        volume.dtype, np.bool_
    )
    if not is_numeric or not np.isfinite(volume).all():
        raise ValueError("prediction_zyx must contain finite numeric values.")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    origin_xyz = _finite_xyz(origin_xyz_mm, label="origin_xyz_mm")
    spacing_xyz = _finite_xyz(spacing_xyz_mm, label="spacing_xyz_mm")
    if np.any(spacing_xyz <= 0.0):
        raise ValueError("spacing_xyz_mm must be strictly positive.")

    mask = volume >= threshold
    foreground_voxels = int(np.count_nonzero(mask))
    shape_zyx = tuple(int(value) for value in volume.shape)
    if foreground_voxels == 0:
        return _empty_centerline_result(shape_zyx, threshold=threshold)

    ndimage, _, _, skeletonize = _postprocessing_dependencies()
    # The explicit false border makes the foreground closed even when it touches
    # the AutoCAR reconstruction boundary and keeps the expensive operations on
    # the tight foreground box rather than the complete 400^3 prediction grid.
    padded_mask, crop_min_zyx = _tight_padded_foreground(mask)
    skeleton = np.asarray(
        skeletonize(padded_mask, method="lee"), dtype=np.bool_
    )
    spacing_zyx = spacing_xyz[::-1]
    distance_to_background_mm = ndimage.distance_transform_edt(
        padded_mask,
        sampling=tuple(float(value) for value in spacing_zyx),
    )

    padded_node_index_zyx = np.argwhere(skeleton).astype(np.int32)
    if padded_node_index_zyx.size == 0:
        # Degenerate components should still have a graph node. Choose the most
        # interior foreground voxel as a transparent, deterministic fallback.
        maximum = np.unravel_index(
            int(np.argmax(distance_to_background_mm)),
            distance_to_background_mm.shape,
        )
        padded_node_index_zyx = np.asarray([maximum], dtype=np.int32)
        skeleton[maximum] = True

    edges = _skeleton_edges(skeleton, padded_node_index_zyx)
    degrees = np.zeros(len(padded_node_index_zyx), dtype=np.int16)
    if edges.size:
        np.add.at(degrees, edges[:, 0], 1)
        np.add.at(degrees, edges[:, 1], 1)
    # 0=isolated, 1=endpoint, 2=regular chain node, 3=junction.
    node_kind = np.where(
        degrees == 0,
        0,
        np.where(degrees == 1, 1, np.where(degrees == 2, 2, 3)),
    ).astype(np.int8)
    component_labels, _ = ndimage.label(
        skeleton,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    component_id = (
        component_labels[tuple(padded_node_index_zyx.T)].astype(np.int32) - 1
    )
    node_radius_mm = distance_to_background_mm[
        tuple(padded_node_index_zyx.T)
    ].astype(np.float32)

    original_node_index_zyx = (
        padded_node_index_zyx
        + crop_min_zyx.astype(np.int32)[None, :]
        - 1
    )
    node_xyz_mm = (
        origin_xyz[None, :]
        + (original_node_index_zyx[:, ::-1] + 0.5)
        * spacing_xyz[None, :]
    ).astype(np.float32)

    return VascularCenterlineGraph(
        source_shape_zyx=shape_zyx,
        threshold=threshold,
        foreground_voxels=foreground_voxels,
        node_index_zyx=original_node_index_zyx.astype(np.int32),
        node_xyz_mm=node_xyz_mm,
        node_radius_mm=node_radius_mm,
        edge_node_indices=edges,
        node_degree=degrees,
        node_kind=node_kind,
        component_id=component_id,
    )


def extract_vascular_graph_surface(
    prediction_zyx: np.ndarray,
    *,
    threshold: float,
    origin_xyz_mm: Sequence[float],
    spacing_xyz_mm: Sequence[float],
) -> VascularGraphSurface:
    """Derive a skeleton graph, radii, and occupancy surface from a prediction."""

    graph = extract_centerline_graph(
        prediction_zyx,
        threshold=threshold,
        origin_xyz_mm=origin_xyz_mm,
        spacing_xyz_mm=spacing_xyz_mm,
    )
    if graph.foreground_voxels == 0:
        return _empty_result(graph.source_shape_zyx, threshold=graph.threshold)

    mask = np.asarray(prediction_zyx) >= graph.threshold
    padded_mask, crop_min_zyx = _tight_padded_foreground(mask)
    _, cKDTree, marching_cubes, _ = _postprocessing_dependencies()
    origin_xyz = _finite_xyz(origin_xyz_mm, label="origin_xyz_mm")
    spacing_xyz = _finite_xyz(spacing_xyz_mm, label="spacing_xyz_mm")
    spacing_zyx = spacing_xyz[::-1]
    vertices_zyx_mm, faces, _, _ = marching_cubes(
        padded_mask.astype(np.float32),
        level=0.5,
        spacing=tuple(float(value) for value in spacing_zyx),
        allow_degenerate=False,
    )
    # marching_cubes treats array samples as lying at integer coordinates. The
    # padded sample at index zero represents a virtual voxel one position below
    # crop_min, and every real voxel sample is located at its physical center.
    padded_sample_zero_zyx_mm = (
        crop_min_zyx.astype(np.float64) - 1.0 + 0.5
    ) * spacing_zyx
    vertices_zyx_mm += padded_sample_zero_zyx_mm[None, :]
    surface_vertices_xyz_mm = (
        vertices_zyx_mm[:, ::-1] + origin_xyz[None, :]
    ).astype(np.float32)
    surface_faces = np.asarray(faces, dtype=np.int32)
    _, nearest_node = cKDTree(graph.node_xyz_mm).query(
        surface_vertices_xyz_mm,
        k=1,
        workers=-1,
    )
    surface_vertex_radius_mm = graph.node_radius_mm[
        np.asarray(nearest_node, dtype=np.int64)
    ].astype(np.float32)

    return VascularGraphSurface(
        source_shape_zyx=graph.source_shape_zyx,
        threshold=graph.threshold,
        foreground_voxels=graph.foreground_voxels,
        node_index_zyx=graph.node_index_zyx,
        node_xyz_mm=graph.node_xyz_mm,
        node_radius_mm=graph.node_radius_mm,
        edge_node_indices=graph.edge_node_indices,
        node_degree=graph.node_degree,
        node_kind=graph.node_kind,
        component_id=graph.component_id,
        surface_vertices_xyz_mm=surface_vertices_xyz_mm,
        surface_faces=surface_faces,
        surface_vertex_radius_mm=surface_vertex_radius_mm,
    )


def _radius_rgb(radius_mm: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(radius_mm, dtype=np.float32)
    finite = values[np.isfinite(values)]
    minimum = float(finite.min()) if finite.size else 0.0
    maximum = float(finite.max()) if finite.size else 1.0
    if maximum <= minimum:
        maximum = minimum + max(abs(minimum) * 0.05, 1e-3)
    normalized = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    anchors = np.asarray(
        [[215, 48, 39], [254, 224, 139], [26, 152, 80]],
        dtype=np.float32,
    )
    lower = normalized <= 0.5
    blend = np.where(lower, normalized * 2.0, (normalized - 0.5) * 2.0)
    colors = np.empty((len(values), 3), dtype=np.float32)
    colors[lower] = (
        anchors[0]
        + blend[lower, None] * (anchors[1] - anchors[0])
    )
    colors[~lower] = (
        anchors[1]
        + blend[~lower, None] * (anchors[2] - anchors[1])
    )
    return np.rint(colors).astype(np.uint8), minimum, maximum


def save_vascular_graph_npz(
    path: Path,
    result: VascularCenterlineGraph,
    *,
    projection_center_offset_xyz_mm: Sequence[float],
) -> None:
    """Save the graph and its radii without conflating it with model output."""

    center_offset = _finite_xyz(
        projection_center_offset_xyz_mm,
        label="projection_center_offset_xyz_mm",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        representation=np.asarray("voxel_skeleton_graph"),
        derivation=np.asarray("threshold_skeletonize_26n_edt_radius"),
        coordinate_frame=np.asarray("projection_centered_xyz_mm"),
        node_index_axis_order=np.asarray("zyx"),
        node_index_zyx=result.node_index_zyx,
        node_xyz_mm=result.node_xyz_mm,
        node_native_xyz_mm=(
            result.node_xyz_mm.astype(np.float64) + center_offset[None, :]
        ).astype(np.float32),
        node_radius_mm=result.node_radius_mm,
        edge_node_indices=result.edge_node_indices,
        node_degree=result.node_degree,
        node_kind=result.node_kind,
        node_kind_labels=np.asarray(
            ["isolated", "endpoint", "regular", "junction"]
        ),
        component_id=result.component_id,
        source_volume_shape_zyx=np.asarray(
            result.source_shape_zyx, dtype=np.int32
        ),
        prediction_threshold=np.asarray(result.threshold, dtype=np.float32),
        foreground_voxels=np.asarray(result.foreground_voxels, dtype=np.int64),
        projection_center_offset_xyz_mm=center_offset.astype(np.float32),
    )


def save_radius_colored_surface_ply(
    path: Path,
    result: VascularGraphSurface,
) -> None:
    """Write a binary PLY mesh with radius and radius-derived vertex colours."""

    vertices = np.asarray(result.surface_vertices_xyz_mm, dtype=np.float32)
    faces = np.asarray(result.surface_faces, dtype=np.int32)
    radii = np.asarray(result.surface_vertex_radius_mm, dtype=np.float32)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError("Cannot save a surface PLY for an empty prediction.")
    colors, _, _ = _radius_rgb(radii)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("radius_mm", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_records = np.empty(len(vertices), dtype=vertex_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertex_records[name] = vertices[:, axis]
    vertex_records["radius_mm"] = radii
    vertex_records["red"] = colors[:, 0]
    vertex_records["green"] = colors[:, 1]
    vertex_records["blue"] = colors[:, 2]
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_records = np.empty(len(faces), dtype=face_dtype)
    face_records["count"] = 3
    face_records["indices"] = faces
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment AutoCAR voxel-derived occupancy surface\n"
        "comment coordinate_frame projection_centered_xyz_mm\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float radius_mm\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        vertex_records.tofile(handle)
        face_records.tofile(handle)


def _subsample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _set_equal_axes(axis: Any, points_xyz: np.ndarray) -> None:
    if points_xyz.size == 0:
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_zlim(-1.0, 1.0)
        return
    minimum = np.nanmin(points_xyz, axis=0)
    maximum = np.nanmax(points_xyz, axis=0)
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)) * 0.55, 0.5)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_graph_surface(
    axis: Any,
    result: VascularGraphSurface,
    *,
    maximum_faces: int,
    maximum_edges: int,
) -> tuple[Any, float, float]:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    vertices = result.surface_vertices_xyz_mm
    faces = result.surface_faces
    face_indices = _subsample_indices(len(faces), maximum_faces)
    plotted_faces = faces[face_indices]
    surface_radius = result.surface_vertex_radius_mm
    _, radius_min, radius_max = _radius_rgb(result.node_radius_mm)
    colormap = LinearSegmentedColormap.from_list(
        "autocar_radius", ["#d73027", "#fee08b", "#1a9850"]
    )
    normalization = Normalize(vmin=radius_min, vmax=radius_max, clip=True)
    if len(plotted_faces):
        face_radius = surface_radius[plotted_faces].mean(axis=1)
        collection = Poly3DCollection(
            vertices[plotted_faces],
            facecolors=colormap(normalization(face_radius)),
            edgecolors="none",
            alpha=0.64,
        )
        axis.add_collection3d(collection)

    edges = result.edge_node_indices
    edge_indices = _subsample_indices(len(edges), maximum_edges)
    plotted_edges = edges[edge_indices]
    if len(plotted_edges):
        segments = result.node_xyz_mm[plotted_edges]
        lines = Line3DCollection(
            segments,
            colors="#17202a",
            linewidths=0.7,
            alpha=0.92,
        )
        axis.add_collection3d(lines)
    critical = result.node_kind != 2
    if np.any(critical):
        critical_indices = np.flatnonzero(critical)
        critical_indices = critical_indices[
            _subsample_indices(len(critical_indices), min(maximum_edges, 3000))
        ]
        axis.scatter(
            result.node_xyz_mm[critical_indices, 0],
            result.node_xyz_mm[critical_indices, 1],
            result.node_xyz_mm[critical_indices, 2],
            c=result.node_radius_mm[critical_indices],
            cmap=colormap,
            norm=normalization,
            s=5.0,
            edgecolors="#111111",
            linewidths=0.25,
            depthshade=False,
        )
    _set_equal_axes(axis, vertices if len(vertices) else result.node_xyz_mm)
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    scalar = ScalarMappable(norm=normalization, cmap=colormap)
    scalar.set_array(result.node_radius_mm)
    return scalar, radius_min, radius_max


def save_graph_surface_visualization(
    path: Path,
    result: VascularGraphSurface,
    *,
    case_id: str,
    maximum_faces: int,
    maximum_edges: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.2, 6.4), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    if result.foreground_voxels == 0:
        axis.text2D(
            0.5,
            0.5,
            f"No predicted foreground at threshold {result.threshold:g}",
            transform=axis.transAxes,
            ha="center",
            va="center",
        )
        axis.set_axis_off()
    else:
        scalar, _, _ = _plot_graph_surface(
            axis,
            result,
            maximum_faces=maximum_faces,
            maximum_edges=maximum_edges,
        )
        colorbar = figure.colorbar(scalar, ax=axis, shrink=0.64, pad=0.02)
        colorbar.set_label("Centerline radius estimate (mm)")
        axis.view_init(elev=24.0, azim=35.0)
    axis.set_title(
        f"Case {case_id}: predicted surface and centerline-radius graph"
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_graph_surface_gif(
    path: Path,
    result: VascularGraphSurface,
    *,
    case_id: str,
    frames: int,
    fps: int,
    maximum_faces: int,
    maximum_edges: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    if result.foreground_voxels == 0:
        raise ValueError("Cannot animate an empty vascular surface.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.0, 6.4))
    axis = figure.add_subplot(111, projection="3d")
    scalar, _, _ = _plot_graph_surface(
        axis,
        result,
        maximum_faces=maximum_faces,
        maximum_edges=maximum_edges,
    )
    colorbar = figure.colorbar(scalar, ax=axis, shrink=0.64, pad=0.02)
    colorbar.set_label("Centerline radius estimate (mm)")
    axis.set_title(
        f"Case {case_id}: predicted surface and centerline-radius graph"
    )

    def rotate(frame_index: int):
        angle = 2.0 * np.pi * float(frame_index) / float(frames)
        axis.view_init(
            elev=24.0 + 5.0 * np.sin(angle),
            azim=360.0 * float(frame_index) / float(frames),
        )
        return (axis,)

    animation = FuncAnimation(figure, rotate, frames=frames, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(figure)


def save_vascular_surface_bundle(
    path: Path,
    prediction_zyx: np.ndarray,
    *,
    case_id: str,
    threshold: float,
    origin_xyz_mm: Sequence[float],
    spacing_xyz_mm: Sequence[float],
    projection_center_offset_xyz_mm: Sequence[float],
    gif_frames: int,
    gif_fps: int,
    maximum_plot_elements: int,
) -> dict[str, Any]:
    """Save graph, radius-aware mesh, and visualizations for one prediction."""

    path.mkdir(parents=True, exist_ok=True)
    result = extract_vascular_graph_surface(
        prediction_zyx,
        threshold=threshold,
        origin_xyz_mm=origin_xyz_mm,
        spacing_xyz_mm=spacing_xyz_mm,
    )
    graph_name = "predicted_centerline_graph.npz"
    surface_name = "predicted_radius_surface.ply"
    image_name = "predicted_surface_centerline_radius.png"
    gif_name = (
        "predicted_surface_centerline_radius.gif"
        if gif_frames > 0 and result.foreground_voxels > 0
        else None
    )
    save_vascular_graph_npz(
        path / graph_name,
        result,
        projection_center_offset_xyz_mm=projection_center_offset_xyz_mm,
    )
    if result.foreground_voxels > 0:
        save_radius_colored_surface_ply(path / surface_name, result)
    save_graph_surface_visualization(
        path / image_name,
        result,
        case_id=case_id,
        maximum_faces=max(1, int(maximum_plot_elements)),
        maximum_edges=max(1, int(maximum_plot_elements)),
    )
    if gif_name is not None:
        save_graph_surface_gif(
            path / gif_name,
            result,
            case_id=case_id,
            frames=int(gif_frames),
            fps=int(gif_fps),
            maximum_faces=max(1, int(maximum_plot_elements)),
            maximum_edges=max(1, int(maximum_plot_elements)),
        )
    manifest = {
        "schema_version": 1,
        "case_id": str(case_id),
        "derivation": "threshold_skeletonize_26n_edt_radius_and_marching_cubes",
        "quantitative_metrics_use_this_postprocessing": False,
        "coordinate_frame": "projection_centered_xyz_mm",
        "prediction_threshold": float(threshold),
        "foreground_voxels": result.foreground_voxels,
        "graph_nodes": int(len(result.node_xyz_mm)),
        "graph_edges": int(len(result.edge_node_indices)),
        "graph_components": result.component_count,
        "surface_vertices": int(len(result.surface_vertices_xyz_mm)),
        "surface_faces": int(len(result.surface_faces)),
        "centerline_graph": graph_name,
        "radius_colored_surface_ply": (
            surface_name if result.foreground_voxels > 0 else None
        ),
        "surface_centerline_radius_image": image_name,
        "surface_centerline_radius_gif": gif_name,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest
