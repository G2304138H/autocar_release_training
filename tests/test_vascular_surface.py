from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("skimage")

from src.geometry.vascular_surface import (
    extract_vascular_graph_surface,
    save_vascular_surface_bundle,
)


def _straight_vessel() -> np.ndarray:
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[2:7, 3:6, 3:6] = 0.9
    return volume


def test_extracts_centerline_radii_and_physical_surface():
    result = extract_vascular_graph_surface(
        _straight_vessel(),
        threshold=0.5,
        origin_xyz_mm=(-4.5, -4.5, -4.5),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
    )

    assert result.foreground_voxels == 45
    assert len(result.node_xyz_mm) >= 2
    assert len(result.edge_node_indices) >= 1
    assert result.component_count == 1
    assert np.all(result.node_radius_mm > 0.0)
    assert np.all(result.node_index_zyx >= np.asarray([2, 3, 3]))
    assert np.all(result.node_index_zyx < np.asarray([7, 6, 6]))
    assert len(result.surface_vertices_xyz_mm) > 0
    assert len(result.surface_faces) > 0
    assert result.surface_vertex_radius_mm.shape == (
        len(result.surface_vertices_xyz_mm),
    )
    np.testing.assert_allclose(
        result.surface_vertices_xyz_mm.min(axis=0),
        (-1.5, -1.5, -2.5),
    )
    np.testing.assert_allclose(
        result.surface_vertices_xyz_mm.max(axis=0),
        (1.5, 1.5, 2.5),
    )


def test_bundle_saves_graph_radius_mesh_and_visualization(tmp_path):
    output = tmp_path / "vascular_surface"
    manifest = save_vascular_surface_bundle(
        output,
        _straight_vessel(),
        case_id="17",
        threshold=0.5,
        origin_xyz_mm=(-4.5, -4.5, -4.5),
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        projection_center_offset_xyz_mm=(10.0, 20.0, 30.0),
        gif_frames=2,
        gif_fps=6,
        maximum_plot_elements=500,
    )

    assert manifest["graph_nodes"] > 0
    assert manifest["surface_faces"] > 0
    assert (output / "predicted_centerline_graph.npz").is_file()
    assert (output / "predicted_radius_surface.ply").is_file()
    assert (output / "predicted_surface_centerline_radius.png").is_file()
    assert (output / "predicted_surface_centerline_radius.gif").is_file()
    assert (
        manifest["surface_centerline_radius_gif"]
        == "predicted_surface_centerline_radius.gif"
    )
    saved_manifest = json.loads((output / "manifest.json").read_text())
    assert saved_manifest == manifest
    with np.load(output / "predicted_centerline_graph.npz") as payload:
        np.testing.assert_allclose(
            payload["node_native_xyz_mm"],
            payload["node_xyz_mm"] + np.asarray([10.0, 20.0, 30.0]),
        )
        assert payload["node_radius_mm"].shape[0] == manifest["graph_nodes"]
        assert payload["edge_node_indices"].shape[0] == manifest["graph_edges"]


def test_empty_prediction_writes_auditable_empty_bundle(tmp_path):
    output = tmp_path / "vascular_surface"
    manifest = save_vascular_surface_bundle(
        output,
        np.zeros((5, 5, 5), dtype=np.float32),
        case_id="18",
        threshold=0.5,
        origin_xyz_mm=(0.0, 0.0, 0.0),
        spacing_xyz_mm=(0.5, 0.5, 0.5),
        projection_center_offset_xyz_mm=(0.0, 0.0, 0.0),
        gif_frames=4,
        gif_fps=2,
        maximum_plot_elements=100,
    )

    assert manifest["foreground_voxels"] == 0
    assert manifest["graph_nodes"] == 0
    assert manifest["radius_colored_surface_ply"] is None
    assert manifest["surface_centerline_radius_gif"] is None
    assert (output / "predicted_centerline_graph.npz").is_file()
    assert (output / "predicted_surface_centerline_radius.png").is_file()
    assert not (output / "predicted_radius_surface.ply").exists()
