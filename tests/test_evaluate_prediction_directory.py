from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("scipy")
pytest.importorskip("skimage")

from src.evaluate_prediction_directory import (
    _RAW_VESSEL_AUTO_KEYS,
    _discover_npz_by_key,
    _discover_prediction_npzs,
    _load_normalized_prediction,
    load_raw_vessel_code,
    main,
)


def _load_prediction_adapter(path: Path, **overrides):
    options = {
        "requested_method": "auto",
        "prediction_key": None,
        "coordinate_frame": "auto",
        "origin_convention": "auto",
        "three_dgrcar_alignment": "auto",
        "case_id_override": None,
    }
    options.update(overrides)
    return _load_normalized_prediction(path, **options)


def test_deepca_adapter_uses_exporter_grid_contract_and_infers_center(tmp_path):
    path = tmp_path / "deepca_case.npz"
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[1, 1, 1] = 1
    np.savez_compressed(
        path,
        vol=volume,
        spacing=np.asarray([1.0, 1.0, 1.0]),
        origin=np.asarray([10.0, 20.0, 30.0]),
        axis_order=np.asarray("ZYX"),
        case_id=np.asarray("lca_0004"),
        view_indices=np.asarray([0, 1]),
        checkpoint=np.asarray("/checkpoints/best.pt"),
    )

    prediction = _load_prediction_adapter(path)

    assert prediction.method == "deepca"
    assert prediction.volume_is_binary_mask is True
    assert prediction.coordinate_frame == "projection_centered"
    assert prediction.origin_convention == "voxel_center"
    np.testing.assert_allclose(
        prediction.metadata["projection_center_offset_xyz_mm"],
        [11.5, 21.5, 31.5],
    )
    np.testing.assert_allclose(prediction.source_grid.origin_xyz_mm, [-2, -2, -2])
    assert prediction.checkpoint == "/checkpoints/best.pt"


def test_3dgrcar_legacy_file_requires_alignment_and_case_override(tmp_path):
    path = tmp_path / "3dgrcar_pred.npz"
    mask = np.zeros((4, 4, 4), dtype=np.bool_)
    mask[1, 1, 1] = True
    np.savez_compressed(
        path,
        prediction_mask_zyx=mask,
        ground_truth_spacing_xyz_m=np.asarray([0.001, 0.001, 0.001]),
        ground_truth_origin_xyz_m=np.asarray([0.0, 0.0, 0.0]),
        projection_center_offset_xyz_m=np.asarray([0.003, 0.006, 0.009]),
        applied_prediction_shift_zyx_voxels=np.asarray([9.0, 6.0, 3.0]),
    )

    with pytest.raises(ValueError, match="omits 3DGR-CAR ground-truth alignment"):
        _load_prediction_adapter(path, case_id_override="7")

    prediction = _load_prediction_adapter(
        path,
        three_dgrcar_alignment="physical",
        case_id_override="7",
    )
    assert prediction.method == "3dgrcar"
    assert prediction.metadata["case_id"] == "7"
    assert prediction.metadata["case_id_source"] == "cli_override"
    assert prediction.coordinate_frame == "projection_centered"
    assert prediction.volume_key == "prediction_mask_zyx"
    np.testing.assert_allclose(
        prediction.source_grid.spacing_xyz_mm, [1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(prediction.source_grid.origin_xyz_mm, [-2, -2, -2])

    discovered = _discover_prediction_npzs(
        path,
        requested_method="3dgrcar",
        prediction_key=None,
        case_id_override="7",
    )
    assert discovered == {"7": path.resolve()}


def _save_prediction(
    path: Path,
    volume_zyx: np.ndarray,
    *,
    case_id: str = "1",
    dataset_split: str = "validation",
    view_indices: tuple[int, int] = (0, 1),
    voxel_size_mm: float = 0.5,
    bbox_min_xyz_mm: tuple[float, float, float] = (-2.0, -2.0, -2.0),
    center_offset_xyz_mm: tuple[float, float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox_min = np.asarray(bbox_min_xyz_mm, dtype=np.float32)
    shape_xyz = np.asarray(volume_zyx.shape[::-1], dtype=np.float32)
    payload: dict[str, np.ndarray] = {
        "prediction_volume_zyx": np.asarray(volume_zyx, dtype=np.float32),
        "volume_axis_order": np.asarray("zyx"),
        "bbox_min_xyz_mm": bbox_min,
        "bbox_max_xyz_mm": bbox_min + shape_xyz * voxel_size_mm,
        "voxel_size_mm": np.asarray(voxel_size_mm, dtype=np.float32),
        "case_id": np.asarray(case_id),
        "dataset_split": np.asarray(dataset_split),
        "view_indices": np.asarray(view_indices, dtype=np.int64),
    }
    if center_offset_xyz_mm is not None:
        payload["projection_center_offset_xyz_mm"] = np.asarray(
            center_offset_xyz_mm, dtype=np.float32
        )
    np.savez_compressed(path, **payload)


def _save_raw_vessel(
    path: Path,
    points_xyz_mm: np.ndarray,
    *,
    radius_mm: float,
    coordinate_frame: str,
    case_id: str = "1",
    center_offset_xyz_mm: tuple[float, float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_xyz_mm, dtype=np.float32)
    vessel = np.concatenate(
        (
            points,
            np.full((len(points), 1), radius_mm, dtype=np.float32),
        ),
        axis=1,
    )[None, ...]
    payload: dict[str, np.ndarray] = {
        "raw_vessel_code_mm": vessel,
        "branch_exists": np.asarray([True]),
        "point_valid_mask": np.ones((1, len(points)), dtype=np.bool_),
        "coordinate_frame": np.asarray(coordinate_frame),
        "case_id": np.asarray(case_id),
    }
    if center_offset_xyz_mm is not None:
        payload["projection_center_offset_xyz_mm"] = np.asarray(
            center_offset_xyz_mm, dtype=np.float32
        )
    np.savez_compressed(path, **payload)


def _save_legacy_stage2_artery(
    path: Path,
    points_xyz_mm: np.ndarray,
    *,
    radius_mm: float,
    center_offset_xyz_mm: tuple[float, float, float],
    case_id: str = "1",
    artery_type: str = "lca",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    branch_capacity = 13 if artery_type == "lca" else 7
    artery = np.zeros((branch_capacity, 200, 4), dtype=np.float32)
    points = np.asarray(points_xyz_mm, dtype=np.float32)
    artery[0, : len(points), :3] = points * np.float32(1.0e-3)
    artery[0, : len(points), 3] = np.float32(radius_mm * 1.0e-3)
    np.savez_compressed(
        path,
        sample_name=np.asarray(f"{artery_type}_{int(case_id):04d}"),
        vessel_type=np.asarray(artery_type),
        artery=artery,
        projection_center_offset=(
            np.asarray(center_offset_xyz_mm, dtype=np.float32)
            * np.float32(1.0e-3)
        ),
        input_scale_to_mm=np.asarray(1000.0, dtype=np.float32),
        # Real legacy files contain this pickle-backed field. The evaluator
        # must ignore it and use the safe sample_name instead.
        source_case_id=np.asarray([case_id], dtype=object),
    )


def _run_directory_evaluation(
    prediction_dir: Path,
    raw_vessel_dir: Path,
    output_dir: Path,
    *extra_args: str,
) -> None:
    assert (
        main(
            [
                "--artery",
                "lca",
                "--prediction-dir",
                str(prediction_dir),
                "--raw-vessel-dir",
                str(raw_vessel_dir),
                "--output-dir",
                str(output_dir),
                "--evaluation-bbox-min-xyz-mm",
                "-2",
                "-2",
                "-2",
                "--evaluation-bbox-max-xyz-mm",
                "2",
                "2",
                "2",
                "--save-masks",
                *extra_args,
            ]
        )
        == 0
    )


def test_legacy_stage2_artery_is_auto_detected_without_pickle(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"
    center_offset = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    centered_points = np.column_stack(
        (
            np.arange(-1.75, 1.5, 0.5, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
        )
    )
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, :7] = 1.0
    _save_prediction(prediction_dir / "validation" / "1.npz", prediction)
    raw_path = raw_dir / "lca" / "lca_0001.npz"
    _save_legacy_stage2_artery(
        raw_path,
        centered_points + center_offset[None, :],
        radius_mm=0.24,
        center_offset_xyz_mm=tuple(float(value) for value in center_offset),
    )
    _save_legacy_stage2_artery(
        raw_dir / "rca" / "rca_0001.npz",
        centered_points + center_offset[None, :],
        radius_mm=0.24,
        center_offset_xyz_mm=tuple(float(value) for value in center_offset),
        artery_type="rca",
    )

    discovered = _discover_npz_by_key(
        raw_dir,
        required_keys=_RAW_VESSEL_AUTO_KEYS,
        label="raw vessel",
        artery="lca",
    )
    assert discovered == {"1": raw_path.resolve()}
    raw = load_raw_vessel_code(
        raw_path,
        vessel_key="auto",
        branch_exists_key="branch_exists",
        point_valid_key="point_valid_mask",
        coordinate_frame="auto",
        scale_to_mm=None,
    )
    assert raw.case_id == "1"
    assert raw.vessel_key == "artery"
    assert raw.coordinate_frame == "native"
    assert raw.scale_to_mm == 1000.0
    assert raw.scale_source == "verified_schema:artery"
    assert raw.branch_exists.tolist() == [True] + [False] * 12
    assert int(raw.active_point_mask.sum()) == 7
    np.testing.assert_allclose(
        raw.vessel_xyzr_mm[0, :7, :3],
        centered_points + center_offset[None, :],
        atol=1e-5,
    )

    _run_directory_evaluation(prediction_dir, raw_dir, output_dir)

    record = json.loads((output_dir / "per_case_metrics.json").read_text())[0]
    assert record["raw_vessel_key"] == "artery"
    assert record["raw_vessel_scale_to_mm"] == 1000.0
    assert record["raw_vessel_scale_source"] == "verified_schema:artery"
    assert record["projection_center_offset_source"] == "raw_vessel_npz"
    assert record["raw_active_branches"] == 1
    assert record["raw_active_points"] == 7
    assert record["dice_3d"] == pytest.approx(1.0)
    assert record["cldice_3d"] == pytest.approx(1.0)


def test_native_imagecas_raw_key_is_auto_detected_in_millimetres(tmp_path):
    path = tmp_path / "lca" / "31" / "vessel_code.npz"
    path.parent.mkdir(parents=True)
    vessel = np.asarray(
        [[1.0, 2.0, 3.0, 0.5], [2.0, 2.0, 3.0, 0.4]],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        branches_xyzr_resampled=vessel,
        branch_exists=np.asarray([True]),
        point_valid_mask=np.asarray([True, True]),
        # This Stage-2-style metadata describes the separate metre-valued
        # artery field, not this fixed-mm raw array, and must not rescale it.
        input_scale_to_mm=np.asarray(1000.0, dtype=np.float32),
    )

    raw = load_raw_vessel_code(
        path,
        vessel_key="auto",
        branch_exists_key="branch_exists",
        point_valid_key="point_valid_mask",
        coordinate_frame="auto",
        scale_to_mm=None,
    )

    assert raw.case_id == "31"
    assert raw.vessel_key == "branches_xyzr_resampled"
    assert raw.vessel_xyzr_mm.shape == (1, 2, 4)
    assert raw.coordinate_frame == "native"
    assert raw.scale_to_mm == 1.0
    assert raw.branch_exists.tolist() == [True]
    assert raw.point_valid.shape == (1, 2)
    assert raw.point_valid.tolist() == [[True, True]]
    np.testing.assert_allclose(raw.vessel_xyzr_mm[0], vessel)


def test_legacy_artery_rejects_conflicting_scale_metadata(tmp_path):
    path = tmp_path / "lca_0001.npz"
    artery = np.zeros((13, 200, 4), dtype=np.float32)
    artery[0, :2] = np.asarray(
        [[0.001, 0.002, 0.003, 0.0005], [0.002, 0.002, 0.003, 0.0004]],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        sample_name=np.asarray("lca_0001"),
        artery=artery,
        input_scale_to_mm=np.asarray(1.0, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="conflicts with the verified 'artery'"):
        load_raw_vessel_code(
            path,
            vessel_key="auto",
            branch_exists_key="branch_exists",
            point_valid_key="point_valid_mask",
            coordinate_frame="auto",
            scale_to_mm=None,
        )


def test_exact_native_vessel_uses_offset_and_writes_complete_outputs(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"

    # At 0.5-mm spacing these are seven consecutive voxel centres. The raw
    # vessel is stored in native coordinates, while the prediction is stored
    # in the projection-centred frame.
    center_offset = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    centered_points = np.column_stack(
        (
            np.arange(-1.75, 1.5, 0.5, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
        )
    )
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, :7] = 1.0
    _save_prediction(
        prediction_dir / "validation" / "1.npz",
        prediction,
        center_offset_xyz_mm=tuple(float(value) for value in center_offset),
    )
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        centered_points + center_offset[None, :],
        radius_mm=0.24,
        coordinate_frame="native",
    )

    _run_directory_evaluation(prediction_dir, raw_dir, output_dir)

    records = json.loads((output_dir / "per_case_metrics.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["voxel_size_mm"] == 0.5
    assert record["source_prediction_voxel_size_mm"] == 0.5
    assert record["evaluation_shape_zyx"] == [8, 8, 8]
    assert record["projection_center_offset_source"] == "prediction_npz"
    np.testing.assert_allclose(
        record["projection_center_offset_xyz_mm"], center_offset
    )
    assert record["dice_3d"] == pytest.approx(1.0)
    assert record["cldice_3d"] == pytest.approx(1.0)
    assert record["centerline_voxel_dice_3d"] == pytest.approx(1.0)
    assert record["centerline_chamfer_distance_mm"] == pytest.approx(
        0.0, abs=1e-6
    )

    summary = json.loads((output_dir / "evaluation_summary.json").read_text())
    assert summary["protocol"]["evaluation_voxel_spacing_xyz_mm"] == [
        0.5,
        0.5,
        0.5,
    ]
    assert summary["macro_dice_3d"] == pytest.approx(1.0)
    assert summary["macro_cldice_3d"] == pytest.approx(1.0)
    assert summary["macro_centerline_chamfer_distance_mm"] == pytest.approx(
        0.0, abs=1e-6
    )

    graph_path = output_dir / "graphs" / "validation" / "1.npz"
    mask_path = output_dir / "masks" / "validation" / "1.npz"
    with np.load(graph_path, allow_pickle=False) as graph:
        assert graph["comparison_coordinate_frame"].item() == "native_xyz_mm"
        assert (
            graph["evaluation_grid_coordinate_frame"].item()
            == "projection_centered_xyz_mm"
        )
        np.testing.assert_allclose(
            graph["evaluation_voxel_spacing_xyz_mm"], [0.5, 0.5, 0.5]
        )
        np.testing.assert_array_equal(graph["evaluation_shape_zyx"], [8, 8, 8])
        np.testing.assert_allclose(
            graph["prediction_node_xyz_mm"],
            graph["prediction_node_projection_centered_xyz_mm"]
            + center_offset[None, :],
        )
        np.testing.assert_allclose(
            graph["prediction_node_xyz_mm"],
            graph["ground_truth_node_xyz_mm"],
            atol=1e-6,
        )
    with np.load(mask_path, allow_pickle=False) as masks:
        assert float(masks["voxel_size_mm"]) == 0.5
        assert masks["prediction_mask_zyx"].shape == (8, 8, 8)
        np.testing.assert_array_equal(
            masks["prediction_mask_zyx"], masks["ground_truth_mask_zyx"]
        )

    assert (output_dir / "per_case_metrics.csv").is_file()
    assert (output_dir / "evaluation_summary.csv").is_file()
    assert (output_dir / "timing_per_case.csv").is_file()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["num_cases"] == 1
    assert manifest["masks"] == "masks/<split>/<case_id>.npz"


def test_one_mm_prediction_is_thresholded_then_resampled_to_half_mm(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"

    prediction = np.zeros((4, 4, 4), dtype=np.float32)
    prediction[1, 1, 1] = 0.75
    _save_prediction(
        prediction_dir / "validation" / "1.npz",
        prediction,
        voxel_size_mm=1.0,
    )
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        np.asarray([[-0.75, -0.75, -0.75]], dtype=np.float32),
        radius_mm=0.2,
        coordinate_frame="projection_centered",
    )

    _run_directory_evaluation(prediction_dir, raw_dir, output_dir)

    record = json.loads((output_dir / "per_case_metrics.json").read_text())[0]
    assert record["source_prediction_shape_zyx"] == [4, 4, 4]
    assert record["source_prediction_voxel_size_mm"] == 1.0
    assert record["evaluation_shape_zyx"] == [8, 8, 8]
    assert record["voxel_size_mm"] == 0.5

    expected = np.zeros((8, 8, 8), dtype=np.uint8)
    expected[2:4, 2:4, 2:4] = 1
    with np.load(
        output_dir / "masks" / "validation" / "1.npz",
        allow_pickle=False,
    ) as masks:
        assert float(masks["voxel_size_mm"]) == 0.5
        predicted_mask = masks["prediction_mask_zyx"].astype(bool)
        ground_truth_mask = masks["ground_truth_mask_zyx"].astype(bool)
        predicted_centerline = masks["prediction_centerline_zyx"].astype(bool)
        ground_truth_centerline = masks[
            "raw_ground_truth_centerline_zyx"
        ].astype(bool)
        np.testing.assert_array_equal(predicted_mask, expected)

    intersection = np.count_nonzero(predicted_mask & ground_truth_mask)
    expected_dice = 2.0 * intersection / (
        np.count_nonzero(predicted_mask) + np.count_nonzero(ground_truth_mask)
    )
    topology_precision = np.count_nonzero(
        predicted_centerline & ground_truth_mask
    ) / np.count_nonzero(predicted_centerline)
    topology_sensitivity = np.count_nonzero(
        ground_truth_centerline & predicted_mask
    ) / np.count_nonzero(ground_truth_centerline)
    expected_cldice = (
        2.0
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity)
    )
    assert record["dice_3d"] == pytest.approx(expected_dice)
    assert record["cldice_3d"] == pytest.approx(expected_cldice)


def test_native_raw_vessel_without_offset_is_rejected(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"

    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, :7] = 1.0
    _save_prediction(prediction_dir / "validation" / "1.npz", prediction)
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        np.asarray([[0.25, 0.25, 0.25]], dtype=np.float32),
        radius_mm=0.2,
        coordinate_frame="native",
    )

    with pytest.raises(
        ValueError,
        match="contain no projection centre offset",
    ):
        _run_directory_evaluation(
            prediction_dir,
            raw_dir,
            output_dir,
        )


def test_native_voxel_ground_truth_requires_proven_center_offset(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    ground_truth_dir = tmp_path / "ground_truth"
    output_dir = tmp_path / "metrics"

    _save_prediction(
        prediction_dir / "validation" / "1.npz",
        np.zeros((8, 8, 8), dtype=np.float32),
    )
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        np.asarray([[0.25, 0.25, 0.25]], dtype=np.float32),
        radius_mm=0.2,
        coordinate_frame="projection_centered",
    )
    ground_truth_dir.mkdir(parents=True)
    np.savez_compressed(
        ground_truth_dir / "lca_0001.npz",
        case_id=np.asarray("1"),
        vol=np.zeros((8, 8, 8), dtype=np.uint8),
        spacing=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        spacing_units=np.asarray("mm"),
    )

    with pytest.raises(
        ValueError,
        match="native voxel ground truth.*no exact projection centre offset",
    ):
        _run_directory_evaluation(
            prediction_dir,
            raw_dir,
            output_dir,
            "--ground-truth-volume-dir",
            str(ground_truth_dir),
        )


def test_saved_prediction_view_pair_must_match_strict_zero_one(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, 1:7] = 1.0
    _save_prediction(
        prediction_dir / "validation" / "1.npz",
        prediction,
        view_indices=(0, 6),
    )
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        np.asarray([[0.25, 0.25, 0.25]], dtype=np.float32),
        radius_mm=0.2,
        coordinate_frame="projection_centered",
    )

    with pytest.raises(ValueError, match=r"requires \[0, 1\]"):
        _run_directory_evaluation(prediction_dir, raw_dir, output_dir)


def test_legacy_prediction_uses_offset_embedded_in_raw_vessel_npz(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"
    center_offset = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    centered_points = np.column_stack(
        (
            np.arange(-1.75, 1.5, 0.5, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
        )
    )
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, :7] = 1.0
    _save_prediction(prediction_dir / "validation" / "1.npz", prediction)
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        centered_points + center_offset[None, :],
        radius_mm=0.24,
        coordinate_frame="native",
        center_offset_xyz_mm=tuple(float(value) for value in center_offset),
    )

    _run_directory_evaluation(prediction_dir, raw_dir, output_dir)

    record = json.loads((output_dir / "per_case_metrics.json").read_text())[0]
    assert record["projection_center_offset_source"] == "raw_vessel_npz"
    assert record["dice_3d"] == pytest.approx(1.0)
    assert record["cldice_3d"] == pytest.approx(1.0)
    assert record["centerline_chamfer_distance_mm"] == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("offset_encoding", ["exact_mm", "legacy_metres"])
def test_legacy_prediction_uses_projection_directory_offset(
    tmp_path, offset_encoding
):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    projection_dir = tmp_path / "projections"
    output_dir = tmp_path / "metrics"

    center_offset = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    centered_points = np.column_stack(
        (
            np.arange(-1.75, 1.5, 0.5, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
            np.full(7, 0.25, dtype=np.float32),
        )
    )
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, :7] = 1.0
    _save_prediction(prediction_dir / "validation" / "1.npz", prediction)
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        centered_points + center_offset[None, :],
        radius_mm=0.24,
        coordinate_frame="native",
    )
    projection_dir.mkdir(parents=True)
    projection_payload: dict[str, np.ndarray] = {
        "case_id": np.asarray("1")
    }
    if offset_encoding == "exact_mm":
        projection_payload["projection_center_offset_xyz_mm"] = center_offset
    else:
        projection_payload["projection_center_offset"] = center_offset / 1000.0
        projection_payload["input_scale_to_mm"] = np.asarray(1000.0)
    np.savez_compressed(
        projection_dir / "lca_0001.npz", **projection_payload
    )

    _run_directory_evaluation(
        prediction_dir,
        raw_dir,
        output_dir,
        "--projection-dir",
        str(projection_dir),
    )

    record = json.loads((output_dir / "per_case_metrics.json").read_text())[0]
    assert record["projection_center_offset_source"] == "projection_npz"
    np.testing.assert_allclose(
        record["projection_center_offset_xyz_mm"], center_offset, atol=1e-5
    )
    assert record["dice_3d"] == pytest.approx(1.0)
    assert record["cldice_3d"] == pytest.approx(1.0)
    assert record["centerline_chamfer_distance_mm"] == pytest.approx(
        0.0, abs=1e-5
    )


def test_mixed_validation_and_test_cases_write_by_split_aggregates(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "metrics"

    centered_points = np.column_stack(
        (
            np.arange(-1.25, 1.5, 0.5, dtype=np.float32),
            np.full(6, 0.25, dtype=np.float32),
            np.full(6, 0.25, dtype=np.float32),
        )
    )
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, 1:7] = 1.0
    for case_id, split in (("1", "validation"), ("2", "test")):
        _save_prediction(
            prediction_dir / split / f"{case_id}.npz",
            prediction,
            case_id=case_id,
            dataset_split=split,
        )
        _save_raw_vessel(
            raw_dir / case_id / "original.npz",
            centered_points,
            radius_mm=0.24,
            coordinate_frame="projection_centered",
            case_id=case_id,
        )

    _run_directory_evaluation(prediction_dir, raw_dir, output_dir)

    summary = json.loads((output_dir / "evaluation_summary.json").read_text())
    assert summary["num_cases"] == 2
    assert set(summary["by_split"]) == {"validation", "test"}
    for split in ("validation", "test"):
        split_summary = summary["by_split"][split]
        assert split_summary["num_cases"] == 1
        assert split_summary["macro_dice_3d"] == pytest.approx(1.0)
        assert split_summary["macro_cldice_3d"] == pytest.approx(1.0)
        assert split_summary["by_split"] == {}
    assert (output_dir / "graphs" / "validation" / "1.npz").is_file()
    assert (output_dir / "graphs" / "test" / "2.npz").is_file()
    assert (output_dir / "masks" / "validation" / "1.npz").is_file()
    assert (output_dir / "masks" / "test" / "2.npz").is_file()


def test_voxel_ground_truth_masks_metrics_to_partial_valid_fov(tmp_path):
    prediction_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw"
    ground_truth_dir = tmp_path / "ground_truth"
    output_dir = tmp_path / "metrics"

    # The voxel GT covers only x=[0, 2) mm. The prediction has an exact vessel
    # in that valid half and an equally sized false-positive line in x<0.
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[4, 4, 0:3] = 1.0
    prediction[4, 4, 4:7] = 1.0
    _save_prediction(prediction_dir / "validation" / "1.npz", prediction)
    raw_points = np.column_stack(
        (
            np.asarray([0.25, 0.75, 1.25], dtype=np.float32),
            np.full(3, 0.25, dtype=np.float32),
            np.full(3, 0.25, dtype=np.float32),
        )
    )
    _save_raw_vessel(
        raw_dir / "1" / "original.npz",
        raw_points,
        radius_mm=0.2,
        coordinate_frame="projection_centered",
        center_offset_xyz_mm=(0.0, 0.0, 0.0),
    )
    ground_truth_dir.mkdir(parents=True)
    ground_truth_xyz = np.zeros((4, 8, 8), dtype=np.uint8)
    ground_truth_xyz[0:3, 4, 4] = 1
    np.savez_compressed(
        ground_truth_dir / "lca_0001.npz",
        case_id=np.asarray("1"),
        vol=ground_truth_xyz,
        spacing=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
        spacing_units=np.asarray("mm"),
    )

    _run_directory_evaluation(
        prediction_dir,
        raw_dir,
        output_dir,
        "--ground-truth-volume-dir",
        str(ground_truth_dir),
        "--ground-truth-origin-xyz-mm",
        "0",
        "-2",
        "-2",
    )

    record = json.loads((output_dir / "per_case_metrics.json").read_text())[0]
    assert record["ground_truth_mask_source"] == "native_voxel_npz"
    assert record["evaluation_valid_fov_fraction"] == pytest.approx(0.5)
    assert record["evaluation_valid_fov_voxels"] == 4 * 8 * 8
    assert record["predicted_foreground_voxels"] == 3
    assert record["ground_truth_foreground_voxels"] == 3
    assert record["dice_3d"] == pytest.approx(1.0)
    assert record["cldice_3d"] == pytest.approx(1.0)

    with np.load(
        output_dir / "masks" / "validation" / "1.npz",
        allow_pickle=False,
    ) as masks:
        prediction_mask = masks["prediction_mask_zyx"].astype(bool)
        valid_fov = masks["evaluation_valid_fov_zyx"].astype(bool)
        assert np.count_nonzero(prediction_mask) == 6
        assert np.count_nonzero(prediction_mask & ~valid_fov) == 3
        assert np.count_nonzero(prediction_mask & valid_fov) == 3
