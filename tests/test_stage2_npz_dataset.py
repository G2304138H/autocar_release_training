import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dataset.stage2_npz import Stage2NPZDataset, Stage2NPZError


def _write_projection(
    path: Path,
    case_id: str = "42",
    num_views: int = 4,
    image_dim: int = 8,
    pixel_spacing_mm: float = 0.65,
) -> None:
    images = np.zeros((num_views, image_dim, image_dim), dtype=np.float32)
    for view_index in range(num_views):
        images[view_index, view_index, view_index + 1] = 1.0
    theta = np.linspace(-30.0, 30.0, num_views, dtype=np.float32)
    phi = np.linspace(5.0, 35.0, num_views, dtype=np.float32)

    # Geometry-produced directions make the metadata-consistency check useful.
    from src.geometry.projection_geometry import ProjectionGeometry

    geometry = ProjectionGeometry.from_angles(
        theta,
        phi,
        image_dim,
        sid_mm=900.0,
        pixel_spacing_mm=pixel_spacing_mm,
    )
    anchor_labels = np.asarray(
        [f"anchor-{view_index}" for view_index in range(num_views)]
    )
    np.savez_compressed(
        path,
        case_id=np.asarray(case_id),
        sample_name=np.asarray(f"lca_{case_id}"),
        images=images,
        theta_deg=theta,
        phi_deg=phi,
        image_dim=np.asarray(image_dim, dtype=np.int32),
        sid=np.asarray(0.9, dtype=np.float32),
        imager_pixel_spacing=np.asarray(pixel_spacing_mm, dtype=np.float32),
        imager_pixel_spacing_units=np.asarray("mm"),
        projection_center_offset=np.asarray(
            [0.001, 0.002, 0.003], dtype=np.float32
        ),
        input_scale_to_mm=np.asarray(1000.0, dtype=np.float32),
        view_directions_world=geometry.view_directions_world,
        view_features=np.arange(num_views * 4, dtype=np.float32).reshape(
            num_views, 4
        ),
        view_indices=np.arange(10, 10 + num_views, dtype=np.int32),
        anchor_clinical_views=anchor_labels,
    )


def _write_voxel(path: Path, case_id=None) -> np.ndarray:
    volume_xyz = np.zeros((4, 5, 6), dtype=np.uint8)
    volume_xyz[1, 2, 3] = 7
    payload = {
        "vol": volume_xyz,
        "spacing": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    }
    if case_id is not None:
        payload["case_id"] = np.asarray(case_id)
    np.savez_compressed(path, **payload)
    return volume_xyz


def _drop_projection_fields(path: Path, *fields: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {
            key: np.asarray(data[key]) for key in data.files if key not in fields
        }
    np.savez_compressed(path, **payload)


class Stage2NPZDatasetTests(unittest.TestCase):
    def test_pairs_by_case_id_and_canonicalises_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "name_does_not_control_pairing.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path)
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="fixed",
                fixed_view_indices=(2, 0),
            )
            sample = dataset[0]

            self.assertEqual(sample["case_id"], "42")
            self.assertEqual(sample["sample_name"], "lca_42")
            self.assertEqual(sample["images"].shape, (2, 1, 8, 8))
            self.assertEqual(int(sample["image_dim"]), 8)
            self.assertEqual(sample["image_dim_source"], "npz")
            np.testing.assert_array_equal(sample["view_indices"], [2, 0])
            np.testing.assert_array_equal(sample["source_view_indices"], [12, 10])
            self.assertEqual(sample["gt_volume_zyx"].shape, (6, 5, 4))
            self.assertEqual(sample["gt_volume_zyx"][3, 2, 1], 1)
            self.assertEqual(int(sample["gt_volume_zyx"].sum()), 1)
            np.testing.assert_allclose(
                sample["gt_spacing_xyz_mm"], [0.4, 0.5, 0.6]
            )
            # Native index (0,0,0) is centred at physical zero, while the
            # VoxelGrid contract stores its lower boundary.
            np.testing.assert_allclose(
                sample["gt_origin_xyz_mm"], [-0.2, -0.25, -0.3]
            )
            np.testing.assert_allclose(
                sample["projection_center_offset_xyz_mm"], [1.0, 2.0, 3.0]
            )
            self.assertEqual(sample["world2pix4x4"].shape, (2, 4, 4))
            self.assertEqual(sample["camera_source_xyz_mm"].shape, (2, 3))
            assert sample["view_labels"] == ("anchor-2", "anchor-0")

    def test_fixed_view_labels_are_validated_per_case(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path)
            _write_voxel(voxel_path)

            matching = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="fixed",
                fixed_view_indices=(0, 3),
                fixed_view_labels=("anchor-0", "anchor-3"),
            )
            assert matching[0]["view_labels"] == ("anchor-0", "anchor-3")

            mismatching = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="fixed",
                fixed_view_indices=(0, 3),
                fixed_view_labels=("wrong", "anchor-3"),
            )
            with self.assertRaisesRegex(Stage2NPZError, "do not match expected"):
                _ = mismatching[0]

    def test_expected_detector_pixel_spacing_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, pixel_spacing_mm=0.55)
            _write_voxel(voxel_path)

            matching = Stage2NPZDataset(
                projection_path,
                voxel_path,
                expected_imager_pixel_spacing_mm=0.55,
            )
            self.assertAlmostEqual(
                float(matching[0]["imager_pixel_spacing_mm"]), 0.55
            )

            mismatching = Stage2NPZDataset(
                projection_path,
                voxel_path,
                expected_imager_pixel_spacing_mm=0.65,
            )
            with self.assertRaisesRegex(Stage2NPZError, "expected 0.65 mm"):
                _ = mismatching[0]

    def test_legacy_geometry_fields_use_explicit_config_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, pixel_spacing_mm=0.55)
            _drop_projection_fields(
                projection_path, "sid", "imager_pixel_spacing"
            )
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                fallback_sid_mm=900.0,
                fallback_imager_pixel_spacing_mm=0.55,
                expected_imager_pixel_spacing_mm=0.55,
            )
            sample = dataset[0]

            self.assertEqual(float(sample["sid_mm"]), 900.0)
            self.assertAlmostEqual(
                float(sample["imager_pixel_spacing_mm"]), 0.55
            )
            self.assertEqual(sample["sid_source"], "config")
            self.assertEqual(
                sample["imager_pixel_spacing_source"], "config"
            )

    def test_npz_geometry_fields_take_precedence_over_fallbacks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, pixel_spacing_mm=0.65)
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                fallback_sid_mm=1200.0,
                fallback_imager_pixel_spacing_mm=0.55,
            )
            sample = dataset[0]

            self.assertEqual(float(sample["sid_mm"]), 900.0)
            self.assertAlmostEqual(
                float(sample["imager_pixel_spacing_mm"]), 0.65
            )
            self.assertEqual(sample["sid_source"], "npz")
            self.assertEqual(sample["imager_pixel_spacing_source"], "npz")

    def test_legacy_projection_derives_image_dim_from_square_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "rca_0042.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, case_id="42", image_dim=8)
            _drop_projection_fields(projection_path, "image_dim")
            _write_voxel(voxel_path)

            sample = Stage2NPZDataset(
                projection_path,
                voxel_path,
                case_id_mode="imagecas_numeric",
            )[0]

            self.assertEqual(int(sample["image_dim"]), 8)
            self.assertEqual(sample["image_dim_source"], "images")

    def test_legacy_geometry_fields_without_fallback_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path)
            _drop_projection_fields(projection_path, "sid")
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(projection_path, voxel_path)
            with self.assertRaisesRegex(Stage2NPZError, "fallback_sid_mm"):
                _ = dataset[0]

    def test_eager_projection_validation_checks_every_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_projection(
                root / "a_good.npz", case_id="42", pixel_spacing_mm=0.55
            )
            _write_projection(
                root / "z_bad.npz", case_id="43", pixel_spacing_mm=0.65
            )
            _write_voxel(root / "42.npz")
            _write_voxel(root / "43.npz")
            dataset = Stage2NPZDataset(
                [root / "a_good.npz", root / "z_bad.npz"],
                [root / "42.npz", root / "43.npz"],
                expected_imager_pixel_spacing_mm=0.55,
            )

            with self.assertRaisesRegex(Stage2NPZError, "expected 0.55 mm"):
                dataset.validate_projection_metadata()

    def test_imagecas_case_ids_join_prefixes_and_zero_padding(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "lca_0001.npz"
            voxel_path = root / "1.npz"
            _write_projection(projection_path, case_id="lca_0001")
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                case_ids=["0001"],
                case_id_mode="imagecas_numeric",
            )
            self.assertEqual(dataset.records[0].case_id, "1")
            self.assertEqual(dataset[0]["case_id"], "1")

    def test_imagecas_projection_without_case_id_uses_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "rca_0001.npz"
            voxel_path = root / "1.npz"
            _write_projection(projection_path, case_id="unused")
            _drop_projection_fields(projection_path, "case_id")
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                case_ids=["1"],
                case_id_mode="imagecas_numeric",
            )

            self.assertEqual(dataset.records[0].case_id, "1")
            self.assertEqual(dataset[0]["case_id"], "1")

    def test_path_like_case_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "voxel.npz"
            _write_projection(projection_path, case_id="../escape")
            _write_voxel(voxel_path, case_id="../escape")

            with self.assertRaisesRegex(Stage2NPZError, "filename-safe"):
                Stage2NPZDataset(projection_path, voxel_path)

    def test_explicit_gt_lower_bound_origin_overrides_derived_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path)
            _write_voxel(voxel_path)

            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                gt_origin_xyz_mm=(10.0, 20.0, 30.0),
            )

            np.testing.assert_array_equal(
                dataset[0]["gt_origin_xyz_mm"], [10.0, 20.0, 30.0]
            )

    def test_voxel_scalar_case_id_takes_precedence_over_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "unrelated_filename.npz"
            _write_projection(projection_path, case_id="case-A")
            _write_voxel(voxel_path, case_id="case-A")

            dataset = Stage2NPZDataset(projection_path, voxel_path)
            self.assertEqual(dataset.records[0].case_id, "case-A")

    def test_random_pair_is_reproducible_and_epoch_dependent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, num_views=7)
            _write_voxel(voxel_path)

            first = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="random_pair",
                random_seed=1234,
            )
            second = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="random_pair",
                random_seed=1234,
            )
            observed_pairs = []
            for epoch in range(8):
                first.set_epoch(epoch)
                second.set_epoch(epoch)
                first_pair = first[0]["view_indices"]
                second_pair = second[0]["view_indices"]
                np.testing.assert_array_equal(first_pair, second_pair)
                self.assertEqual(len(np.unique(first_pair)), 2)
                observed_pairs.append(tuple(first_pair.tolist()))
            self.assertGreater(len(set(observed_pairs)), 1)

    def test_random_pair_respects_minimum_view_angle(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            voxel_path = root / "42.npz"
            _write_projection(projection_path, num_views=7)
            _write_voxel(voxel_path)
            dataset = Stage2NPZDataset(
                projection_path,
                voxel_path,
                view_mode="random_pair",
                random_seed=7,
                minimum_pair_angle_deg=20.0,
            )

            for epoch in range(5):
                dataset.set_epoch(epoch)
                self.assertGreaterEqual(dataset[0]["pair_angle_deg"], 20.0)

    def test_case_filter_ignores_unpaired_cases_outside_split(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selected = root / "selected_projection.npz"
            unselected = root / "unselected_projection.npz"
            voxel = root / "42.npz"
            _write_projection(selected, case_id="42")
            _write_projection(unselected, case_id="99")
            _write_voxel(voxel)

            dataset = Stage2NPZDataset(
                [selected, unselected],
                voxel,
                case_ids=["42"],
            )

            self.assertEqual([record.case_id for record in dataset.records], ["42"])

    def test_duplicate_projection_case_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.npz"
            second = root / "second.npz"
            voxel = root / "42.npz"
            _write_projection(first, case_id="42")
            _write_projection(second, case_id="42")
            _write_voxel(voxel)

            with self.assertRaisesRegex(
                Stage2NPZError, "Duplicate projection files"
            ):
                Stage2NPZDataset([first, second], voxel)

    def test_missing_pair_and_invalid_schema_raise_clear_errors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            projection_path = root / "projection.npz"
            wrong_voxel = root / "99.npz"
            _write_projection(projection_path)
            _write_voxel(wrong_voxel)

            with self.assertRaisesRegex(Stage2NPZError, "No voxel NPZ"):
                Stage2NPZDataset(projection_path, wrong_voxel)

            matching_voxel = root / "42.npz"
            _write_voxel(matching_voxel)
            np.savez_compressed(
                projection_path,
                case_id=np.asarray("42"),
                images=np.zeros((2, 8), dtype=np.float32),
            )
            dataset = Stage2NPZDataset(projection_path, matching_voxel)
            with self.assertRaisesRegex(Stage2NPZError, r"\[V,H,W\]"):
                _ = dataset[0]

    @unittest.skipUnless(
        Path("/Users/renyu/Desktop/lca_0001.npz").is_file()
        and Path("/Users/renyu/Desktop/1.npz").is_file(),
        "attached NPZ samples are unavailable",
    )
    def test_attached_samples_satisfy_training_contract(self):
        dataset = Stage2NPZDataset(
            "/Users/renyu/Desktop/lca_0001.npz",
            "/Users/renyu/Desktop/1.npz",
            view_mode="fixed",
            fixed_view_indices=(0, 1),
        )
        sample = dataset[0]

        self.assertEqual(sample["case_id"], "1")
        self.assertEqual(sample["images"].shape, (2, 1, 256, 256))
        self.assertEqual(sample["gt_volume_zyx"].shape, (275, 512, 512))
        self.assertEqual(sample["gt_volume_zyx"].dtype, np.uint8)
        self.assertEqual(int(sample["gt_volume_zyx"].sum()), 58120)
        np.testing.assert_allclose(
            sample["gt_spacing_xyz_mm"],
            [0.376953125, 0.376953125, 0.5],
        )
        np.testing.assert_allclose(
            sample["gt_origin_xyz_mm"],
            [-0.1884765625, -0.1884765625, -0.25],
        )
        np.testing.assert_allclose(
            sample["projection_center_offset_xyz_mm"],
            [116.2403, 114.1458, 59.56715],
            atol=1e-4,
        )

        # The attached branch positions provide direct evidence for the
        # centre-at-index*spacing convention: every valid centreline point's
        # containing voxel is foreground with the derived lower boundary.
        points_xyz_mm = sample["raw_vessel_code_mm"][
            sample["point_valid_mask"], :3
        ]
        indices_xyz = np.floor(
            (points_xyz_mm - sample["gt_origin_xyz_mm"])
            / sample["gt_spacing_xyz_mm"]
        ).astype(np.int64)
        shape_xyz = np.asarray(sample["gt_volume_zyx"].shape[::-1])
        self.assertTrue(np.all((indices_xyz >= 0) & (indices_xyz < shape_xyz)))
        self.assertTrue(
            np.all(
                sample["gt_volume_zyx"][
                    indices_xyz[:, 2],
                    indices_xyz[:, 1],
                    indices_xyz[:, 0],
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
