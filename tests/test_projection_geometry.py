import unittest
from pathlib import Path

import numpy as np

from src.geometry.projection_geometry import ProjectionGeometry


class ProjectionGeometryTests(unittest.TestCase):
    def test_zero_angles_match_generator_axes_and_distances(self):
        geometry = ProjectionGeometry.from_angles(
            theta_deg=np.array([0.0]),
            phi_deg=np.array([0.0]),
            image_dim=256,
            sid_mm=900.0,
            pixel_spacing_mm=0.65,
        )

        np.testing.assert_allclose(
            geometry.source_xyz_mm[0], [0.0, 0.0, -750.0], atol=1e-6
        )
        np.testing.assert_allclose(
            geometry.detector_center_xyz_mm[0], [0.0, 0.0, 150.0], atol=1e-6
        )
        np.testing.assert_allclose(
            geometry.detector_x_xyz[0], [0.0, 1.0, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(
            geometry.detector_y_xyz[0], [-1.0, 0.0, 0.0], atol=1e-6
        )
        np.testing.assert_allclose(
            geometry.view_directions_world[0], [0.0, 0.0, 1.0], atol=1e-6
        )

        expected_principal = 256.0 * (128.0 - 1.0) / 255.0
        np.testing.assert_allclose(
            geometry.project_points_xy(np.zeros((1, 3)), 0),
            [[expected_principal, expected_principal]],
            atol=2e-5,
        )

    def test_detector_pixel_round_trip_and_rc_convention(self):
        geometry = ProjectionGeometry.from_angles(
            theta_deg=np.array([37.0]),
            phi_deg=np.array([21.0]),
            image_dim=256,
            sid_mm=900.0,
            pixel_spacing_mm=0.65,
        )
        pixels_xy = np.array(
            [[0.0, 0.0], [42.25, 193.5], [255.0, 255.0]], dtype=np.float32
        )
        detector_points = geometry.detector_points_from_pixels_xy(pixels_xy, 0)

        np.testing.assert_allclose(
            geometry.project_points_xy(detector_points, 0),
            pixels_xy,
            atol=2e-4,
        )
        np.testing.assert_allclose(
            geometry.project_points_rc(detector_points, 0),
            pixels_xy[:, ::-1],
            atol=2e-4,
        )

        origins, directions = geometry.pixels_xy_to_rays(pixels_xy, 0)
        np.testing.assert_allclose(
            origins,
            np.broadcast_to(geometry.source_xyz_mm[0], origins.shape),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1), np.ones(3), atol=1e-6
        )

    def test_world2pix_matrix_matches_projection_method(self):
        geometry = ProjectionGeometry.from_angles(
            theta_deg=np.array([37.0, -82.0]),
            phi_deg=np.array([21.0, 104.0]),
            image_dim=128,
            sid_mm=900.0,
            pixel_spacing_mm=0.7,
        )
        points = np.array(
            [[10.0, 20.0, 30.0], [-40.0, 50.0, 10.0], [30.0, -20.0, -10.0]]
        )
        homogeneous = np.concatenate([points, np.ones((3, 1))], axis=1)

        for view_index in range(2):
            projected = homogeneous @ geometry.world2pix4x4[view_index].T
            matrix_xy = projected[:, :2] / projected[:, 3, None]
            np.testing.assert_allclose(
                matrix_xy,
                geometry.project_points_xy(points, view_index),
                atol=2e-5,
            )

        gram_x = np.sum(geometry.detector_x_xyz**2, axis=1)
        gram_y = np.sum(geometry.detector_y_xyz**2, axis=1)
        cross = np.sum(
            geometry.detector_x_xyz * geometry.detector_y_xyz, axis=1
        )
        np.testing.assert_allclose(gram_x, np.ones(2), atol=1e-6)
        np.testing.assert_allclose(gram_y, np.ones(2), atol=1e-6)
        np.testing.assert_allclose(cross, np.zeros(2), atol=1e-6)

    def test_invalid_camera_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must exceed"):
            ProjectionGeometry.from_angles(
                theta_deg=[0.0],
                phi_deg=[0.0],
                image_dim=256,
                sid_mm=700.0,
                pixel_spacing_mm=0.65,
            )
        with self.assertRaisesRegex(ValueError, "same number"):
            ProjectionGeometry.from_angles(
                theta_deg=[0.0, 1.0],
                phi_deg=[0.0],
                image_dim=256,
                sid_mm=900.0,
                pixel_spacing_mm=0.65,
            )

    @unittest.skipUnless(
        Path("/Users/renyu/Desktop/lca_0001.npz").is_file(),
        "attached projection sample is unavailable",
    )
    def test_attached_vessel_code_reprojects_inside_all_masks(self):
        path = Path("/Users/renyu/Desktop/lca_0001.npz")
        with np.load(path, allow_pickle=False) as data:
            geometry = ProjectionGeometry.from_angles(
                theta_deg=data["theta_deg"],
                phi_deg=data["phi_deg"],
                image_dim=int(data["image_dim"]),
                sid_mm=float(data["sid"]) * 1000.0,
                pixel_spacing_mm=float(data["imager_pixel_spacing"]),
            )
            np.testing.assert_allclose(
                geometry.view_directions_world,
                data["view_directions_world"],
                atol=1e-6,
            )

            branches = data["projected_branch_indices"]
            valid = data["point_valid_mask"][branches]
            points_xyz_mm = data["raw_vessel_code_mm"][branches, :, :3][valid]
            points_xyz_mm -= (
                data["projection_center_offset"]
                * float(data["input_scale_to_mm"])
            )
            masks = data["images"]

            for view_index in range(geometry.num_views):
                points_rc = np.rint(
                    geometry.project_points_rc(points_xyz_mm, view_index)
                ).astype(np.int64)
                in_bounds = (
                    (points_rc[:, 0] >= 0)
                    & (points_rc[:, 0] < masks.shape[1])
                    & (points_rc[:, 1] >= 0)
                    & (points_rc[:, 1] < masks.shape[2])
                )
                self.assertTrue(in_bounds.all())

                local_hits = []
                for row, column in points_rc:
                    local_hits.append(
                        masks[
                            view_index,
                            max(0, row - 2) : min(masks.shape[1], row + 3),
                            max(0, column - 2) : min(
                                masks.shape[2], column + 3
                            ),
                        ].any()
                    )
                self.assertTrue(np.asarray(local_hits).all())


if __name__ == "__main__":
    unittest.main()
