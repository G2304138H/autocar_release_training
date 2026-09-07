"""Projection geometry used to generate the Stage-2 vessel masks.

This module is a self-contained NumPy port of ``get_local_params``,
``ray_image_intersection`` and ``convert3D_to_pixels`` from the data generator.
Unlike that generator, all world distances here are expressed in millimetres.

Pixel coordinates are deliberately explicit throughout this module:

* ``xy`` means ``(column, row)`` and is the convention used by
  :attr:`ProjectionGeometry.world2pix4x4`.
* ``rc`` means ``(row, column)`` and can be used to index an image directly.

The legacy renderer first computed a vertically flipped detector ``y`` value
and then flipped it again while writing the NumPy image.  The two operations
cancel.  Consequently, the final mask row increases along ``detector_y_xyz``.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


DEFAULT_SOURCE_TO_ISOCENTER_MM = 750.0
_COORDINATE_SYSTEM_CHANGE = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def _as_finite_vector(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return values


def _rotation_matrices(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Return the generator's active rotations with shape ``[V, 3, 3]``."""

    coordinate_change = _COORDINATE_SYSTEM_CHANGE
    coordinate_change_inverse = np.linalg.inv(coordinate_change)
    rotations = []

    for theta_value, phi_value in zip(theta_deg, phi_deg):
        theta = np.deg2rad(theta_value)
        phi = np.deg2rad(phi_value)

        theta_rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # This sign convention is intentionally identical to the generator.
        phi_rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(phi), np.sin(phi)],
                [0.0, -np.sin(phi), np.cos(phi)],
            ],
            dtype=np.float64,
        )

        rotation_theta = (
            coordinate_change @ theta_rotation @ coordinate_change_inverse
        )
        rotation_phi = (
            coordinate_change @ phi_rotation @ coordinate_change_inverse
        )
        rotations.append(rotation_theta @ rotation_phi)

    return np.stack(rotations, axis=0)


@dataclass(frozen=True)
class ProjectionGeometry:
    """Biplane/cone-beam geometry for one or more Stage-2 views.

    Use :meth:`from_angles` rather than constructing this class directly.
    World coordinates are centred at the projection isocentre and measured in
    millimetres.  ``detector_x_xyz`` points toward increasing image columns and
    ``detector_y_xyz`` points toward increasing image rows.
    """

    theta_deg: np.ndarray
    phi_deg: np.ndarray
    image_dim: int
    sid_mm: float
    source_to_isocenter_mm: float
    pixel_spacing_mm: float
    detector_center_xyz_mm: np.ndarray
    source_xyz_mm: np.ndarray
    detector_x_xyz: np.ndarray
    detector_y_xyz: np.ndarray
    view_directions_world: np.ndarray
    world2pix4x4: np.ndarray

    @classmethod
    def from_angles(
        cls,
        theta_deg: np.ndarray,
        phi_deg: np.ndarray,
        image_dim: int,
        sid_mm: float,
        pixel_spacing_mm: float,
        source_to_isocenter_mm: float = DEFAULT_SOURCE_TO_ISOCENTER_MM,
    ) -> "ProjectionGeometry":
        """Construct the exact geometry used by ``render_fixed_views``.

        Args:
            theta_deg: Per-view generator theta angles in degrees.
            phi_deg: Per-view generator phi angles in degrees.
            image_dim: Width and height of the square detector in pixels.
            sid_mm: Source-to-detector distance in millimetres.
            pixel_spacing_mm: Detector pixel spacing recorded by the dataset.
            source_to_isocenter_mm: Fixed source-to-isocentre distance.  The
                Stage-2 generator uses 750 mm.
        """

        theta = _as_finite_vector("theta_deg", theta_deg)
        phi = _as_finite_vector("phi_deg", phi_deg)
        if theta.shape != phi.shape:
            raise ValueError(
                "theta_deg and phi_deg must contain the same number of views."
            )

        if isinstance(image_dim, bool) or int(image_dim) != image_dim:
            raise ValueError("image_dim must be a positive integer.")
        image_dim = int(image_dim)
        if image_dim <= 1:
            raise ValueError("image_dim must be greater than one.")

        sid_mm = float(sid_mm)
        source_to_isocenter_mm = float(source_to_isocenter_mm)
        pixel_spacing_mm = float(pixel_spacing_mm)
        scalar_values = (sid_mm, source_to_isocenter_mm, pixel_spacing_mm)
        if not np.isfinite(scalar_values).all():
            raise ValueError("Camera distances and pixel spacing must be finite.")
        if source_to_isocenter_mm <= 0.0:
            raise ValueError("source_to_isocenter_mm must be positive.")
        if sid_mm <= source_to_isocenter_mm:
            raise ValueError(
                "sid_mm must exceed source_to_isocenter_mm so that the detector "
                "is on the opposite side of the isocentre."
            )
        if pixel_spacing_mm <= 0.0:
            raise ValueError("pixel_spacing_mm must be positive.")

        rotations = _rotation_matrices(theta, phi)
        detector_distance_mm = sid_mm - source_to_isocenter_mm

        detector_direction = np.einsum(
            "vij,j->vi", rotations, np.array([0.0, 0.0, 1.0])
        )
        detector_centers = detector_direction * detector_distance_mm
        sources = -detector_direction * source_to_isocenter_mm
        detector_x = np.einsum(
            "vij,j->vi", rotations, np.array([0.0, 1.0, 0.0])
        )
        detector_y = np.einsum(
            "vij,j->vi", rotations, np.array([-1.0, 0.0, 0.0])
        )

        matrices = _make_world2pix4x4(
            sources=sources,
            detector_centers=detector_centers,
            detector_x=detector_x,
            detector_y=detector_y,
            image_dim=image_dim,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        return cls(
            theta_deg=theta.astype(np.float32),
            phi_deg=phi.astype(np.float32),
            image_dim=image_dim,
            sid_mm=sid_mm,
            source_to_isocenter_mm=source_to_isocenter_mm,
            pixel_spacing_mm=pixel_spacing_mm,
            detector_center_xyz_mm=detector_centers.astype(np.float32),
            source_xyz_mm=sources.astype(np.float32),
            detector_x_xyz=detector_x.astype(np.float32),
            detector_y_xyz=detector_y.astype(np.float32),
            view_directions_world=detector_direction.astype(np.float32),
            world2pix4x4=matrices.astype(np.float32),
        )

    @property
    def num_views(self) -> int:
        return int(self.theta_deg.size)

    @property
    def detector_width_mm(self) -> float:
        return self.pixel_spacing_mm * self.image_dim

    @property
    def detector_edge_extent_mm(self) -> float:
        """Legacy distance between the detector coordinates labelled 0 and N."""

        return self.detector_width_mm * (self.image_dim - 1) / self.image_dim

    @property
    def detector_pixel_scale(self) -> float:
        """Legacy continuous pixels per millimetre on the detector plane."""

        return self.image_dim / self.detector_edge_extent_mm

    @property
    def principal_pixel_xy(self) -> np.ndarray:
        """Pixel coordinate to which the centred isocentre projects."""

        centre = self.image_dim * (self.image_dim / 2.0 - 1.0) / (
            self.image_dim - 1.0
        )
        return np.array([centre, centre], dtype=np.float32)

    @property
    def projection_matrices3x4(self) -> np.ndarray:
        """Projective matrices whose outputs are ``(column, row, scale)``."""

        return self.world2pix4x4[:, [0, 1, 3], :]

    def _validate_view_index(self, view_index: int) -> int:
        view_index = int(view_index)
        if not 0 <= view_index < self.num_views:
            raise IndexError(
                f"view_index {view_index} is outside [0, {self.num_views})."
            )
        return view_index

    def project_points_xy(
        self, points_xyz_mm: np.ndarray, view_index: int
    ) -> np.ndarray:
        """Project world points to continuous ``(column, row)`` coordinates.

        Points on the plane through the source parallel to the detector cannot
        be projected and are returned as ``NaN``.
        """

        view_index = self._validate_view_index(view_index)
        points = np.asarray(points_xyz_mm, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("points_xyz_mm must have shape [..., 3].")
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        homogeneous = np.concatenate(
            [flat_points, np.ones((flat_points.shape[0], 1))], axis=1
        )
        projected = homogeneous @ self.world2pix4x4[view_index].astype(
            np.float64
        ).T
        denominator = projected[:, 3]
        xy = np.full((flat_points.shape[0], 2), np.nan, dtype=np.float64)
        valid = np.abs(denominator) > 1e-9
        xy[valid] = projected[valid, :2] / denominator[valid, None]
        return xy.reshape(original_shape + (2,)).astype(np.float32)

    def project_points_rc(
        self, points_xyz_mm: np.ndarray, view_index: int
    ) -> np.ndarray:
        """Project world points to continuous ``(row, column)`` coordinates."""

        return self.project_points_xy(points_xyz_mm, view_index)[..., ::-1]

    def detector_points_from_pixels_xy(
        self,
        pixels_xy: np.ndarray,
        view_index: int,
        pixel_offset: float = 0.0,
    ) -> np.ndarray:
        """Map continuous ``(column, row)`` coordinates to the detector plane.

        The generator rounded projected coordinates directly to integer mask
        indices, so ``pixel_offset=0`` exactly inverts its convention.  Pass
        ``0.5`` only when a downstream algorithm intentionally casts through
        conventional pixel-cell centres.
        """

        view_index = self._validate_view_index(view_index)
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        if pixels.ndim == 0 or pixels.shape[-1] != 2:
            raise ValueError("pixels_xy must have shape [..., 2].")
        if not np.isfinite(pixels).all():
            raise ValueError("pixels_xy contains NaN or infinite values.")
        pixel_offset = float(pixel_offset)
        if not np.isfinite(pixel_offset):
            raise ValueError("pixel_offset must be finite.")

        principal = self.principal_pixel_xy.astype(np.float64)
        displacement = (pixels + pixel_offset - principal) / self.detector_pixel_scale
        return (
            self.detector_center_xyz_mm[view_index].astype(np.float64)
            + displacement[..., 0, None]
            * self.detector_x_xyz[view_index].astype(np.float64)
            + displacement[..., 1, None]
            * self.detector_y_xyz[view_index].astype(np.float64)
        ).astype(np.float32)

    def pixels_xy_to_rays(
        self,
        pixels_xy: np.ndarray,
        view_index: int,
        pixel_offset: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ray origins and unit directions for detector pixels."""

        view_index = self._validate_view_index(view_index)
        detector_points = self.detector_points_from_pixels_xy(
            pixels_xy, view_index, pixel_offset=pixel_offset
        ).astype(np.float64)
        directions = detector_points - self.source_xyz_mm[view_index].astype(
            np.float64
        )
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("A detector point coincides with the camera source.")
        directions = (directions / norms).astype(np.float32)
        origins = np.broadcast_to(
            self.source_xyz_mm[view_index], directions.shape
        ).copy()
        return origins, directions

    def pixels_rc_to_rays(
        self,
        pixels_rc: np.ndarray,
        view_index: int,
        pixel_offset: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return rays for image-index coordinates supplied as ``(row, column)``."""

        pixels = np.asarray(pixels_rc)
        if pixels.ndim == 0 or pixels.shape[-1] != 2:
            raise ValueError("pixels_rc must have shape [..., 2].")
        return self.pixels_xy_to_rays(
            pixels[..., ::-1], view_index, pixel_offset=pixel_offset
        )


def _make_world2pix4x4(
    sources: np.ndarray,
    detector_centers: np.ndarray,
    detector_x: np.ndarray,
    detector_y: np.ndarray,
    image_dim: int,
    pixel_spacing_mm: float,
) -> np.ndarray:
    """Build matrices equivalent to the generator's ray-plane projection."""

    detector_width_mm = pixel_spacing_mm * image_dim
    detector_edge_extent_mm = detector_width_mm * (image_dim - 1) / image_dim
    pixel_scale = image_dim / detector_edge_extent_mm
    principal = image_dim * (image_dim / 2.0 - 1.0) / (image_dim - 1.0)

    matrices = np.zeros((sources.shape[0], 4, 4), dtype=np.float64)
    for view_index in range(sources.shape[0]):
        source = sources[view_index]
        detector_center = detector_centers[view_index]
        axis_x = detector_x[view_index]
        axis_y = detector_y[view_index]
        normal = np.cross(axis_x, axis_y)
        source_to_detector_normal = np.dot(normal, detector_center - source)

        denominator = np.concatenate([normal, [-np.dot(normal, source)]])
        x_from_source = np.concatenate([axis_x, [-np.dot(axis_x, source)]])
        y_from_source = np.concatenate([axis_y, [-np.dot(axis_y, source)]])

        matrices[view_index, 0] = (
            principal * denominator
            + pixel_scale * source_to_detector_normal * x_from_source
        )
        matrices[view_index, 1] = (
            principal * denominator
            + pixel_scale * source_to_detector_normal * y_from_source
        )
        # This follows the existing AutoCAR/PyTorch3D-style 4x4 contract:
        # rows 0 and 1 are pixel numerators and row 3 is the denominator.
        matrices[view_index, 2, 3] = 1.0
        matrices[view_index, 3] = denominator

    return matrices
