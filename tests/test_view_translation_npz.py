from __future__ import annotations

import math

import numpy as np
import pytest

from src.view_translation_npz import resolve_evaluation_view_translation
from src.view_translation_robustness_npz import (
    resolve_view_translation_robustness_plan,
    translation_condition_id,
    translation_vector_mm,
)


def test_default_translation_plan_has_nine_total_magnitude_conditions():
    plan = resolve_view_translation_robustness_plan({"eval_num_views": 2})

    assert len(plan["conditions"]) == 9
    assert plan["patterns"] == ["y", "xz", "xyz"]
    assert plan["magnitudes_mm"] == [5.0, 10.0, 20.0]
    for condition in plan["conditions"]:
        assert np.linalg.norm(condition["translation_xyz_mm"]) == pytest.approx(
            condition["magnitude_mm"]
        )


def test_translation_vectors_follow_fixed_positive_patient_directions():
    assert translation_vector_mm("y", 5.0) == (0.0, 5.0, 0.0)
    assert translation_vector_mm("xz", 10.0) == pytest.approx(
        (10.0 / math.sqrt(2.0), 0.0, 10.0 / math.sqrt(2.0))
    )
    assert translation_vector_mm("xyz", 20.0) == pytest.approx(
        (20.0 / math.sqrt(3.0),) * 3
    )
    assert translation_condition_id("xz", 10.0) == "xz_translation_10mm"


def test_child_translation_requires_positive_vector_and_second_input():
    options = resolve_evaluation_view_translation(
        {
            "evaluation_view_translation": {
                "translation_xyz_mm": [0.0, 5.0, 0.0]
            }
        }
    )
    assert options["enabled"] is True
    assert options["translation_magnitude_mm"] == 5.0

    with pytest.raises(ValueError, match="positive-direction"):
        resolve_evaluation_view_translation(
            {
                "evaluation_view_translation": {
                    "translation_xyz_mm": [0.0, -5.0, 0.0]
                }
            }
        )
    with pytest.raises(ValueError, match="must be 1"):
        resolve_evaluation_view_translation(
            {
                "evaluation_view_translation": {
                    "translation_xyz_mm": [0.0, 5.0, 0.0],
                    "perturbed_input_position": 0,
                }
            }
        )


def test_translation_plan_requires_exactly_two_views():
    with pytest.raises(ValueError, match="eval_num_views=2"):
        resolve_view_translation_robustness_plan({"eval_num_views": 1})


def test_translation_replaces_only_second_image_and_keeps_camera_matrix(tmp_path):
    pytest.importorskip("skimage")
    torch = pytest.importorskip("torch")
    from src.geometry.projection_geometry import ProjectionGeometry
    from src.view_translation_npz import (
        _projection_vessel,
        _render_surface_mask,
        apply_evaluation_view_translation,
    )

    artery_m = np.zeros((1, 7, 4), dtype=np.float32)
    artery_m[0, :, 0] = np.linspace(-0.02, 0.02, 7)
    artery_m[0, :, 2] = 0.005
    artery_m[0, :, 3] = 0.002
    projection_path = tmp_path / "case_1.npz"
    np.savez_compressed(
        projection_path,
        artery=artery_m,
        projected_branch_indices=np.asarray([0], dtype=np.int32),
        projection_center_offset=np.zeros(3, dtype=np.float32),
        mask_render_mode=np.asarray("filled"),
    )
    geometry = ProjectionGeometry.from_angles(
        theta_deg=np.asarray([0.0, 20.0], dtype=np.float32),
        phi_deg=np.asarray([0.0, 10.0], dtype=np.float32),
        image_dim=96,
        sid_mm=900.0,
        pixel_spacing_mm=0.55,
        source_to_isocenter_mm=750.0,
    )
    vessel = _projection_vessel(projection_path, num_circle_points=120)
    stored_images = np.stack(
        [
            _render_surface_mask(
                vessel["surface_rings_xyz_mm"],
                geometry=geometry,
                view_index=view_index,
                render_mode="filled",
            )
            for view_index in range(2)
        ]
    )
    sample = {
        "case_id": "1",
        "projection_path": str(projection_path),
        "images": torch.from_numpy(stored_images[:, None]),
        "view_indices": torch.tensor([0, 1]),
        "theta_deg": torch.from_numpy(geometry.theta_deg),
        "phi_deg": torch.from_numpy(geometry.phi_deg),
        "world2pix4x4": torch.from_numpy(geometry.world2pix4x4),
        "image_dim": torch.tensor(96),
        "sid_mm": torch.tensor(900.0),
        "imager_pixel_spacing_mm": torch.tensor(0.55),
        "source_to_isocenter_mm": torch.tensor(750.0),
    }
    options = resolve_evaluation_view_translation(
        {
            "evaluation_view_translation": {
                "translation_xyz_mm": [0.0, 5.0, 0.0],
                "minimum_clean_rerender_dice": 0.999,
            }
        }
    )

    translated, record = apply_evaluation_view_translation(sample, options)

    torch.testing.assert_close(translated["images"][0], sample["images"][0])
    assert not torch.equal(translated["images"][1], sample["images"][1])
    torch.testing.assert_close(
        translated["world2pix4x4"], sample["world2pix4x4"]
    )
    assert record is not None
    assert record["clean_rerender_dice_vs_stored"] == pytest.approx(1.0)
    assert record["theta_change_deg"] == 0.0
    assert record["phi_change_deg"] == 0.0
    assert record["selected_source_view_indices"] == [0, 1]
