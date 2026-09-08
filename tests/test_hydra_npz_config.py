from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
from hydra import compose, initialize_config_dir


def test_stage2_experiment_composes_a_consistent_paper_feature_width():
    config_directory = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_directory.resolve())
    ):
        config = compose(
            config_name="train.yaml", overrides=["experiment=stage2_npz"]
        )

    reconstruction = config.model.recon_net
    per_view = reconstruction.encoder2d.out_ch + int(
        reconstruction.ray_casting.include_distance_feature
    )
    assert reconstruction.expected_view_count == 2
    assert reconstruction.encoder2d.working_image_dim == 512
    assert reconstruction.ray_casting.fusion == "concat"
    assert reconstruction.ray_casting.candidate_mode == "voxel_grid"
    assert reconstruction.ray_casting.voxel_chunk_size > 0
    assert reconstruction.ray_casting.distance_sampling == "nearest"
    assert reconstruction.unet3d.in_channels == 2 * per_view
    assert reconstruction.unet3d.architecture == "34C"
    assert config.model.optimizer._target_ == "torch.optim.Adam"
    assert config.model.optimizer.lr == pytest.approx(3e-4)
    assert config.model.scheduler is None
    assert config.data.batch_size == 1
    assert list(config.data.evaluation_view_indices) == [0, 6]
    assert list(config.data.evaluation_view_labels) == [
        "RAO 25, CAU 35",
        "LAO 5, CRA 40",
    ]
    assert config.data.minimum_train_pair_angle_deg == pytest.approx(30.0)
    assert config.trainer.check_val_every_n_epoch == 1
    assert config.trainer.precision == "32-true"


def test_stage2_gpu_debug_profile_does_not_force_spconv_onto_cpu():
    config_directory = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_directory.resolve())
    ):
        config = compose(
            config_name="train.yaml",
            overrides=["experiment=stage2_npz", "debug=stage2_gpu"],
        )

    assert config.model.recon_net.sparse_backend == "spconv"
    assert config.trainer.accelerator == "gpu"
    assert config.trainer.devices == 1
    assert config.trainer.limit_train_batches == 1
    assert config.trainer.limit_val_batches == 1
    assert config.test is False


@pytest.mark.parametrize(
    ("experiment", "artery", "pixel_spacing", "projection_suffix"),
    [
        (
            "stage2_npz_lca",
            "lca",
            0.65,
            "vessel_code_stage_2_lca_paired/anchors",
        ),
        (
            "stage2_npz_rca",
            "rca",
            0.55,
            "imagecas_autocar_6/stage_2_imagecas_all_branch",
        ),
    ],
)
def test_imagecas_artery_experiments_compose_cluster_paths(
    experiment, artery, pixel_spacing, projection_suffix
):
    config_directory = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_directory.resolve())
    ):
        config = compose(
            config_name="train.yaml", overrides=[f"experiment={experiment}"]
        )

    assert config.task_name == f"train_autocar_{artery}"
    assert config.data.case_id_mode == "imagecas_numeric"
    assert config.data.expected_imager_pixel_spacing_mm == pytest.approx(
        pixel_spacing
    )
    assert str(config.data.projection_source).endswith(projection_suffix)
    assert str(config.data.voxel_source).endswith(f"imagecas_voxel/{artery}")
    assert str(config.data.split_json).endswith("split.json")
    assert list(config.data.evaluation_view_indices) == [0, 6]
    assert config.trainer.precision == "32-true"
    if artery == "lca":
        assert list(config.data.excluded_train_case_ids) == ["288", "421"]
        assert config.data.fallback_imager_pixel_spacing_mm == pytest.approx(0.65)
        assert config.data.fallback_sid_mm == pytest.approx(900.0)
        assert list(config.data.evaluation_view_labels) == [
            "RAO 25, CAU 35",
            "LAO 5, CRA 40",
        ]
    else:
        assert list(config.data.excluded_train_case_ids) == [
            "0288",
            "0421",
            "0909",
            "0108",
            "0207",
            "0324",
        ]
        assert config.data.fallback_imager_pixel_spacing_mm == pytest.approx(0.55)
        assert config.data.fallback_sid_mm == pytest.approx(900.0)
        assert config.data.evaluation_view_labels is None
    assert config.data.source_to_isocenter_mm == pytest.approx(750.0)
