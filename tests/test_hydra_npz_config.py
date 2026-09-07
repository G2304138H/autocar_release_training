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
