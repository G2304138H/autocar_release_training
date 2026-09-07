from scripts import train_imagecas_npz as launcher


def test_launcher_uses_supplied_lca_defaults():
    args = launcher._parser().parse_args(
        ["--artery", "lca", "--skip-preflight"]
    )
    settings = launcher._resolved_settings(args)

    assert settings["experiment"] == "stage2_npz_lca"
    assert str(settings["projection_source"]).endswith(
        "vessel_code_stage_2_lca_paired/anchors"
    )
    assert str(settings["voxel_source"]).endswith("imagecas_voxel/lca")
    assert settings["pixel_spacing_mm"] == 0.65
    assert settings["fallback_pixel_spacing_mm"] == 0.65
    assert settings["fallback_sid_mm"] == 900.0
    assert settings["view_labels"] == (
        "RAO 25, CAU 35",
        "LAO 5, CRA 40",
    )


def test_launcher_builds_rca_debug_command_without_a_shell():
    args = launcher._parser().parse_args(
        [
            "--artery",
            "rca",
            "--skip-preflight",
            "--max-epochs",
            "3",
            "--",
            "debug=stage2_gpu",
        ]
    )
    settings = launcher._resolved_settings(args)
    command = launcher._hydra_command(args, settings)

    assert command[:3] == [launcher.sys.executable, "-m", "src.train"]
    assert "experiment=stage2_npz_rca" in command
    assert "data.expected_imager_pixel_spacing_mm=0.55" in command
    assert "data.fallback_imager_pixel_spacing_mm=0.55" in command
    assert "data.fallback_sid_mm=900" in command
    assert "data.source_to_isocenter_mm=750" in command
    assert "trainer.max_epochs=3" in command
    assert "debug=stage2_gpu" in command
    assert "--" not in command
