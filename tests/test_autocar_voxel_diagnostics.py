import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.modules.autocar_voxel_pl import AutoCARVoxelLit


def test_batch_identity_formats_single_collated_case():
    identity = AutoCARVoxelLit._batch_identity(
        {
            "case_id": ["23"],
            "projection_path": ["/data/lca_0023.npz"],
        }
    )

    assert identity == (
        "case_id='23', projection_path='/data/lca_0023.npz'"
    )


def test_batch_identity_handles_tensor_case_ids_and_missing_paths():
    identity = AutoCARVoxelLit._batch_identity(
        {"case_id": torch.tensor([1, 2])}
    )

    assert "case_ids=('1', '2')" in identity
    assert "projection_paths=('<missing>',)" in identity
