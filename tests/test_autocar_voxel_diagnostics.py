from functools import partial

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.modules import autocar_voxel_pl
from src.modules.autocar_voxel_pl import AutoCARVoxelLit


class _DummyAutoCAR(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.sparse_backend = "raw"


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


def test_batch_identity_includes_selected_view_context():
    identity = AutoCARVoxelLit._batch_identity(
        {
            "case_id": ["23"],
            "projection_path": ["/data/lca_0023.npz"],
            "view_indices": torch.tensor([[0, 6]]),
            "source_view_indices": torch.tensor([[10, 16]]),
            "view_labels": [("RAO 25, CAU 35",), ("LAO 5, CRA 40",)],
            "pair_angle_deg": torch.tensor([79.94]),
        }
    )

    assert "view_indices=[[0, 6]]" in identity
    assert "source_view_indices=[[10, 16]]" in identity
    assert "RAO 25, CAU 35" in identity
    assert "pair_angle_deg=[79.940" in identity


def test_nonfinite_tensor_diagnostics_report_parameter_statistics():
    failures = AutoCARVoxelLit._nonfinite_tensors(
        [
            (
                "gradient:layer.weight",
                torch.tensor([1.0, float("nan"), float("inf")]),
            )
        ]
    )

    assert len(failures) == 1
    assert "gradient:layer.weight" in failures[0]
    assert "finite=1/3" in failures[0]
    assert "nan=1" in failures[0]
    assert "+inf=1" in failures[0]


def test_numerical_debug_aborts_on_first_nonfinite_gradient(monkeypatch):
    monkeypatch.setattr(autocar_voxel_pl, "AutoCAR", _DummyAutoCAR)
    model = AutoCARVoxelLit(
        recon_net={"profile": "test"},
        optimizer=partial(torch.optim.Adam, lr=3e-4),
        numerical_debug=True,
    )
    batch = {
        "case_id": ["102"],
        "projection_path": ["/data/rca_0102.npz"],
        "view_indices": torch.tensor([[1, 5]]),
        "source_view_indices": torch.tensor([[1, 5]]),
        "pair_angle_deg": torch.tensor([62.5]),
    }
    identity = model._batch_identity(batch)
    model._set_numerical_debug_context("training", batch, 432)
    model._numerical_debug_accumulation_cases.append(identity)
    model.recon_net.weight.grad = torch.tensor([float("nan")])

    with pytest.raises(FloatingPointError) as failure:
        model.on_after_backward()

    message = str(failure.value)
    assert "before the optimizer could update model weights" in message
    assert "case_id='102'" in message
    assert "view_indices=[[1, 5]]" in message
    assert "gradient:recon_net.weight" in message
