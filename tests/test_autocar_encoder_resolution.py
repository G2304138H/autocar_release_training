import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("rootutils")

from src.modules.autocar import AutoCAR


class _RecordingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = tuple(inputs.shape)
        return torch.stack((inputs, 2.0 * inputs), dim=2)


def _resolution_adapter(working_image_dim: int | None) -> AutoCAR:
    model = AutoCAR.__new__(AutoCAR)
    torch.nn.Module.__init__(model)
    model.encoder_working_image_dim = working_image_dim
    model.encoder2d = _RecordingEncoder()
    return model


def test_encoder_uses_paper_resolution_and_restores_native_grid():
    model = _resolution_adapter(512)
    inputs = torch.rand(1, 2, 256, 256, requires_grad=True)

    features = model._encode_at_working_resolution(inputs)

    assert model.encoder2d.input_shape == (1, 2, 512, 512)
    assert features.shape == (1, 2, 2, 256, 256)
    features.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_encoder_native_mode_avoids_unrequested_resizing():
    model = _resolution_adapter(None)
    inputs = torch.rand(1, 2, 256, 256)

    features = model._encode_at_working_resolution(inputs)

    assert model.encoder2d.input_shape == (1, 2, 256, 256)
    assert features.shape == (1, 2, 2, 256, 256)


@pytest.mark.parametrize("shape", [(2, 256, 256), (1, 2, 1, 256, 256)])
def test_encoder_input_resize_rejects_non_view_batches(shape):
    with pytest.raises(ValueError, match=r"\[B,V,H,W\]"):
        AutoCAR._resize_encoder_input(torch.zeros(shape), (512, 512))
