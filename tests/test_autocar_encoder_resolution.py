import pytest
import sys
import types


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


def test_variable_view_model_accepts_one_and_seven_views(monkeypatch):
    class _FakeUNet(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()

        def forward(self, volume):
            return volume

    fake_spconv = types.ModuleType("src.modules.spconv_unet")
    fake_spconv.SpconvUNet18A = _FakeUNet
    fake_spconv.SpconvUNet34C = _FakeUNet
    monkeypatch.setitem(sys.modules, "src.modules.spconv_unet", fake_spconv)
    model = AutoCAR({
        "sparse_backend": "spconv",
        "expected_view_count": None,
        "min_view_count": 1,
        "max_view_count": 7,
        "encoder2d": {"out_ch": 12, "input": "mask"},
        "ray_casting": {
            "bbox_min": [-1, -1, -1], "bbox_max": [1, 1, 1], "LODs": [1],
            "support_views": "all", "adaptive_support_views": False,
            "fusion": "mean", "include_distance_feature": True,
        },
        "unet3d": {"in_channels": 13, "out_channels": 2},
    })
    monkeypatch.setattr(
        model, "_encode_at_working_resolution",
        lambda images: images[:, :, None].expand(-1, -1, 12, -1, -1),
    )
    monkeypatch.setattr(
        model.ray_casting, "forward",
        lambda distance, features, matrices: (features, torch.ones(1, 3)),
    )
    for view_count in (1, 7):
        masks = torch.ones(1, view_count, 8, 8)
        matrices = torch.eye(4).repeat(1, view_count, 1, 1)
        prediction, _ = model(masks, matrices)
        assert prediction.shape == (1, view_count, 12, 8, 8)
    with pytest.raises(ValueError, match="1 to 7 views"):
        model(torch.ones(1, 8, 8, 8), torch.eye(4).repeat(1, 8, 1, 1))
