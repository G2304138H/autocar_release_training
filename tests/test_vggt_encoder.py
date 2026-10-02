"""CPU contract tests with a tiny substitute for the expensive VGGT backbone."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.modules.vggt_encoder import FrozenVGGTEncoder


class TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images):
        assert images.shape[2] == 3
        batch, views, _, height, width = images.shape
        count = height // 14 * (width // 14) + 1
        # Depend on all supplied views, retaining separate cases.
        value = images.mean(dim=(1, 2, 3, 4)).view(batch, 1, 1, 1)
        return [value.expand(batch, views, count, 2048) * self.scale], 1


class FrozenVGGTTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType("vggt.models.aggregator")
        module.Aggregator = TinyAggregator
        self.patch = patch.dict(sys.modules, {"vggt.models.aggregator": module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_views_gradients_freezing_and_external_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"aggregator.scale": torch.tensor(3.0)}, path)
            encoder = FrozenVGGTEncoder(pretrained_path=str(path), adapter_channels=4)
            encoder.train()
            self.assertFalse(encoder.backbone.training)
            for count in (1, 2, 4, 7):
                encoder.zero_grad(set_to_none=True)
                output = encoder(torch.rand(2, count, 28, 42))
                self.assertEqual(output.shape, (2, count, 12, 28, 42))
                output.square().mean().backward()
                self.assertIsNone(encoder.backbone.scale.grad)
                self.assertTrue(all(p.grad is not None for p in encoder.adapter.parameters()))
            self.assertEqual(encoder.backbone.scale.item(), 3.0)
            saved = encoder.state_dict()
            self.assertFalse(any(key.startswith("backbone.") for key in saved))
            self.assertNotIn("pretrained_loaded", saved)
            restored = FrozenVGGTEncoder(pretrained_path=str(path), adapter_channels=4)
            restored.load_state_dict(saved, strict=True)
            masks = torch.rand(1, 2, 28, 42)
            torch.testing.assert_close(restored(masks), encoder(masks))

    def test_nested_restore_and_strict_adapter_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"aggregator.scale": torch.tensor(3.0)}, path)
            original = nn.Sequential(FrozenVGGTEncoder(pretrained_path=str(path), adapter_channels=4))
            masks = torch.rand(1, 1, 28, 28)
            expected = original(masks)
            saved = original.state_dict()
            self.assertFalse(any("backbone." in key for key in saved))
            restored = nn.Sequential(FrozenVGGTEncoder(pretrained_path=str(path), adapter_channels=4))
            restored.load_state_dict(saved)
            torch.testing.assert_close(expected, restored(masks))
            del saved["0.adapter.0.weight"]
            with self.assertRaisesRegex(RuntimeError, "adapter.0.weight"):
                restored.load_state_dict(saved)

    def test_legacy_full_checkpoint_loads_without_external_file(self):
        encoder = FrozenVGGTEncoder(pretrained_path="/missing/model.pt", adapter_channels=4)
        saved = encoder.state_dict()
        saved["backbone.scale"] = torch.tensor(3.0)
        saved["pretrained_loaded"] = torch.tensor(True)
        encoder.load_state_dict(saved)
        self.assertEqual(encoder.backbone.scale.item(), 3.0)
        encoder(torch.rand(1, 1, 28, 28))
        self.assertNotIn("backbone.scale", encoder.state_dict())

    def test_compact_checkpoint_requires_external_weights(self):
        encoder = FrozenVGGTEncoder(pretrained_path="/missing/model.pt")
        with self.assertRaises(FileNotFoundError):
            encoder.load_state_dict(encoder.state_dict())

    def test_rejects_incomplete_pretraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"depth_head.unused": torch.tensor(1.)}, path)
            encoder = FrozenVGGTEncoder(pretrained_path=str(path))
            with self.assertRaises(RuntimeError):
                encoder(torch.rand(1, 1, 28, 28))
            self.assertFalse(encoder.pretrained_loaded.item())

    def test_rejects_non_patch_aligned_dimensions(self):
        encoder = FrozenVGGTEncoder()
        with self.assertRaisesRegex(ValueError, "multiples of 14"):
            encoder(torch.rand(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
