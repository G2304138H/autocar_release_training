"""Frozen VGGT multi-view features with a trainable dense convolutional adapter."""
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FrozenVGGTEncoder(nn.Module):
    def __init__(self, out_ch=12, pretrained_path=None, pretrained_repo="facebook/VGGT-1B",
                 backbone_precision="bf16", adapter_channels=64):
        super().__init__()
        try:
            from vggt.models.aggregator import Aggregator
        except ImportError as error:
            raise ImportError("Install requirements/vggt.txt to use the VGGT encoder.") from error
        if backbone_precision not in {"bf16", "fp32"}:
            raise ValueError("VGGT backbone_precision must be bf16 or fp32.")
        self.backbone = Aggregator()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.pretrained_path = pretrained_path
        self.pretrained_repo = pretrained_repo
        self.backbone_precision = backbone_precision
        # Persist readiness alongside the full backbone state. Restored AutoCAR
        # checkpoints are self-contained and never need the original weight file.
        self.register_buffer("pretrained_loaded", torch.tensor(False))
        self.adapter = nn.Sequential(
            nn.Conv2d(2048, adapter_channels, 1), nn.GELU(),
            nn.Conv2d(adapter_channels, adapter_channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(adapter_channels, out_ch, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _ensure_pretrained(self):
        if bool(self.pretrained_loaded.item()):
            return
        if self.pretrained_path:
            path = Path(self.pretrained_path).expanduser()
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(self.pretrained_repo, filename="model.pt")
        print(f"Loading frozen VGGT backbone from {path}", flush=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        prefix = "aggregator."
        if any(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix)}
        # Reject incomplete/mismatched weights instead of freezing random layers.
        self.backbone.load_state_dict(state, strict=True)
        self.pretrained_loaded.fill_(True)

    def forward(self, masks):
        if masks.ndim != 4:
            raise ValueError("VGGT expects masks [B,V,H,W].")
        batch, views, height, width = masks.shape
        if height % 14 or width % 14:
            raise ValueError("VGGT working image dimensions must be multiples of 14.")
        self._ensure_pretrained()
        images = masks.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
        use_bf16 = images.is_cuda and self.backbone_precision == "bf16"
        if use_bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("VGGT bf16 requires a supported GPU; set backbone_precision=fp32.")
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        # no_grad, rather than inference_mode: adapter backward must save tokens.
        with torch.no_grad(), context:
            layers, patch_start = self.backbone(images)
            tokens = layers[-1][:, :, patch_start:]
        grid_h, grid_w = height // 14, width // 14
        if tokens.shape[2:] != (grid_h * grid_w, 2048):
            raise RuntimeError(f"Unexpected VGGT patch features: {tuple(tokens.shape)}")
        features = tokens.reshape(batch * views, grid_h, grid_w, 2048)
        features = features.permute(0, 3, 1, 2).contiguous().float()
        del layers, tokens
        features = self.adapter(features)
        features = F.interpolate(features, (height, width), mode="bilinear", align_corners=False)
        return features.reshape(batch, views, features.shape[1], height, width)
