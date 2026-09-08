"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import rootutils
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.modules.encoder2d import StackedHourGlassEncoder
from src.modules.ray_casting import SparseBackwardProjection


class AutoCAR(torch.nn.Module):
    """module for AutoCAR in PyTorch.
    - Used for both training and inference.
    - Without dependency on PyTorch Lightning.
    - Only include reconstrution, no rendering or loss calculation.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = OmegaConf.create(cfg)
        self.sparse_backend = str(cfg.get("sparse_backend", "minkowski"))
        self.expected_view_count = int(cfg.get("expected_view_count", 2))
        if self.expected_view_count < 1:
            raise ValueError("expected_view_count must be positive.")
        encoder_channels = int(cfg.encoder2d.out_ch)
        self.encoder_input = str(
            cfg.encoder2d.get("input", "legacy_exp_distance")
        )
        if self.encoder_input not in {"mask", "legacy_exp_distance"}:
            raise ValueError(
                "encoder2d.input must be 'mask' or 'legacy_exp_distance'."
            )
        working_image_dim = cfg.encoder2d.get("working_image_dim")
        if working_image_dim is None:
            self.encoder_working_image_dim = None
        else:
            if isinstance(working_image_dim, bool):
                raise ValueError("encoder2d.working_image_dim must be an integer.")
            self.encoder_working_image_dim = int(working_image_dim)
            if (
                self.encoder_working_image_dim != working_image_dim
                or self.encoder_working_image_dim < 256
                or self.encoder_working_image_dim % 256 != 0
            ):
                raise ValueError(
                    "encoder2d.working_image_dim must be a positive multiple "
                    "of 256 so the four-level hourglass has aligned shapes."
                )
        self.encoder2d = StackedHourGlassEncoder(encoder_channels)
        self.ray_casting = SparseBackwardProjection(
            cfg.ray_casting.bbox_min,
            cfg.ray_casting.bbox_max,
            cfg.ray_casting.LODs,
            max_pixel_distance=cfg.ray_casting.get(
                "max_pixel_distance", 0.5
            ),
            support_views=cfg.ray_casting.get("support_views", 2),
            fusion=cfg.ray_casting.get("fusion", "mean"),
            backend=self.sparse_backend,
            ray_chunk_size=cfg.ray_casting.get("ray_chunk_size", 4096),
            pixel_center_offset=cfg.ray_casting.get(
                "pixel_center_offset", 0.0
            ),
            projection_pixel_order=cfg.ray_casting.get(
                "projection_pixel_order", "rc"
            ),
            include_distance_feature=cfg.ray_casting.get(
                "include_distance_feature", False
            ),
            candidate_mode=cfg.ray_casting.get("candidate_mode", "ray"),
            voxel_chunk_size=cfg.ray_casting.get(
                "voxel_chunk_size", 262_144
            ),
            distance_sampling=cfg.ray_casting.get(
                "distance_sampling", "nearest"
            ),
        )
        expected_input_channels = self.ray_casting.output_channels(
            encoder_channels, self.expected_view_count
        )
        configured_input_channels = int(cfg.unet3d.in_channels)
        if configured_input_channels != expected_input_channels:
            raise ValueError(
                "unet3d.in_channels does not match the backward-projection "
                f"feature width: configured {configured_input_channels}, "
                f"expected {expected_input_channels}."
            )
        architecture = str(cfg.unet3d.get("architecture", "18A")).upper()
        if self.sparse_backend == "spconv":
            from src.modules.spconv_unet import (
                SpconvUNet18A,
                SpconvUNet34C,
            )

            unet_types = {"18A": SpconvUNet18A, "34C": SpconvUNet34C}
        elif self.sparse_backend == "minkowski":
            from src.modules.minkunet import MinkUNet18A, MinkUNet34C

            unet_types = {"18A": MinkUNet18A, "34C": MinkUNet34C}
        else:
            raise ValueError(
                "AutoCAR requires sparse_backend='spconv' or 'minkowski'; "
                f"got {self.sparse_backend!r}. Use backend='raw' only when "
                "testing SparseBackwardProjection directly."
            )
        try:
            unet_type = unet_types[architecture]
        except KeyError as error:
            raise ValueError(
                f"Unknown unet3d.architecture {architecture!r}; "
                "expected '18A' or '34C'."
            ) from error
        self.unet3d = unet_type(
            configured_input_channels, cfg.unet3d.out_channels
        )

    @staticmethod
    def _resize_encoder_input(
        encoder_input: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Resize `[B,V,H,W]` views without mixing cases or views."""

        if encoder_input.ndim != 4:
            raise ValueError(
                "Encoder input must be [B,V,H,W], got "
                f"{tuple(encoder_input.shape)}."
            )
        batch_size, view_count, height, width = encoder_input.shape
        if (height, width) == output_hw:
            return encoder_input
        flattened = encoder_input.reshape(
            batch_size * view_count, 1, height, width
        )
        resized = F.interpolate(
            flattened,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, view_count, *output_hw)

    @staticmethod
    def _resize_encoder_features(
        features: torch.Tensor,
        output_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Restore `[B,V,C,H,W]` features to the native detector grid."""

        if features.ndim != 5:
            raise ValueError(
                "Encoder features must be [B,V,C,H,W], got "
                f"{tuple(features.shape)}."
            )
        batch_size, view_count, channels, height, width = features.shape
        if (height, width) == output_hw:
            return features
        flattened = features.reshape(
            batch_size * view_count, channels, height, width
        )
        resized = F.interpolate(
            flattened,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(
            batch_size, view_count, channels, *output_hw
        )

    def _encode_at_working_resolution(
        self, encoder_input: torch.Tensor
    ) -> torch.Tensor:
        native_hw = tuple(int(value) for value in encoder_input.shape[-2:])
        if self.encoder_working_image_dim is None:
            working_hw = native_hw
        else:
            working_hw = (
                self.encoder_working_image_dim,
                self.encoder_working_image_dim,
            )
        working_input = self._resize_encoder_input(encoder_input, working_hw)
        working_features = self.encoder2d(working_input)
        return self._resize_encoder_features(working_features, native_hw)

    def forward(self, masks, world2pix4x4):
        """
        masks: B x V x H x W, pytorch tensor
        world2pix4x4: B x V x 4 x 4, pytorch tensor
        """
        # B = masks.shape[0]
        # V = masks.shape[1]
        # H, W = masks.shape[2:]

        masks_float = masks.to(dtype=torch.float32)
        if self.encoder_input == "mask":
            distance_maps = self.ray_casting.distance_maps_from_masks(
                masks_float
            )
            encoder_input = masks_float
        else:
            # Preserve the released checkpoint path without making Kornia a
            # dependency of the maintained NPZ environment.
            import kornia

            distance_maps = kornia.contrib.distance_transform(masks_float, 3)
            encoder_input = torch.exp(-distance_maps)

        feature = self._encode_at_working_resolution(encoder_input)
        # Features are restored to the native detector grid before ray casting.
        if feature.shape[-2:] != masks.shape[-2:]:
            raise RuntimeError(
                "Encoder features do not match the native detector grid: "
                f"features={tuple(feature.shape[-2:])}, "
                f"masks={tuple(masks.shape[-2:])}."
            )

        if masks.shape[1] != self.expected_view_count:
            raise ValueError(
                f"Expected {self.expected_view_count} views, got {masks.shape[1]}."
            )
        sparse_volume, world_coords = self.ray_casting(
            distance_maps, feature, world2pix4x4
        )
        if world_coords.shape[0] == 0:
            raise RuntimeError(
                "Sparse backward projection produced an empty visual hull; "
                "check camera geometry, masks, and max_pixel_distance."
            )
        pred = self.unet3d(sparse_volume)
        return pred, world_coords


if __name__ == "__main__":
    import tqdm
    from src.dataset.imagecas_enchanced import ImageCASEnhancedDataset

    dataset = ImageCASEnhancedDataset(
        "data/imagecas_left_surface", "data/filelist.txt"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=lambda x: x
    )
    model = AutoCAR().cuda()
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # print(data)
        model(data)
        # if i > 10:
        #     break
