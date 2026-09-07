"""
This file is released under  CC BY-NC-ND 4.0 license.
If you have any question, please contact zhuyh19@mails.tsinghua.edu.cn
"""

import rootutils
import torch
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

        feature = self.encoder2d(encoder_input)
        # feature: B x V x C x H x W

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
