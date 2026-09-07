"""Voxel-supervised AutoCAR training module for Stage-2 NPZ data."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch
from lightning import LightningModule

from src.geometry.voxel_grid import VoxelGrid, resample_binary_volume_nearest
from src.metrics import masked_dice_3d, masked_ssim_3d, structural_similarity_3d
from src.modules.autocar import AutoCAR
from src.modules.loss import DiceLossLogit
from src.modules.sparse_utils import (
    rasterize_sparse_channel,
    sparse_coordinates_bxyz,
    sparse_features,
)
from src.modules.voxel_supervision import sample_sparse_volume_targets


class AutoCARVoxelLit(LightningModule):
    """Train reconstruction directly from supplied projections and GT voxels."""

    def __init__(
        self,
        recon_net,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        *,
        prediction_threshold: float = 0.5,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ssim_window_size: int = 7,
        ssim_chunk_depth: int = 8,
        ssim_every_n_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.recon_net = AutoCAR(recon_net)
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_dice = DiceLossLogit()
        if not 0.0 <= float(prediction_threshold) <= 1.0:
            raise ValueError("prediction_threshold must lie in [0,1].")
        if not all(
            math.isfinite(float(value))
            for value in (bce_weight, dice_weight)
        ):
            raise ValueError("Loss weights must be finite.")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative with a positive sum.")
        if (
            isinstance(ssim_window_size, bool)
            or int(ssim_window_size) != ssim_window_size
            or int(ssim_window_size) < 3
            or int(ssim_window_size) % 2 == 0
        ):
            raise ValueError("ssim_window_size must be an odd integer >= 3.")
        if (
            isinstance(ssim_chunk_depth, bool)
            or int(ssim_chunk_depth) != ssim_chunk_depth
            or int(ssim_chunk_depth) < 1
        ):
            raise ValueError("ssim_chunk_depth must be a positive integer.")
        if (
            isinstance(ssim_every_n_epochs, bool)
            or int(ssim_every_n_epochs) != ssim_every_n_epochs
            or int(ssim_every_n_epochs) < 1
        ):
            raise ValueError("ssim_every_n_epochs must be a positive integer.")

    @property
    def sparse_backend(self) -> str:
        return self.recon_net.sparse_backend

    def on_train_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        if hasattr(datamodule, "set_train_epoch"):
            datamodule.set_train_epoch(self.current_epoch)

    def _forward_batch(self, batch: Dict[str, Any]):
        masks = batch["images"].to(dtype=torch.float32)
        if masks.ndim != 5 or masks.shape[2] != 1:
            raise ValueError(
                f"Expected images [B,V,1,H,W], got {tuple(masks.shape)}."
            )
        matrices = batch["world2pix4x4"].to(dtype=torch.float32)
        return self.recon_net(masks[:, :, 0], matrices)

    def _sample_voxel_targets(
        self, prediction, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample native GT and return a native-FOV validity mask."""

        coordinates = sparse_coordinates_bxyz(prediction, self.sparse_backend)
        features = sparse_features(prediction)
        device = features.device
        coordinates = coordinates.to(device=device)
        bbox_min = self.recon_net.ray_casting.bbox_min.to(device=device)
        voxel_size = self.recon_net.ray_casting.voxel_size
        return sample_sparse_volume_targets(
            coordinates,
            reference_features=features,
            bbox_min_xyz_mm=bbox_min,
            voxel_size_mm=voxel_size,
            volumes_zyx=batch["gt_volume_zyx"],
            spacing_xyz_mm=batch["gt_spacing_xyz_mm"],
            origin_xyz_mm=batch["gt_origin_xyz_mm"],
            target_to_source_offset_xyz_mm=batch[
                "projection_center_offset_xyz_mm"
            ],
        )

    def model_step(self, batch: Dict[str, Any]):
        prediction, _ = self._forward_batch(batch)
        logits = sparse_features(prediction)[:, 0]
        if logits.numel() == 0:
            raise RuntimeError(
                "Sparse backward projection produced no supported voxels; "
                "check camera geometry, masks, and support_views."
            )
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "The reconstruction network produced non-finite sparse logits."
            )
        targets, valid_targets = self._sample_voxel_targets(prediction, batch)
        if not torch.any(valid_targets):
            raise RuntimeError(
                "No sparse prediction coordinates fall inside the native "
                "ground-truth field of view; check offsets and grid origins."
            )
        valid_logits = logits[valid_targets]
        valid_values = targets[valid_targets]
        loss_bce = self.criterion_bce(valid_logits, valid_values)
        loss_dice = self.criterion_dice(valid_logits, valid_values)
        loss = (
            float(self.hparams.bce_weight) * loss_bce
            + float(self.hparams.dice_weight) * loss_dice
        )
        for name, value in (
            ("BCE loss", loss_bce),
            ("Dice loss", loss_dice),
            ("combined loss", loss),
        ):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"The {name} is non-finite.")
        return (
            loss,
            loss_bce,
            loss_dice,
            prediction,
            targets,
            valid_targets,
        )

    @staticmethod
    def _batch_identity(batch: Dict[str, Any]) -> str:
        """Format case and source paths for progress and failure messages."""

        def values(key: str) -> tuple[str, ...]:
            raw = batch.get(key, "<missing>")
            if isinstance(raw, str):
                return (raw,)
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu()
                if raw.ndim == 0:
                    return (str(raw.item()),)
                return tuple(str(item) for item in raw.tolist())
            try:
                return tuple(str(item) for item in raw)
            except TypeError:
                return (str(raw),)

        case_ids = values("case_id")
        projection_paths = values("projection_path")
        return (
            f"case_id={case_ids[0]!r}, projection_path={projection_paths[0]!r}"
            if len(case_ids) == len(projection_paths) == 1
            else f"case_ids={case_ids!r}, projection_paths={projection_paths!r}"
        )

    def _raise_step_failure(
        self,
        phase: str,
        batch: Dict[str, Any],
        batch_idx: int,
        error: Exception,
    ) -> None:
        identity = self._batch_identity(batch)
        raise RuntimeError(
            f"{phase} failed at epoch={self.current_epoch + 1}, "
            f"batch={batch_idx + 1}, {identity}: "
            f"{type(error).__name__}: {error}"
        ) from error

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        try:
            return self._training_step(batch, batch_idx)
        except Exception as error:
            self._raise_step_failure("training", batch, batch_idx, error)

    def _training_step(self, batch: Dict[str, Any], batch_idx: int):
        (
            loss,
            loss_bce,
            loss_dice,
            prediction,
            _,
            valid_targets,
        ) = self.model_step(batch)
        batch_size = int(batch["images"].shape[0])
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_bce", loss_bce, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_dice", loss_dice, on_epoch=True, batch_size=batch_size)
        self._log_sparse_diagnostics(
            "train",
            prediction,
            valid_targets,
            batch,
            batch_size=batch_size,
            on_step=True,
            sync_dist=False,
        )
        return loss

    def _log_sparse_diagnostics(
        self,
        prefix: str,
        prediction,
        valid_targets: torch.Tensor,
        batch: Dict[str, Any],
        *,
        batch_size: int,
        on_step: bool,
        sync_dist: bool,
    ) -> None:
        """Log visual-hull size and native-FOV coverage for auditability."""

        sparse_voxels = sparse_features(prediction).shape[0]
        self.log(
            f"{prefix}/sparse_voxels",
            float(sparse_voxels),
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log(
            f"{prefix}/valid_sparse_fraction",
            valid_targets.to(dtype=torch.float32).mean(),
            on_step=on_step,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        if "pair_angle_deg" in batch:
            self.log(
                f"{prefix}/pair_angle_deg",
                batch["pair_angle_deg"].to(dtype=torch.float32).mean(),
                on_step=on_step,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def _dense_case_metrics(
        self, prediction, batch: Dict[str, Any], *, include_ssim: bool
    ) -> list[dict[str, float]]:
        shape_xyz = self.recon_net.ray_casting.spatial_shape_xyz
        dense_batch_tensor = rasterize_sparse_channel(
            prediction,
            backend=self.sparse_backend,
            spatial_shape_xyz=shape_xyz,
            batch_size=int(batch["images"].shape[0]),
            output_device="cpu",
        )
        # NumPy cannot represent bfloat16. Preserve float16 otherwise to avoid
        # an additional 256 MiB float32 copy for the default 400^3 grid.
        if dense_batch_tensor.dtype == torch.bfloat16:
            dense_batch_tensor = dense_batch_tensor.float()
        dense_batch = dense_batch_tensor.detach().numpy()
        target_grid = VoxelGrid(
            shape_zyx=tuple(reversed(shape_xyz)),
            spacing_xyz_mm=(self.recon_net.ray_casting.voxel_size,) * 3,
            origin_xyz_mm=tuple(
                float(value)
                for value in self.recon_net.ray_casting.bbox_min.detach().cpu()
            ),
        )
        results = []
        for batch_index, prediction_zyx in enumerate(dense_batch):
            gt_zyx = batch["gt_volume_zyx"][batch_index].detach().cpu().numpy()
            spacing_xyz = batch["gt_spacing_xyz_mm"][batch_index].detach().cpu().numpy()
            origin_xyz = batch["gt_origin_xyz_mm"][batch_index].detach().cpu().numpy()
            offset_xyz = batch["projection_center_offset_xyz_mm"][
                batch_index
            ].detach().cpu().numpy()
            source_grid = VoxelGrid(
                shape_zyx=gt_zyx.shape,
                spacing_xyz_mm=tuple(float(value) for value in spacing_xyz),
                origin_xyz_mm=tuple(float(value) for value in origin_xyz),
            )
            aligned_gt, valid_fov = resample_binary_volume_nearest(
                gt_zyx,
                source_grid,
                target_grid,
                target_to_source_offset_xyz_mm=offset_xyz,
            )
            case_metrics: dict[str, float] = {
                "masked_dice_3d": masked_dice_3d(
                    aligned_gt,
                    prediction_zyx,
                    mask=valid_fov,
                    prediction_threshold=float(self.hparams.prediction_threshold),
                )
            }
            if include_ssim:
                case_metrics["ssim_3d"] = float(
                    structural_similarity_3d(
                        aligned_gt,
                        prediction_zyx,
                        data_range=1.0,
                        window_size=int(self.hparams.ssim_window_size),
                        roi_mask=valid_fov,
                        chunk_depth=int(self.hparams.ssim_chunk_depth),
                    )
                )
                foreground_windows = aligned_gt | (
                    prediction_zyx >= float(self.hparams.prediction_threshold)
                )
                try:
                    case_metrics["masked_ssim_3d"] = masked_ssim_3d(
                        aligned_gt,
                        prediction_zyx,
                        mask=foreground_windows,
                        roi_mask=valid_fov,
                        data_range=1.0,
                        window_size=int(self.hparams.ssim_window_size),
                        chunk_depth=int(self.hparams.ssim_chunk_depth),
                    )
                except ValueError:
                    pass
            results.append(case_metrics)
        return results

    def _evaluation_step(self, batch: Dict[str, Any], prefix: str) -> None:
        (
            loss,
            loss_bce,
            loss_dice,
            prediction,
            _,
            valid_targets,
        ) = self.model_step(batch)
        is_test = prefix == "test"
        include_ssim = is_test or (
            (self.current_epoch + 1) % int(self.hparams.ssim_every_n_epochs) == 0
        )
        metrics = self._dense_case_metrics(
            prediction, batch, include_ssim=include_ssim
        )
        batch_size = len(metrics)
        self.log(f"{prefix}/loss", loss, on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/loss_bce", loss_bce, on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/loss_dice", loss_dice, on_epoch=True, batch_size=batch_size)
        self._log_sparse_diagnostics(
            prefix,
            prediction,
            valid_targets,
            batch,
            batch_size=batch_size,
            on_step=False,
            sync_dist=True,
        )
        for name in sorted({key for item in metrics for key in item}):
            values = [item[name] for item in metrics if name in item]
            self.log(
                f"{prefix}/{name}",
                float(np.mean(values)),
                on_epoch=True,
                prog_bar=name == "masked_dice_3d",
                batch_size=len(values),
                sync_dist=True,
            )

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        phase = "validation"
        if getattr(self.trainer, "sanity_checking", False):
            phase = "validation sanity check"
        self.print(
            f"[AutoCAR] {phase}: epoch={self.current_epoch + 1}, "
            f"batch={batch_idx + 1}, {self._batch_identity(batch)}",
            flush=True,
        )
        try:
            self._evaluation_step(batch, "val")
        except Exception as error:
            self._raise_step_failure(phase, batch, batch_idx, error)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.print(
            f"[AutoCAR] test: epoch={self.current_epoch + 1}, "
            f"batch={batch_idx + 1}, {self._batch_identity(batch)}",
            flush=True,
        )
        try:
            self._evaluation_step(batch, "test")
        except Exception as error:
            self._raise_step_failure("test", batch, batch_idx, error)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/masked_dice_3d",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }
