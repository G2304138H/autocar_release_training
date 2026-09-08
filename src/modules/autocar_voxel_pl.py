"""Voxel-supervised AutoCAR training module for Stage-2 NPZ data."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

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
        numerical_debug: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.recon_net = AutoCAR(recon_net)
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_dice = DiceLossLogit()
        self._numerical_debug_context: tuple[str, int, str] | None = None
        self._numerical_debug_accumulation_cases: list[str] = []
        self._numerical_debug_last_optimizer_cases: tuple[str, ...] = ()
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
        if bool(self.hparams.numerical_debug):
            self._assert_finite_model_state("before forward")
        masks = batch["images"].to(dtype=torch.float32)
        if masks.ndim != 5 or masks.shape[2] != 1:
            raise ValueError(
                f"Expected images [B,V,1,H,W], got {tuple(masks.shape)}."
            )
        matrices = batch["world2pix4x4"].to(dtype=torch.float32)
        output = self.recon_net(masks[:, :, 0], matrices)
        if bool(self.hparams.numerical_debug):
            # BatchNorm buffers can change during a training forward pass.
            self._assert_finite_model_state("after forward")
        return output

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
            statistics = (
                f" ({self._tensor_statistics(logits)})"
                if bool(self.hparams.numerical_debug)
                else ""
            )
            raise FloatingPointError(
                "The reconstruction network produced non-finite sparse logits"
                f"{statistics}."
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
        """Format source identity and selected-view metadata for diagnostics."""

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
        identity = (
            f"case_id={case_ids[0]!r}, projection_path={projection_paths[0]!r}"
            if len(case_ids) == len(projection_paths) == 1
            else f"case_ids={case_ids!r}, projection_paths={projection_paths!r}"
        )
        metadata = []
        for key in (
            "view_indices",
            "source_view_indices",
            "view_labels",
            "pair_angle_deg",
        ):
            if key not in batch:
                continue
            raw = batch[key]
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().tolist()
            elif isinstance(raw, np.ndarray):
                raw = raw.tolist()
            metadata.append(f"{key}={raw!r}")
        return identity + (", " + ", ".join(metadata) if metadata else "")

    @staticmethod
    def _tensor_values(tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach()
        return detached.coalesce().values() if detached.is_sparse else detached

    @classmethod
    def _tensor_statistics(cls, tensor: torch.Tensor) -> str:
        """Return bounded statistics without copying a healthy model tensor."""

        values = cls._tensor_values(tensor)
        finite = torch.isfinite(values)
        total_count = values.numel()
        finite_count = int(finite.sum().item())
        if values.is_floating_point() or values.is_complex():
            nan_count = int(torch.isnan(values).sum().item())
            posinf_count = int(torch.isposinf(values).sum().item())
            neginf_count = int(torch.isneginf(values).sum().item())
        else:
            nan_count = posinf_count = neginf_count = 0
        summary = (
            f"shape={tuple(values.shape)}, dtype={values.dtype}, "
            f"finite={finite_count}/{total_count}, nan={nan_count}, "
            f"+inf={posinf_count}, -inf={neginf_count}"
        )
        if finite_count:
            finite_values = values[finite]
            if finite_values.is_complex():
                finite_values = finite_values.abs()
            finite_values = finite_values.to(dtype=torch.float64)
            summary += (
                f", finite_min={finite_values.min().item():.6g}, "
                f"finite_max={finite_values.max().item():.6g}, "
                f"finite_mean={finite_values.mean().item():.6g}"
            )
        return summary

    @classmethod
    def _nonfinite_tensors(
        cls,
        tensors: Iterable[tuple[str, torch.Tensor]],
        *,
        limit: int = 5,
    ) -> list[str]:
        if limit <= 0:
            return []
        grouped: dict[
            torch.device, list[tuple[str, torch.Tensor, torch.Tensor]]
        ] = {}
        for name, tensor in tensors:
            values = cls._tensor_values(tensor)
            grouped.setdefault(values.device, []).append(
                (name, tensor, torch.isfinite(values).all())
            )

        failures = []
        # Resolve all checks on a device in one synchronization rather than
        # synchronizing once for every layer in this intentionally thorough mode.
        for entries in grouped.values():
            finite_by_tensor = torch.stack([entry[2] for entry in entries])
            failed_indices = (
                (~finite_by_tensor).nonzero(as_tuple=False).flatten().cpu().tolist()
            )
            for index in failed_indices:
                name, tensor, _ = entries[index]
                failures.append(f"{name}: {cls._tensor_statistics(tensor)}")
                if len(failures) == limit:
                    return failures
        return failures

    def _set_numerical_debug_context(
        self,
        phase: str,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        if bool(self.hparams.numerical_debug):
            self._numerical_debug_context = (
                phase,
                batch_idx,
                self._batch_identity(batch),
            )

    def _numerical_debug_location(self) -> str:
        context = self._numerical_debug_context
        if context is None:
            return f"epoch={self.current_epoch + 1}, batch=<unknown>"
        phase, batch_idx, identity = context
        return (
            f"phase={phase}, epoch={self.current_epoch + 1}, "
            f"batch={batch_idx + 1}, {identity}"
        )

    def _last_optimizer_context(self) -> str:
        if not self._numerical_debug_last_optimizer_cases:
            return "no preceding optimizer-step cases were recorded"
        return (
            "preceding optimizer-step cases=["
            + "; ".join(self._numerical_debug_last_optimizer_cases)
            + "]"
        )

    def _assert_finite_model_state(self, when: str) -> None:
        failures = self._nonfinite_tensors(
            (
                (f"parameter:{name}", parameter)
                for name, parameter in self.named_parameters()
            )
        )
        failures.extend(
            self._nonfinite_tensors(
                ((f"buffer:{name}", buffer) for name, buffer in self.named_buffers()),
                limit=max(0, 5 - len(failures)),
            )
        )
        if failures:
            raise FloatingPointError(
                f"Numerical troubleshooting found non-finite model state {when}; "
                f"{self._numerical_debug_location()}; "
                f"{self._last_optimizer_context()}; affected tensors: "
                + " | ".join(failures)
            )

    def _assert_finite_gradients(self, when: str) -> None:
        failures = self._nonfinite_tensors(
            (
                (f"gradient:{name}", parameter.grad)
                for name, parameter in self.named_parameters()
                if parameter.grad is not None
            )
        )
        if failures:
            accumulated = tuple(self._numerical_debug_accumulation_cases)
            raise FloatingPointError(
                f"Numerical troubleshooting found non-finite gradients {when}, "
                "before the optimizer could update model weights; "
                f"{self._numerical_debug_location()}; "
                "accumulated cases=["
                + "; ".join(accumulated)
                + "]; affected tensors: "
                + " | ".join(failures)
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
        self._set_numerical_debug_context("training", batch, batch_idx)
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
        if bool(self.hparams.numerical_debug):
            self._numerical_debug_accumulation_cases.append(
                self._batch_identity(batch)
            )
        return loss

    def on_after_backward(self) -> None:
        """Stop at the first case that introduces a non-finite gradient."""

        if bool(self.hparams.numerical_debug):
            self._assert_finite_gradients("after backward")

    def on_before_optimizer_step(self, optimizer) -> None:
        """Make the final finite check before Adam mutates parameters/state."""

        if not bool(self.hparams.numerical_debug):
            return
        self._assert_finite_gradients("at optimizer-step boundary")
        self._numerical_debug_last_optimizer_cases = tuple(
            self._numerical_debug_accumulation_cases
        )
        self._numerical_debug_accumulation_cases.clear()

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
        self._set_numerical_debug_context(phase, batch, batch_idx)
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
        self._set_numerical_debug_context("test", batch, batch_idx)
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
