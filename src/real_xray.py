"""AutoCAR on real X-ray angiograms (CoronaryDominance keyframes exported by angioseg-nnunet step 08).

Same inputs, view choice, camera, centring and outputs as the PPCAR-Net `real-xray` branch (see
src/real_xray_common.py). AutoCAR part:
  * camera matrices from src.geometry.projection_geometry.ProjectionGeometry.from_angles with the case's SID / SOD /
    detector pixel (acquisition camera) - the same contract as Stage2NPZDataset, isocentre at the world origin;
  * the trained network (recon_net: 2D encoder, sparse back-projection, sparse 3D U-Net) as in src/predict_npz.py,
    rasterised on the checkpoint's box (bbox_min..bbox_max, LOD voxel size); occupancy = sigmoid >= 0.5;
  * the radius-coloured surface comes from src.geometry.vascular_surface.extract_vascular_graph_surface.

Usage (from the repository root; spconv needs a CUDA GPU):
  python -m src.real_xray --config configs/real_xray/lca.json --checkpoint /path/to/best.ckpt \
      --input_dir /dataset/reny0012/angioseg/recon_inputs/xravos_A_v2/lca \
      --output_dir /dataset/reny0012/angioseg/recon_outputs/autocar_xravos_A_v2/lca --limit 5 --output_mode summary
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.geometry.projection_geometry import ProjectionGeometry
from src.modules.autocar_voxel_pl import AutoCARVoxelLit
from src.modules.sparse_utils import rasterize_sparse_channel
from src.real_xray_common import Prediction, base_parser, config_from_args, run


class AutoCARProjector:
    def __init__(self, theta_deg: float, phi_deg: float, cam: dict):
        self.geometry = _geometry([theta_deg], [phi_deg], cam)
        self.source_mm = np.asarray(self.geometry.source_xyz_mm[0], np.float64)

    def project_rc(self, points_mm: np.ndarray) -> np.ndarray:
        return np.asarray(self.geometry.project_points_rc(np.asarray(points_mm, np.float64), 0), np.float64)


def _geometry(theta_deg, phi_deg, cam) -> ProjectionGeometry:
    return ProjectionGeometry.from_angles(theta_deg=np.asarray(theta_deg, np.float64),
                                          phi_deg=np.asarray(phi_deg, np.float64), image_dim=int(cam["image_dim"]),
                                          sid_mm=float(cam["sid_mm"]), pixel_spacing_mm=float(cam["pixel_mm"]),
                                          source_to_isocenter_mm=float(cam["sod_mm"]))


class AutoCARBackend:
    name = "AutoCAR"

    def __init__(self, config: dict):
        ckpt = config.get("checkpoint")
        if not ckpt:
            raise ValueError("set the checkpoint (config 'checkpoint' or --checkpoint)")
        dev = config.get("device") or "auto"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model = AutoCARVoxelLit.load_from_checkpoint(ckpt, map_location="cpu")
        if self.model.sparse_backend == "spconv" and self.device.type != "cuda":
            raise RuntimeError("this checkpoint uses spconv, which needs a CUDA GPU")
        self.model.to(self.device)
        self.model.freeze()
        rn = self.model.recon_net
        self.min_views, self.max_views = int(rn.min_view_count), int(rn.max_view_count)
        self.threshold = float(self.model.hparams.get("prediction_threshold", 0.5))
        self.checkpoint = str(Path(ckpt).resolve())

    def describe(self) -> dict:
        rc = self.model.recon_net.ray_casting
        return dict(method=self.name, checkpoint=self.checkpoint, device=str(self.device),
                    sparse_backend=self.model.sparse_backend, views=[self.min_views, self.max_views],
                    bbox_min_xyz_mm=[float(x) for x in rc.bbox_min.detach().cpu()],
                    bbox_max_xyz_mm=[float(x) for x in rc.bbox_max.detach().cpu()], voxel_size_mm=rc.voxel_size,
                    max_pixel_distance=rc.max_pixel_distance, support_views=rc.support_views, fusion=rc.fusion,
                    threshold=self.threshold)

    def predict(self, images, theta_deg, phi_deg, cam) -> Prediction:
        rn = self.model.recon_net
        rc = rn.ray_casting
        geo = _geometry(theta_deg, phi_deg, cam)
        masks = torch.from_numpy(np.asarray(images, np.float32)[None]).to(self.device)
        mats = torch.from_numpy(np.asarray(geo.world2pix4x4, np.float32)[None]).to(self.device)
        shape_xyz = tuple(int(v) for v in rc.spatial_shape_xyz)
        vs = float(rc.voxel_size)
        empty = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            try:
                pred, _ = rn(masks, mats)
                dense = rasterize_sparse_channel(pred, backend=self.model.sparse_backend, spatial_shape_xyz=shape_xyz,
                                                 batch_size=1, output_device="cpu")[0].float().numpy()
            except RuntimeError as e:
                if "empty visual hull" not in str(e):
                    raise
                empty = True                    # the input views share no consistent 3D support
                dense = np.zeros(shape_xyz[::-1], np.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        bbox_min = rc.bbox_min.detach().cpu().numpy().astype(np.float64)
        return Prediction(volume_zyx=dense, origin_center_xyz_mm=bbox_min + 0.5 * vs, spacing_xyz_mm=np.full(3, vs),
                          threshold=self.threshold, seconds=seconds,
                          info=dict(empty_visual_hull=empty, bbox_min_xyz_mm=bbox_min.tolist(), voxel_size_mm=vs))

    def projector(self, theta_deg, phi_deg, cam):
        return AutoCARProjector(theta_deg, phi_deg, cam)

    def surface(self, mask_zyx, origin_center_xyz_mm, spacing_xyz_mm):
        from src.geometry.vascular_surface import extract_vascular_graph_surface
        sp = np.asarray(spacing_xyz_mm, np.float64)
        s = extract_vascular_graph_surface(mask_zyx.astype(np.float32), threshold=0.5,
                                           origin_xyz_mm=np.asarray(origin_center_xyz_mm, np.float64) - 0.5 * sp,
                                           spacing_xyz_mm=sp)   # AutoCAR origin = lower voxel boundary
        return s.surface_vertices_xyz_mm, s.surface_faces, s.surface_vertex_radius_mm


def main(argv=None):
    a = base_parser(__doc__).parse_args(argv)
    config = config_from_args(a)
    run(config, AutoCARBackend(config), a.shard, a.num_shards)


if __name__ == "__main__":
    main()
