"""Real X-ray (CoronaryDominance) inference for voxel reconstruction methods - shared, model-independent part.

The same file is used by the DeepCA and AutoCAR repositories. It mirrors the PPCAR-Net `real-xray` branch
(vessel_code/real_xray), so the three methods read the same cases, get the same input views and the same camera, and
write the same summary PNG / summary.csv:

  input   : one .npz per study x artery from angioseg-nnunet step 08 (scripts/export_recon_inputs.py)
  camera  : "acquisition" (default; SID/SOD of the case's X-ray system, detector pixel derived so the exported 256x256
            grid keeps its isocentre pixel size) or "training" (SID 900, SOD 750 mm, 0.65 / 0.55 mm)
  centring: every view's mask is moved to the image centre before inference (default mask_centroid); all pictures
            are drawn on the original keyframes (the shift is folded into model_to_full)
  views   : the same automatic choice as PPCAR-Net (one view per direction, largest smallest angle between them)

A method plugs in through a small backend object (see `Backend`): it predicts an occupancy volume from the input
views and the camera, gives a projector for the same camera convention, and extracts a radius-coloured surface with
the method's own post-processing.

Only numpy, scipy, scikit-image, matplotlib and Pillow are used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy import ndimage

FORMAT_VERSION = 1
# Camera of the Stage-2 training renders (the projection producer used by vessel-code, DeepCA and AutoCAR; mm)
MODEL_CAMERA = {
    "lca": dict(image_dim=256, sid_mm=900.0, sod_mm=750.0, pixel_mm=0.65),
    "rca": dict(image_dim=256, sid_mm=900.0, sod_mm=750.0, pixel_mm=0.55),
}
# CoronaryDominance acquisition systems (dataset authors' reference values, medians; mm)
DEFAULT_ACQUISITION_GEOMETRY = {
    "allura": {"sid_mm": 1137.0, "sod_mm": 765.0},
    "azurion": {"sid_mm": 1195.0, "sod_mm": 810.0},
    "default": {"sid_mm": 1137.0, "sod_mm": 765.0},
}
CENTERING_MODES = ("none", "mask_centroid", "bbox_center")
SHEET_COLUMNS = ["X-ray (keyframe)", "X-ray + reprojected 3D", "segmentation (input mask)",
                 "segmentation vs reprojected 3D", "3D prediction, this view", "3D prediction, radius"]
SUMMARY_FIELDS = ["case_name", "artery", "method", "seg_method", "subset", "system", "iso_pixel_mm", "camera",
                  "camera_sid_mm", "camera_sod_mm", "camera_pixel_mm", "input_centering", "centering_shift_px",
                  "n_views", "input_views", "input_clinical_views", "heldout_views", "dice_input", "dice_heldout",
                  "dice_input_aligned", "dice_heldout_aligned", "foreground_voxels", "inference_seconds", "status"]

PRED_RGB = np.array([1.0, 0.55, 0.0])      # predicted silhouette on the X-ray (orange)
TP_RGB = np.array([0.1, 0.85, 0.2])        # predicted and segmented
FP_RGB = np.array([1.0, 0.15, 0.15])       # predicted only
FN_RGB = np.array([1.0, 1.0, 1.0])         # segmented only
SURFACE_RGB = np.array([0.84, 0.15, 0.16])
BACKGROUND_RGB = np.array([0.07, 0.07, 0.08])


# ============================================================================= backend interface
@dataclass
class Prediction:
    """Occupancy on a regular grid in the centred camera frame (XYZ mm, isocentre at 0). Array order ZYX."""
    volume_zyx: np.ndarray               # probability / raw score; foreground = volume >= threshold
    origin_center_xyz_mm: np.ndarray     # centre of voxel [0,0,0]
    spacing_xyz_mm: np.ndarray
    threshold: float
    seconds: float = float("nan")
    info: dict = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(self.volume_zyx) >= self.threshold


class Projector(Protocol):
    source_mm: np.ndarray                                    # (3,) X-ray source position

    def project_rc(self, points_mm: np.ndarray) -> np.ndarray:   # (N,3) -> (N,2) (row, col) in the 256 grid
        ...


class Backend(Protocol):
    name: str
    min_views: int
    max_views: int

    def predict(self, images: np.ndarray, theta_deg: np.ndarray, phi_deg: np.ndarray, cam: dict) -> Prediction: ...

    def projector(self, theta_deg: float, phi_deg: float, cam: dict) -> Projector: ...

    def surface(self, mask_zyx: np.ndarray, origin_center_xyz_mm, spacing_xyz_mm):
        """-> (vertices_xyz_mm (N,3), faces (F,3), vertex_radius_mm (N,)) with the method's own post-processing."""
        ...

    def describe(self) -> dict: ...


# ============================================================================= cases
def model_iso_pixel_mm(artery: str) -> float:
    c = MODEL_CAMERA[artery]
    return c["pixel_mm"] * c["sod_mm"] / c["sid_mm"]


def inference_camera(case: "RealXrayCase", config: dict) -> dict:
    """camera "training": the checkpoint's camera. camera "acquisition": SID/SOD of the case's system (config
    `acquisition_geometry`, keyed by step 08's `system`), detector pixel = isocentre pixel x SID / SOD so the
    exported grid keeps its size (unless `imager_pixel_spacing_mm` is set; a mismatch is reported in `warning`)."""
    mode = config.get("camera", "acquisition")
    cam = dict(MODEL_CAMERA[case.artery_type], mode=mode, system=case.meta.get("system", ""))
    if mode == "training":
        return cam
    if mode != "acquisition":
        raise ValueError("camera must be 'training' or 'acquisition'")
    table = {**DEFAULT_ACQUISITION_GEOMETRY, **config.get("acquisition_geometry", {})}
    geo = table.get(cam["system"]) or table.get("default")
    sid, sod = float(geo["sid_mm"]), float(geo["sod_mm"])
    if not 0 < sod < sid:
        raise ValueError(f"acquisition geometry needs 0 < SOD < SID, got SID {sid}, SOD {sod} mm")
    grid_px = model_iso_pixel_mm(case.artery_type) * sid / sod
    px = float(config.get("imager_pixel_spacing_mm") or grid_px)
    cam.update(sid_mm=sid, sod_mm=sod, pixel_mm=px)
    if abs(px / grid_px - 1) > 0.01:
        cam["warning"] = (f"imager_pixel_spacing_mm {px:.4f} gives {px * sod / sid:.4f} mm at the isocentre, "
                          f"but the exported images use {model_iso_pixel_mm(case.artery_type):.4f}")
    return cam


@dataclass
class RealXrayCase:
    path: Path
    name: str
    artery_type: str
    images: np.ndarray            # (V,256,256) binary keyframe masks in model space
    raw_images: np.ndarray        # (V,256,256) keyframe X-ray in model space, [0,1]
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    raw_full: np.ndarray          # (V,Hmax,Wmax) uint8, original orientation
    label_full: np.ndarray
    full_shape: np.ndarray        # (V,2)
    model_to_full: np.ndarray     # (V,3,3)
    clinical_views: list
    angle_group: np.ndarray
    keyframe_vessel_px: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        return int(self.images.shape[0])


def load_case(path, artery_type: str) -> RealXrayCase:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        if "input_format" not in z.files or str(z["input_format"]) != "real_xray_keyframes":
            raise ValueError(f"{path}: not a real X-ray keyframe case (input_format missing)")
        if int(z["format_version"]) > FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {int(z['format_version'])} is newer than this reader")
        artery = str(z["vessel_type"]).lower()
        if artery != artery_type:
            raise ValueError(f"{path}: case artery {artery} != requested {artery_type}")
        cam = MODEL_CAMERA[artery]
        for key, ref in (("image_dim", cam["image_dim"]), ("sid", cam["sid_mm"] / 1000.0),
                         ("imager_pixel_spacing", cam["pixel_mm"])):
            if key in z.files and not np.isclose(float(z[key]), ref):
                raise ValueError(f"{path}: {key}={float(z[key])} but the {artery.upper()} model expects {ref}")
        images = np.asarray(z["images"], np.float32)
        V = images.shape[0]
        if images.shape[1:] != (cam["image_dim"],) * 2 or images.min() < 0 or images.max() > 1:
            raise ValueError(f"{path}: images must be (V,256,256) in [0,1]")
        meta_keys = ["case_id", "study_id", "subset", "seg_method", "seg_dir", "mask_key", "orientation",
                     "center_mode", "scale_mode", "scale_full_to_model", "src_iso_pixel_mm", "angle_convention",
                     "system", "scale_calibration"]
        meta = {k: (z[k].item() if z[k].ndim == 0 else z[k].tolist()) for k in meta_keys if k in z.files}
        for k in ("primary_angle_deg", "secondary_angle_deg", "keyframe_index", "n_frames",
                  "crop_clipped_fraction", "nearest_anchor_deg", "video_stem", "video_path"):
            if k in z.files:
                meta[k] = z[k].tolist()
        return RealXrayCase(
            path=path, name=str(z["case_name"]) if "case_name" in z.files else path.stem, artery_type=artery,
            images=images, raw_images=np.asarray(z["raw_images"], np.float32),
            theta_deg=np.asarray(z["theta_deg"], np.float64), phi_deg=np.asarray(z["phi_deg"], np.float64),
            raw_full=np.asarray(z["raw_full"]), label_full=np.asarray(z["label_full"]),
            full_shape=np.asarray(z["full_shape"], np.int64), model_to_full=np.asarray(z["model_to_full"], np.float64),
            clinical_views=[str(s) for s in z["clinical_views"]],
            angle_group=np.asarray(z["angle_group"], np.int64) if "angle_group" in z.files else np.arange(V),
            keyframe_vessel_px=(np.asarray(z["keyframe_vessel_px"]) if "keyframe_vessel_px" in z.files
                                else images.reshape(V, -1).sum(1)),
            meta=meta,
        )


def _directions(case: RealXrayCase) -> np.ndarray:
    p = np.deg2rad(np.asarray(case.meta.get("primary_angle_deg", 90 - case.theta_deg), np.float64))
    s = np.deg2rad(np.asarray(case.meta.get("secondary_angle_deg", 90 - case.phi_deg), np.float64))
    return np.stack([np.sin(p) * np.cos(s), np.cos(p) * np.cos(s), np.sin(s)], 1)


def select_input_views(case: RealXrayCase, num_views: int, strategy: str = "max_separation",
                       max_useful_separation_deg: float = 90.0) -> list:
    """Identical to PPCAR-Net (vessel_code/real_xray/data.py), so all methods get the same input views."""
    groups: dict = {}
    for i in range(case.num_views):
        g = int(case.angle_group[i])
        if g not in groups or case.keyframe_vessel_px[i] > case.keyframe_vessel_px[groups[g]]:
            groups[g] = i
    reps = sorted(groups.values())
    if num_views > len(reps):
        raise ValueError(f"{case.name}: {num_views} input views requested but only {len(reps)} distinct directions")
    if strategy == "first":
        firsts = []
        for i in range(case.num_views):
            if int(case.angle_group[i]) not in {int(case.angle_group[j]) for j in firsts}:
                firsts.append(i)
        return firsts[:num_views]
    if strategy != "max_separation":
        raise ValueError(f"unknown view selection {strategy}")
    if num_views == 1:
        return [max(reps, key=lambda i: case.keyframe_vessel_px[i])]
    d = _directions(case)
    best, best_key = None, None
    for combo in combinations(reps, num_views):
        ang = min(np.degrees(np.arccos(np.clip(d[a] @ d[b], -1, 1))) for a, b in combinations(combo, 2))
        key = (min(ang, max_useful_separation_deg), int(sum(case.keyframe_vessel_px[i] for i in combo)))
        if best_key is None or key > best_key:
            best, best_key = list(combo), key
    return best


def resolve_input_views(case: RealXrayCase, config: dict) -> list:
    spec = config.get("input_views", "auto")
    if isinstance(spec, dict):
        spec = spec.get(case.name, "auto")
    if spec == "auto":
        return select_input_views(case, int(config.get("num_views", 2)), config.get("view_selection", "max_separation"))
    idx = [int(i) for i in spec]
    if not idx or len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) >= case.num_views:
        raise ValueError(f"{case.name}: invalid input_views {idx} for {case.num_views} views")
    return idx


# ============================================================================= 2D helpers
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else float("nan")


def translation_to_align(pred: np.ndarray, target: np.ndarray, max_shift: int = 48, sigma: float = 2.0):
    """Integer (d_row, d_col) that best moves `pred` onto `target` (cross-correlation of blurred masks)."""
    a = ndimage.gaussian_filter((pred > 0.5).astype(np.float64), sigma)
    b = ndimage.gaussian_filter((target > 0.5).astype(np.float64), sigma)
    if a.sum() == 0 or b.sum() == 0:
        return 0, 0
    H, W = a.shape
    corr = np.real(np.fft.ifft2(np.fft.fft2(b, s=(2 * H, 2 * W)) * np.conj(np.fft.fft2(a, s=(2 * H, 2 * W)))))
    corr = np.fft.fftshift(corr)
    win = corr[H - max_shift:H + max_shift + 1, W - max_shift:W + max_shift + 1]
    r, c = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(r - max_shift), int(c - max_shift)


def shift_mask(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    return ndimage.shift(mask.astype(np.float32), (d_row, d_col), order=0, mode="constant", cval=0.0)


def shift_matrix(d_row: float, d_col: float) -> np.ndarray:
    """Homogeneous model->model map for content moved by (d_row, d_col): new(u) = old(u - d)."""
    T = np.eye(3)
    T[0, 2], T[1, 2] = -d_row, -d_col
    return T


def model_to_full(mask_model: np.ndarray, model_to_full_matrix: np.ndarray, full_shape) -> np.ndarray:
    inv = np.linalg.inv(model_to_full_matrix)
    out = ndimage.affine_transform(mask_model.astype(np.float32), inv[:2, :2], offset=inv[:2, 2],
                                   output_shape=tuple(int(s) for s in full_shape), order=1, mode="constant")
    return out > 0.5


def center_views(case: RealXrayCase, views: list, mode: str, min_px: int = 20) -> dict:
    """Move each view's vessel mask to the image centre (in place), as PPCAR-Net does before inference.
    mask_centroid: mean of the mask pixels; bbox_center: centre of the 2-98 percentile box. The raw X-ray moves with
    the mask and model_to_full is updated, so overlays on the original frames stay correct."""
    shifts = {}
    for i in views:
        m = case.images[i] > 0.5
        if mode == "none" or m.sum() < min_px:
            shifts[i] = [0, 0]
            continue
        r, c = np.nonzero(m)
        if mode == "mask_centroid":
            cr, cc = r.mean(), c.mean()
        elif mode == "bbox_center":
            cr = 0.5 * (np.percentile(r, 2) + np.percentile(r, 98))
            cc = 0.5 * (np.percentile(c, 2) + np.percentile(c, 98))
        else:
            raise ValueError(f"input_centering must be one of {CENTERING_MODES}")
        H, W = m.shape
        dr, dc = int(round((H - 1) / 2 - cr)), int(round((W - 1) / 2 - cc))
        if dr or dc:
            case.images[i] = shift_mask(case.images[i], dr, dc)
            case.raw_images[i] = ndimage.shift(case.raw_images[i], (dr, dc), order=1).astype(np.float32)
            case.model_to_full[i] = case.model_to_full[i] @ shift_matrix(dr, dc)
        shifts[i] = [dr, dc]
    return shifts


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


def overlay_on_xray(raw_u8: np.ndarray, pred: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    g = raw_u8.astype(np.float32) / 255.0 if raw_u8.dtype == np.uint8 else np.clip(raw_u8, 0, 1)
    rgb = np.repeat(g[..., None], 3, -1)
    rgb[pred] = (1 - alpha) * rgb[pred] + alpha * PRED_RGB
    rgb[_outline(pred)] = PRED_RGB
    return np.clip(rgb, 0, 1)


def overlay_on_mask(seg: np.ndarray, pred: np.ndarray) -> np.ndarray:
    seg, pred = seg > 0, pred > 0
    rgb = np.zeros(seg.shape + (3,), np.float32)
    rgb[seg & ~pred] = FN_RGB
    rgb[seg & pred] = TP_RGB
    rgb[pred & ~seg] = FP_RGB
    return rgb


def save_png(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)


def save_panel(tiles, titles, path: Path, suptitle: str, legend=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    n = len(tiles)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(max(3.8 * n, 7.6), 4.9), squeeze=False)
    for ax, tile, title in zip(axes[0], tiles, titles):
        ax.imshow(tile, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(textwrap.fill(suptitle, width=int(max(3.8 * n, 7.6) * 11)), fontsize=9.5, y=0.985)
    if legend:
        fig.text(0.5, 0.015, legend, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def radius_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("radius", ["#d73027", "#fee08b", "#1a9850"])


# ============================================================================= 3D -> 2D
def voxel_centres_mm(mask_zyx: np.ndarray, origin_center_xyz_mm, spacing_xyz_mm, sub: int = 1) -> np.ndarray:
    """Foreground voxel centres (XYZ mm); with sub > 1 each voxel is sampled sub^3 times (for gap-free splatting)."""
    idx = np.argwhere(mask_zyx)[:, ::-1].astype(np.float64)                 # (N,3) xyz index
    sp = np.asarray(spacing_xyz_mm, np.float64)
    pts = np.asarray(origin_center_xyz_mm, np.float64)[None] + idx * sp[None]
    if sub <= 1 or len(pts) == 0:
        return pts
    off = (np.arange(sub) + 0.5) / sub - 0.5
    o = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3) * sp[None]
    return (pts[:, None, :] + o[None]).reshape(-1, 3)


def inside_points_mm(pred: Prediction, sub: int, chunk: int = 400_000):
    """Points (XYZ mm) filling the predicted 0.5 iso-surface (the surface drawn in the 3D panels): each voxel next to
    the foreground is sampled sub^3 times and a sample is kept where the trilinearly interpolated occupancy >= 0.5."""
    mask = pred.mask
    m = mask.astype(np.float32)
    cand = np.argwhere(ndimage.binary_dilation(mask)).astype(np.float64)       # zyx
    off = (np.arange(sub) + 0.5) / sub - 0.5 if sub > 1 else np.zeros(1)
    o = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    sp = np.asarray(pred.spacing_xyz_mm, np.float64)
    org = np.asarray(pred.origin_center_xyz_mm, np.float64)
    for s in range(0, len(cand), chunk):
        f = (cand[s:s + chunk, None, :] + o[None]).reshape(-1, 3)
        keep = ndimage.map_coordinates(m, f.T, order=1, mode="constant", cval=0.0) >= 0.5
        yield org[None] + f[keep][:, ::-1] * sp[None]


def reproject_volume(pred: Prediction, projector: Projector, image_dim: int, iso_pixel_mm: float) -> np.ndarray:
    """Silhouette (256x256) of the predicted occupancy: points filling the 0.5 iso-surface, sampled at most half an
    isocentre pixel apart, are projected to their nearest detector pixel."""
    out = np.zeros((image_dim, image_dim), np.float32)
    if not pred.mask.any():
        return out
    sub = int(np.clip(np.ceil(2 * float(np.max(pred.spacing_xyz_mm)) / iso_pixel_mm), 1, 6))
    for pts in inside_points_mm(pred, sub):
        rc = projector.project_rc(pts)
        ok = np.isfinite(rc).all(1)
        rc = np.rint(rc[ok]).astype(np.int64)
        ok = (rc[:, 0] >= 0) & (rc[:, 0] < image_dim) & (rc[:, 1] >= 0) & (rc[:, 1] < image_dim)
        out[rc[ok, 0], rc[ok, 1]] = 1
    return out


def display_volume(pred: Prediction, target_mm: float):
    """Binary volume for drawing the surface: block-max pooled to about `target_mm` voxels (keeps the mesh small)."""
    mask = pred.mask
    sp = np.asarray(pred.spacing_xyz_mm, np.float64)
    f = int(max(1, np.floor(target_mm / float(sp.max()) + 1e-6)))
    if f == 1:
        return mask, np.asarray(pred.origin_center_xyz_mm, np.float64), sp
    pad = [(0, (-s) % f) for s in mask.shape]
    m = np.pad(mask, pad)
    Z, Y, X = (s // f for s in m.shape)
    pooled = m.reshape(Z, f, Y, f, X, f).max((1, 3, 5))
    origin = np.asarray(pred.origin_center_xyz_mm, np.float64) + 0.5 * (f - 1) * sp
    return pooled, origin, sp * f


def render_mesh_view(vertices_mm, faces, vertex_radius, projector: Projector, model_to_full_matrix, full_shape,
                     shift_rc=(0, 0), radius_range=None):
    """The predicted surface seen along one view's X-ray direction, on that view's full-resolution frame (same
    projection and position as the reprojected silhouette). Painter's algorithm: triangles nearer the detector are
    drawn last (seen from the detector side, as in the clinical display). Returns (plain, radius) RGB."""
    from PIL import Image, ImageDraw
    H, W = (int(x) for x in full_shape)
    plain = np.ones((H, W, 3)) * BACKGROUND_RGB
    if len(faces) == 0:
        return plain, plain.copy()
    rc = projector.project_rc(np.asarray(vertices_mm, np.float64)) + np.asarray(shift_rc, float)[None]
    full = (np.asarray(model_to_full_matrix) @ np.c_[rc, np.ones(len(rc))].T).T[:, :2]
    tri = np.asarray(vertices_mm, np.float64)[faces]                         # (F,3,3)
    centre = tri.mean(1)
    ray = centre - np.asarray(projector.source_mm, np.float64)[None]
    depth = np.linalg.norm(ray, axis=1)
    ray /= np.maximum(depth[:, None], 1e-12)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    shade = 0.30 + 0.70 * np.abs((nrm * ray).sum(1))
    ftri = full[faces]                                                       # (F,3,2) (row, col)
    ok = np.isfinite(ftri).all((1, 2))
    order = np.argsort(depth)
    order = order[ok[order]]
    fr = np.asarray(vertex_radius, np.float64)[faces].mean(1)
    lo, hi = radius_range if radius_range else (float(fr.min()), float(fr.max()))
    col_r = radius_colormap()(np.clip((fr - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3]
    imgs = []
    for colours in (np.repeat(SURFACE_RGB[None], len(faces), 0), col_r):
        canvas = Image.new("RGB", (W, H), tuple(int(255 * x) for x in BACKGROUND_RGB))
        dr = ImageDraw.Draw(canvas)
        rgb = (np.clip(colours * shade[:, None], 0, 1) * 255).astype(np.uint8)
        pts = ftri[:, :, ::-1]                                               # (x=col, y=row)
        for i in order:
            p = pts[i]
            dr.polygon([(p[0, 0], p[0, 1]), (p[1, 0], p[1, 1]), (p[2, 0], p[2, 1])], fill=tuple(rgb[i].tolist()))
        imgs.append(np.asarray(canvas, np.float32) / 255.0)
    return imgs[0], imgs[1]


def save_surface_gifs(vertices_mm, faces, vertex_radius, dest: Path, title: str, frames: int, fps: int,
                      max_faces: int = 60000) -> tuple:
    """Rotating 3D surface: plain (red) and radius-coloured GIFs. Returns the radius range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if len(faces) == 0:
        return None, None
    rng = np.random.default_rng(0)
    sel = np.arange(len(faces)) if len(faces) <= max_faces else np.sort(rng.choice(len(faces), max_faces, False))
    tri = np.asarray(vertices_mm)[faces[sel]]
    fr = np.asarray(vertex_radius)[faces[sel]].mean(1)
    lo, hi = float(np.min(vertex_radius)), float(np.max(vertex_radius))
    cm = radius_colormap()
    v = np.asarray(vertices_mm)
    c, half = v.mean(0), 0.55 * float((v.max(0) - v.min(0)).max())
    for name, colours, bar in (("surface_3d.gif", np.repeat(SURFACE_RGB[None], len(sel), 0), False),
                               ("surface_3d_radius.gif", cm(np.clip((fr - lo) / max(hi - lo, 1e-6), 0, 1))[:, :3], True)):
        fig = plt.figure(figsize=(6.4, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colours, edgecolors="none", linewidths=0))
        ax.set(xlim=(c[0] - half, c[0] + half), ylim=(c[1] - half, c[1] + half), zlim=(c[2] - half, c[2] + half),
               xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
        ax.set_title(title + (": radius" if bar else ""), fontsize=10)
        if bar:
            fig.colorbar(ScalarMappable(Normalize(lo, hi), cm), ax=ax, shrink=0.6, pad=0.06).set_label("radius (mm)")

        def rot(k, ax=ax):
            ax.view_init(elev=20.0, azim=360.0 * k / frames)
            return (ax,)
        FuncAnimation(fig, rot, frames=frames, blit=False).save(dest / name, writer=PillowWriter(fps=fps), dpi=90)
        plt.close(fig)
    return lo, hi


# ============================================================================= per case
def save_case_sheet(case, method, surf, inputs, views, masks_shown, shifts, projectors, cam, aligned, pred,
                    path: Path) -> Path:
    """One PNG per case: one row per view (keyframe), six panels (see SHEET_COLUMNS), all in the original
    orientation and resolution of the keyframe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    verts, faces, vrad = surf
    rng = (float(np.min(vrad)), float(np.max(vrad))) if len(vrad) else (0.0, 1.0)
    V = case.num_views
    Ht = V * 2.6 + 1.3
    fig, axes = plt.subplots(V, 6, figsize=(6 * 2.55 + 2.0, Ht), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.995, top=1 - 0.80 / Ht, bottom=0.55 / Ht, wspace=0.04, hspace=0.06)
    for i in range(V):
        H, W = case.full_shape[i]
        raw = case.raw_full[i, :H, :W]
        seg = case.label_full[i, :H, :W] > 0
        p = model_to_full(masks_shown[i], case.model_to_full[i], (H, W))
        plain, rad = render_mesh_view(verts, faces, vrad, projectors[i], case.model_to_full[i], (H, W),
                                      shift_rc=shifts[i], radius_range=rng)
        panels = [np.repeat((raw.astype(np.float32) / 255.0)[..., None], 3, -1), overlay_on_xray(raw, p),
                  np.repeat(seg[..., None].astype(np.float32), 3, -1), overlay_on_mask(seg, p), plain, rad]
        for j, (ax, im) in enumerate(zip(axes[i], panels)):
            ax.imshow(im, interpolation="antialiased")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(SHEET_COLUMNS[j], fontsize=9)
        v = views[i]
        is_input = v["role"] == "input"
        kf = (case.meta.get("keyframe_index") or [None] * V)[i]
        nf = (case.meta.get("n_frames") or [None] * V)[i]
        d = v["dice_aligned"] if aligned else v["dice"]
        label = (f"v{i}  {case.clinical_views[i]}\n"
                 f"{'INPUT to 3D reconstruction' if is_input else 'NOT input (held-out)'}"
                 f"{chr(10) + '(same angle as an input)' if v['same_direction_as_an_input'] else ''}\n"
                 f"keyframe {kf}/{nf}   Dice {d:.2f}")
        axes[i, 0].set_ylabel(label, fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8,
                              color="#11703a" if is_input else "#a34a00", fontweight="bold" if is_input else "normal")
        for ax in axes[i]:
            for sp in ax.spines.values():
                sp.set_edgecolor("#11703a" if is_input else "#cccccc")
                sp.set_linewidth(2.2 if is_input else 0.8)
    md = {g: [v["dice_aligned"] if aligned else v["dice"] for v in views if v["role"] == g] for g in ("input", "heldout")}
    mean = {g: (float(np.nanmean(x)) if x else float("nan")) for g, x in md.items()}
    ctr = case.meta.get("input_centering", "none")
    fig.suptitle(
        f"{case.name}   [{method}  |  seg {case.meta.get('seg_method', '?')}]   {case.meta.get('system', '')} "
        f"{float(case.meta.get('src_iso_pixel_mm', float('nan'))):.3f} mm/px   camera {cam.get('mode', '')} "
        f"(SID {cam['sid_mm']:.0f}, SOD {cam['sod_mm']:.0f} mm)   {int(pred.mask.sum())} voxels\n"
        f"inputs: {', '.join(f'v{k}' for k in inputs)}   mean Dice{' (2D-aligned)' if aligned else ''}: "
        f"input {mean['input']:.2f}, held-out {mean['heldout']:.2f}"
        + (f"   | views centred on the mask ({ctr}) for the model; shown at the original position"
           if ctr != "none" else ""), fontsize=10.5, y=1 - 0.12 / Ht, va="top")
    fig.text(0.01, 0.12 / Ht, "orange = reprojected 3D occupancy;  green = both, red = reprojection only, white = "
             "segmentation only\n3D panels: predicted surface seen along this view's X-ray direction (original image "
             "orientation)" + (";  reprojections shifted by their best 2D translation" if aligned else ""),
             fontsize=8, va="bottom")
    cax = fig.add_axes((0.80, 0.22 / Ht, 0.18, 0.09 / Ht))
    cb = fig.colorbar(ScalarMappable(Normalize(*rng), radius_colormap()), cax=cax, orientation="horizontal")
    cb.ax.set_title("radius (mm)", fontsize=8, pad=2)
    cb.ax.tick_params(labelsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def visualize_case(case, backend, pred: Prediction, cam, inputs, dest: Path, config: dict) -> dict:
    input_set = set(inputs)
    input_groups = {int(case.angle_group[i]) for i in inputs}
    align = bool(config.get("align_reprojection", True))
    max_shift = int(config.get("max_align_shift_px", 48))
    dim = int(cam["image_dim"])
    iso = cam["pixel_mm"] * cam["sod_mm"] / cam["sid_mm"]
    projectors = [backend.projector(float(case.theta_deg[i]), float(case.phi_deg[i]), cam) for i in range(case.num_views)]
    reproj = np.stack([reproject_volume(pred, projectors[i], dim, iso) for i in range(case.num_views)])
    aligned = np.zeros_like(reproj)
    views = []
    for i in range(case.num_views):
        d = translation_to_align(reproj[i], case.images[i], max_shift)
        aligned[i] = shift_mask(reproj[i], *d)
        views.append(dict(index=i, clinical_view=case.clinical_views[i], theta_deg=float(case.theta_deg[i]),
                          phi_deg=float(case.phi_deg[i]), role="input" if i in input_set else "heldout",
                          same_direction_as_an_input=bool(int(case.angle_group[i]) in input_groups and i not in input_set),
                          dice=dice(reproj[i], case.images[i]), dice_aligned=dice(aligned[i], case.images[i]),
                          align_shift_px=list(d)))
    shown = aligned if align else reproj
    shifts = [tuple(v["align_shift_px"]) if align else (0, 0) for v in views]

    vol, origin, spacing = display_volume(pred, float(config.get("display_voxel_mm", 1.0)))
    surf = backend.surface(vol, origin, spacing) if vol.any() else (np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros(0))
    full_mode = config.get("output_mode", "full") == "full"
    tag = f"{case.name} [{backend.name} | seg {case.meta.get('seg_method', '?')}]"
    if full_mode:
        rdir = dest / "reprojection"
        for group in ("input", "heldout"):
            ids = [v["index"] for v in views if v["role"] == group]
            tx, tm, titles = [], [], []
            for i in ids:
                H, W = case.full_shape[i]
                pf = model_to_full(shown[i], case.model_to_full[i], (H, W))
                raw, seg = case.raw_full[i, :H, :W], case.label_full[i, :H, :W] > 0
                xr, mk = overlay_on_xray(raw, pf), overlay_on_mask(seg, pf)
                save_png(xr, rdir / "views" / f"v{i}_{group}_xray.png")
                save_png(mk, rdir / "views" / f"v{i}_{group}_mask.png")
                tx.append(xr)
                tm.append(mk)
                v = views[i]
                note = " (same angle as an input)" if v["same_direction_as_an_input"] else ""
                score = (f"Dice {v['dice_aligned']:.3f} after 2D shift ({v['align_shift_px'][0]:+d},"
                         f"{v['align_shift_px'][1]:+d}) px; {v['dice']:.3f} without") if align else f"Dice {v['dice']:.3f}"
                titles.append(f"v{i}: {case.clinical_views[i]}{note}\n{score}")
            if ids:
                what = "input views" if group == "input" else "held-out views (not given to the model)"
                save_panel(tx, titles, rdir / f"{group}_views_on_xray.png",
                           f"{tag}: reconstruction reprojected onto the {what}",
                           "orange = reprojected predicted occupancy (original image orientation)")
                save_panel(tm, titles, rdir / f"{group}_views_on_mask.png",
                           f"{tag}: reconstruction vs segmentation, {what}",
                           "green = both, red = reprojection only, white = segmentation only")
        if config.get("save_gifs", True):
            save_surface_gifs(*surf, dest, f"{case.name} [{backend.name}]", int(config.get("gif_frames", 60)),
                              int(config.get("gif_fps", 12)))
    sheet = save_case_sheet(case, backend.name, surf, inputs, views, shown, shifts, projectors, cam, align, pred,
                            dest / "summary.png")
    if config.get("_summary_dir"):
        import shutil
        Path(config["_summary_dir"]).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sheet, Path(config["_summary_dir"]) / f"{case.name}.png")

    def mean(key, g):
        x = [v[key] for v in views if v["role"] == g and np.isfinite(v[key])]
        return float(np.mean(x)) if x else None
    vr = surf[2]
    return dict(
        case_name=case.name, artery=case.artery_type.upper(), method=backend.name, npz=str(case.path), meta=case.meta,
        camera=cam, input_views=list(inputs), input_clinical_views=[case.clinical_views[i] for i in inputs],
        heldout_views=[v["index"] for v in views if v["role"] == "heldout"], views=views, overlays_aligned=align,
        foreground_voxels=int(pred.mask.sum()), inference_seconds=pred.seconds, prediction_info=pred.info,
        radius_range_mm=[float(vr.min()), float(vr.max())] if len(vr) else None,
        mean_dice=dict(input=mean("dice", "input"), heldout=mean("dice", "heldout"),
                       input_aligned=mean("dice_aligned", "input"), heldout_aligned=mean("dice_aligned", "heldout")),
    )


def summary_row(s: dict) -> dict:
    fmt = lambda x: "" if x is None else f"{x:.4f}"  # noqa: E731
    cam, md, meta = s["camera"], s["mean_dice"], s["meta"]
    return dict(case_name=s["case_name"], artery=s["artery"], method=s["method"], seg_method=meta.get("seg_method", ""),
                subset=meta.get("subset", ""), system=meta.get("system", ""), iso_pixel_mm=meta.get("src_iso_pixel_mm", ""),
                camera=cam.get("mode", ""), camera_sid_mm=round(cam["sid_mm"], 1), camera_sod_mm=round(cam["sod_mm"], 1),
                camera_pixel_mm=round(cam["pixel_mm"], 4), input_centering=meta.get("input_centering", "none"),
                centering_shift_px=" ".join(f"{k}:{v[0]},{v[1]}" for k, v in (meta.get("centering_shift_px") or {}).items()),
                n_views=len(s["views"]), input_views=";".join(map(str, s["input_views"])),
                input_clinical_views=" | ".join(s["input_clinical_views"]),
                heldout_views=";".join(map(str, s["heldout_views"])),
                dice_input=fmt(md["input"]), dice_heldout=fmt(md["heldout"]),
                dice_input_aligned=fmt(md["input_aligned"]), dice_heldout_aligned=fmt(md["heldout_aligned"]),
                foreground_voxels=s["foreground_voxels"], inference_seconds=fmt(s["inference_seconds"]), status="ok")


def write_summary(out: Path, rows: list, shard: int) -> None:
    with open(out / f"summary_shard{shard:03d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    merged = []
    for p in sorted(out.glob("summary_shard*.csv")):
        with open(p, newline="") as f:
            merged.extend(csv.DictReader(f))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)


def validate_config(config: dict) -> None:
    if config.get("artery_type") not in ("rca", "lca"):
        raise ValueError("artery_type must be rca or lca")
    if config.get("output_mode", "full") not in ("full", "summary"):
        raise ValueError("output_mode must be full or summary")
    if config.get("input_centering", "mask_centroid") not in CENTERING_MODES:
        raise ValueError(f"input_centering must be one of {CENTERING_MODES}")
    if config.get("centering_views", "all") not in ("all", "inputs"):
        raise ValueError("centering_views must be all or inputs")
    for key in ("input_dir", "output_dir"):
        if not config.get(key):
            raise ValueError(f"{key} is required")


def run(config: dict, backend: Backend, shard: int = 0, num_shards: int = 1) -> Path:
    validate_config(config)
    artery = config["artery_type"]
    files = sorted(Path(config["input_dir"]).glob("*.npz"))
    if config.get("case_ids"):
        wanted = set(map(str, config["case_ids"]))
        missing = wanted - {p.stem for p in files}
        if missing:
            raise ValueError(f"Missing requested cases: {sorted(missing)}")
        files = [p for p in files if p.stem in wanted]
    files = [p for k, p in enumerate(files) if k % num_shards == shard]
    if config.get("limit"):
        files = files[: int(config["limit"])]
    if not files:
        raise ValueError(f"No cases to run in {config['input_dir']}")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    config = dict(config, _summary_dir=str(out / "_summary"))
    (out / "run.json").write_text(json.dumps({"config": config, "backend": backend.describe()}, indent=2,
                                             default=str) + "\n")
    print(f"[real_xray:{backend.name}] {len(files)} cases (shard {shard}/{num_shards}) -> {out}", flush=True)
    rows, t0 = [], time.time()
    for n, path in enumerate(files, 1):
        dest = out / path.stem
        done = dest / "case_summary.json"
        if done.exists() and not config.get("overwrite", False):
            rows.append(summary_row(json.loads(done.read_text())))
            continue
        try:
            case = load_case(path, artery)
            inputs = sorted(resolve_input_views(case, config))
            if not backend.min_views <= len(inputs) <= backend.max_views:
                raise ValueError(f"{backend.name} checkpoint takes {backend.min_views}-{backend.max_views} views, "
                                 f"got {len(inputs)} (set num_views)")
            cam = inference_camera(case, config)
            if cam.get("warning"):
                print(f"  [camera] {case.name}: {cam['warning']}", flush=True)
            ctr = config.get("input_centering", "mask_centroid")
            ctr_views = inputs if config.get("centering_views", "all") == "inputs" else list(range(case.num_views))
            case.meta["input_centering"] = ctr
            case.meta["centering_shift_px"] = {str(k): v for k, v in center_views(case, ctr_views, ctr).items()}
            pred = backend.predict(case.images[inputs], case.theta_deg[inputs], case.phi_deg[inputs], cam)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / "prediction.npz", method=np.asarray(backend.name),
                                mask_zyx=pred.mask.astype(np.uint8), volume_axis_order=np.asarray("zyx"),
                                origin_center_xyz_mm=np.asarray(pred.origin_center_xyz_mm, np.float64),
                                spacing_xyz_mm=np.asarray(pred.spacing_xyz_mm, np.float64),
                                threshold=np.asarray(pred.threshold), input_views=np.asarray(inputs),
                                theta_deg=case.theta_deg, phi_deg=case.phi_deg,
                                camera_sid_mm=cam["sid_mm"], camera_sod_mm=cam["sod_mm"],
                                camera_pixel_mm=cam["pixel_mm"], camera_mode=np.asarray(cam["mode"]),
                                frame=np.asarray("Stage-2 camera frame, isocentre at 0, mm"))
            summary = visualize_case(case, backend, pred, cam, inputs, dest, config)
            done.write_text(json.dumps(summary, indent=2, default=float) + "\n")
            rows.append(summary_row(summary))
            md = summary["mean_dice"]
            print(f"  {n}/{len(files)} {case.name}: inputs {inputs} | {summary['foreground_voxels']} voxels | Dice "
                  f"input {md['input']} held-out {md['heldout']} | {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep going over a large set; the error is recorded in summary.csv
            import traceback
            traceback.print_exc()
            rows.append(dict(case_name=path.stem, artery=artery.upper(), method=backend.name,
                             status=f"error: {type(e).__name__}: {e}"))
    write_summary(out, rows, shard)
    print(f"[real_xray:{backend.name}] done -> {out / 'summary.csv'}", flush=True)
    return out


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/real_xray/<lca|rca>.json")
    p.add_argument("--checkpoint", help="override config checkpoint")
    p.add_argument("--input_dir", help="step-08 folder with <Study>_<ART>.npz (override config)")
    p.add_argument("--output_dir", help="override config output_dir")
    p.add_argument("--num_views", type=int, help="input views per case (default 2, as in training)")
    p.add_argument("--camera", choices=["acquisition", "training"])
    p.add_argument("--output_mode", choices=["full", "summary"],
                   help="full: GIFs + per-view overlays + summary.png; summary: only summary.png per case")
    p.add_argument("--input_centering", choices=list(CENTERING_MODES))
    p.add_argument("--device")
    p.add_argument("--limit", type=int, help="only the first N cases (test run)")
    p.add_argument("--case_ids", nargs="+", help="only these cases (file names without .npz)")
    p.add_argument("--overwrite", action="store_true", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p


def config_from_args(a) -> dict:
    config = json.loads(Path(a.config).read_text())
    for key in ("checkpoint", "input_dir", "output_dir", "num_views", "camera", "output_mode", "input_centering",
                "device", "limit", "case_ids", "overwrite"):
        if getattr(a, key, None) is not None:
            config[key] = getattr(a, key)
    return config
