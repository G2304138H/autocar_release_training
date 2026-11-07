"""Real X-ray inference (shared part) on a synthetic step-08 case: no checkpoint, no GPU.

A visual-hull backend built on this repository's own camera checks that the reprojection, the centring and the
summary outputs agree with the method's projection convention.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from src import real_xray_common as C
from src.real_xray import AutoCARProjector as Projector

REPO = Path(__file__).resolve().parents[1]


def _case(tmp: Path):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 160)[:, None]
    branches = [np.array([-20., 0, 15]) + np.array([40., 8, -35]) * t + np.sin(np.pi * t) * [5, -4, 3],
                np.array([-20., 0, 15]) + np.array([30., -28, -12]) * t + np.sin(np.pi * t) * [-3, 5, 4]]
    pts = []
    for b in branches:
        for p in b:
            d = rng.normal(size=(40, 3))
            pts.append(p + 1.6 * d / np.linalg.norm(d, axis=1, keepdims=True) * rng.uniform(0, 1, (40, 1)))
    pts = np.concatenate(pts)
    prim, sec = np.array([-30., 40., 0., -30.]), np.array([-25., 25., 40., -25.])
    theta, phi = (90 - prim + 180) % 360 - 180, 90 - sec
    iso = C.model_iso_pixel_mm("lca")
    g = C.DEFAULT_ACQUISITION_GEOMETRY["allura"]
    cam = dict(C.MODEL_CAMERA["lca"], mode="acquisition", system="allura", sid_mm=g["sid_mm"], sod_mm=g["sod_mm"],
               pixel_mm=iso * g["sid_mm"] / g["sod_mm"])
    V = len(theta)
    imgs = np.zeros((V, 256, 256), np.float32)
    for i in range(V):
        rc = np.rint(Projector(theta[i], phi[i], cam).project_rc(pts)).astype(int)
        imgs[i, rc[:, 0], rc[:, 1]] = 1
        imgs[i] = ndimage.binary_closing(imgs[i] > 0, np.ones((3, 3))) | (imgs[i] > 0)
    M = np.array([[2., 0, 0.5], [0, 2., 0.5], [0, 0, 1]])
    up = np.stack([np.kron(m, np.ones((2, 2))) for m in imgs]) > 0
    d = tmp / "inputs"
    d.mkdir(parents=True)
    np.savez_compressed(d / "Study00001_LCA.npz", input_format="real_xray_keyframes", format_version=1,
                        vessel_type="lca", case_name="Study00001_LCA", images=imgs, raw_images=1 - 0.5 * imgs,
                        theta_deg=theta, phi_deg=phi, raw_full=(200 - 120 * up).astype(np.uint8),
                        label_full=up.astype(np.uint8), full_shape=np.full((V, 2), 512),
                        model_to_full=np.repeat(M[None], V, 0),
                        clinical_views=np.array([f"P{p:+.0f} S{s:+.0f}" for p, s in zip(prim, sec)]),
                        angle_group=np.array([0, 1, 2, 0]), keyframe_vessel_px=imgs.reshape(V, -1).sum(1),
                        primary_angle_deg=prim, secondary_angle_deg=sec, system="allura", seg_method="synthetic",
                        src_iso_pixel_mm=0.23, keyframe_index=np.arange(V), n_frames=np.full(V, 40))
    return d, cam


class HullBackend:
    """Voxels whose centre projects onto the mask in every input view (this repository's camera)."""
    name = "hull"
    min_views, max_views = 1, 7

    def describe(self):
        return {}

    def predict(self, images, theta, phi, cam):
        n, sp = 120, 0.6
        ax = (np.arange(n) - (n - 1) / 2) * sp
        z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")
        pts = np.stack([x, y, z], -1).reshape(-1, 3)
        keep = np.ones(len(pts), bool)
        for im, t, p in zip(images, theta, phi):
            rc = np.rint(Projector(t, p, cam).project_rc(pts))
            ok = np.isfinite(rc).all(1) & (rc >= 0).all(1) & (rc < 256).all(1)
            hit = np.zeros(len(pts), bool)
            hit[ok] = im[rc[ok, 0].astype(int), rc[ok, 1].astype(int)] > 0.5
            keep &= hit
        return C.Prediction(keep.reshape(n, n, n).astype(np.float32), np.full(3, ax[0]), np.full(3, sp), 0.5, 0.0)

    def projector(self, t, p, cam):
        return Projector(t, p, cam)

    def surface(self, mask, origin, spacing):
        from skimage.measure import marching_cubes
        sp = np.asarray(spacing)[::-1]
        v, f, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), 0.5, spacing=tuple(sp))
        xyz = (v - sp)[:, ::-1] + np.asarray(origin)
        return xyz, f, np.full(len(xyz), 1.5)


class RealXrayTest(unittest.TestCase):
    def test_hull_reprojects_onto_its_inputs_and_outputs_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d, _ = _case(tmp)
            cfg = json.loads((REPO / "configs/real_xray/lca.json").read_text())
            cfg.update(input_dir=str(d), output_dir=str(tmp / "out"), output_mode="summary", input_centering="none")
            C.run(cfg, HullBackend())
            s = json.loads((tmp / "out/Study00001_LCA/case_summary.json").read_text())
            self.assertEqual(s["input_views"], [0, 1])
            for v in s["views"]:
                if v["role"] == "input":
                    self.assertGreater(v["dice"], 0.8)        # a visual hull covers its own inputs
            self.assertTrue((tmp / "out/_summary/Study00001_LCA.png").is_file())
            self.assertIn("ok", (tmp / "out/summary.csv").read_text())

    def test_centring_keeps_the_mapping_to_the_original_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, _ = _case(Path(tmp))
            case = C.load_case(d / "Study00001_LCA.npz", "lca")
            before = case.images[0].copy()
            shifts = C.center_views(case, [0], "mask_centroid")
            r, c = np.nonzero(case.images[0] > 0.5)
            self.assertLessEqual(abs(r.mean() - 127.5), 0.6)
            back = C.model_to_full(case.images[0], case.model_to_full[0], (512, 512))
            orig = C.model_to_full(before, np.array([[2., 0, 0.5], [0, 2., 0.5], [0, 0, 1]]), (512, 512))
            self.assertGreater(C.dice(back, orig), 0.98, shifts)


if __name__ == "__main__":
    unittest.main()
