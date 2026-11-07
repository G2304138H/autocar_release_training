# AutoCAR on real X-ray angiograms (CoronaryDominance)

Inference only: the trained ImageCAS AutoCAR checkpoints are applied to real angiograms, as a comparison for the
PPCAR-Net `real-xray` branch. Everything except the model is shared with that branch, so the three methods see
the same cases, input views, camera and centring and write the same outputs.

## Input

One `.npz` per study x artery from angioseg-nnunet step 08 (`scripts/export_recon_inputs.py`), e.g.
`/dataset/reny0012/angioseg/recon_inputs/xravos_A_v2/lca/Study00001_LCA.npz`: every view's keyframe mask in the
256x256 model space (`images`), `theta_deg` / `phi_deg` (paper convention), the raw X-ray and segmentation at full
resolution, and the map back to the original frame (`model_to_full`).

## What happens per case

1. **Input views**: one view per viewing direction (the one with most vessel), then the `num_views` (default 2,
   as in training) with the largest smallest angle between them, the same rule as PPCAR-Net.
2. **Centring** (`input_centering`, default `mask_centroid`): every view's mask is moved to the image centre before
   inference. `bbox_center` is less pulled by a catheter segment; `none` keeps the raw C-arm position. The shift is
   folded into `model_to_full`, so all pictures are drawn on the original keyframes.
3. **Camera** (`camera`, default `acquisition`): SID / SOD of the case's X-ray system (Allura 1137 / 765 mm,
   Azurion 1195 / 810 mm; `acquisition_geometry`), detector pixel = isocentre pixel of the exported grid
   (0.5417 mm LCA / 0.4583 mm RCA) x SID / SOD. `training` uses the training camera (900 / 750 mm, 0.65 / 0.55 mm).
   Both feed the method's own Stage-2 camera code, so the back-projection uses the real geometry.
4. **Reconstruction**: camera matrices from `ProjectionGeometry.from_angles` with the case's SID / SOD / pixel (the same contract as `Stage2NPZDataset`, isocentre at the origin); `recon_net` (2D encoder, sparse back-projection, sparse 3D U-Net) as in `src/predict_npz.py`, rasterised on the checkpoint's box and voxel size; sigmoid >= 0.5 is vessel. If the input views share no 3D support (empty visual hull) the prediction is empty and the case is still drawn. The surface and radius come from `src.geometry.vascular_surface.extract_vascular_graph_surface` (drawn on a 1 mm grid, `display_voxel_mm`). A two-view checkpoint needs `num_views` 2; a variable-view checkpoint takes 1-7.
5. **Evaluation in 2D**: the predicted occupancy (its 0.5 iso-surface) is projected into every view, inputs and
   held-out, and compared with the segmentation (Dice, and Dice after the best 2D translation).

## Output (`<output_dir>/`)

- `_summary/<case>.png` and `<case>/summary.png`: one row per view (INPUT / NOT input); columns: X-ray, X-ray +
  reprojected 3D, segmentation, segmentation vs reprojection, predicted surface seen along that view, the same
  coloured by radius.
- `summary.csv`: one row per case (camera, centring shifts, input views, mean Dice for input and held-out views,
  foreground voxels, inference time, status).
- `<case>/prediction.npz` (binary volume, grid, camera), `<case>/case_summary.json`.
- `output_mode: full` adds `surface_3d.gif`, `surface_3d_radius.gif` and per-view overlays in `reprojection/`.

## Run

```bash
cd /export/home2/reny0012/code/autocar_release_training
# 5 LCA cases, summary PNG only
sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,LIMIT=5,OUTPUT_MODE=summary,CKPT=/path/to/lca.ckpt slurm/real_xray_infer.sbatch
# all RCA cases
sbatch --export=ALL,ART=rca,METHOD=smart_seg_semantic_v2,OUTPUT_MODE=summary,CKPT=/path/to/rca.ckpt slurm/real_xray_infer.sbatch
```

Directly: `python -m src.real_xray --config configs/real_xray/lca.json --checkpoint ... --input_dir ... --output_dir ...`
(`--limit`, `--case_ids`, `--output_mode`, `--camera`, `--input_centering`, `--num_views`, `--device`).

Code: `src/real_xray_common.py` (shared with the other methods; numpy / scipy / scikit-image / matplotlib only) and
`src/real_xray.py` (the AutoCAR part). Test without a checkpoint: `pytest tests/test_real_xray.py`.
