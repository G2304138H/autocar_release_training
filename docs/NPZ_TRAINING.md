# Training and evaluating AutoCAR on paired NPZ data

This document defines the maintained path for training a static two-view
AutoCAR reconstruction model from pre-rendered vessel masks and voxel ground
truth. Dynamic graph construction and voxel-to-graph post-processing are not
part of this path.

The released ImageCAS mesh-rendering pipeline remains available as a legacy
reference. The NPZ path does not require PyTorch3D, `pysdf`, trimesh, napari, or
MinkowskiEngine. Its data alignment and 3D metrics are NumPy implementations,
so scikit-image, nibabel, and OpenCV are not primary dependencies either.

## 1. Environment profiles

The primary profile targets:

- Linux on x86-64 with an NVIDIA GPU
- Python 3.11
- PyTorch 2.4.1 built for CUDA 12.4
- spconv 2.3.8 from the `spconv-cu124` wheel

This profile is intended for retraining. A released MinkowskiEngine checkpoint
is not assumed to load into the spconv port because sparse-backend parameter
layouts differ. The Python 3.8 legacy profile applies only to a pristine
checkout of upstream commit `99ca485` (the released code); this maintained
branch targets Python 3.11 and uses modern syntax, so it is not compatible with
that legacy environment.

A host whose driver or system toolkit reports CUDA 12.5 can use the cu124
wheels: the driver, local toolkit, and runtime bundled into the PyTorch wheel
do not need identical minor-version strings. Install and verify the environment
as follows:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/cuda124.txt
python scripts/check_environment.py
python scripts/check_environment.py --sparse-smoke-test
```

The last command deliberately runs the complete spconv `SpconvUNet34C` on a
small synthetic sparse grid and checks a CUDA backward pass. It exits non-zero
if CUDA, spconv, the full 34C network, or gradient execution fails. The ordinary
checker is diagnostic and does not fail merely because optional GPU packages
are absent.

For CPU-only geometry, data, alignment, and metric development:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Linux CPU wheel:
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# On macOS, use instead: python -m pip install torch==2.4.1
python -m pip install -r requirements/cpu.txt
python scripts/check_environment.py
pytest
```

See [the environment profile guide](../requirements/README.md) for the isolated
legacy CUDA 11.3/MinkowskiEngine environment and an explanation of the CUDA
version numbers reported by the driver, toolkit, and PyTorch.

### Cluster virtual environment at `/export/home2/reny0012/vir_env`

Run these commands from the `codex/npz-training` checkout. They use only the
standard-library `venv` module and pip; conda and a local CUDA toolkit build are
not required:

```bash
cd /path/to/autocar_release
python3.11 -m venv /export/home2/reny0012/vir_env
/export/home2/reny0012/vir_env/bin/python -m pip install --upgrade pip setuptools wheel
/export/home2/reny0012/vir_env/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
/export/home2/reny0012/vir_env/bin/python -m pip install -r requirements/cuda124.txt
/export/home2/reny0012/vir_env/bin/python scripts/check_environment.py
/export/home2/reny0012/vir_env/bin/python scripts/check_environment.py --sparse-smoke-test
```

If that environment directory already exists, first run its `bin/python -V`.
Do not mix a pre-existing Python 3.8/legacy MinkowskiEngine environment with
this Python 3.11/spconv profile. A driver reporting CUDA 12.5 is compatible
with the installed cu124 wheels; `torch.version.cuda` should report `12.4`.

## 2. Expected dataset layout

Projection filenames and voxel filenames do not have to share a stem. Pair
them through the scalar `case_id` stored in each projection NPZ.

One practical layout is:

```text
data/stage2/
  projections/
    lca_0001.npz
    lca_0002.npz
  voxels/
    1.npz
    2.npz
  splits.json
```

The split manifest is a JSON object with non-empty, disjoint `train`, `val`,
and `test` case-ID lists:

```json
{
  "train": ["1", "2"],
  "val": ["3"],
  "test": ["4"]
}
```

Splits must be made by case, not by view, so projections of one anatomy cannot
leak between training and evaluation.

The supplied schema-v2 manifests instead store paths in the top-level split
lists. The maintained LCA/RCA configurations set
`case_id_mode=imagecas_numeric`, which maps both supported forms to the same
numeric physical-case ID used by `<case_number>.npz` voxel files:

```text
.../rca_0508.npz             -> 508
.../lca/23/prefix_02.npz     -> 23
```

The attached manifests validate as 602/75/76 train/validation/test physical
cases for LCA and 600/75/74 for RCA, with no overlap after normalization.

The cluster-specific inputs are configured in
`configs/data/stage2_npz_lca.yaml` and
`configs/data/stage2_npz_rca.yaml`. The loader reads detector spacing from each
projection NPZ and checks it against the declared artery-level invariant:
0.65 mm for LCA and 0.55 mm for RCA. Per-case metadata takes precedence when
present. For legacy RCA files only, the configuration supplies the known
0.55 mm spacing and 900 mm SID when those keys are absent; every sample records
whether each value came from the NPZ or configuration.

### Projection NPZ contract

Required fields:

| Field | Meaning |
| --- | --- |
| `case_id` | Scalar case identifier used to find the GT NPZ |
| `images` | Binary/soft projection masks with shape `[V,H,W]` |
| `theta_deg`, `phi_deg` | One camera angle pair per view |
| `image_dim` | Detector image size in pixels |
| `sid` | Source-to-detector distance in metres; may be omitted only when `fallback_sid_mm` is configured |
| `imager_pixel_spacing` | Detector pixel spacing in millimetres; may be omitted only when `fallback_imager_pixel_spacing_mm` is configured |
| `projection_center_offset` | XYZ reconstruction centre in source units |

`input_scale_to_mm` is used to convert the centre offset to millimetres and
defaults to 1000 when absent. The sample data also contains view features,
world directions, clinical-anchor labels, and vessel-code annotations; these
are retained as metadata but are not required to supervise occupancy. The
maintained `[0,6]` protocol additionally requires `anchor_clinical_views` so
that those positions are verified per case. The source generator used a fixed
source-to-isocentre distance of 750 mm; the loader exposes this as an explicit
constructor/configuration value rather than silently treating it as file data.

### Ground-truth NPZ contract

Required fields:

| Field | Meaning |
| --- | --- |
| `vol` | Binary volume stored in nibabel-style XYZ axis order |
| `spacing` | XYZ voxel spacing in millimetres |

The loader converts `vol[X,Y,Z]` to the canonical model/evaluation order
`gt_volume_zyx[Z,Y,X]`. It does not reorder `spacing`, which remains XYZ. For
these ImageCAS-style arrays, voxel index `(x,y,z)` is centred at physical
`(x,y,z) * spacing_xyz_mm`. Because the common `VoxelGrid` API represents an
origin as a lower boundary, the loader derives the default native origin per
case as `-0.5 * spacing_xyz_mm`.

## 3. Loader output

`src.dataset.stage2_npz.Stage2NPZDataset` validates the paired files. Its public
constructor is:

```python
Stage2NPZDataset(
    projection_source,
    voxel_source,
    *,
    view_mode="all",                 # "all", "fixed", or "random_pair"
    fixed_view_indices=None,
    fixed_view_labels=None,
    random_seed=0,
    minimum_pair_angle_deg=0.0,
    output_type="numpy",             # "numpy" or "torch"
    case_ids=None,
    case_id_mode="literal",          # or "imagecas_numeric"
    expected_imager_pixel_spacing_mm=None,
    fallback_imager_pixel_spacing_mm=None,
    fallback_sid_mm=None,
    gt_origin_xyz_mm=None,
    source_to_isocenter_mm=750.0,
)
```

It emits the following batch fields:

```text
case_id                         string
images                          [selected_V, 1, H, W] float32
gt_volume_zyx                   [Z, Y, X] bool/float32
gt_spacing_xyz_mm               [3] float32
gt_origin_xyz_mm                [3] float32, native lower boundary
projection_center_offset_xyz_mm [3] float32
world2pix4x4                    [selected_V, 4, 4] float32
camera_source_xyz_mm            [selected_V, 3] float32
detector_center_xyz_mm          [selected_V, 3] float32
detector_x_xyz/detector_y_xyz   [selected_V, 3] float32
sid_mm / sid_source             scalar float32 / "npz" or "config"
imager_pixel_spacing_mm         scalar float32
imager_pixel_spacing_source     "npz" or "config"
view_directions_world           [selected_V, 3] float32
view_indices                    [selected_V] int64
pair_angle_deg                  scalar float32 (two-view samples only)
```

`case_ids` restricts discovery to a declared split and rejects missing or
duplicate requested IDs. `minimum_pair_angle_deg` filters candidate pairs by
their world-direction separation; it applies to `random_pair` selection and
defaults to zero in the dataset API. `gt_origin_xyz_mm=None` derives the
ImageCAS lower boundary `-0.5 * gt_spacing_xyz_mm` independently for each case.
Pass an explicit XYZ vector only when the source volume genuinely has a
different lower-bound origin; the override is a boundary, not a voxel centre.
`fixed_view_labels` makes fixed-index evaluation fail early when a case is
missing the expected `anchor_clinical_views` metadata or has different labels.
`expected_imager_pixel_spacing_mm` likewise turns an artery/path mix-up into a
clear error before training. The fallback values are explicitly millimetres;
they are consulted only for a missing field and never overwrite a value stored
in a case file. The maintained RCA configuration uses 900 mm SID, 750 mm
source-to-isocentre distance, and 0.55 mm detector spacing. LCA keeps both
fallbacks disabled because its current files provide the metadata.

Supported view policies are:

- `random_pair`: deterministic two-view sampling keyed by seed, case, and
  epoch for training. The maintained data module sets a 30-degree minimum
  separation to avoid nearly redundant views.
- `fixed`: one predeclared pair for validation and primary comparison.
- `all`: retain all available views for pair-ensemble experiments.

The primary validation protocol uses fixed slots `[0,6]`. In the supplied view
bank these are the `RAO 25, CAU 35` and `LAO 5, CRA 40` clinical anchors; their
attached-case separation is 80.22 degrees. This pair is selected from camera
metadata, not from reconstruction labels, and is inside the training policy's
30-degree support. The maintained data-module configuration verifies these two
labels in every validation and test case before treating the slots as shared
semantics.

AutoCAR is natively a two-view model. Every comparison must therefore receive
the same two selected views. A model using all seven projections has a larger
information budget and must be reported as a separate protocol. Likewise,
evaluating and fusing all 21 pairs would be an ensemble extension, not the
native AutoCAR result.

### Attached-case sparsity audit

The default `max_pixel_distance=0.5` is a provisional, memory-safe starting
point, not a paper-reported hyperparameter. With nearest-pixel sampling and the
strict `< 0.5` test, every accepted detector sample lands on a foreground pixel,
so both EDT feature channels are identically zero; the channels remain in the
model input so a wider threshold can be selected without changing the network
shape. The table below uses the corrected native GT origin, exhaustive
voxel-centre enumeration, nearest-pixel Eq. 5 gating, and the half-open detector
bounds implemented by the maintained path:

| Attached-case pair | Separation | Pixel threshold | Sparse voxels | GT-vessel coverage |
| --- | ---: | ---: | ---: | ---: |
| `[0,6]` (primary) | 80.22 deg | 0.5 | 179,303 | 98.338% |
| `[0,6]` (primary) | 80.22 deg | 1.5 | 270,477 | 99.423% |
| `[0,1]` (audit) | 28.10 deg | 0.5 | 494,527 | 98.583% |
| `[0,1]` (audit) | 28.10 deg | 1.5 | 702,857 | 99.589% |

At threshold 1.5, 16 of the 21 attached-case pairs satisfy the declared
30-degree minimum. Their median hull is 329,433 voxels and the largest is
520,416 voxels for pair `[5,6]` (43.08 degrees); the primary `[0,6]` pair has
270,477 voxels. Threshold 1.5 is the scientifically preferable candidate on
this audit because its primary-pair GT coverage is 99.423% and its EDT channels
remain informative. Do not make it the default until a full CUDA backward pass
has been profiled on worst-case eligible pair `[5,6]`. Predeclare the threshold
and pair rule before a comparison; do not tune them separately on test cases.

## 4. Coordinate conventions

All internal physical geometry uses millimetres and XYZ coordinate vectors.
Dense tensor axes use ZYX.

The projection adapter reconstructs each source position, detector centre, and
detector basis directly from `theta_deg`, `phi_deg`, SID, source-to-isocentre
distance, image dimensions, and pixel spacing. The stored projection centre is
subtracted to enter AutoCAR's centred world coordinates.

`ProjectionGeometry.from_angles(...)` is the single camera-construction API.
It exposes conventional 3x4 projection matrices and 4x4 matrices, point
projection in `(column, row)` or `(x, y)` order, detector-point construction,
and pixel-to-ray conversion. Training code should use this adapter rather than
reconstructing camera conventions independently.

The reconstruction grid uses lower-bound origin plus voxel-centre locations:

```text
xyz_mm = origin_xyz_mm + (index_xyz + 0.5) * spacing_xyz_mm
```

For the native GT convention, the default `origin_xyz_mm = -0.5 * spacing`
makes this simplify to `xyz_mm = index_xyz * spacing_xyz_mm`. This half-voxel
shift is essential: setting the native lower boundary to zero would displace
every GT centre by half a voxel. The attached case supports this convention:
all 1,000 valid points in `raw_vessel_code_mm` fall in foreground voxels when
mapped using the derived lower boundary. This default is evidenced for the
generated ImageCAS-style NPZ files used here; an arbitrary NPZ or NIfTI export
must provide its actual affine or lower-bound origin rather than inheriting the
ImageCAS convention.

Do not compare a centred prediction array directly with the native GT array.
For evaluation, map prediction-grid centres back into the GT physical frame by
adding `projection_center_offset_xyz_mm`, then resample the binary GT with
nearest-neighbour interpolation. This operation also produces a `valid_fov`
mask identifying target voxels covered by the source volume.

## 5. Training contract

The maintained training path is:

1. Load a pair of pre-rendered 2D masks and its voxel GT.
2. Encode each mask with the released 2D hourglass network.
3. Stream bounded 3D voxel centres and reproject them into each detector.
4. Apply nearest-pixel Eq. 5 membership in every view, intersect the surviving
   coordinates, and bilinearly sample the learned encoder features.
5. Concatenate the per-view distance-plus-encoder features and run the sparse
   3D U-Net.
6. Query the aligned GT at the returned sparse coordinates.
7. Optimise occupancy using BCE plus soft Dice.
8. Rasterise validation predictions on one declared physical grid and evaluate
   them with the shared volume metrics.

### Fidelity boundary and known discrepancies

The publication and released repository are not a complete, internally
consistent training specification. The maintained configuration makes the
following choices explicit:

| Item | Paper | Released code/config | Maintained NPZ path |
| --- | --- | --- | --- |
| 2D encoder width | 12 channels | 16 channels | 12 channels |
| Sparse backbone | MinkUNet34C | hard-coded MinkUNet18A | spconv 34C port |
| Two output channels | channel 0 centreness, channel 1 occupancy | training reads channel 0 as occupancy | channel 0 occupancy; channel 1 reserved |
| Optimisation | writes “ADMM”, learning rate `3e-4`, batch 8, accumulation 4, synchronized BN on four A10s | Adam, base `4e-4` (experiment overrides include `1e-3`) | Adam `3e-4`, no scheduler; released optimizer family at the paper's rate |
| Eq. 5 radius `epsilon_v` | numeric value omitted | not recoverable from released module | strict `< 0.5` pixels by default, to be profiled and declared |
| Sparse candidates and Eq. 7 sampling | sampled ray bundles; interpolation unspecified | module absent | exhaustive chunked voxel centres; nearest EDT, bilinear encoder features |

Consequently, this path is suitable for retraining and a controlled voxel
reconstruction comparison, but it is not claimed to reproduce the authors'
undisclosed training run or to load their MinkowskiEngine weights exactly.
Occupancy-only supervision is intentional for the requested voxel benchmark.
The projection NPZ includes vessel-code metadata, but the release does not
provide a complete, testable recipe for reproducing the paper's sparse
centreness labels from these files; that auxiliary target is not needed for
masked-Dice/SSIM occupancy comparison.

The supplied NPZ files contain seven pre-rendered projections. Random pair
selection therefore operates only over those views and does not reproduce the
paper's online pose-cluster sampling or thin-plate-spline augmentation. This is
a deliberate dataset-adapter boundary and must be reported when comparing the
retrained model with paper numbers.

Methods Eq. 7 is implemented at each voxel centre. The paper defines Eq. 5 on
integer active pixels but does not specify Eq. 7 interpolation. The maintained
default therefore samples the EDT with nearest-pixel, ties-to-even rounding
(matching the supplied generator's `numpy.rint`) and samples learned encoder
features with bilinear `grid_sample(..., align_corners=False)`. Integer image
coordinates are pixel centres and the valid detector extent reaches one half
pixel beyond each edge. Voxels whose centres reproject outside that extent are
discarded. `distance_sampling=bilinear` remains an explicitly labelled
continuous-EDT ablation and computes the extra EDT halo required to avoid
artificial dilation from a prematurely capped distance map.

The `LODs` argument is retained for configuration compatibility, but only its
last (finest) value is implemented. The maintained configuration therefore uses
`LODs: [0.5]`; it does not claim progressive coarse-to-fine reconstruction.

Native GT shapes currently require micro-batch 1. The maintained single-GPU
experiment accumulates four steps for effective batch 4, uses `32-true`
precision, and clips gradients at 0.5. Full precision is required for the
tested RTX 6000 Ada/spconv 2.3.8 environment: FP16 failed in spconv's
implicit-GEMM algorithm tuner. Clipping is a practical stability choice, not a
setting reported by the paper.

The target command interface is:

```bash
python -m src.train experiment=stage2_npz \
  data.projection_source=/path/to/projections \
  data.voxel_source=/path/to/voxels \
  data.split_json=/path/to/splits.json
```

For the supplied cluster paths, use the Python launcher. Its preflight
normalizes the path-based split entries, checks every case has exactly one
projection/voxel pair, validates detector spacing, and loads one case from
each split:

```bash
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery lca --preflight-only
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery rca --preflight-only
```

Run a one-batch end-to-end GPU check for each artery before committing a long
job:

```bash
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery lca --max-epochs 1 -- debug=stage2_gpu
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery rca --max-epochs 1 -- debug=stage2_gpu
```

Then launch the independent full training runs:

```bash
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery lca --max-epochs 200
/export/home2/reny0012/vir_env/bin/python scripts/train_imagecas_npz.py --artery rca --max-epochs 200
```

The experiments use separate task names (`train_autocar_lca` and
`train_autocar_rca`), so checkpoints and TensorBoard logs do not collide.
Resume with `--checkpoint /absolute/path/to/last.ckpt`. Start with
`--num-workers 0`; after the first successful epoch, a small positive value can
be benchmarked if the cluster's shared-memory limits permit it.

For a one-batch end-to-end CUDA check, use the dedicated GPU debug profile:

```bash
python -m src.train experiment=stage2_npz debug=stage2_gpu \
  data.projection_source=/path/to/projections \
  data.voxel_source=/path/to/voxels \
  data.split_json=/path/to/splits.json
```

The upstream `debug=default`/`debug=fdr` profiles force CPU and are therefore
incompatible with the spconv model. `debug=stage2_gpu` retains CUDA, uses full
precision, disables subprocess workers, and runs one train and validation
batch with anomaly detection.

The NPZ experiment configuration must record the reconstruction bounds, voxel
size, ray step/LODs, view policy, pair, feature-fusion rule, occupancy threshold,
and random seed. These values are part of the scientific result, not merely
implementation details.

## 6. Validation metrics

The shared implementation is in `src.metrics.volume`.

### Masked Dice

`masked_dice_3d` thresholds prediction and GT (default `0.5`) and computes Dice
inside `valid_fov`:

```text
2 * sum(valid_fov * prediction * ground_truth)
-------------------------------------------------
sum(valid_fov * prediction) + sum(valid_fov * ground_truth)
```

Using the valid physical field of view as the mask retains penalties for false
positives. A GT-foreground-only mask must not be used for the primary Dice
result. Empty/empty is defined as 1.

### 3D SSIM

`structural_similarity_3d` uses probabilities in `[0,1]`, `data_range=1`, and a
default uniform `7 x 7 x 7` window. The implementation computes the result in
depth chunks to avoid materialising several full-resolution floating-point
volumes. Only windows fully contained in the valid FOV contribute, preventing
out-of-volume padding from leaking into the score.

Global 3D SSIM is strongly background dominated for sparse vessels and must
not be interpreted alone. On the attached aligned volume an all-zero predictor
has Dice 0 but global SSIM about 0.995534. The evaluator therefore also reports
`masked_ssim_3d`, using the predeclared union of predicted and GT foreground as
the window-centre selector (the same all-zero prediction scores about
`2.33e-6`). Report global SSIM because it is part of the requested protocol,
but pair it with masked Dice and vessel-window masked SSIM.

`compute_volume_metrics` returns at least:

- `masked_dice_3d`
- `ssim_3d`
- foreground, intersection, and valid-voxel counts needed to audit the score

Thresholds must be predeclared or chosen once on the validation set. Never
optimise a threshold independently for each test case.

Export a checkpoint's fixed-pair dense predictions and evaluate every selected
case in one command:

```bash
python -m src.predict_npz \
  --checkpoint /path/to/checkpoints/best.ckpt \
  --projections /path/to/projections \
  --voxels /path/to/voxels \
  --split-json /path/to/splits.json \
  --split test \
  --output-directory /path/to/predictions \
  --view-indices 0 6 \
  --evaluate
```

Use the case-level split manifest for the primary aggregate. Repeatable
`--case-id ID` flags are available for explicitly identified diagnostic cases
and are mutually exclusive with `--split-json`. Unrestricted export is allowed,
but `--evaluate` refuses to run without a split manifest or explicit case IDs
so train/validation/test cases cannot be silently mixed. The command writes one
`<case_id>.npz` probability grid per case, one `<case_id>.metrics.json` when
`--evaluate` is enabled, and `summary.json`. The summary contains the case
manifest plus aggregate mean, sample standard deviation, and standard error
for masked Dice, global 3D SSIM, and vessel-window masked 3D SSIM when defined.
Existing prediction files are protected unless `--overwrite` is supplied.
CUDA inference defaults to `--precision 32` for the tested RTX 6000
Ada/spconv 2.3.8 environment. `--precision 16-mixed` remains an explicit
experimental override for environments where spconv's implicit-GEMM tuner
supports it. Every prediction NPZ records its axis order, bounds,
voxel size, case ID, selected views, and pair angle. `summary.json` also records
the checkpoint, device, inference precision, output dtype, sparse backend, and
the complete sparse-projection protocol (candidate mode, distance sampling,
threshold, support rule, fusion, bounds, and voxel size), selected case IDs,
split name, and a SHA-256 digest of the split manifest.

By default the exporter also verifies the two expected clinical-anchor labels.
`--skip-anchor-label-check` exists for a separately documented dataset schema;
using it removes the semantic-slot safeguard and should be recorded as a
protocol change.

For a complete config-driven validation-and-test run, copy either
`configs/eval_npz_paper_metric_template.json` or
`configs/eval_npz_visualisation_template.json`, fill in the checkpoint and
dataset paths, and run:

```bash
python -m src.eval_npz --config /path/to/eval_config.json
```

This runner follows the parametric evaluator's conventions. `eval_split` may
be `val`, `test`, or `val_test`; `num_eval_cases` accepts a positive integer or
`"all"`; and `eval_case_ids` can select named diagnostic cases without leaving
the requested split. Every selected case is reconstructed directly from the
fixed input views and saved under
`predictions/final/{validation,test}/<case_id>.npz`. The output also includes a
prediction manifest, synchronized model-forward timing, `performance_*` JSON
files, a resolved configuration, and an audited evaluation record.

`evaluation_mode: "paper_metric"` writes per-case JSON/CSV, aggregate macro
mean/standard-error and micro Dice summaries, and a split comparison chart
under `metrics/`. As in the parametric evaluator, it resamples native GT and
the thresholded prediction onto an endpoint-aligned `128 x 128 x 128` grid
covering the full native CT field of view. The comparison masks are saved under
`metrics/voxel_masks/` when `paper_metric_save_masks` is true. The original
full-resolution AutoCAR probability grid is always retained under
`predictions/`. `evaluation_mode: "visualisation"` writes an input-view panel,
orthogonal probability/GT overlays, an optional rotating 3D GIF, case metrics,
and an artifact manifest under
`visualization/<case_id>/<split>/final/`. Use `max_visualizations` to cap these
bundles without limiting prediction or metric generation. Set
`visualization_gif_frames` to `0` to skip GIF rendering while retaining the
static monitor images.

For split manifests that contain ImageCAS source paths (rather than already
normalized IDs), set `case_id_mode` to `"imagecas_numeric"`. Set
`expected_imager_pixel_spacing_mm` to the artery-specific detector spacing
when it is known (`0.65` for the supplied LCA inputs and `0.55` for RCA).

The framework-independent evaluator accepts one exported probability or logit
volume from AutoCAR or another method at a time. Its command interface is:

```bash
python -m src.evaluate_npz \
  --prediction /path/to/prediction.npz \
  --prediction-key prediction_volume_zyx \
  --ground-truth /path/to/voxels/1.npz \
  --projection /path/to/projections/lca_0001.npz \
  --output /path/to/metrics/1.json
```

Use `--prediction-domain logit` when the exported array has not passed through
a sigmoid. Exported NPZ files supply and validate `volume_axis_order`,
`bbox_min_xyz_mm`, `bbox_max_xyz_mm`, `voxel_size_mm`, case ID, and view indices.
Explicit geometry flags must agree with any embedded metadata. A bare `.npy`
or metadata-free NPZ must declare `--prediction-axis-order`,
`--bbox-min-xyz-mm`, and `--voxel-size-mm`; this prevents a differently shaped
comparison grid from being silently interpreted as the AutoCAR default. Keep
the resolved grid, threshold, SSIM window, and SSIM chunk depth identical
between methods. The evaluator uses the same per-case native-origin default as
the loader. Use `--ground-truth-origin-xyz-mm X Y Z` only to supply a different
native lower boundary explicitly.

Use the same evaluator, grid, pair, threshold, and FOV definition for AutoCAR
and the comparison model.

## 7. Required tests before training

CPU tests must establish:

- projection metadata validation and case-ID matching;
- XYZ-to-ZYX conversion and anisotropic spacing handling;
- camera reprojection agreement with supplied vessel-code/mask samples;
- exhaustive chunked voxel enumeration, nearest-EDT membership, bilinear
  feature sampling, two-view intersection, and the legacy ray path;
- identity and deliberately shifted physical-volume alignment;
- exact Dice examples, empty-volume behaviour, mask semantics, SSIM symmetry,
  and chunked-versus-reference SSIM agreement;
- a model forward pass and gradient flow to the 2D features.

GPU acceptance requires:

1. `python scripts/check_environment.py --sparse-smoke-test` passes.
2. The model completes a finite forward/backward step.
3. A one-case training run can intentionally overfit.
4. A short multi-case run writes and reloads a checkpoint.
5. Evaluation emits per-case metrics and aggregate mean, standard deviation,
   and standard error without changing the declared protocol.
