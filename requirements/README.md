# Environment profiles

The maintained training path targets Python 3.11 and prebuilt CUDA 12.4
wheels. PyTorch is intentionally installed in a separate command so pip uses
the official PyTorch wheel index without forcing every other dependency to be
resolved from that index.

## Primary GPU profile (Linux/NVIDIA)

This profile is appropriate when `nvidia-smi` reports driver support for CUDA
12.4 or newer, including a host whose locally installed toolkit is CUDA 12.5.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/cuda124.txt
python scripts/check_environment.py
python scripts/check_environment.py --sparse-smoke-test
```

To install at the requested shared-filesystem location, create the environment
there and invoke its interpreter explicitly (activation is optional):

```bash
python3.11 -m venv /export/home2/reny0012/vir_env
/export/home2/reny0012/vir_env/bin/python -m pip install --upgrade pip setuptools wheel
/export/home2/reny0012/vir_env/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
/export/home2/reny0012/vir_env/bin/python -m pip install -r requirements/cuda124.txt
```

The `spconv-cu124` wheel requires an x86-64 Linux environment with a recent
enough glibc. CUDA 12.4 wheels for spconv 2.3.8 use the `manylinux_2_28` tag.
Run the diagnostic before launching training.

## CPU development profile

The CPU profile supports dataset, camera, coordinate-alignment, and metric
tests. It does not represent production training performance.

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

Do not request `--sparse-smoke-test` in the CPU profile. The default diagnostic
is intentionally successful when CUDA and optional sparse packages are absent.

## Legacy reproduction profile (pristine upstream only)

The released network used Python 3.8, PyTorch 1.11, CUDA 11.3, PyTorch3D, and a
source-built MinkowskiEngine. Keep it in a separate container or conda
environment; it is not a dependency of the maintained NPZ pipeline. This
profile applies only to a pristine checkout of upstream commit `99ca485`. The
maintained branch targets Python 3.11 and uses modern syntax, so it is not
compatible with this Python 3.8 environment.

```bash
conda create -n autocar-legacy python=3.8.13
conda activate autocar-legacy
conda install pytorch==1.11.0 torchvision cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
python -m pip install -r requirements/legacy-cu113.txt
```

MinkowskiEngine must then be built against an installed CUDA 11.3 toolkit with
`nvcc`, as described by the upstream repository. It must not be imported by the
new spconv-based path. Use the pristine upstream checkout and this legacy
profile when inspecting or reproducing an author's MinkowskiEngine checkpoint;
sparse-backend parameter layouts are not assumed to be checkpoint-compatible.
The maintained cu124/spconv branch is intended for retraining.

## Why the CUDA version numbers differ

- `nvidia-smi` reports the newest CUDA API level supported by the installed
  NVIDIA driver. It does not prove that the matching CUDA toolkit is installed.
- `nvcc --version` reports the compiler/toolkit found on `PATH`. Prebuilt
  PyTorch and spconv wheels normally do not need this compiler.
- `torch.version.cuda` reports the CUDA runtime used to build the installed
  PyTorch wheel. For the primary profile it should report `12.4`.

It is therefore expected for a working machine to show 12.5 in `nvidia-smi` or
`nvcc` and 12.4 in PyTorch. The NVIDIA driver must be new enough for the wheel's
runtime; the three strings do not need to be identical.

References:

- [PyTorch installation and archived cu124 commands](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [spconv installation](https://github.com/traveller59/spconv)
- [MinkowskiEngine installation](https://github.com/NVIDIA/MinkowskiEngine)
