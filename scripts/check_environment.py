#!/usr/bin/env python3
"""Report the AutoCAR runtime environment without requiring ML packages.

The default command is diagnostic only and exits successfully even when CUDA,
PyTorch, or spconv are absent.  Pass ``--sparse-smoke-test`` to request a tiny
spconv forward/backward pass; a requested smoke test fails with a non-zero exit
code when its prerequisites are unavailable or the operation fails.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


COMMAND_TIMEOUT_SECONDS = 8


def _run_command(command: Sequence[str]) -> Dict[str, Any]:
    """Run a read-only diagnostic command and turn every failure into data."""

    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": list(command),
            "reason": "executable not found on PATH",
        }

    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "command": list(command),
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return {
        "available": True,
        "command": list(command),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _first_match(pattern: str, value: str) -> Optional[str]:
    match = re.search(pattern, value, flags=re.IGNORECASE)
    return match.group(1) if match else None


def parse_nvcc_version(output: str) -> Optional[str]:
    """Extract the toolkit release from ``nvcc --version`` output."""

    return _first_match(r"\brelease\s+([0-9]+(?:\.[0-9]+)*)", output)


def parse_nvidia_smi_cuda_version(output: str) -> Optional[str]:
    """Extract nvidia-smi's driver-supported CUDA version."""

    return _first_match(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", output)


def _version_at_least(value: Optional[str], required: str) -> Optional[bool]:
    """Compare numeric dotted versions, returning None when parsing fails."""

    if value is None:
        return None
    try:
        parsed = tuple(int(part) for part in value.split("."))
        minimum = tuple(int(part) for part in required.split("."))
    except ValueError:
        return None
    width = max(len(parsed), len(minimum))
    parsed += (0,) * (width - len(parsed))
    minimum += (0,) * (width - len(minimum))
    return parsed >= minimum


def collect_nvcc() -> Dict[str, Any]:
    result = _run_command(["nvcc", "--version"])
    combined = "\n".join(
        part for part in (result.get("stdout", ""), result.get("stderr", "")) if part
    )
    result["toolkit_version"] = parse_nvcc_version(combined)
    return result


def _parse_csv_rows(output: str, columns: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(columns):
            rows.append(dict(zip(columns, values)))
    return rows


def collect_nvidia_smi() -> Dict[str, Any]:
    summary = _run_command(["nvidia-smi"])
    raw_summary = summary.pop("stdout", "")
    summary["driver_supported_cuda"] = parse_nvidia_smi_cuda_version(
        raw_summary
    )
    if summary.get("ok"):
        # The default output can contain an unrelated process list. Do not
        # retain it in a report that may be attached to an issue or CI log.
        summary.pop("stderr", None)

    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    columns = [
        "index",
        "name",
        "driver_version",
        "compute_capability",
        "memory_total_mib",
    ]
    if query.get("ok"):
        summary["gpus"] = _parse_csv_rows(query.get("stdout", ""), columns)
    else:
        # Older drivers may not support the compute_cap query field.
        fallback = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        fallback_columns = ["index", "name", "driver_version", "memory_total_mib"]
        summary["gpus"] = (
            _parse_csv_rows(fallback.get("stdout", ""), fallback_columns)
            if fallback.get("ok")
            else []
        )
        summary["query_warning"] = query.get("stderr") or query.get("reason")
    return summary


def _distribution_version(candidates: Sequence[str]) -> Dict[str, Any]:
    """Find the installed distribution without importing its extension modules."""

    for name in candidates:
        try:
            return {
                "installed": True,
                "distribution": name,
                "version": importlib.metadata.version(name),
            }
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # Metadata should never make the checker fail.
            return {
                "installed": None,
                "distribution": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {"installed": False, "distribution": None, "version": None}


def collect_sparse_packages() -> Dict[str, Any]:
    return {
        "spconv": _distribution_version(
            [
                "spconv-cu126",
                "spconv-cu124",
                "spconv-cu121",
                "spconv-cu120",
                "spconv-cu118",
                "spconv",
            ]
        ),
        "cumm": _distribution_version(
            [
                "cumm-cu126",
                "cumm-cu124",
                "cumm-cu121",
                "cumm-cu120",
                "cumm-cu118",
                "cumm",
            ]
        ),
        "minkowski_engine": _distribution_version(
            ["MinkowskiEngine", "minkowskiengine"]
        ),
    }


def collect_torch() -> Dict[str, Any]:
    metadata = _distribution_version(["torch"])
    result: Dict[str, Any] = dict(metadata)
    if not metadata.get("installed"):
        return result

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        result.update(
            {
                "import_ok": False,
                "import_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result

    result.update(
        {
            "import_ok": True,
            "version": getattr(torch, "__version__", metadata.get("version")),
            # This is the CUDA runtime against which the PyTorch wheel was built.
            "built_cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        }
    )

    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        result["cuda_available"] = False
        result["cuda_error"] = f"{type(exc).__name__}: {exc}"

    try:
        result["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:
        result["cudnn_error"] = f"{type(exc).__name__}: {exc}"

    try:
        mps = getattr(torch.backends, "mps", None)
        result["mps_built"] = bool(mps is not None and mps.is_built())
        result["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception as exc:
        result["mps_error"] = f"{type(exc).__name__}: {exc}"

    result["devices"] = []
    if result.get("cuda_available"):
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                capability = torch.cuda.get_device_capability(index)
                result["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": f"{capability[0]}.{capability[1]}",
                        "memory_total_mib": round(properties.total_memory / (1024**2)),
                    }
                )
        except Exception as exc:
            result["device_query_error"] = f"{type(exc).__name__}: {exc}"
    return result


def compatibility_notes(report: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    platform_info = report["platform"]
    if platform_info["system"] != "Linux" or platform_info["machine"].lower() not in {
        "x86_64",
        "amd64",
    }:
        notes.append(
            "The primary spconv CUDA profile targets x86-64 Linux; use this "
            "host only for CPU-side development."
        )
    libc_name = (platform_info.get("libc_name") or "").lower()
    libc_version = platform_info.get("libc_version")
    if libc_name == "glibc" and _version_at_least(libc_version, "2.28") is False:
        notes.append(
            f"spconv-cu124 uses a manylinux_2_28 wheel, but this host reports "
            f"glibc {libc_version}."
        )

    python_version = report["python"]["version_info"]
    if python_version[:2] != [3, 11]:
        notes.append(
            "The primary profile is pinned to Python 3.11; this interpreter is "
            f"{python_version[0]}.{python_version[1]}."
        )

    torch_info = report["torch"]
    if not torch_info.get("installed"):
        notes.append("PyTorch is not installed; CPU-only diagnostics are still valid.")
    elif not torch_info.get("import_ok"):
        notes.append("PyTorch is installed but failed to import; inspect torch.import_error.")
    elif not torch_info.get("cuda_available"):
        notes.append("PyTorch cannot currently access CUDA; GPU training is unavailable here.")

    smi_version = report["nvidia_smi"].get("driver_supported_cuda")
    nvcc_version = report["nvcc"].get("toolkit_version")
    torch_cuda = torch_info.get("built_cuda_runtime")
    if any((smi_version, nvcc_version, torch_cuda)):
        notes.append(
            "nvidia-smi, nvcc, and torch.version.cuda describe different layers; "
            "their minor versions need not be identical."
        )
    if _version_at_least(smi_version, "12.4") is False:
        notes.append(
            f"The NVIDIA driver reports CUDA {smi_version}; the primary cu124 "
            "profile requires driver support for CUDA 12.4 or newer."
        )
    if torch_cuda and not str(torch_cuda).startswith("12.4"):
        notes.append(
            "The primary wheel profile expects PyTorch cu124; this wheel "
            f"reports CUDA {torch_cuda}."
        )
    return notes


def collect_report() -> Dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    report: Dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "libc_name": libc_name or None,
            "libc_version": libc_version or None,
        },
        "python": {
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
        },
        "nvidia_smi": collect_nvidia_smi(),
        "nvcc": collect_nvcc(),
        "torch": collect_torch(),
        "sparse_packages": collect_sparse_packages(),
    }
    report["notes"] = compatibility_notes(report)
    return report


def run_sparse_smoke_test() -> Dict[str, Any]:
    """Run the complete spconv U-Net on CUDA and verify backward gradients."""

    result: Dict[str, Any] = {"requested": True, "ok": False}
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        result["error"] = f"PyTorch import failed: {type(exc).__name__}: {exc}"
        return result

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        result["error"] = (
            "CUDA availability check failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return result
    if not cuda_available:
        result["error"] = "torch.cuda.is_available() is False"
        return result

    try:
        spconv = importlib.import_module("spconv.pytorch")
    except Exception as exc:
        result["error"] = f"spconv import failed: {type(exc).__name__}: {exc}"
        return result

    try:
        device = torch.device("cuda:0")
        axis = torch.arange(16, dtype=torch.int32, device=device)
        coordinates_zyx = torch.cartesian_prod(axis, axis, axis)
        indices = torch.cat(
            [
                torch.zeros(
                    (coordinates_zyx.shape[0], 1),
                    dtype=torch.int32,
                    device=device,
                ),
                coordinates_zyx,
            ],
            dim=1,
        )
        features = torch.randn(
            indices.shape[0],
            2,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        sparse_input = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[16, 16, 16],
            batch_size=1,
        )
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.modules.spconv_unet import SpconvUNet34C

        model = SpconvUNet34C(2, 2).to(device).eval()
        output = model(sparse_input)
        loss = output.features.square().mean()
        loss.backward()

        feature_grad_ok = features.grad is not None and bool(
            torch.isfinite(features.grad).all().item()
        )
        parameter_grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        parameter_grad_ok = bool(parameter_grads) and all(
            grad is not None and bool(torch.isfinite(grad).all().item())
            for grad in parameter_grads
        )
        if not feature_grad_ok or not parameter_grad_ok:
            raise RuntimeError("forward completed but finite backward gradients were not produced")

        result.update(
            {
                "ok": True,
                "device": torch.cuda.get_device_name(0),
                "input_points": int(features.shape[0]),
                "output_points": int(output.features.shape[0]),
                "network": "SpconvUNet34C",
                "loss": float(loss.detach().cpu()),
                "feature_gradient_finite": feature_grad_ok,
                "parameter_gradients_finite": parameter_grad_ok,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return result


def _status(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def format_human(report: Dict[str, Any]) -> str:
    torch_info = report["torch"]
    sparse_packages = report["sparse_packages"]
    lines = [
        "AutoCAR environment report",
        f"OS: {report['platform']['platform']} ({report['platform']['machine']})",
        f"Python: {report['python']['version']} ({report['python']['executable']})",
        "nvidia-smi: "
        + (
            "available; driver-supported CUDA "
            + (report["nvidia_smi"].get("driver_supported_cuda") or "unknown")
            if report["nvidia_smi"].get("available")
            else "not found"
        ),
        "nvcc: "
        + (
            f"available; toolkit {report['nvcc'].get('toolkit_version') or 'unknown'}"
            if report["nvcc"].get("available")
            else "not found"
        ),
        "PyTorch: "
        + (
            f"{torch_info.get('version')}; built CUDA "
            f"{torch_info.get('built_cuda_runtime') or 'none'}; "
            f"CUDA available {_status(torch_info.get('cuda_available'))}"
            if torch_info.get("installed")
            else "not installed"
        ),
        "spconv: "
        + (
            f"{sparse_packages['spconv'].get('distribution')} "
            f"{sparse_packages['spconv'].get('version')}"
            if sparse_packages["spconv"].get("installed")
            else "not installed"
        ),
        "cumm: "
        + (
            f"{sparse_packages['cumm'].get('distribution')} "
            f"{sparse_packages['cumm'].get('version')}"
            if sparse_packages["cumm"].get("installed")
            else "not installed"
        ),
    ]
    for device in torch_info.get("devices", []):
        lines.append(
            "GPU {index}: {name}; compute {compute_capability}; {memory_total_mib} MiB".format(
                **device
            )
        )
    for note in report.get("notes", []):
        lines.append(f"Note: {note}")
    smoke = report.get("sparse_smoke_test")
    if smoke is not None:
        lines.append(
            "spconv forward/backward smoke test: "
            + ("passed" if smoke.get("ok") else f"failed ({smoke.get('error', 'unknown error')})")
        )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of text"
    )
    parser.add_argument(
        "--sparse-smoke-test",
        action="store_true",
        help="run a tiny spconv CUDA forward/backward pass (requires a GPU)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = collect_report()
    if args.sparse_smoke_test:
        report["sparse_smoke_test"] = run_sparse_smoke_test()

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_human(report))

    smoke = report.get("sparse_smoke_test")
    return 1 if smoke is not None and not smoke.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
