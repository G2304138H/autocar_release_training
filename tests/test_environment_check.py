"""CPU-safe tests for the standalone environment diagnostic."""

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


def _minimal_report():
    return {
        "platform": {"platform": "TestOS", "machine": "test-machine"},
        "python": {
            "version": "3.11.0",
            "version_info": [3, 11, 0],
            "executable": "/test/python",
        },
        "nvidia_smi": {"available": False, "gpus": []},
        "nvcc": {"available": False, "toolkit_version": None},
        "torch": {"installed": False, "devices": []},
        "sparse_packages": {
            "spconv": {"installed": False},
            "cumm": {"installed": False},
            "minkowski_engine": {"installed": False},
        },
        "notes": [],
    }


def test_parse_cuda_versions():
    assert (
        environment.parse_nvidia_smi_cuda_version(
            "NVIDIA-SMI 555.42  Driver Version: 555.42  CUDA Version: 12.5"
        )
        == "12.5"
    )
    assert (
        environment.parse_nvcc_version(
            "Cuda compilation tools, release 12.4, V12.4.131"
        )
        == "12.4"
    )
    assert environment.parse_nvcc_version("nvcc unavailable") is None
    assert environment._version_at_least("12.5", "12.4") is True
    assert environment._version_at_least("12.3", "12.4") is False
    assert environment._version_at_least(None, "12.4") is None


def test_missing_command_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda _name: None)

    result = environment._run_command(["not-present", "--version"])

    assert result["available"] is False
    assert "not found" in result["reason"]


def test_human_output_handles_cpu_only_report():
    output = environment.format_human(_minimal_report())

    assert "nvidia-smi: not found" in output
    assert "PyTorch: not installed" in output
    assert "spconv: not installed" in output


def test_json_main_is_diagnostic_on_cpu(monkeypatch, capsys):
    report = _minimal_report()
    monkeypatch.setattr(environment, "collect_report", lambda: report)

    assert environment.main(["--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["platform"]["platform"] == "TestOS"


def test_requested_sparse_smoke_failure_sets_nonzero_exit(monkeypatch, capsys):
    report = _minimal_report()
    monkeypatch.setattr(environment, "collect_report", lambda: report)
    monkeypatch.setattr(
        environment,
        "run_sparse_smoke_test",
        lambda: {"requested": True, "ok": False, "error": "no CUDA"},
    )

    assert environment.main(["--json", "--sparse-smoke-test"]) == 1
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["sparse_smoke_test"]["error"] == "no CUDA"
