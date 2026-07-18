# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the archiveable CUDA validation capture script."""

from types import SimpleNamespace

import numpy as np

from scripts.capture_cuda_validation import (
    build_cuda_validation_report,
    collect_cuda_environment,
    write_json_report,
)
from tests._fake_cupy_backend import install_fake_cupy_module


class _FakeRuntime:
    """Small CUDA runtime facade for validation-report tests."""

    @staticmethod
    def getDeviceCount():
        """Return one fake CUDA device."""
        return 1

    @staticmethod
    def runtimeGetVersion():
        """Return a CUDA 12.2-style integer version."""
        return 12020

    @staticmethod
    def driverGetVersion():
        """Return a CUDA 12.3-style integer driver version."""
        return 12030

    @staticmethod
    def getDeviceProperties(device_id):
        """Return stable fake device properties."""
        assert device_id == 0
        return {
            "name": b"Fake CUDA Device\x00",
            "major": 6,
            "minor": 1,
            "totalGlobalMem": 4 * 1024**3,
            "multiProcessorCount": 5,
        }


class _FakeDevice:
    """Fake CuPy device descriptor."""

    compute_capability = "61"

    def __init__(self, device_id):
        """Store the requested device identifier."""
        self.device_id = device_id


class _FakeNullStream:
    """Fake default stream with a no-op synchronization method."""

    @staticmethod
    def synchronize():
        """Synchronize the fake stream."""
        return None


def _install_validation_cupy(monkeypatch):
    """Install fake CuPy with the CUDA metadata surface used by the script."""
    fake_cupy = install_fake_cupy_module(monkeypatch)
    fake_cupy.__version__ = "14.1.1-test"
    fake_cupy.cuda = SimpleNamespace(
        runtime=_FakeRuntime(),
        Device=_FakeDevice,
        Stream=SimpleNamespace(null=_FakeNullStream()),
        nvrtc=SimpleNamespace(getVersion=lambda: (12, 2)),
    )
    return fake_cupy


def test_collect_cuda_environment_marks_import_failure_as_unavailable(monkeypatch):
    """Missing CuPy should produce structured unavailable metadata."""
    def fail_import(name):
        """Raise ImportError for the CuPy import requested by the script."""
        if name == "cupy":
            raise ImportError("CuPy intentionally unavailable")
        raise AssertionError(name)

    monkeypatch.setattr("scripts.capture_cuda_validation.importlib.import_module", fail_import)
    environment = collect_cuda_environment()

    assert environment["cupy_importable"] is False
    assert environment["cuda_usable"] is False
    assert environment["cuda_smoke_test"]["status"] == "unavailable"
    assert environment["cuda_smoke_test"]["error_type"] == "ImportError"


def test_cuda_validation_report_records_parity_and_claim_boundaries(monkeypatch, tmp_path):
    """Fake-CuPy execution should produce a complete machine-readable report."""
    fake_cupy = _install_validation_cupy(monkeypatch)

    report = build_cuda_validation_report(
        models=["itfrs", "vqrs", "owafrs"],
        block_sizes=[2],
        atol=1e-12,
        cupy_module=fake_cupy,
    )

    assert report["status"] == "success"
    assert report["environment"]["cuda_usable"] is True
    assert report["environment"]["cuda_runtime_version"] == "12.2"
    assert report["environment"]["cuda_driver_version"] == "12.3"
    assert report["environment"]["devices"][0]["name"] == "Fake CUDA Device"
    assert report["summary"]["case_count"] == 6
    assert report["summary"]["failed_cases"] == 0
    assert report["claim_boundaries"]["owafrs_gpu_approximation_accumulators"] is False
    assert report["claim_boundaries"]["full_gpu_native_pipeline_claimed"] is False
    assert all(row["numerical_parity"] for row in report["parity_cases"])

    owafrs_rows = [row for row in report["parity_cases"] if row["model"] == "owafrs"]
    assert owafrs_rows
    assert all(row["used_gpu_similarity_blocks"] is True for row in owafrs_rows)
    assert all(row["used_gpu_approximation_accumulators"] is False for row in owafrs_rows)

    output_path = tmp_path / "cuda_validation.json"
    write_json_report(report, output_path)
    payload = output_path.read_text(encoding="utf-8")
    assert '"schema_version": "1.0"' in payload
    assert '"status": "success"' in payload
    assert np.isfinite(report["summary"]["maximum_observed_absolute_difference"])
