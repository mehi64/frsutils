# SPDX-License-Identifier: BSD-3-Clause
"""Capture an archiveable real-CUDA validation report for FRsutils.

The script records platform, package, CUDA, and GPU metadata together with a
NumPy-dense versus CuPy-blockwise numerical parity matrix for all public models.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frsutils import compute_approximations  # noqa: E402


VALIDATION_X = np.asarray(
    [
        [0.00, 0.00, 0.10],
        [0.08, 0.18, 0.12],
        [0.24, 0.10, 0.22],
        [0.52, 0.62, 0.58],
        [0.68, 0.72, 0.66],
        [0.84, 0.77, 0.81],
        [0.93, 0.88, 0.95],
        [1.00, 0.82, 0.91],
    ],
    dtype=np.float64,
)
VALIDATION_Y = np.asarray(["cold", "cold", "cold", "warm", "warm", "warm", "hot", "hot"])
EXPECTED_GPU_ACCUMULATORS = {"itfrs": True, "vqrs": True, "owafrs": False}

MODEL_CONFIGS: Dict[str, Sequence[Mapping[str, Any]]] = {
    "itfrs": (
        {"similarity": "linear"},
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "yager",
            "ub_tnorm_p": 1.7,
            "lb_implicator_name": "lukasiewicz",
        },
    ),
    "vqrs": (
        {"similarity": "linear"},
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "lb_fuzzy_quantifier_name": "linear",
            "lb_fuzzy_quantifier_alpha": 0.2,
            "lb_fuzzy_quantifier_beta": 1.0,
            "ub_fuzzy_quantifier_name": "quadratic",
            "ub_fuzzy_quantifier_alpha": 0.0,
            "ub_fuzzy_quantifier_beta": 0.6,
        },
    ),
    "owafrs": (
        {"similarity": "linear"},
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "yager",
            "ub_tnorm_p": 1.7,
            "lb_implicator_name": "lukasiewicz",
            "lb_owa_method_name": "harmonic",
            "ub_owa_method_name": "exponential",
            "ub_owa_method_base": 1.3,
        },
    ),
}


def _source_project_version() -> Optional[str]:
    """Read the project version from the source checkout when available."""
    pyproject_path = _REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    match = re.search(
        r"(?ms)^\[project\].*?^version\s*=\s*[\"']([^\"']+)[\"']",
        pyproject_path.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def _distribution_version(name: str) -> Optional[str]:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return _source_project_version() if name == "frsutils" else None


def _normalize_compute_capability(value: Any) -> Optional[str]:
    """Normalize CUDA compute capability into dotted major/minor form."""
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit() and len(text) >= 2:
        return f"{text[:-1]}.{text[-1]}"
    return text or None


def _capture_cupy_config(cp: Any) -> Optional[str]:
    """Capture ``cupy.show_config()`` output when the function is available."""
    show_config = getattr(cp, "show_config", None)
    if not callable(show_config):
        return None
    stream = io.StringIO()
    try:
        with redirect_stdout(stream):
            show_config()
    except Exception:
        return None
    return stream.getvalue().strip() or None


def _run_command(command: Sequence[str]) -> Dict[str, Any]:
    """Run a metadata command without raising when the command is unavailable."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "status": "unavailable",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "command": list(command),
        "status": "success" if completed.returncode == 0 else "failed",
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _decode_cuda_version(raw_version: Any) -> Optional[str]:
    """Convert CUDA integer versions such as 12020 into ``12.2``."""
    try:
        value = int(raw_version)
    except (TypeError, ValueError):
        return None
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def _safe_device_properties(cp: Any, device_id: int) -> Dict[str, Any]:
    """Return JSON-safe CuPy device properties for one CUDA device."""
    try:
        raw = dict(cp.cuda.runtime.getDeviceProperties(device_id))
    except Exception:
        raw = {}

    properties: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = key.decode("utf-8", errors="replace") if isinstance(key, bytes) else str(key)
        if isinstance(value, bytes):
            normalized_value: Any = value.decode("utf-8", errors="replace").rstrip("\x00")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized_value = value
        else:
            normalized_value = str(value)
        properties[normalized_key] = normalized_value
    return properties


def collect_cuda_environment(cupy_module: Any = None) -> Dict[str, Any]:
    """Collect platform, package, CUDA, and GPU metadata.

    Parameters
    ----------
    cupy_module : module or None, default=None
        Optional injected CuPy-compatible module used by tests. When omitted,
        CuPy is imported lazily.

    Returns
    -------
    dict
        JSON-safe environment metadata and CUDA smoke-test status.
    """
    environment: Dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "frsutils_version": _distribution_version("frsutils"),
        "cupy_version": None,
        "cupy_distribution_version": _distribution_version("cupy-cuda12x"),
        "cuda_runtime_header_distribution_version": _distribution_version("nvidia-cuda-runtime-cu12"),
        "cuda_pathfinder_distribution_version": _distribution_version("cuda-pathfinder"),
        "cupy_show_config": None,
        "cupy_importable": False,
        "cuda_usable": False,
        "cuda_device_count": 0,
        "devices": [],
        "cuda_runtime_version": None,
        "cuda_driver_version": None,
        "nvrtc_version": None,
        "cuda_smoke_test": {"status": "not_run", "error_type": None, "error_message": None},
        "commands": {
            "nvidia_smi": _run_command(["nvidia-smi"]),
            "nvcc_version": _run_command(["nvcc", "--version"]),
        },
    }

    try:
        cp = cupy_module if cupy_module is not None else importlib.import_module("cupy")
    except Exception as exc:
        environment["cuda_smoke_test"] = {
            "status": "unavailable",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        return environment

    environment["cupy_importable"] = True
    environment["cupy_version"] = str(getattr(cp, "__version__", "unknown"))
    environment["cupy_show_config"] = _capture_cupy_config(cp)

    try:
        runtime = cp.cuda.runtime
        device_count = int(runtime.getDeviceCount())
        environment["cuda_device_count"] = device_count
        environment["cuda_runtime_version"] = _decode_cuda_version(runtime.runtimeGetVersion())
        environment["cuda_driver_version"] = _decode_cuda_version(runtime.driverGetVersion())

        try:
            nvrtc_version = cp.cuda.nvrtc.getVersion()
            environment["nvrtc_version"] = ".".join(str(part) for part in nvrtc_version)
        except Exception:
            environment["nvrtc_version"] = None

        devices = []
        for device_id in range(device_count):
            properties = _safe_device_properties(cp, device_id)
            try:
                compute_capability = _normalize_compute_capability(
                    cp.cuda.Device(device_id).compute_capability
                )
            except Exception:
                major = properties.get("major")
                minor = properties.get("minor")
                compute_capability = f"{major}.{minor}" if major is not None and minor is not None else None
            devices.append(
                {
                    "device_id": device_id,
                    "name": properties.get("name"),
                    "compute_capability": compute_capability,
                    "total_global_memory_bytes": properties.get("totalGlobalMem"),
                    "multiprocessor_count": properties.get("multiProcessorCount"),
                }
            )
        environment["devices"] = devices

        if device_count < 1:
            raise RuntimeError("No CUDA device is available.")

        probe = cp.asarray([0.0, 1.0, 2.0, 3.0], dtype=cp.float64)
        squared_sum = cp.sum(probe * probe)
        cp.cuda.Stream.null.synchronize()
        observed = float(np.asarray(cp.asnumpy(squared_sum)))
        if observed != 14.0:
            raise RuntimeError(f"Unexpected CUDA smoke-test result: {observed!r}")
    except Exception as exc:
        environment["cuda_smoke_test"] = {
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        return environment

    environment["cuda_usable"] = True
    environment["cuda_smoke_test"] = {
        "status": "success",
        "error_type": None,
        "error_message": None,
    }
    return environment


def _max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return the maximum absolute difference between two public arrays."""
    if actual.size == 0 and expected.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def run_cupy_parity_matrix(
    *,
    models: Iterable[str],
    block_sizes: Iterable[int],
    atol: float,
) -> Sequence[Dict[str, Any]]:
    """Compare dense NumPy and blockwise CuPy across public model contracts.

    Parameters
    ----------
    models : iterable of str
        Public model aliases to validate.
    block_sizes : iterable of int
        Positive block sizes used for CuPy-backed execution.
    atol : float
        Maximum accepted absolute difference for every public result array.

    Returns
    -------
    sequence of dict
        One JSON-safe validation row per model/configuration/block-size case.
    """
    rows = []
    for model in models:
        normalized_model = str(model).strip().lower()
        if normalized_model not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported validation model: {model!r}")

        for config_index, config in enumerate(MODEL_CONFIGS[normalized_model]):
            dense = compute_approximations(
                VALIDATION_X,
                VALIDATION_Y,
                model=normalized_model,
                engine="dense",
                backend="numpy",
                **dict(config),
            )
            for block_size in block_sizes:
                gpu = compute_approximations(
                    VALIDATION_X,
                    VALIDATION_Y,
                    model=normalized_model,
                    engine="blockwise",
                    backend="cupy",
                    block_size=int(block_size),
                    **dict(config),
                )
                differences = {
                    "lower": _max_abs_diff(gpu.lower, dense.lower),
                    "upper": _max_abs_diff(gpu.upper, dense.upper),
                    "boundary": _max_abs_diff(gpu.boundary, dense.boundary),
                    "positive_region": _max_abs_diff(gpu.positive_region, dense.positive_region),
                }
                expected_accumulators = EXPECTED_GPU_ACCUMULATORS[normalized_model]
                metadata_valid = (
                    gpu.backend == "cupy"
                    and gpu.used_blockwise is True
                    and gpu.used_gpu_similarity_blocks is True
                    and gpu.used_gpu_approximation_accumulators is expected_accumulators
                )
                numerical_valid = all(value <= atol for value in differences.values())
                rows.append(
                    {
                        "status": "success" if numerical_valid and metadata_valid else "failed",
                        "model": normalized_model,
                        "config_index": config_index,
                        "config": dict(config),
                        "block_size": int(block_size),
                        "absolute_tolerance": float(atol),
                        "max_abs_differences": differences,
                        "numerical_parity": numerical_valid,
                        "metadata_contract_valid": metadata_valid,
                        "resolved_backend": gpu.backend,
                        "used_gpu_similarity_blocks": gpu.used_gpu_similarity_blocks,
                        "used_gpu_approximation_accumulators": gpu.used_gpu_approximation_accumulators,
                        "expected_gpu_approximation_accumulators": expected_accumulators,
                        "public_output_type": "numpy.ndarray",
                        "public_output_dtypes": {
                            "lower": str(gpu.lower.dtype),
                            "upper": str(gpu.upper.dtype),
                            "boundary": str(gpu.boundary.dtype),
                            "positive_region": str(gpu.positive_region.dtype),
                        },
                    }
                )
    return rows


def build_cuda_validation_report(
    *,
    models: Sequence[str],
    block_sizes: Sequence[int],
    atol: float,
    cupy_module: Any = None,
) -> Dict[str, Any]:
    """Build a complete archiveable CUDA validation report.

    Parameters
    ----------
    models : sequence of str
        Public model aliases to validate.
    block_sizes : sequence of int
        Positive block sizes for parity validation.
    atol : float
        Maximum accepted absolute difference.
    cupy_module : module or None, default=None
        Optional injected CuPy-compatible module used by tests.

    Returns
    -------
    dict
        Environment metadata, claim boundaries, and numerical validation rows.
    """
    environment = collect_cuda_environment(cupy_module=cupy_module)
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_type": "frsutils_real_cuda_numerical_parity",
        "status": "unavailable",
        "environment": environment,
        "settings": {
            "models": [str(model).strip().lower() for model in models],
            "block_sizes": [int(value) for value in block_sizes],
            "absolute_tolerance": float(atol),
            "n_samples": int(VALIDATION_X.shape[0]),
            "n_features": int(VALIDATION_X.shape[1]),
        },
        "claim_boundaries": {
            "public_outputs_are_numpy": True,
            "itfrs_gpu_similarity_blocks": True,
            "itfrs_gpu_approximation_accumulators": True,
            "vqrs_gpu_similarity_blocks": True,
            "vqrs_gpu_approximation_accumulators": True,
            "owafrs_gpu_similarity_blocks": True,
            "owafrs_gpu_approximation_accumulators": False,
            "full_gpu_native_pipeline_claimed": False,
            "performance_speedup_claimed": False,
        },
        "parity_cases": [],
        "summary": {
            "case_count": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "maximum_observed_absolute_difference": None,
        },
    }

    if not environment["cuda_usable"]:
        return report

    try:
        rows = list(run_cupy_parity_matrix(models=models, block_sizes=block_sizes, atol=atol))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        return report

    observed_differences = [
        float(value)
        for row in rows
        for value in row["max_abs_differences"].values()
    ]
    successful_cases = sum(row["status"] == "success" for row in rows)
    failed_cases = len(rows) - successful_cases
    report["parity_cases"] = rows
    report["summary"] = {
        "case_count": len(rows),
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "maximum_observed_absolute_difference": max(observed_differences, default=0.0),
    }
    report["status"] = "success" if failed_cases == 0 else "failed"
    return report


def write_json_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a CUDA validation report as deterministic indented JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parse_csv_values(value: str) -> Sequence[str]:
    """Parse non-empty comma-separated CLI values."""
    values = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_positive_ints(value: str) -> Sequence[int]:
    """Parse comma-separated positive integers for CLI block sizes."""
    try:
        values = [int(item) for item in _parse_csv_values(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Block sizes must be integers.") from exc
    if any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Block sizes must be positive.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for CUDA validation capture."""
    parser = argparse.ArgumentParser(
        description="Capture an archiveable FRsutils real-CUDA validation report."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("cuda_validation_report.json"),
        help="JSON output path, default: cuda_validation_report.json.",
    )
    parser.add_argument(
        "--models",
        type=_parse_csv_values,
        default=list(MODEL_CONFIGS),
        help="Comma-separated models, default: itfrs,vqrs,owafrs.",
    )
    parser.add_argument(
        "--block-sizes",
        type=_parse_positive_ints,
        default=[1, 3, 16],
        help="Comma-separated positive block sizes, default: 1,3,16.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Maximum accepted absolute difference, default: 1e-12.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero status when a usable CUDA device is unavailable.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CUDA validation capture and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.atol < 0.0:
        parser.error("--atol must be non-negative.")

    report = build_cuda_validation_report(
        models=args.models,
        block_sizes=args.block_sizes,
        atol=float(args.atol),
    )
    write_json_report(report, args.output_json)
    print(f"CUDA validation status: {report['status']}")
    print(f"JSON report: {args.output_json}")

    if report["status"] == "success":
        return 0
    if report["status"] == "unavailable" and not args.require_cuda:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised by CLI invocation.
    raise SystemExit(main())
