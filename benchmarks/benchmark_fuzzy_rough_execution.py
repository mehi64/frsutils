# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6 benchmark suite for dense/blockwise/GPU fuzzy-rough execution.

This module supports performance measurement and is not part of the stable public API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Allow running this file directly from a source checkout without installing the
# package first. Installed-package usage still works because the import below is
# tried after the repository root has been added only when needed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from FRsutils.api import compute_approximations  # noqa: E402


@dataclass(frozen=True)
class BenchmarkScenario:
    """Immutable benchmark scenario definition.
    
    Parameters
    ----------
    name : object
        Public scenario name written to reports.
    engine : object
        FRsutils approximation engine alias.
    backend : object
        FRsutils backend alias.
    uses_block_size : object
        True if the scenario should be expanded over block sizes.
    """

    name: str
    engine: str
    backend: str
    uses_block_size: bool


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Flat benchmark row written to JSON/CSV reports.
    
    Parameters
    ----------
    status : object
        "success", "skipped", or "failed".
    scenario : object
        Scenario alias.
    model : object
        Fuzzy-rough model alias.
    n_samples : object
        Number of benchmark samples.
    n_features : object
        Number of benchmark features.
    block_size : object
        Block size for blockwise scenarios; None for dense.
    backend : object
        Requested backend alias.
    resolved_backend : object
        Backend reported by FRsutils result metadata.
    engine : object
        Execution engine reported by FRsutils result metadata.
    repeats : object
        Number of timed repetitions attempted.
    median_runtime_seconds : object
        Median runtime across successful repetitions.
    mean_runtime_seconds : object
        Mean runtime across successful repetitions.
    min_runtime_seconds : object
        Minimum runtime across successful repetitions.
    max_runtime_seconds : object
        Maximum runtime across successful repetitions.
    python_peak_memory_bytes : object
        Peak Python allocator memory observed via tracemalloc.
    max_abs_error_lower : object
        Max absolute lower-approximation error vs dense reference.
    max_abs_error_upper : object
        Max absolute upper-approximation error vs dense reference.
    max_abs_error_positive_region : object
        Max absolute positive-region error vs dense reference.
    used_blockwise : object
        Result metadata flag.
    used_gpu_similarity_blocks : object
        Result metadata flag.
    used_gpu_approximation_accumulators : object
        Result metadata flag.
    error_type : object
        Exception type for skipped/failed cases.
    error_message : object
        Exception message for skipped/failed cases.
    """

    status: str
    scenario: str
    model: str
    n_samples: int
    n_features: int
    block_size: Optional[int]
    backend: str
    resolved_backend: Optional[str]
    engine: Optional[str]
    repeats: int
    median_runtime_seconds: Optional[float]
    mean_runtime_seconds: Optional[float]
    min_runtime_seconds: Optional[float]
    max_runtime_seconds: Optional[float]
    python_peak_memory_bytes: Optional[int]
    max_abs_error_lower: Optional[float]
    max_abs_error_upper: Optional[float]
    max_abs_error_positive_region: Optional[float]
    used_blockwise: Optional[bool]
    used_gpu_similarity_blocks: Optional[bool]
    used_gpu_approximation_accumulators: Optional[bool]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


SCENARIOS: Dict[str, BenchmarkScenario] = {
    "dense_numpy": BenchmarkScenario(
        name="dense_numpy",
        engine="dense",
        backend="numpy",
        uses_block_size=False,
    ),
    "blockwise_numpy": BenchmarkScenario(
        name="blockwise_numpy",
        engine="blockwise",
        backend="numpy",
        uses_block_size=True,
    ),
    "blockwise_cupy": BenchmarkScenario(
        name="blockwise_cupy",
        engine="blockwise",
        backend="cupy",
        uses_block_size=True,
    ),
}


class BenchmarkConfigurationError(ValueError):
    """Raised when benchmark CLI/configuration values are invalid."""


def parse_csv_values(value: str) -> List[str]:
    """Parse a comma-separated string into non-empty normalized tokens.
        
        Parameters
        ----------
        value : str
            Comma-separated input string.
        
        Returns
        -------
        List[str]
            Lower-case tokens.
        
        Raises
        ------
        BenchmarkConfigurationError
            If no tokens are provided.
        
    """
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise BenchmarkConfigurationError("At least one value must be provided.")
    return tokens


def parse_int_values(value: str) -> List[int]:
    """Parse a comma-separated string into positive integers.
        
        Parameters
        ----------
        value : str
            Comma-separated integer string.
        
        Returns
        -------
        List[int]
            Positive integer values.
        
        Raises
        ------
        BenchmarkConfigurationError
            If a value is invalid or non-positive.
        
    """
    values: List[int] = []
    for token in parse_csv_values(value):
        try:
            parsed = int(token)
        except ValueError as exc:
            raise BenchmarkConfigurationError(f"Invalid integer value: {token!r}") from exc
        if parsed < 1:
            raise BenchmarkConfigurationError("Integer benchmark values must be positive.")
        values.append(parsed)
    return values


def make_synthetic_dataset(
    *,
    n_samples: int,
    n_features: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a deterministic numeric benchmark dataset.
        
        The generator creates two mildly separated classes while keeping all features
        in a comparable numeric range. It intentionally avoids scikit-learn so the
        benchmark script has no extra runtime dependency beyond FRsutils itself.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_features : int
            Number of numeric features.
        random_state : int
            RNG seed.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple `(X, y)`.
        
    """
    if n_samples < 2:
        raise BenchmarkConfigurationError("n_samples must be at least 2.")
    if n_features < 1:
        raise BenchmarkConfigurationError("n_features must be positive.")

    rng = np.random.default_rng(random_state)
    y = np.arange(n_samples, dtype=int) % 2
    rng.shuffle(y)

    X = rng.normal(loc=0.0, scale=0.18, size=(n_samples, n_features))
    X += y[:, None] * 0.35
    X -= X.min(axis=0, keepdims=True)
    denom = X.max(axis=0, keepdims=True)
    denom[denom == 0.0] = 1.0
    X = X / denom
    return X.astype(float, copy=False), y


def _scenario_is_optional_cupy_failure(exc: BaseException, scenario: BenchmarkScenario) -> bool:
    """Return True when an exception should be treated as an optional CuPy skip.
        
        Parameters
        ----------
        exc : BaseException
            Exception raised while running a benchmark case.
        scenario : BenchmarkScenario
            Scenario being executed.
        
        Returns
        -------
        bool
            True when the row should be reported as skipped.
        
    """
    if scenario.backend != "cupy":
        return False
    message = str(exc).lower()
    return isinstance(exc, ImportError) or "cupy" in message or "cuda" in message or "gpu" in message


def _max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Compute max absolute numerical error as a plain float.
        
        Parameters
        ----------
        candidate : np.ndarray
            Candidate output array.
        reference : np.ndarray
            Dense reference output array.
        
        Returns
        -------
        float
            Maximum absolute error.
        
    """
    return float(np.max(np.abs(np.asarray(candidate) - np.asarray(reference))))


def _runtime_summary(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Summarize successful runtime measurements.
        
        Parameters
        ----------
        values : Sequence[float]
            Runtime values in seconds.
        
        Returns
        -------
        Tuple[float, float, float, float]
            `(median, mean, min, max)`.
        
    """
    return (
        float(statistics.median(values)),
        float(statistics.mean(values)),
        float(min(values)),
        float(max(values)),
    )


def _execute_once(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    scenario: BenchmarkScenario,
    block_size: Optional[int],
):
    """Execute one FRsutils approximation case through the public API.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label vector.
        model : str
            Fuzzy-rough model alias.
        scenario : BenchmarkScenario
            Execution scenario.
        block_size : Optional[int]
            Optional block size for blockwise scenarios.
        
        Returns
        -------
        object
            FuzzyRoughApproximationResult.
        
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "similarity": "linear",
        "engine": scenario.engine,
        "backend": scenario.backend,
    }
    if scenario.uses_block_size:
        kwargs["block_size"] = int(block_size or 1)
    return compute_approximations(X, y, **kwargs)


def benchmark_one_case(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    scenario: BenchmarkScenario,
    block_size: Optional[int],
    repeats: int,
    dense_reference: Optional[Any],
) -> BenchmarkCaseResult:
    """Time and validate one benchmark case.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label vector.
        model : str
            Fuzzy-rough model alias.
        scenario : BenchmarkScenario
            Execution scenario.
        block_size : Optional[int]
            Optional block size for blockwise scenarios.
        repeats : int
            Number of timed repetitions.
        dense_reference : Optional[Any]
            Dense reference result for numerical equivalence.
        
        Returns
        -------
        BenchmarkCaseResult
            BenchmarkCaseResult row.
        
    """
    if repeats < 1:
        raise BenchmarkConfigurationError("repeats must be positive.")

    runtimes: List[float] = []
    peak_memory = 0
    last_result = None

    try:
        for _ in range(repeats):
            tracemalloc.start()
            start_time = time.perf_counter()
            last_result = _execute_once(X, y, model=model, scenario=scenario, block_size=block_size)
            elapsed = time.perf_counter() - start_time
            _, current_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            runtimes.append(elapsed)
            peak_memory = max(peak_memory, int(current_peak))
    except Exception as exc:  # pragma: no cover - CuPy/CUDA failures are environment-specific
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        status = "skipped" if _scenario_is_optional_cupy_failure(exc, scenario) else "failed"
        return BenchmarkCaseResult(
            status=status,
            scenario=scenario.name,
            model=model,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            block_size=block_size if scenario.uses_block_size else None,
            backend=scenario.backend,
            resolved_backend=None,
            engine=scenario.engine,
            repeats=repeats,
            median_runtime_seconds=None,
            mean_runtime_seconds=None,
            min_runtime_seconds=None,
            max_runtime_seconds=None,
            python_peak_memory_bytes=None,
            max_abs_error_lower=None,
            max_abs_error_upper=None,
            max_abs_error_positive_region=None,
            used_blockwise=None,
            used_gpu_similarity_blocks=None,
            used_gpu_approximation_accumulators=None,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )

    median_time, mean_time, min_time, max_time = _runtime_summary(runtimes)

    if dense_reference is None:
        max_error_lower = 0.0
        max_error_upper = 0.0
        max_error_positive_region = 0.0
    else:
        max_error_lower = _max_abs_error(last_result.lower, dense_reference.lower)
        max_error_upper = _max_abs_error(last_result.upper, dense_reference.upper)
        max_error_positive_region = _max_abs_error(last_result.positive_region, dense_reference.positive_region)

    return BenchmarkCaseResult(
        status="success",
        scenario=scenario.name,
        model=model,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
        block_size=block_size if scenario.uses_block_size else None,
        backend=scenario.backend,
        resolved_backend=str(getattr(last_result, "backend", scenario.backend)),
        engine=str(getattr(last_result, "engine", scenario.engine)),
        repeats=repeats,
        median_runtime_seconds=median_time,
        mean_runtime_seconds=mean_time,
        min_runtime_seconds=min_time,
        max_runtime_seconds=max_time,
        python_peak_memory_bytes=peak_memory,
        max_abs_error_lower=max_error_lower,
        max_abs_error_upper=max_error_upper,
        max_abs_error_positive_region=max_error_positive_region,
        used_blockwise=bool(getattr(last_result, "used_blockwise", False)),
        used_gpu_similarity_blocks=bool(getattr(last_result, "used_gpu_similarity_blocks", False)),
        used_gpu_approximation_accumulators=bool(
            getattr(last_result, "used_gpu_approximation_accumulators", False)
        ),
    )


def run_benchmark_suite(
    *,
    models: Sequence[str],
    sample_sizes: Sequence[int],
    n_features: int,
    block_sizes: Sequence[int],
    scenario_names: Sequence[str],
    repeats: int,
    random_state: int,
) -> Dict[str, Any]:
    """Execute the Phase 6 benchmark matrix.
        
        Parameters
        ----------
        models : Sequence[str]
            Fuzzy-rough model aliases.
        sample_sizes : Sequence[int]
            Sample sizes to generate.
        n_features : int
            Number of synthetic features.
        block_sizes : Sequence[int]
            Block sizes for blockwise scenarios.
        scenario_names : Sequence[str]
            Scenario aliases from SCENARIOS.
        repeats : int
            Timed repetitions per case.
        random_state : int
            Base RNG seed.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with metadata and benchmark row dictionaries.
        
    """
    normalized_models = [model.strip().lower() for model in models]
    normalized_scenarios = [name.strip().lower() for name in scenario_names]

    unknown_scenarios = [name for name in normalized_scenarios if name not in SCENARIOS]
    if unknown_scenarios:
        raise BenchmarkConfigurationError(f"Unknown scenario(s): {', '.join(unknown_scenarios)}")

    rows: List[BenchmarkCaseResult] = []
    for size_index, n_samples in enumerate(sample_sizes):
        X, y = make_synthetic_dataset(
            n_samples=int(n_samples),
            n_features=int(n_features),
            random_state=int(random_state) + size_index,
        )
        for model in normalized_models:
            dense_reference = _execute_once(
                X,
                y,
                model=model,
                scenario=SCENARIOS["dense_numpy"],
                block_size=None,
            )
            for scenario_name in normalized_scenarios:
                scenario = SCENARIOS[scenario_name]
                candidate_block_sizes: Iterable[Optional[int]] = block_sizes if scenario.uses_block_size else [None]
                for block_size in candidate_block_sizes:
                    rows.append(
                        benchmark_one_case(
                            X,
                            y,
                            model=model,
                            scenario=scenario,
                            block_size=int(block_size) if block_size is not None else None,
                            repeats=int(repeats),
                            dense_reference=dense_reference,
                        )
                    )

    return {
        "metadata": {
            "benchmark_phase": "phase_6",
            "description": "Dense/blockwise/GPU execution benchmark suite for FRsutils fuzzy-rough approximations.",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "models": normalized_models,
            "sample_sizes": [int(value) for value in sample_sizes],
            "n_features": int(n_features),
            "block_sizes": [int(value) for value in block_sizes],
            "scenarios": normalized_scenarios,
            "repeats": int(repeats),
            "random_state": int(random_state),
            "memory_note": "python_peak_memory_bytes is measured with tracemalloc and does not fully capture NumPy/CuPy native allocator memory.",
        },
        "results": [asdict(row) for row in rows],
    }


def _make_json_safe(value: Any) -> Any:
    """Convert non-standard numeric values into JSON-safe objects.
        
        Parameters
        ----------
        value : Any
            Candidate value.
        
        Returns
        -------
        Any
            JSON-safe value.
        
    """
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    return value


def write_json_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a benchmark report to JSON.
        
        Parameters
        ----------
        report : Mapping[str, Any]
            Benchmark report dictionary.
        output_path : Path
            Target JSON path.
        
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(_make_json_safe(dict(report)), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def write_csv_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write benchmark result rows to CSV.
        
        Parameters
        ----------
        report : Mapping[str, Any]
            Benchmark report dictionary.
        output_path : Path
            Target CSV path.
        
    """
    rows = list(report.get("results", []))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark script.
        
        Returns
        -------
        argparse.ArgumentParser
            Configured argparse parser.
        
    """
    parser = argparse.ArgumentParser(description="Run FRsutils Phase 6 execution benchmarks.")
    parser.add_argument("--models", default="itfrs,vqrs,owafrs", help="Comma-separated model aliases.")
    parser.add_argument("--sample-sizes", default="128,256", help="Comma-separated positive sample sizes.")
    parser.add_argument("--n-features", type=int, default=8, help="Number of synthetic numeric features.")
    parser.add_argument("--block-sizes", default="64,128", help="Comma-separated positive block sizes.")
    parser.add_argument(
        "--scenarios",
        default="dense_numpy,blockwise_numpy,blockwise_cupy",
        help=f"Comma-separated scenarios. Available: {', '.join(sorted(SCENARIOS))}.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per case.")
    parser.add_argument("--random-state", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.
        
        Parameters
        ----------
        argv : Optional[Sequence[str]]
            Optional argument list; defaults to sys.argv.
        
        Returns
        -------
        int
            Process exit code.
        
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = run_benchmark_suite(
            models=parse_csv_values(args.models),
            sample_sizes=parse_int_values(args.sample_sizes),
            n_features=int(args.n_features),
            block_sizes=parse_int_values(args.block_sizes),
            scenario_names=parse_csv_values(args.scenarios),
            repeats=int(args.repeats),
            random_state=int(args.random_state),
        )
    except BenchmarkConfigurationError as exc:
        parser.error(str(exc))
        return 2

    if args.output_json is not None:
        write_json_report(report, args.output_json)
    if args.output_csv is not None:
        write_csv_report(report, args.output_csv)

    success_count = sum(1 for row in report["results"] if row["status"] == "success")
    skipped_count = sum(1 for row in report["results"] if row["status"] == "skipped")
    failed_count = sum(1 for row in report["results"] if row["status"] == "failed")

    print(
        "FRsutils Phase 6 benchmark complete: "
        f"{success_count} success, {skipped_count} skipped, {failed_count} failed."
    )
    if args.output_json is not None:
        print(f"JSON report: {args.output_json}")
    if args.output_csv is not None:
        print(f"CSV report: {args.output_csv}")

    return 1 if failed_count else 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
