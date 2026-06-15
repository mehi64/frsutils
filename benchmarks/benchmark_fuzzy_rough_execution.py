# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark suite for dense/blockwise/GPU fuzzy-rough execution.

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

from frsutils import compute_approximations  # noqa: E402


@dataclass(frozen=True)
class BenchmarkScenario:
    """Immutable benchmark scenario definition.
    
    Parameters
    ----------
    name : object
        Public scenario name written to reports.
    engine : object
        frsutils approximation engine alias.
    backend : object
        frsutils backend alias.
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
        Backend reported by frsutils result metadata.
    engine : object
        Execution engine reported by frsutils result metadata.
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
    max_abs_error_boundary: Optional[float]
    max_abs_error_positive_region: Optional[float]
    used_blockwise: Optional[bool]
    used_gpu_similarity_blocks: Optional[bool]
    used_gpu_approximation_accumulators: Optional[bool]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class PairedBackendComparisonResult:
    """Comparison row for two blockwise backend executions.

    Parameters
    ----------
    status : str
        "success", "skipped", or "failed" for the paired comparison.
    model : str
        Fuzzy-rough model alias.
    n_samples : int
        Number of analyzed samples.
    n_features : int
        Number of analyzed features.
    block_size : int
        Block size used for both blockwise runs.
    reference_backend : str
        Backend used as the numerical reference, usually ``"numpy"``.
    candidate_backend : str
        Backend compared against the reference, usually ``"cupy"``.
    reference_median_runtime_seconds : Optional[float]
        Median runtime for the reference backend.
    candidate_median_runtime_seconds : Optional[float]
        Median runtime for the candidate backend.
    speedup_reference_over_candidate : Optional[float]
        ``reference_runtime / candidate_runtime`` when both runtimes are valid.
    max_abs_diff_lower : Optional[float]
        Max absolute lower-approximation difference between backends.
    max_abs_diff_upper : Optional[float]
        Max absolute upper-approximation difference between backends.
    max_abs_diff_boundary : Optional[float]
        Max absolute boundary-region difference between backends.
    max_abs_diff_positive_region : Optional[float]
        Max absolute positive-region difference between backends.
    error_type : Optional[str]
        Error type for skipped/failed comparisons.
    error_message : Optional[str]
        Error message for skipped/failed comparisons.
    """

    status: str
    model: str
    n_samples: int
    n_features: int
    block_size: int
    reference_backend: str
    candidate_backend: str
    reference_scenario: str
    candidate_scenario: str
    reference_status: str
    candidate_status: str
    reference_median_runtime_seconds: Optional[float]
    candidate_median_runtime_seconds: Optional[float]
    reference_mean_runtime_seconds: Optional[float]
    candidate_mean_runtime_seconds: Optional[float]
    speedup_reference_over_candidate: Optional[float]
    max_abs_diff_lower: Optional[float]
    max_abs_diff_upper: Optional[float]
    max_abs_diff_boundary: Optional[float]
    max_abs_diff_positive_region: Optional[float]
    reference_used_gpu_similarity_blocks: Optional[bool]
    candidate_used_gpu_similarity_blocks: Optional[bool]
    reference_used_gpu_approximation_accumulators: Optional[bool]
    candidate_used_gpu_approximation_accumulators: Optional[bool]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkDataset:
    """Dataset used by the benchmark suite.

    Parameters
    ----------
    X : np.ndarray
        Numeric feature matrix.
    y : np.ndarray
        One-dimensional label vector.
    name : str
        Short dataset identifier written to reports.
    source : str
        Dataset source description written to report metadata.
    """

    X: np.ndarray
    y: np.ndarray
    name: str
    source: str


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



def parse_backend_pair(value: str) -> Tuple[str, str]:
    """Parse a comma-separated backend pair.

    Parameters
    ----------
    value : str
        Comma-separated backend aliases.

    Returns
    -------
    Tuple[str, str]
        Reference and candidate backend aliases.

    Raises
    ------
    BenchmarkConfigurationError
        If exactly two backend aliases are not provided.
    """
    tokens = parse_csv_values(value)
    if len(tokens) != 2:
        raise BenchmarkConfigurationError("Exactly two comparison backends are required.")
    return tokens[0], tokens[1]


def make_synthetic_dataset(
    *,
    n_samples: int,
    n_features: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a deterministic numeric benchmark dataset.
        
        The generator creates two mildly separated classes while keeping all features
        in a comparable numeric range. It intentionally avoids scikit-learn so the
        benchmark script has no extra runtime dependency beyond frsutils itself.
        
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


def _read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV file as a header row and data rows.

    Parameters
    ----------
    path : Path
        CSV file path.

    Returns
    -------
    header, rows : tuple[list[str], list[list[str]]]
        Header names and non-empty data rows.
    """
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise BenchmarkConfigurationError(f"CSV file is empty: {path}") from exc
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise BenchmarkConfigurationError(f"CSV file has no data rows: {path}")
    return header, rows


def _resolve_target_column(header: Sequence[str], target_column: str) -> int:
    """Resolve a target column name or integer index.

    Parameters
    ----------
    header : Sequence[str]
        CSV header row.
    target_column : str
        Column name or zero-based integer index.

    Returns
    -------
    int
        Zero-based target column index.
    """
    if not target_column:
        raise BenchmarkConfigurationError("--target-column is required with --input-csv.")

    stripped = str(target_column).strip()
    try:
        index = int(stripped)
    except ValueError:
        normalized_header = [name.strip() for name in header]
        if stripped not in normalized_header:
            raise BenchmarkConfigurationError(
                f"Target column {stripped!r} was not found in CSV header."
            )
        index = normalized_header.index(stripped)

    if index < 0 or index >= len(header):
        raise BenchmarkConfigurationError(
            f"Target column index {index} is out of range for {len(header)} columns."
        )
    return index


def load_csv_dataset(path: Path, *, target_column: str, name: Optional[str] = None) -> BenchmarkDataset:
    """Load a numeric feature matrix and labels from a CSV file.

    Parameters
    ----------
    path : Path
        CSV file with a header row.
    target_column : str
        Target column name or zero-based column index.
    name : Optional[str]
        Optional report dataset name.

    Returns
    -------
    BenchmarkDataset
        Loaded benchmark dataset.
    """
    if not path.exists():
        raise BenchmarkConfigurationError(f"CSV file does not exist: {path}")

    header, rows = _read_csv_rows(path)
    target_index = _resolve_target_column(header, target_column)

    expected_width = len(header)
    feature_rows: List[List[float]] = []
    labels: List[str] = []
    for row_number, row in enumerate(rows, start=2):
        if len(row) != expected_width:
            raise BenchmarkConfigurationError(
                f"CSV row {row_number} has {len(row)} columns; expected {expected_width}."
            )
        feature_values: List[float] = []
        for column_index, value in enumerate(row):
            if column_index == target_index:
                labels.append(value.strip())
                continue
            try:
                feature_values.append(float(value))
            except ValueError as exc:
                column_name = header[column_index].strip() or str(column_index)
                raise BenchmarkConfigurationError(
                    f"Non-numeric feature value in row {row_number}, column {column_name!r}."
                ) from exc
        feature_rows.append(feature_values)

    X = np.asarray(feature_rows, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] < 1:
        raise BenchmarkConfigurationError("CSV input must contain at least one feature column.")
    return BenchmarkDataset(
        X=X,
        y=y,
        name=name or path.stem,
        source=f"csv:{path}",
    )


def load_npy_dataset(
    *,
    x_path: Path,
    y_path: Path,
    name: Optional[str] = None,
) -> BenchmarkDataset:
    """Load a feature matrix and labels from two NumPy ``.npy`` files.

    Parameters
    ----------
    x_path : Path
        Path to a two-dimensional numeric feature matrix.
    y_path : Path
        Path to a one-dimensional label vector.
    name : Optional[str]
        Optional report dataset name.

    Returns
    -------
    BenchmarkDataset
        Loaded benchmark dataset.
    """
    if not x_path.exists():
        raise BenchmarkConfigurationError(f"X .npy file does not exist: {x_path}")
    if not y_path.exists():
        raise BenchmarkConfigurationError(f"y .npy file does not exist: {y_path}")

    X = np.asarray(np.load(x_path), dtype=float)
    y = np.asarray(np.load(y_path))
    if X.ndim != 2:
        raise BenchmarkConfigurationError("--input-npy-x must contain a 2D feature matrix.")
    if y.ndim != 1:
        raise BenchmarkConfigurationError("--input-npy-y must contain a 1D label vector.")
    if X.shape[0] != y.shape[0]:
        raise BenchmarkConfigurationError("X and y .npy files must have the same sample count.")
    return BenchmarkDataset(
        X=X,
        y=y,
        name=name or x_path.stem,
        source=f"npy:X={x_path};y={y_path}",
    )


def _validate_loaded_dataset(dataset: BenchmarkDataset) -> BenchmarkDataset:
    """Validate a loaded benchmark dataset and normalize array shapes.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Candidate dataset.

    Returns
    -------
    BenchmarkDataset
        Validated dataset with NumPy arrays.
    """
    X = np.asarray(dataset.X, dtype=float)
    y = np.asarray(dataset.y)
    if X.ndim != 2:
        raise BenchmarkConfigurationError("Benchmark X must be a 2D feature matrix.")
    if y.ndim != 1:
        raise BenchmarkConfigurationError("Benchmark y must be a 1D label vector.")
    if X.shape[0] != y.shape[0]:
        raise BenchmarkConfigurationError("Benchmark X and y sample counts must match.")
    if X.shape[0] < 2:
        raise BenchmarkConfigurationError("Benchmark dataset must contain at least 2 samples.")
    if X.shape[1] < 1:
        raise BenchmarkConfigurationError("Benchmark dataset must contain at least 1 feature.")
    return BenchmarkDataset(X=X, y=y, name=dataset.name, source=dataset.source)


def _slice_dataset(dataset: BenchmarkDataset, n_samples: int) -> BenchmarkDataset:
    """Return a prefix sample of a loaded benchmark dataset.

    Parameters
    ----------
    dataset : BenchmarkDataset
        Source dataset.
    n_samples : int
        Prefix size.

    Returns
    -------
    BenchmarkDataset
        Dataset prefix used for one benchmark size.
    """
    if n_samples < 2:
        raise BenchmarkConfigurationError("sample sizes must be at least 2 for loaded datasets.")
    if n_samples > dataset.X.shape[0]:
        raise BenchmarkConfigurationError(
            f"Requested sample size {n_samples} exceeds loaded dataset size {dataset.X.shape[0]}."
        )
    return BenchmarkDataset(
        X=dataset.X[:n_samples],
        y=dataset.y[:n_samples],
        name=dataset.name,
        source=dataset.source,
    )


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
    """Execute one frsutils approximation case through the public API.
        
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


def _benchmark_one_case_with_result(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    scenario: BenchmarkScenario,
    block_size: Optional[int],
    repeats: int,
    dense_reference: Optional[Any],
) -> Tuple[BenchmarkCaseResult, Optional[Any]]:
    """Time and validate one benchmark case and keep the last public result.
        
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
            max_abs_error_boundary=None,
            max_abs_error_positive_region=None,
            used_blockwise=None,
            used_gpu_similarity_blocks=None,
            used_gpu_approximation_accumulators=None,
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        ), None

    median_time, mean_time, min_time, max_time = _runtime_summary(runtimes)

    if dense_reference is None:
        max_error_lower = None
        max_error_upper = None
        max_error_boundary = None
        max_error_positive_region = None
    else:
        max_error_lower = _max_abs_error(last_result.lower, dense_reference.lower)
        max_error_upper = _max_abs_error(last_result.upper, dense_reference.upper)
        max_error_boundary = _max_abs_error(last_result.boundary, dense_reference.boundary)
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
        max_abs_error_boundary=max_error_boundary,
        max_abs_error_positive_region=max_error_positive_region,
        used_blockwise=bool(getattr(last_result, "used_blockwise", False)),
        used_gpu_similarity_blocks=bool(getattr(last_result, "used_gpu_similarity_blocks", False)),
        used_gpu_approximation_accumulators=bool(
            getattr(last_result, "used_gpu_approximation_accumulators", False)
        ),
    ), last_result


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
        Benchmark result row without the internal public result object.
    """
    row, _ = _benchmark_one_case_with_result(
        X,
        y,
        model=model,
        scenario=scenario,
        block_size=block_size,
        repeats=repeats,
        dense_reference=dense_reference,
    )
    return row


def _build_blockwise_scenario_for_backend(backend: str) -> BenchmarkScenario:
    """Build a blockwise scenario for a backend alias.

    Parameters
    ----------
    backend : str
        Backend alias, such as ``"numpy"`` or ``"cupy"``.

    Returns
    -------
    BenchmarkScenario
        Blockwise scenario using the requested backend.
    """
    normalized_backend = str(backend).strip().lower()
    if not normalized_backend:
        raise BenchmarkConfigurationError("Backend aliases must be non-empty.")
    scenario_name = f"blockwise_{normalized_backend}"
    return SCENARIOS.get(
        scenario_name,
        BenchmarkScenario(
            name=scenario_name,
            engine="blockwise",
            backend=normalized_backend,
            uses_block_size=True,
        ),
    )


def _safe_speedup(reference_runtime: Optional[float], candidate_runtime: Optional[float]) -> Optional[float]:
    """Return ``reference_runtime / candidate_runtime`` when valid.

    Parameters
    ----------
    reference_runtime : Optional[float]
        Reference backend runtime.
    candidate_runtime : Optional[float]
        Candidate backend runtime.

    Returns
    -------
    Optional[float]
        Speedup ratio, or None when unavailable.
    """
    if reference_runtime is None or candidate_runtime is None or candidate_runtime <= 0.0:
        return None
    return float(reference_runtime / candidate_runtime)


def compare_blockwise_backend_pair(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    block_size: int,
    repeats: int,
    reference_backend: str = "numpy",
    candidate_backend: str = "cupy",
) -> PairedBackendComparisonResult:
    """Compare two blockwise public-API backends on the same dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    model : str
        Fuzzy-rough model alias.
    block_size : int
        Block size used by both blockwise executions.
    repeats : int
        Timed repetitions per backend.
    reference_backend : str, default="numpy"
        Backend used as the numerical reference.
    candidate_backend : str, default="cupy"
        Backend compared against the reference.

    Returns
    -------
    PairedBackendComparisonResult
        Runtime and numerical-difference comparison row.
    """
    reference_scenario = _build_blockwise_scenario_for_backend(reference_backend)
    candidate_scenario = _build_blockwise_scenario_for_backend(candidate_backend)

    reference_row, reference_result = _benchmark_one_case_with_result(
        X,
        y,
        model=model,
        scenario=reference_scenario,
        block_size=int(block_size),
        repeats=int(repeats),
        dense_reference=None,
    )
    candidate_row, candidate_result = _benchmark_one_case_with_result(
        X,
        y,
        model=model,
        scenario=candidate_scenario,
        block_size=int(block_size),
        repeats=int(repeats),
        dense_reference=None,
    )

    base_kwargs: Dict[str, Any] = {
        "model": model,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "block_size": int(block_size),
        "reference_backend": reference_scenario.backend,
        "candidate_backend": candidate_scenario.backend,
        "reference_scenario": reference_scenario.name,
        "candidate_scenario": candidate_scenario.name,
        "reference_status": reference_row.status,
        "candidate_status": candidate_row.status,
        "reference_median_runtime_seconds": reference_row.median_runtime_seconds,
        "candidate_median_runtime_seconds": candidate_row.median_runtime_seconds,
        "reference_mean_runtime_seconds": reference_row.mean_runtime_seconds,
        "candidate_mean_runtime_seconds": candidate_row.mean_runtime_seconds,
        "speedup_reference_over_candidate": _safe_speedup(
            reference_row.median_runtime_seconds,
            candidate_row.median_runtime_seconds,
        ),
        "reference_used_gpu_similarity_blocks": reference_row.used_gpu_similarity_blocks,
        "candidate_used_gpu_similarity_blocks": candidate_row.used_gpu_similarity_blocks,
        "reference_used_gpu_approximation_accumulators": reference_row.used_gpu_approximation_accumulators,
        "candidate_used_gpu_approximation_accumulators": candidate_row.used_gpu_approximation_accumulators,
    }

    if reference_row.status != "success" or candidate_row.status != "success":
        status = "skipped" if "skipped" in {reference_row.status, candidate_row.status} else "failed"
        error_type = candidate_row.error_type or reference_row.error_type
        error_message = candidate_row.error_message or reference_row.error_message
        return PairedBackendComparisonResult(
            status=status,
            max_abs_diff_lower=None,
            max_abs_diff_upper=None,
            max_abs_diff_boundary=None,
            max_abs_diff_positive_region=None,
            error_type=error_type,
            error_message=error_message,
            **base_kwargs,
        )

    return PairedBackendComparisonResult(
        status="success",
        max_abs_diff_lower=_max_abs_error(candidate_result.lower, reference_result.lower),
        max_abs_diff_upper=_max_abs_error(candidate_result.upper, reference_result.upper),
        max_abs_diff_boundary=_max_abs_error(candidate_result.boundary, reference_result.boundary),
        max_abs_diff_positive_region=_max_abs_error(
            candidate_result.positive_region,
            reference_result.positive_region,
        ),
        **base_kwargs,
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
    dataset: Optional[BenchmarkDataset] = None,
    skip_dense_reference: bool = False,
    compare_blockwise_backends: bool = False,
    comparison_backends: Tuple[str, str] = ("numpy", "cupy"),
) -> Dict[str, Any]:
    """Execute the benchmark matrix.
        
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
        dataset : Optional[BenchmarkDataset]
            Optional loaded dataset. If omitted, synthetic datasets are generated.
        skip_dense_reference : bool
            If True, do not compute dense reference rows before candidate cases.
        compare_blockwise_backends : bool
            If True, add paired blockwise backend comparisons to the report.
        comparison_backends : Tuple[str, str]
            Pair of backend aliases used for paired comparison.
        
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

    if len(comparison_backends) != 2:
        raise BenchmarkConfigurationError("comparison_backends must contain exactly two backend aliases.")

    loaded_dataset = _validate_loaded_dataset(dataset) if dataset is not None else None

    rows: List[BenchmarkCaseResult] = []
    comparisons: List[PairedBackendComparisonResult] = []
    observed_feature_counts: List[int] = []
    for size_index, n_samples in enumerate(sample_sizes):
        if loaded_dataset is None:
            X, y = make_synthetic_dataset(
                n_samples=int(n_samples),
                n_features=int(n_features),
                random_state=int(random_state) + size_index,
            )
            dataset_name = "synthetic"
            dataset_source = "synthetic"
        else:
            dataset_prefix = _slice_dataset(loaded_dataset, int(n_samples))
            X, y = dataset_prefix.X, dataset_prefix.y
            dataset_name = dataset_prefix.name
            dataset_source = dataset_prefix.source
        observed_feature_counts.append(int(X.shape[1]))
        for model in normalized_models:
            dense_reference = None
            if not skip_dense_reference:
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
            if compare_blockwise_backends:
                for block_size in block_sizes:
                    comparisons.append(
                        compare_blockwise_backend_pair(
                            X,
                            y,
                            model=model,
                            block_size=int(block_size),
                            repeats=int(repeats),
                            reference_backend=str(comparison_backends[0]),
                            candidate_backend=str(comparison_backends[1]),
                        )
                    )

    return {
        "metadata": {
            "benchmark_suite": "public_api_execution",
            "description": "Dense/blockwise/GPU execution benchmark suite for frsutils fuzzy-rough approximations.",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "models": normalized_models,
            "sample_sizes": [int(value) for value in sample_sizes],
            "n_features": int(observed_feature_counts[0] if observed_feature_counts else n_features),
            "dataset_name": dataset_name if rows else (loaded_dataset.name if loaded_dataset else "synthetic"),
            "dataset_source": dataset_source if rows else (loaded_dataset.source if loaded_dataset else "synthetic"),
            "block_sizes": [int(value) for value in block_sizes],
            "scenarios": normalized_scenarios,
            "repeats": int(repeats),
            "random_state": int(random_state),
            "dense_reference_enabled": not skip_dense_reference,
            "paired_backend_comparison_enabled": bool(compare_blockwise_backends),
            "comparison_backends": [str(value).strip().lower() for value in comparison_backends],
            "memory_note": "python_peak_memory_bytes is measured with tracemalloc and does not fully capture NumPy/CuPy native allocator memory.",
        },
        "results": [asdict(row) for row in rows],
        "comparisons": [asdict(row) for row in comparisons],
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



def write_comparison_csv_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write paired backend comparison rows to CSV.

    Parameters
    ----------
    report : Mapping[str, Any]
        Benchmark report dictionary.
    output_path : Path
        Target CSV path for comparison rows.
    """
    rows = list(report.get("comparisons", []))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_optional_float(value: Optional[float], *, precision: int = 6) -> str:
    """Format an optional floating-point value for console output.

    Parameters
    ----------
    value : Optional[float]
        Optional numeric value.
    precision : int
        Number of digits after the decimal point.

    Returns
    -------
    str
        Formatted value or ``"n/a"``.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}g}"


def print_comparison_summary(report: Mapping[str, Any]) -> None:
    """Print a compact runtime and numerical-difference comparison summary.

    Parameters
    ----------
    report : Mapping[str, Any]
        Benchmark report dictionary.
    """
    comparisons = list(report.get("comparisons", []))
    if not comparisons:
        return

    print("\nPaired blockwise backend comparison:")
    header = (
        "model",
        "n",
        "features",
        "block",
        "status",
        "ref_s",
        "cand_s",
        "speedup",
        "max_diff_pos",
    )
    print(" | ".join(header))
    print("-" * 88)
    for row in comparisons:
        print(
            " | ".join(
                [
                    str(row.get("model")),
                    str(row.get("n_samples")),
                    str(row.get("n_features")),
                    str(row.get("block_size")),
                    str(row.get("status")),
                    _format_optional_float(row.get("reference_median_runtime_seconds")),
                    _format_optional_float(row.get("candidate_median_runtime_seconds")),
                    _format_optional_float(row.get("speedup_reference_over_candidate")),
                    _format_optional_float(row.get("max_abs_diff_positive_region")),
                ]
            )
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark script.
        
        Returns
        -------
        argparse.ArgumentParser
            Configured argparse parser.
        
    """
    parser = argparse.ArgumentParser(description="Run frsutils execution benchmarks.")
    parser.add_argument("--models", default="itfrs,vqrs,owafrs", help="Comma-separated model aliases.")
    parser.add_argument(
        "--sample-sizes",
        default=None,
        help=(
            "Comma-separated positive sample sizes. Defaults to 128,256 for synthetic "
            "data and to the full dataset size for loaded CSV/NPY data."
        ),
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=None,
        help=(
            "Number of synthetic samples for a single generated benchmark dataset. "
            "Use --sample-sizes for multiple generated sizes."
        ),
    )
    parser.add_argument("--n-features", type=int, default=8, help="Number of synthetic numeric features.")
    parser.add_argument("--block-sizes", default="64,128", help="Comma-separated positive block sizes.")
    parser.add_argument(
        "--scenarios",
        default="dense_numpy,blockwise_numpy,blockwise_cupy",
        help=f"Comma-separated scenarios. Available: {', '.join(sorted(SCENARIOS))}.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions per case.")
    parser.add_argument("--random-state", type=int, default=42, help="Base RNG seed.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Optional CSV dataset path with a header row.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target column name or zero-based index for --input-csv.",
    )
    parser.add_argument(
        "--input-npy-x",
        type=Path,
        default=None,
        help="Optional .npy feature-matrix path.",
    )
    parser.add_argument(
        "--input-npy-y",
        type=Path,
        default=None,
        help="Optional .npy label-vector path.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name written to report metadata.",
    )
    parser.add_argument(
        "--skip-dense-reference",
        action="store_true",
        help=(
            "Skip dense NumPy reference computation. This is useful for large datasets "
            "where only blockwise CPU/GPU runtime comparison is needed. Numerical error "
            "fields are left empty when no dense reference is computed."
        ),
    )
    parser.add_argument(
        "--compare-blockwise-backends",
        action="store_true",
        help=(
            "Run paired blockwise backend comparisons on the same X/y, model, and "
            "block size. Defaults to comparing NumPy as reference against CuPy."
        ),
    )
    parser.add_argument(
        "--comparison-backends",
        default="numpy,cupy",
        help="Two comma-separated backend aliases for paired comparison, default: numpy,cupy.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional case-level CSV output path.")
    parser.add_argument(
        "--output-comparison-csv",
        type=Path,
        default=None,
        help="Optional paired backend comparison CSV output path.",
    )
    return parser


def _load_dataset_from_cli_args(args: argparse.Namespace) -> Optional[BenchmarkDataset]:
    """Load an optional benchmark dataset from CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    Optional[BenchmarkDataset]
        Loaded dataset, or None for synthetic benchmark data.
    """
    uses_csv = args.input_csv is not None
    uses_npy = args.input_npy_x is not None or args.input_npy_y is not None
    if uses_csv and uses_npy:
        raise BenchmarkConfigurationError("Use either --input-csv or --input-npy-x/--input-npy-y, not both.")
    if uses_csv:
        return load_csv_dataset(
            Path(args.input_csv),
            target_column=str(args.target_column or ""),
            name=args.dataset_name,
        )
    if uses_npy:
        if args.input_npy_x is None or args.input_npy_y is None:
            raise BenchmarkConfigurationError("Both --input-npy-x and --input-npy-y are required together.")
        return load_npy_dataset(
            x_path=Path(args.input_npy_x),
            y_path=Path(args.input_npy_y),
            name=args.dataset_name,
        )
    return None


def _sample_sizes_from_cli_args(
    args: argparse.Namespace,
    dataset: Optional[BenchmarkDataset],
) -> List[int]:
    """Resolve benchmark sample sizes from CLI arguments and dataset context.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    dataset : Optional[BenchmarkDataset]
        Optional loaded dataset.

    Returns
    -------
    List[int]
        Sample sizes for the benchmark matrix.
    """
    if args.sample_sizes and args.synthetic_samples is not None:
        raise BenchmarkConfigurationError("Use either --sample-sizes or --synthetic-samples, not both.")
    if args.sample_sizes:
        return parse_int_values(args.sample_sizes)
    if dataset is not None:
        return [int(dataset.X.shape[0])]
    if args.synthetic_samples is not None:
        if int(args.synthetic_samples) < 2:
            raise BenchmarkConfigurationError("--synthetic-samples must be at least 2.")
        return [int(args.synthetic_samples)]
    return [128, 256]


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
        dataset = _load_dataset_from_cli_args(args)
        report = run_benchmark_suite(
            models=parse_csv_values(args.models),
            sample_sizes=_sample_sizes_from_cli_args(args, dataset),
            n_features=int(args.n_features),
            block_sizes=parse_int_values(args.block_sizes),
            scenario_names=parse_csv_values(args.scenarios),
            repeats=int(args.repeats),
            random_state=int(args.random_state),
            dataset=dataset,
            skip_dense_reference=bool(args.skip_dense_reference),
            compare_blockwise_backends=bool(args.compare_blockwise_backends),
            comparison_backends=parse_backend_pair(args.comparison_backends),
        )
    except BenchmarkConfigurationError as exc:
        parser.error(str(exc))
        return 2

    if args.output_json is not None:
        write_json_report(report, args.output_json)
    if args.output_csv is not None:
        write_csv_report(report, args.output_csv)
    if args.output_comparison_csv is not None:
        write_comparison_csv_report(report, args.output_comparison_csv)

    success_count = sum(1 for row in report["results"] if row["status"] == "success")
    skipped_count = sum(1 for row in report["results"] if row["status"] == "skipped")
    failed_count = sum(1 for row in report["results"] if row["status"] == "failed")
    comparison_failed_count = sum(
        1 for row in report.get("comparisons", []) if row["status"] == "failed"
    )

    print(
        "frsutils benchmark complete: "
        f"{success_count} success, {skipped_count} skipped, {failed_count} failed."
    )
    if args.output_json is not None:
        print(f"JSON report: {args.output_json}")
    if args.output_csv is not None:
        print(f"CSV report: {args.output_csv}")
    if args.output_comparison_csv is not None:
        print(f"Comparison CSV report: {args.output_comparison_csv}")
    print_comparison_summary(report)

    return 1 if failed_count or comparison_failed_count else 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
