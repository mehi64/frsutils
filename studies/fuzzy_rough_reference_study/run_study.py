# SPDX-License-Identifier: BSD-3-Clause
"""Run the reproducible frsutils reference study on public datasets.

The study uses only the public ``frsutils`` approximation API for scientific
computations. It records real-dataset approximation summaries, dense/blockwise
numerical agreement, runtimes, per-sample scores, environment metadata, a
synthetic execution benchmark, and deterministic figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_fuzzy_rough_execution import (  # noqa: E402
    run_benchmark_suite,
    write_csv_report,
    write_json_report,
)
from frsutils import compute_approximations  # noqa: E402


@dataclass(frozen=True)
class StudyDataset:
    """Prepared public dataset and binary study target.

    Parameters
    ----------
    name : str
        Stable dataset/task identifier written to outputs.
    task : str
        Target-construction rule from the study configuration.
    X : ndarray of shape (n_samples, n_features)
        Min-max scaled feature matrix.
    y : ndarray of shape (n_samples,)
        Binary target consumed by frsutils.
    original_y : ndarray of shape (n_samples,)
        Original scikit-learn labels retained for traceability.
    source : str
        Human-readable source description.
    """

    name: str
    task: str
    X: np.ndarray
    y: np.ndarray
    original_y: np.ndarray
    source: str


class StudyConfigurationError(ValueError):
    """Raised when the reference-study configuration is invalid."""


def load_study_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a reference-study JSON configuration.

    Parameters
    ----------
    path : Path
        JSON configuration path.

    Returns
    -------
    config : dict
        Parsed configuration mapping.

    Raises
    ------
    StudyConfigurationError
        If required top-level sections are missing or malformed.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise StudyConfigurationError("Study configuration must be a JSON object.")
    for key in ("datasets", "models", "similarity", "benchmark"):
        if key not in config:
            raise StudyConfigurationError(f"Missing required configuration key: {key}")
    if not config["datasets"] or not config["models"]:
        raise StudyConfigurationError("At least one dataset and one model are required.")
    return config


def _load_raw_dataset(loader_name: str):
    """Load a bundled scikit-learn dataset by stable alias."""
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }
    try:
        return loaders[loader_name]()
    except KeyError as exc:
        raise StudyConfigurationError(f"Unknown dataset loader: {loader_name}") from exc


def load_study_dataset(spec: Mapping[str, Any]) -> StudyDataset:
    """Prepare one configured real dataset and binary target.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Dataset specification from ``study_config.json``.

    Returns
    -------
    dataset : StudyDataset
        Scaled feature matrix and traceable binary labels.

    Notes
    -----
    All features are transformed independently to ``[0, 1]`` using
    ``MinMaxScaler``. One-vs-rest and pairwise tasks retain original labels in
    the machine-readable per-sample output.
    """
    name = str(spec["name"])
    loader_name = str(spec["loader"])
    task = str(spec["task"])
    raw = _load_raw_dataset(loader_name)
    X = np.asarray(raw.data, dtype=float)
    original_y = np.asarray(raw.target)

    if task == "original_binary":
        y = original_y.astype(int, copy=True)
    elif task == "one_vs_rest":
        positive_class = spec["positive_class"]
        y = (original_y == positive_class).astype(int)
    elif task == "pairwise_binary":
        included = np.asarray(spec["include_classes"])
        positive_class = spec["positive_class"]
        mask = np.isin(original_y, included)
        X = X[mask]
        original_y = original_y[mask]
        y = (original_y == positive_class).astype(int)
    else:
        raise StudyConfigurationError(f"Unknown task type: {task}")

    if np.unique(y).size != 2:
        raise StudyConfigurationError(f"Dataset task {name!r} did not produce two classes.")

    X_scaled = MinMaxScaler().fit_transform(X).astype(float, copy=False)
    return StudyDataset(
        name=name,
        task=task,
        X=X_scaled,
        y=np.asarray(y, dtype=int),
        original_y=np.asarray(original_y),
        source=f"scikit-learn:{loader_name}",
    )


def _model_call_kwargs(config: Mapping[str, Any], model_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build public API keyword arguments for one model configuration."""
    similarity = config["similarity"]
    kwargs: dict[str, Any] = {
        "model": str(model_spec["name"]),
        "similarity": str(similarity["name"]),
        "backend": "numpy",
    }
    if "sigma" in similarity:
        kwargs["similarity_sigma"] = float(similarity["sigma"])
    kwargs.update(dict(model_spec.get("parameters", {})))
    return kwargs


def _timed_approximation(
    dataset: StudyDataset,
    *,
    kwargs: Mapping[str, Any],
    engine: str,
    block_size: int,
    repeats: int,
) -> tuple[Any, list[float]]:
    """Execute one public approximation case repeatedly and return timings."""
    if repeats < 1:
        raise StudyConfigurationError("repeats must be positive.")

    result = None
    runtimes: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compute_approximations(
            dataset.X,
            dataset.y,
            engine=engine,
            block_size=block_size,
            **dict(kwargs),
        )
        runtimes.append(float(time.perf_counter() - start))
    return result, runtimes


def _array_summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Return stable scalar summary fields for one approximation array."""
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _max_abs_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return maximum absolute difference between two result arrays."""
    return float(np.max(np.abs(np.asarray(candidate) - np.asarray(reference))))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write deterministic CSV rows with an explicit field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _runtime_rows(
    dataset: StudyDataset,
    model: str,
    engine: str,
    block_size: Optional[int],
    runtimes: Sequence[float],
) -> list[dict[str, Any]]:
    """Build repeat-level runtime rows for one study case."""
    return [
        {
            "dataset": dataset.name,
            "model": model,
            "engine": engine,
            "block_size": block_size if engine == "blockwise" else None,
            "repeat": index,
            "runtime_seconds": runtime,
        }
        for index, runtime in enumerate(runtimes, start=1)
    ]


def _validate_public_result(result: Any, *, dataset_name: str, model: str, engine: str) -> None:
    """Validate finite public result arrays and expected score ranges."""
    for field in ("lower", "upper", "boundary", "positive_region"):
        values = np.asarray(getattr(result, field), dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite {field} values for {dataset_name}/{model}/{engine}.")
    for field in ("lower", "upper", "positive_region"):
        values = np.asarray(getattr(result, field), dtype=float)
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise RuntimeError(f"Out-of-range {field} values for {dataset_name}/{model}/{engine}.")


def run_reference_analysis(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Run real-dataset approximation analysis and write machine-readable outputs.

    Parameters
    ----------
    config : Mapping[str, Any]
        Validated study configuration.
    output_dir : Path
        Directory that receives CSV outputs.

    Returns
    -------
    tables : dict
        In-memory table rows used by figure and manifest generation.

    Raises
    ------
    RuntimeError
        If dense and blockwise outputs differ beyond the configured tolerance.
    """
    repeats = int(config.get("repeats", 3))
    block_size = int(config.get("block_size", 64))
    atol = float(config.get("equivalence_atol", 1e-12))
    top_k = int(config.get("top_k_boundary_gap", 10))

    summary_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    for dataset_spec in config["datasets"]:
        dataset = load_study_dataset(dataset_spec)
        class_counts = np.bincount(dataset.y, minlength=2)

        for model_spec in config["models"]:
            kwargs = _model_call_kwargs(config, model_spec)
            model = str(model_spec["name"])
            dense, dense_times = _timed_approximation(
                dataset,
                kwargs=kwargs,
                engine="dense",
                block_size=block_size,
                repeats=repeats,
            )
            blockwise, blockwise_times = _timed_approximation(
                dataset,
                kwargs=kwargs,
                engine="blockwise",
                block_size=block_size,
                repeats=repeats,
            )
            _validate_public_result(dense, dataset_name=dataset.name, model=model, engine="dense")
            _validate_public_result(blockwise, dataset_name=dataset.name, model=model, engine="blockwise")

            errors = {
                "max_abs_error_lower": _max_abs_error(blockwise.lower, dense.lower),
                "max_abs_error_upper": _max_abs_error(blockwise.upper, dense.upper),
                "max_abs_error_boundary": _max_abs_error(blockwise.boundary, dense.boundary),
                "max_abs_error_positive_region": _max_abs_error(
                    blockwise.positive_region,
                    dense.positive_region,
                ),
            }
            passed = all(value <= atol for value in errors.values())
            equivalence_rows.append(
                {
                    "dataset": dataset.name,
                    "model": model,
                    "n_samples": int(dataset.X.shape[0]),
                    "n_features": int(dataset.X.shape[1]),
                    "block_size": block_size,
                    "atol": atol,
                    **errors,
                    "passed": passed,
                }
            )
            if not passed:
                raise RuntimeError(
                    f"Dense/blockwise equivalence failed for {dataset.name}/{model}: {errors}"
                )

            summary_row: dict[str, Any] = {
                "dataset": dataset.name,
                "task": dataset.task,
                "source": dataset.source,
                "n_samples": int(dataset.X.shape[0]),
                "n_features": int(dataset.X.shape[1]),
                "class_0_count": int(class_counts[0]),
                "class_1_count": int(class_counts[1]),
                "model": model,
                "similarity": kwargs["similarity"],
                "similarity_sigma": kwargs.get("similarity_sigma"),
                "dense_median_runtime_seconds": float(statistics.median(dense_times)),
                "blockwise_median_runtime_seconds": float(statistics.median(blockwise_times)),
                "block_size": block_size,
            }
            for field in ("lower", "upper", "boundary", "positive_region"):
                summary_row.update(_array_summary(field, getattr(dense, field)))
            summary_rows.append(summary_row)

            runtime_rows.extend(_runtime_rows(dataset, model, "dense", None, dense_times))
            runtime_rows.extend(
                _runtime_rows(dataset, model, "blockwise", block_size, blockwise_times)
            )

            for sample_index in range(dataset.X.shape[0]):
                sample_rows.append(
                    {
                        "dataset": dataset.name,
                        "sample_index": sample_index,
                        "original_label": dataset.original_y[sample_index],
                        "binary_label": int(dataset.y[sample_index]),
                        "model": model,
                        "lower": float(dense.lower[sample_index]),
                        "upper": float(dense.upper[sample_index]),
                        "boundary": float(dense.boundary[sample_index]),
                        "positive_region": float(dense.positive_region[sample_index]),
                    }
                )

            ranked_indices = np.argsort(np.abs(np.asarray(dense.boundary, dtype=float)))[::-1][:top_k]
            for rank, sample_index in enumerate(ranked_indices, start=1):
                boundary_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": model,
                        "rank": rank,
                        "sample_index": int(sample_index),
                        "original_label": dataset.original_y[sample_index],
                        "binary_label": int(dataset.y[sample_index]),
                        "lower": float(dense.lower[sample_index]),
                        "upper": float(dense.upper[sample_index]),
                        "boundary": float(dense.boundary[sample_index]),
                        "absolute_boundary_gap": float(abs(dense.boundary[sample_index])),
                        "positive_region": float(dense.positive_region[sample_index]),
                    }
                )

    summary_fields = list(summary_rows[0].keys())
    equivalence_fields = list(equivalence_rows[0].keys())
    runtime_fields = list(runtime_rows[0].keys())
    sample_fields = list(sample_rows[0].keys())
    boundary_fields = list(boundary_rows[0].keys())

    _write_csv(output_dir / "approximation_summary.csv", summary_rows, summary_fields)
    _write_csv(output_dir / "dense_blockwise_equivalence.csv", equivalence_rows, equivalence_fields)
    _write_csv(output_dir / "runtime_results.csv", runtime_rows, runtime_fields)
    _write_csv(output_dir / "sample_scores.csv", sample_rows, sample_fields)
    _write_csv(output_dir / "highest_boundary_gap_samples.csv", boundary_rows, boundary_fields)

    return {
        "summary": summary_rows,
        "equivalence": equivalence_rows,
        "runtime": runtime_rows,
        "samples": sample_rows,
        "boundary": boundary_rows,
    }


def run_execution_benchmark(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Run the existing public-API execution benchmark with the study profile."""
    benchmark = config["benchmark"]
    report = run_benchmark_suite(
        models=[str(item["name"]) for item in config["models"]],
        sample_sizes=[int(value) for value in benchmark["sample_sizes"]],
        n_features=int(benchmark["n_features"]),
        block_sizes=[int(value) for value in benchmark["block_sizes"]],
        scenario_names=[str(value) for value in benchmark["scenarios"]],
        repeats=int(benchmark["repeats"]),
        random_state=int(benchmark["random_seed"]),
    )
    report["metadata"]["reference_study_profile"] = "fuzzy_rough_reference_study"
    write_json_report(report, output_dir / "benchmark_results.json")
    write_csv_report(report, output_dir / "benchmark_results.csv")

    failed = [row for row in report["results"] if row["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Execution benchmark contains failed cases: {failed}")
    return report


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    """Return best-effort Git revision and worktree-state metadata."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"git_commit": revision, "git_worktree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_worktree_dirty": None}


def _package_version(distribution: str, module: Optional[Any] = None) -> Optional[str]:
    """Return an installed distribution version with an optional module fallback."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return getattr(module, "__version__", None) if module is not None else None


def capture_environment(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Write execution environment metadata and a minimal version lock file."""
    import sklearn

    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_schema_version": config.get("schema_version"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "matplotlib_version": _package_version("matplotlib"),
        "frsutils_version": _package_version("frsutils"),
        "random_seed": int(config.get("random_seed", 42)),
        "backend": "numpy",
        "command": (
            "python studies/fuzzy_rough_reference_study/run_study.py "
            "--config studies/fuzzy_rough_reference_study/study_config.json"
        ),
        **_git_metadata(_REPO_ROOT),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lock_lines = [
        f"numpy=={np.__version__}",
        f"scikit-learn=={sklearn.__version__}",
    ]
    if environment["matplotlib_version"]:
        lock_lines.append(f"matplotlib=={environment['matplotlib_version']}")
    if environment["frsutils_version"]:
        lock_lines.append(f"frsutils=={environment['frsutils_version']}")
    (output_dir / "requirements-study.txt").write_text(
        "\n".join(lock_lines) + "\n",
        encoding="utf-8",
    )
    return environment


def create_figures(tables: Mapping[str, Sequence[Mapping[str, Any]]], *, output_dir: Path) -> None:
    """Create deterministic runtime and positive-region figures."""
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = list(tables["summary"])
    labels = [f"{row['dataset']}\n{row['model']}" for row in summary_rows]
    dense = [float(row["dense_median_runtime_seconds"]) for row in summary_rows]
    blockwise = [float(row["blockwise_median_runtime_seconds"]) for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, dense, width, label="dense NumPy")
    ax.bar(x + width / 2, blockwise, width, label="blockwise NumPy")
    ax.set_ylabel("Median runtime (seconds)")
    ax.set_title("frsutils reference-study execution time")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "runtime_by_dataset_and_model.png", dpi=160)
    plt.close(fig)

    sample_rows = list(tables["samples"])
    grouped: MutableMapping[str, list[float]] = {}
    for row in sample_rows:
        key = f"{row['dataset']}\n{row['model']}"
        grouped.setdefault(key, []).append(float(row["positive_region"]))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(list(grouped.values()), tick_labels=list(grouped.keys()), showfliers=False)
    ax.set_ylabel("Positive-region score")
    ax.set_title("Per-sample positive-region distributions")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figure_dir / "positive_region_distributions.png", dpi=160)
    plt.close(fig)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(config: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Write a checksum manifest for all generated study artifacts."""
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "study_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "study": "fuzzy_rough_reference_study",
        "schema_version": config.get("schema_version"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": files,
    }
    (output_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_reference_study(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    include_benchmark: bool = True,
    include_figures: bool = True,
) -> dict[str, Any]:
    """Run the complete reproducible reference-study workflow.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed study configuration.
    output_dir : Path
        Directory for generated artifacts.
    include_benchmark : bool, default=True
        Whether to run the synthetic dense/blockwise benchmark profile.
    include_figures : bool, default=True
        Whether to render PNG figures with Matplotlib.

    Returns
    -------
    report : dict
        In-memory summaries of generated tables, environment, benchmark, and
        manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tables = run_reference_analysis(config, output_dir=output_dir)
    benchmark_report = (
        run_execution_benchmark(config, output_dir=output_dir) if include_benchmark else None
    )
    environment = capture_environment(config, output_dir=output_dir)
    if include_figures:
        create_figures(tables, output_dir=output_dir)
    manifest = write_manifest(config, output_dir=output_dir)
    return {
        "tables": tables,
        "benchmark": benchmark_report,
        "environment": environment,
        "manifest": manifest,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the reference study."""
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the frsutils fuzzy-rough reference study.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_dir / "study_config.json",
        help="Study JSON configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir / "results",
        help="Directory for generated study artifacts.",
    )
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the synthetic benchmark profile.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip Matplotlib figure generation.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the study from command-line arguments and return an exit code."""
    args = build_arg_parser().parse_args(argv)
    config = load_study_config(args.config)
    report = run_reference_study(
        config,
        output_dir=args.output_dir,
        include_benchmark=not args.skip_benchmark,
        include_figures=not args.skip_figures,
    )
    equivalence = report["tables"]["equivalence"]
    print(
        "frsutils reference study complete: "
        f"{len(equivalence)} dense/blockwise comparisons passed; "
        f"artifacts written to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
