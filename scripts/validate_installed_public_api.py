# SPDX-License-Identifier: BSD-3-Clause
"""Validate the installed FRsutils public API in a release-like environment.

The validator exercises dense and blockwise computations, scorer fitted-state
contracts, default logging silence, and optional read-only installation checks.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any

import numpy as np
from sklearn.base import clone


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RESULT_ARRAY_NAMES = (
    "lower",
    "upper",
    "boundary",
    "signed_boundary",
    "positive_region",
)
_FITTED_ATTRIBUTE_NAMES = (
    "result_",
    "positive_region_",
    "lower_",
    "upper_",
    "signed_boundary_",
    "boundary_",
    "n_samples_in_",
)

_VALIDATION_X = np.asarray(
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
_VALIDATION_Y = np.asarray(
    ["cold", "cold", "cold", "warm", "warm", "warm", "hot", "hot"],
    dtype=object,
)
_MODEL_CONFIGS: dict[str, Mapping[str, Any]] = {
    "itfrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    },
    "vqrs": {
        "lb_fuzzy_quantifier_name": "quadratic",
        "lb_fuzzy_quantifier_alpha": 0.2,
        "lb_fuzzy_quantifier_beta": 1.0,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.0,
        "ub_fuzzy_quantifier_beta": 0.6,
    },
    "owafrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "harmonic",
    },
}


def _prepare_import_path(*, require_installed: bool) -> None:
    """Expose the source checkout only for non-installed validation runs."""
    repository_root = str(_REPOSITORY_ROOT)
    if not require_installed and repository_root not in sys.path:
        sys.path.insert(0, repository_root)


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is located inside ``parent``."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _has_write_permission_bits(path: Path) -> bool:
    """Return whether a path has any POSIX write-permission bit set."""
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return bool(path.stat().st_mode & write_bits)


def _find_writable_path(root: Path) -> Path | None:
    """Return the first path with write bits below ``root``, if any."""
    candidates = [root, *sorted(root.rglob("*"))]
    for candidate in candidates:
        if _has_write_permission_bits(candidate):
            return candidate
    return None


def _snapshot_directory(root: Path) -> tuple[str, ...]:
    """Return a deterministic relative-path snapshot of a directory tree."""
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
        )
    )


def _call_silently(function: Callable[[], Any], *, description: str) -> Any:
    """Run a callable and fail when it writes to standard output or error."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = function()

    observed_stdout = stdout.getvalue()
    observed_stderr = stderr.getvalue()
    if observed_stdout or observed_stderr:
        raise RuntimeError(
            f"{description} produced unexpected output: "
            f"stdout={observed_stdout!r}, stderr={observed_stderr!r}."
        )
    return result


def _validate_result(result: Any, *, model: str, engine: str) -> None:
    """Validate one public approximation result and its metadata contract."""
    expected_shape = (_VALIDATION_Y.size,)
    for attribute_name in _RESULT_ARRAY_NAMES:
        values = getattr(result, attribute_name)
        if not isinstance(values, np.ndarray):
            raise TypeError(
                f"{model}/{engine} result.{attribute_name} is not a NumPy array."
            )
        if values.shape != expected_shape:
            raise ValueError(
                f"{model}/{engine} result.{attribute_name} has shape "
                f"{values.shape!r}, expected {expected_shape!r}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{model}/{engine} result.{attribute_name} contains non-finite values."
            )

    np.testing.assert_allclose(
        result.signed_boundary,
        result.upper - result.lower,
        rtol=0.0,
        atol=1e-12,
    )
    if result.signed_boundary is not result.boundary:
        raise RuntimeError("signed_boundary must alias the legacy boundary array.")
    if result.model != model:
        raise RuntimeError(
            f"Result model metadata is {result.model!r}, expected {model!r}."
        )
    if result.engine != engine:
        raise RuntimeError(
            f"Result engine metadata is {result.engine!r}, expected {engine!r}."
        )
    if result.used_blockwise is not (engine == "blockwise"):
        raise RuntimeError(
            f"Result used_blockwise metadata is inconsistent for {model}/{engine}."
        )


def _validate_model_case(
    compute_approximations: Callable[..., Any],
    *,
    model: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate dense/blockwise parity for one public fuzzy-rough model."""
    common_options = {
        "model": model,
        "similarity": "linear",
        **dict(config),
    }
    dense = _call_silently(
        lambda: compute_approximations(
            _VALIDATION_X,
            _VALIDATION_Y,
            engine="dense",
            **common_options,
        ),
        description=f"{model} dense computation",
    )
    blockwise = _call_silently(
        lambda: compute_approximations(
            _VALIDATION_X,
            _VALIDATION_Y,
            engine="blockwise",
            block_size=3,
            backend="numpy",
            **common_options,
        ),
        description=f"{model} blockwise computation",
    )

    _validate_result(dense, model=model, engine="dense")
    _validate_result(blockwise, model=model, engine="blockwise")

    maximum_absolute_difference = 0.0
    for attribute_name in _RESULT_ARRAY_NAMES:
        dense_values = getattr(dense, attribute_name)
        blockwise_values = getattr(blockwise, attribute_name)
        np.testing.assert_allclose(
            blockwise_values,
            dense_values,
            rtol=0.0,
            atol=1e-12,
        )
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            float(np.max(np.abs(blockwise_values - dense_values))),
        )

    return {
        "model": model,
        "dense_blockwise_parity": True,
        "maximum_absolute_difference": maximum_absolute_difference,
        "n_samples": int(_VALIDATION_Y.size),
        "block_size": 3,
    }


def _validate_scorer(
    scorer_class: type,
    *,
    model: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate fitted-state and cloning contracts for one public scorer."""
    scorer = scorer_class(
        model=model,
        similarity="linear",
        engine="blockwise",
        block_size=3,
        **dict(config),
    )
    _call_silently(
        lambda: scorer.fit(_VALIDATION_X, _VALIDATION_Y),
        description=f"{model} scorer fit",
    )

    missing_attributes = [
        name for name in _FITTED_ATTRIBUTE_NAMES if not hasattr(scorer, name)
    ]
    if missing_attributes:
        raise RuntimeError(
            f"{model} scorer is missing fitted attributes: {missing_attributes}."
        )
    if scorer.boundary_ is not scorer.signed_boundary_:
        raise RuntimeError("boundary_ must alias signed_boundary_.")
    np.testing.assert_allclose(
        scorer.signed_boundary_,
        scorer.upper_ - scorer.lower_,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scorer.positive_region_,
        scorer.result_.positive_region,
        rtol=0.0,
        atol=1e-12,
    )
    if scorer.n_samples_in_ != _VALIDATION_Y.size:
        raise RuntimeError("n_samples_in_ does not match the fitted label count.")

    cloned = clone(scorer)
    leaked_attributes = [
        name for name in _FITTED_ATTRIBUTE_NAMES if hasattr(cloned, name)
    ]
    if leaked_attributes:
        raise RuntimeError(
            f"sklearn.clone copied fitted attributes: {leaked_attributes}."
        )

    return {
        "model": model,
        "fitted_attribute_contract": True,
        "clone_is_unfitted": True,
        "boundary_alias_identity": True,
    }


def _distribution_version(distribution_name: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_environment(package_module: Any) -> dict[str, Any]:
    """Collect environment metadata for the validation report."""
    package_file = Path(package_module.__file__).resolve()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scikit_learn_version": _distribution_version("scikit-learn"),
        "frsutils_version": _distribution_version("frsutils"),
        "frsutils_file": str(package_file),
        "working_directory": str(Path.cwd().resolve()),
        "cupy_installed": _distribution_version("cupy-cuda12x") is not None,
    }


def validate_public_api(
    *,
    require_installed: bool = False,
    require_package_read_only: bool = False,
    require_cwd_read_only: bool = False,
) -> dict[str, Any]:
    """Validate the public API and return a machine-readable report.

    Parameters
    ----------
    require_installed : bool, default=False
        Reject a ``frsutils`` import resolved from this source checkout.
    require_package_read_only : bool, default=False
        Require every path in the imported package tree to have no POSIX write
        permission bits.
    require_cwd_read_only : bool, default=False
        Require the current working directory to have no POSIX write permission
        bits.

    Returns
    -------
    report : dict
        Environment metadata and successful validation results.
    """
    current_directory = Path.cwd().resolve()
    before_snapshot = _snapshot_directory(current_directory)
    _prepare_import_path(require_installed=require_installed)

    package_module = _call_silently(
        lambda: importlib.import_module("frsutils"),
        description="frsutils import",
    )
    package_file = Path(package_module.__file__).resolve()
    package_directory = package_file.parent

    if require_installed and _path_is_within(package_file, _REPOSITORY_ROOT):
        raise RuntimeError(
            "frsutils resolved from the source checkout instead of an installed "
            f"distribution: {package_file}."
        )
    if require_package_read_only:
        writable_path = _find_writable_path(package_directory)
        if writable_path is not None:
            raise RuntimeError(
                "The imported package tree is not read-only; write bits remain on "
                f"{writable_path}."
            )
    if require_cwd_read_only and _has_write_permission_bits(current_directory):
        raise RuntimeError(
            f"The working directory is not read-only: {current_directory}."
        )

    compute_approximations = getattr(package_module, "compute_approximations")
    scorer_class = getattr(package_module, "FuzzyRoughPositiveRegionScorer")

    model_results = [
        _validate_model_case(
            compute_approximations,
            model=model,
            config=config,
        )
        for model, config in _MODEL_CONFIGS.items()
    ]
    scorer_results = [
        _validate_scorer(
            scorer_class,
            model=model,
            config=config,
        )
        for model, config in _MODEL_CONFIGS.items()
    ]

    after_snapshot = _snapshot_directory(current_directory)
    if after_snapshot != before_snapshot:
        before_set = set(before_snapshot)
        after_set = set(after_snapshot)
        raise RuntimeError(
            "Public API validation changed the working directory contents: "
            f"added={sorted(after_set - before_set)}, "
            f"removed={sorted(before_set - after_set)}."
        )

    return {
        "status": "success",
        "environment": _collect_environment(package_module),
        "requirements": {
            "installed_distribution_required": require_installed,
            "package_read_only_required": require_package_read_only,
            "cwd_read_only_required": require_cwd_read_only,
        },
        "checks": {
            "import_silent": True,
            "public_calls_silent": True,
            "working_directory_unchanged": True,
            "dense_blockwise_models_validated": len(model_results),
            "scorer_models_validated": len(scorer_results),
        },
        "model_results": model_results,
        "scorer_results": scorer_results,
    }


def _write_report(report: Mapping[str, Any], output_path: Path | None) -> None:
    """Write a JSON report to a file or standard output."""
    payload = json.dumps(report, indent=2, sort_keys=True)
    if output_path is None:
        print(payload)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + os.linesep, encoding="utf-8")
    print(f"Public API validation report written to {output_path}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for installed API validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable validation report.",
    )
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="Reject imports resolved from the source checkout.",
    )
    parser.add_argument(
        "--require-package-read-only",
        action="store_true",
        help="Require the imported package tree to have no write bits.",
    )
    parser.add_argument(
        "--require-cwd-read-only",
        action="store_true",
        help="Require the current working directory to have no write bits.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run installed public API validation and return a process exit status."""
    args = parse_args(argv)
    try:
        report = validate_public_api(
            require_installed=args.require_installed,
            require_package_read_only=args.require_package_read_only,
            require_cwd_read_only=args.require_cwd_read_only,
        )
    except Exception as exc:
        report = {
            "status": "failed",
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        _write_report(report, args.output_json)
        return 1

    _write_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
