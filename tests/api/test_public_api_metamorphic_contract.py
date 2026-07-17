# SPDX-License-Identifier: BSD-3-Clause
"""Metamorphic and cross-engine contracts for public fuzzy-rough APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from frsutils import build_similarity_matrix, compute_approximations
from frsutils.api.results import FuzzyRoughApproximationResult


MODELS = ("itfrs", "vqrs", "owafrs")
ENGINES = ("dense", "blockwise")
RESULT_FIELDS = ("lower", "upper", "boundary", "positive_region")


def _build_dataset(n_samples: int, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic finite dataset with at least two represented classes."""
    rng = np.random.default_rng(1000 * n_samples + n_features)
    X = rng.random((n_samples, n_features), dtype=float)
    labels = np.arange(n_samples, dtype=int) % min(3, n_samples)
    return X, labels


def _compute(
    X: np.ndarray,
    y: Any,
    *,
    model: str,
    engine: str,
    block_size: int = 3,
    return_similarity_matrix: bool = False,
) -> FuzzyRoughApproximationResult:
    """Compute a result through one public engine using stable test settings."""
    return compute_approximations(
        X,
        y,
        model=model,
        similarity="linear",
        engine=engine,
        block_size=block_size,
        return_similarity_matrix=return_similarity_matrix,
    )


def _assert_result_arrays_close(
    actual: FuzzyRoughApproximationResult,
    expected: FuzzyRoughApproximationResult,
    *,
    expected_order: np.ndarray | None = None,
) -> None:
    """Assert equality of all public approximation arrays, optionally reordered."""
    for field in RESULT_FIELDS:
        expected_values = getattr(expected, field)
        if expected_order is not None:
            expected_values = expected_values[expected_order]
        np.testing.assert_allclose(
            getattr(actual, field),
            expected_values,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
def test_sample_permutation_equivariance(model: str, engine: str) -> None:
    """Reordering aligned samples reorders every approximation by the same permutation."""
    X, y = _build_dataset(8, 4)
    permutation = np.array([5, 0, 7, 3, 1, 6, 2, 4])

    baseline = _compute(
        X,
        y,
        model=model,
        engine=engine,
        return_similarity_matrix=True,
    )
    permuted = _compute(
        X[permutation],
        y[permutation],
        model=model,
        engine=engine,
        return_similarity_matrix=True,
    )

    _assert_result_arrays_close(
        permuted,
        baseline,
        expected_order=permutation,
    )
    np.testing.assert_allclose(
        permuted.similarity_matrix,
        baseline.similarity_matrix[np.ix_(permutation, permutation)],
        rtol=1e-12,
        atol=1e-12,
    )


LABEL_REPRESENTATIONS: tuple[tuple[str, Callable[[np.ndarray], Any]], ...] = (
    (
        "python-list-of-strings",
        lambda labels: [f"class-{value}" for value in labels],
    ),
    (
        "numpy-string-array",
        lambda labels: np.asarray([f"class-{value}" for value in labels]),
    ),
    (
        "numpy-object-array",
        lambda labels: np.asarray(
            [f"class-{value}" for value in labels],
            dtype=object,
        ),
    ),
)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "label_kind,label_factory",
    LABEL_REPRESENTATIONS,
    ids=[case[0] for case in LABEL_REPRESENTATIONS],
)
def test_label_renaming_and_representations_are_invariant(
    model: str,
    engine: str,
    label_kind: str,
    label_factory: Callable[[np.ndarray], Any],
) -> None:
    """Equivalent class partitions produce identical results for list and string labels."""
    del label_kind
    X, numeric_labels = _build_dataset(8, 3)
    renamed_labels = label_factory(numeric_labels)

    numeric_result = _compute(X, numeric_labels, model=model, engine=engine)
    renamed_result = _compute(X, renamed_labels, model=model, engine=engine)

    _assert_result_arrays_close(renamed_result, numeric_result)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "n_samples,n_features",
    [(2, 1), (3, 2), (5, 5), (8, 1), (8, 2), (11, 5)],
)
def test_dense_and_blockwise_match_across_shapes_and_block_boundaries(
    model: str,
    n_samples: int,
    n_features: int,
) -> None:
    """Blockwise results match dense results across representative data shapes."""
    X, y = _build_dataset(n_samples, n_features)
    dense = _compute(X, y, model=model, engine="dense")

    for block_size in sorted({1, 2, 3, n_samples, n_samples + 2}):
        blockwise = _compute(
            X,
            y,
            model=model,
            engine="blockwise",
            block_size=block_size,
        )

        _assert_result_arrays_close(blockwise, dense)
        assert blockwise.engine == "blockwise"
        assert blockwise.block_size == block_size
        assert blockwise.used_blockwise is True


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
def test_feature_column_permutation_invariance(model: str, engine: str) -> None:
    """Reordering feature columns does not change T-norm aggregated similarities."""
    X, y = _build_dataset(8, 5)
    feature_permutation = np.array([4, 1, 3, 0, 2])

    baseline = _compute(X, y, model=model, engine=engine)
    reordered = _compute(
        X[:, feature_permutation],
        y,
        model=model,
        engine=engine,
    )

    _assert_result_arrays_close(reordered, baseline)


@pytest.mark.parametrize("model", MODELS)
def test_dense_precomputed_similarity_respects_sample_permutation(model: str) -> None:
    """The dense precomputed-matrix path preserves sample permutation semantics."""
    X, y = _build_dataset(8, 4)
    permutation = np.array([2, 7, 0, 5, 3, 1, 6, 4])
    similarity_matrix = build_similarity_matrix(X, similarity="linear")

    baseline = compute_approximations(
        None,
        y,
        model=model,
        similarity_matrix=similarity_matrix,
    )
    permuted = compute_approximations(
        None,
        y[permutation],
        model=model,
        similarity_matrix=similarity_matrix[np.ix_(permutation, permutation)],
    )

    _assert_result_arrays_close(
        permuted,
        baseline,
        expected_order=permutation,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("engine", ENGINES)
def test_public_approximation_does_not_mutate_feature_or_label_inputs(
    model: str,
    engine: str,
) -> None:
    """Public approximation execution leaves caller-owned arrays unchanged."""
    X, y = _build_dataset(8, 4)
    X_before = X.copy()
    y_before = y.copy()

    _compute(X, y, model=model, engine=engine, block_size=3)

    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
