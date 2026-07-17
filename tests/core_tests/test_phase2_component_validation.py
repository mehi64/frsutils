# SPDX-License-Identifier: BSD-3-Clause
"""Validation contracts for core similarities, T-norms, and implicators."""

from __future__ import annotations

import numpy as np
import pytest

from frsutils.core.implicators import Implicator
from frsutils.core.similarities import (
    GaussianSimilarity,
    LinearSimilarity,
    Similarity,
    calculate_similarity_matrix,
)
from frsutils.core.tnorms import TNorm, YagerTNorm


IMPLICATOR_NAMES = (
    "lukasiewicz",
    "goedel",
    "kleenedienes",
    "reichenbach",
    "goguen",
    "rescher",
    "yager",
    "weber",
    "fodor",
)


def test_similarity_call_accepts_scalar_inputs() -> None:
    """Ordinary similarity calls support a scalar feature comparison."""
    assert LinearSimilarity()(0.25, 0.75) == pytest.approx(0.5)
    assert GaussianSimilarity(sigma=0.5)(0.25, 0.75) == pytest.approx(
        np.exp(-0.5)
    )


@pytest.mark.parametrize("similarity", [LinearSimilarity(), GaussianSimilarity(0.5)])
def test_similarity_call_rejects_ambiguous_1d_vector_inputs(similarity: Similarity) -> None:
    """Feature-vector aggregation must be delegated to an explicit T-norm."""
    with pytest.raises(ValueError, match="scalar inputs or a 2D pairwise"):
        similarity(np.array([0.0, 0.5]), np.array([0.25, 0.75]))


@pytest.mark.parametrize(
    "sigma",
    [True, np.bool_(False), np.nan, np.inf, -np.inf],
)
def test_gaussian_similarity_rejects_boolean_and_non_finite_sigma(sigma: object) -> None:
    """Gaussian sigma must be a finite positive non-boolean real number."""
    with pytest.raises(ValueError, match="finite positive number"):
        GaussianSimilarity(sigma=sigma)


@pytest.mark.parametrize(
    "p",
    [True, np.bool_(False), np.nan, np.inf, -np.inf],
)
def test_yager_tnorm_rejects_boolean_and_non_finite_p(p: object) -> None:
    """Yager p must be a finite positive non-boolean real number."""
    with pytest.raises(ValueError, match="finite positive real number"):
        YagerTNorm(p=p)


@pytest.mark.parametrize("implicator_name", IMPLICATOR_NAMES)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_implicators_reject_non_finite_scalar_inputs(
    implicator_name: str,
    invalid_value: float,
) -> None:
    """Every registered implicator rejects non-finite scalar memberships."""
    implicator = Implicator.create(implicator_name)

    with pytest.raises(ValueError, match="finite values"):
        implicator(invalid_value, 0.5)
    with pytest.raises(ValueError, match="finite values"):
        implicator(0.5, invalid_value)


@pytest.mark.parametrize("implicator_name", IMPLICATOR_NAMES)
def test_implicators_reject_non_finite_array_inputs(implicator_name: str) -> None:
    """Vectorized implicator validation also rejects non-finite memberships."""
    implicator = Implicator.create(implicator_name)
    a = np.array([0.0, np.nan], dtype=float)
    b = np.array([1.0, 0.5], dtype=float)

    with pytest.raises(ValueError, match="finite values"):
        implicator(a, b)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_core_similarity_matrix_builder_rejects_non_finite_features(
    invalid_value: float,
) -> None:
    """The low-level dense similarity builder rejects non-finite feature data."""
    X = np.array([[0.0], [invalid_value]], dtype=float)

    with pytest.raises(ValueError, match="finite numeric values"):
        calculate_similarity_matrix(
            X,
            LinearSimilarity(),
            TNorm.create("minimum"),
        )


def test_core_similarity_matrix_builder_rejects_non_numeric_features() -> None:
    """The low-level dense builder reports non-numeric arrays as invalid data."""
    X = np.array([["zero"], ["one"]], dtype=object)

    with pytest.raises(ValueError, match="finite numeric values"):
        calculate_similarity_matrix(
            X,
            LinearSimilarity(),
            TNorm.create("minimum"),
        )
