# SPDX-License-Identifier: BSD-3-Clause
"""Shared input-contract tests for dense fuzzy-rough core models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from frsutils.core.fuzzy_quantifiers import LinearFuzzyQuantifier
from frsutils.core.implicators import LukasiewiczImplicator
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.models.owafrs import OWAFRS
from frsutils.core.models.vqrs import VQRS
from frsutils.core.owa_weights import LinearOWAWeights
from frsutils.core.tnorms import MinTNorm


def _build_itfrs(similarity_matrix: np.ndarray, labels: np.ndarray) -> ITFRS:
    """Build a dense ITFRS model for shared contract tests."""
    return ITFRS(
        similarity_matrix,
        labels,
        ub_tnorm=MinTNorm(),
        lb_implicator=LukasiewiczImplicator(),
    )


def _build_vqrs(similarity_matrix: np.ndarray, labels: np.ndarray) -> VQRS:
    """Build a dense VQRS model for shared contract tests."""
    return VQRS(
        similarity_matrix,
        labels,
        lb_fuzzy_quantifier=LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        ub_fuzzy_quantifier=LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )


def _build_owafrs(similarity_matrix: np.ndarray, labels: np.ndarray) -> OWAFRS:
    """Build a dense OWAFRS model for shared contract tests."""
    return OWAFRS(
        similarity_matrix,
        labels,
        ub_tnorm=MinTNorm(),
        lb_implicator=LukasiewiczImplicator(),
        ub_owa_method=LinearOWAWeights(),
        lb_owa_method=LinearOWAWeights(),
    )


MODEL_BUILDERS: tuple[Callable[[np.ndarray, np.ndarray], object], ...] = (
    _build_itfrs,
    _build_vqrs,
    _build_owafrs,
)


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
@pytest.mark.parametrize(
    "similarity_matrix, labels",
    [
        (np.empty((0, 0), dtype=float), np.array([], dtype=int)),
        (np.array([[1.0]], dtype=float), np.array(["only"], dtype=object)),
    ],
)
def test_dense_core_models_require_at_least_two_samples(
    builder: Callable[[np.ndarray, np.ndarray], object],
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> None:
    """All dense core models enforce the same minimum-size contract."""
    with pytest.raises(ValueError, match="at least two samples"):
        builder(similarity_matrix, labels)


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
def test_dense_core_models_reject_non_1d_labels(
    builder: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    """All dense core models reject ambiguous multidimensional labels."""
    with pytest.raises(ValueError, match="labels must be a 1D"):
        builder(
            np.eye(2, dtype=float),
            np.array([["a"], ["b"]], dtype=object),
        )


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_dense_core_models_reject_non_finite_similarity_values(
    builder: Callable[[np.ndarray, np.ndarray], object],
    invalid_value: float,
) -> None:
    """All dense core models reject non-finite fuzzy-relation entries."""
    similarity_matrix = np.eye(2, dtype=float)
    similarity_matrix[0, 1] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        builder(similarity_matrix, np.array(["a", "b"], dtype=object))


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
def test_dense_core_models_reject_non_numeric_similarity_values(
    builder: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    """All dense core models reject non-numeric relation matrices clearly."""
    similarity_matrix = np.array(
        [["one", "zero"], ["zero", "one"]],
        dtype=object,
    )

    with pytest.raises(ValueError, match="numeric finite values"):
        builder(similarity_matrix, np.array(["a", "b"], dtype=object))


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_dense_core_models_reject_out_of_range_similarity_values(
    builder: Callable[[np.ndarray, np.ndarray], object],
    invalid_value: float,
) -> None:
    """All dense core models reject relation degrees outside [0, 1]."""
    similarity_matrix = np.eye(2, dtype=float)
    similarity_matrix[0, 1] = invalid_value

    with pytest.raises(ValueError, match="range"):
        builder(similarity_matrix, np.array(["a", "b"], dtype=object))


@pytest.mark.parametrize("builder", MODEL_BUILDERS)
def test_dense_core_models_accept_finite_asymmetric_non_reflexive_relations(
    builder: Callable[[np.ndarray, np.ndarray], object],
) -> None:
    """Low-level models accept finite fuzzy relations without symmetry/reflexivity."""
    relation = np.array(
        [
            [0.8, 0.4],
            [0.2, 0.9],
        ],
        dtype=float,
    )
    model = builder(relation, np.array(["same", "same"], dtype=object))

    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.isfinite(lower).all()
    assert np.isfinite(upper).all()


def test_itfrs_uses_rows_as_query_samples_for_asymmetric_relations() -> None:
    """ITFRS aggregates row ``i`` to produce the result for sample ``i``."""
    relation = np.array(
        [
            [0.7, 0.9, 0.2],
            [0.1, 0.8, 0.6],
            [0.4, 0.3, 0.9],
        ],
        dtype=float,
    )
    labels = np.array(["a", "a", "b"], dtype=object)
    model = _build_itfrs(relation, labels)

    np.testing.assert_allclose(model.lower_approximation(), [0.8, 0.4, 0.6])
    np.testing.assert_allclose(model.upper_approximation(), [0.9, 0.1, 0.0])
    assert model.relation_orientation == "rows_are_queries"

    transposed = _build_itfrs(relation.T, labels)
    assert not np.allclose(
        model.lower_approximation(),
        transposed.lower_approximation(),
    )


def test_vqrs_uses_rows_as_query_samples_for_asymmetric_relations() -> None:
    """VQRS computes supporting and total evidence from each relation row."""
    relation = np.array(
        [
            [0.7, 0.9, 0.2],
            [0.1, 0.8, 0.6],
            [0.4, 0.3, 0.9],
        ],
        dtype=float,
    )
    labels = np.array(["a", "a", "b"], dtype=object)
    identity_quantifier = LinearFuzzyQuantifier(alpha=0.0, beta=1.0)
    model = VQRS(
        relation,
        labels,
        lb_fuzzy_quantifier=identity_quantifier,
        ub_fuzzy_quantifier=identity_quantifier,
    )

    expected = np.array([0.9 / 1.1, 0.1 / 0.7, 0.0], dtype=float)
    np.testing.assert_allclose(model.lower_approximation(), expected)
    np.testing.assert_allclose(model.upper_approximation(), expected)
    assert model.relation_orientation == "rows_are_queries"

    transposed = VQRS(
        relation.T,
        labels,
        lb_fuzzy_quantifier=identity_quantifier,
        ub_fuzzy_quantifier=identity_quantifier,
    )
    assert not np.allclose(
        model.lower_approximation(),
        transposed.lower_approximation(),
    )


def test_owafrs_uses_rows_as_query_samples_for_asymmetric_relations() -> None:
    """OWAFRS sorts and aggregates each relation row for its query sample."""
    relation = np.array(
        [
            [0.7, 0.9, 0.2],
            [0.1, 0.8, 0.6],
            [0.4, 0.3, 0.9],
        ],
        dtype=float,
    )
    labels = np.array(["a", "a", "b"], dtype=object)
    model = _build_owafrs(relation, labels)

    np.testing.assert_allclose(
        model.lower_approximation(),
        [13.0 / 15.0, 3.0 / 5.0, 19.0 / 30.0],
    )
    np.testing.assert_allclose(
        model.upper_approximation(),
        [3.0 / 5.0, 1.0 / 15.0, 0.0],
    )
    assert model.relation_orientation == "rows_are_queries"

    transposed = _build_owafrs(relation.T, labels)
    assert not np.allclose(
        model.lower_approximation(),
        transposed.lower_approximation(),
    )
