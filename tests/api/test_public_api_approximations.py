# SPDX-License-Identifier: BSD-3-Clause
"""Tests for FRsutils public approximation API."""

import numpy as np

from FRsutils.api import (
    FuzzyRoughApproximationResult,
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)


X_SMALL = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.8, 0.8],
        [0.9, 0.9],
    ],
    dtype=float,
)
Y_SMALL = np.array([0, 0, 1, 1])


def test_compute_approximations_returns_named_result():
    result = compute_approximations(X_SMALL, Y_SMALL, model="itfrs")

    assert isinstance(result, FuzzyRoughApproximationResult)
    assert result.lower.shape == (len(Y_SMALL),)
    assert result.upper.shape == (len(Y_SMALL),)
    assert result.boundary.shape == (len(Y_SMALL),)
    assert result.positive_region.shape == (len(Y_SMALL),)
    assert result.model == "itfrs"


def test_convenience_wrappers_match_full_result():
    result = compute_approximations(X_SMALL, Y_SMALL, model="itfrs")

    np.testing.assert_allclose(compute_lower_approximation(X_SMALL, Y_SMALL), result.lower)
    np.testing.assert_allclose(compute_upper_approximation(X_SMALL, Y_SMALL), result.upper)
    np.testing.assert_allclose(compute_boundary_region(X_SMALL, Y_SMALL), result.boundary)
    np.testing.assert_allclose(compute_positive_region(X_SMALL, Y_SMALL), result.positive_region)


def test_compute_approximations_accepts_precomputed_similarity_matrix():
    similarity_matrix = build_similarity_matrix(X_SMALL, similarity="linear")

    result = compute_approximations(
        X=None,
        y=Y_SMALL,
        model="itfrs",
        similarity_matrix=similarity_matrix,
        return_similarity_matrix=True,
    )

    np.testing.assert_allclose(result.similarity_matrix, similarity_matrix)
    assert result.positive_region.shape == (len(Y_SMALL),)


def test_compute_positive_region_supports_alternative_models():
    scores = compute_positive_region(X_SMALL, Y_SMALL, model="vqrs")

    assert scores.shape == (len(Y_SMALL),)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
