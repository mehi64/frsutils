# SPDX-License-Identifier: BSD-3-Clause
"""Phase 3 tests for the public positive-region scorer."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from FRsutils.api import (
    FuzzyRoughApproximationResult,
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_positive_region,
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


def test_positive_region_scorer_fit_score_matches_function_api():
    """@brief fit_score returns the same values as compute_positive_region."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")

    fitted_scores = scorer.fit_score(X_SMALL, Y_SMALL)
    function_scores = compute_positive_region(X_SMALL, Y_SMALL, model="itfrs", similarity="linear")

    np.testing.assert_allclose(fitted_scores, function_scores)
    np.testing.assert_allclose(scorer.score_samples(), function_scores)
    assert scorer.n_samples_in_ == len(Y_SMALL)


def test_positive_region_scorer_exposes_full_result():
    """@brief The scorer stores the named approximation result after fitting."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    scorer.fit(X_SMALL, Y_SMALL)

    result = scorer.as_result()

    assert isinstance(result, FuzzyRoughApproximationResult)
    np.testing.assert_allclose(result.positive_region, scorer.positive_region_)
    np.testing.assert_allclose(result.lower, scorer.lower_)
    np.testing.assert_allclose(result.upper, scorer.upper_)
    np.testing.assert_allclose(result.boundary, scorer.boundary_)


def test_positive_region_scorer_accepts_precomputed_similarity_matrix():
    """@brief Downstream packages can fit the scorer from a reused matrix."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity_matrix=sim,
        return_similarity_matrix=True,
    )

    scores = scorer.fit_score(X=None, y=Y_SMALL)

    assert scores.shape == (len(Y_SMALL),)
    np.testing.assert_allclose(scorer.result_.similarity_matrix, sim)


def test_positive_region_scorer_is_sklearn_parameter_compatible():
    """@brief The scorer supports sklearn-style get_params/set_params."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    params = scorer.get_params()

    assert params["model"] == "itfrs"
    assert params["similarity"] == "linear"

    scorer.set_params(model="vqrs", lb_fuzzy_quantifier_alpha=0.2)
    assert scorer.model == "vqrs"
    assert scorer.lb_fuzzy_quantifier_alpha == 0.2
    assert scorer.fit_score(X_SMALL, Y_SMALL).shape == (len(Y_SMALL),)


def test_positive_region_scorer_rejects_unfitted_access():
    """@brief Unfitted scorer access fails with sklearn's NotFittedError."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs")

    with pytest.raises(NotFittedError):
        scorer.score_samples()

    with pytest.raises(NotFittedError):
        scorer.as_result()
