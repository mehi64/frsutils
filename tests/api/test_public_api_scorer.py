# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the public positive-region scorer."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from frsutils import (
    FuzzyRoughApproximationResult,
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_positive_region,
    normalize_flat_config_to_nested,
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


MODEL_SCORER_CONFIGS = {
    "itfrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    },
    "vqrs": {
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.0,
        "lb_fuzzy_quantifier_beta": 0.5,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.1,
        "ub_fuzzy_quantifier_beta": 0.8,
    },
    "owafrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "harmonic",
    },
}


def test_positive_region_scorer_fit_score_matches_function_api():
    """fit_score returns the same values as compute_positive_region."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")

    fitted_scores = scorer.fit_score(X_SMALL, Y_SMALL)
    function_scores = compute_positive_region(X_SMALL, Y_SMALL, model="itfrs", similarity="linear")

    np.testing.assert_allclose(fitted_scores, function_scores)
    np.testing.assert_allclose(scorer.score_samples(), function_scores)
    assert scorer.n_samples_in_ == len(Y_SMALL)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_positive_region_scorer_matches_function_api_for_each_model(model):
    """The scorer forwards model-specific flat params for every public model."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model=model,
        similarity="linear",
        **MODEL_SCORER_CONFIGS[model],
    )

    fitted_scores = scorer.fit_score(X_SMALL, Y_SMALL)
    function_scores = compute_positive_region(
        X_SMALL,
        Y_SMALL,
        model=model,
        similarity="linear",
        **MODEL_SCORER_CONFIGS[model],
    )

    assert isinstance(fitted_scores, np.ndarray)
    assert fitted_scores.shape == (len(Y_SMALL),)
    np.testing.assert_allclose(fitted_scores, function_scores)


def test_positive_region_scorer_exposes_full_result():
    """The scorer stores the named approximation result after fitting."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    scorer.fit(X_SMALL, Y_SMALL)

    result = scorer.as_result()

    assert isinstance(result, FuzzyRoughApproximationResult)
    np.testing.assert_allclose(result.positive_region, scorer.positive_region_)
    np.testing.assert_allclose(result.lower, scorer.lower_)
    np.testing.assert_allclose(result.upper, scorer.upper_)
    np.testing.assert_allclose(result.boundary, scorer.boundary_)


def test_positive_region_scorer_accepts_precomputed_similarity_matrix():
    """Downstream packages can fit the scorer from a reused matrix."""
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
    """The scorer supports sklearn-style get_params/set_params."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    params = scorer.get_params()

    assert params["model"] == "itfrs"
    assert params["similarity"] == "linear"

    scorer.set_params(model="vqrs", lb_fuzzy_quantifier_alpha=0.2)
    assert scorer.model == "vqrs"
    assert scorer.lb_fuzzy_quantifier_alpha == 0.2
    assert scorer.fit_score(X_SMALL, Y_SMALL).shape == (len(Y_SMALL),)


def test_positive_region_scorer_is_sklearn_clone_compatible():
    """sklearn.clone preserves constructor params without fitted state."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        ub_owa_method_name="linear",
        lb_owa_method_name="harmonic",
    )
    scorer.fit(X_SMALL, Y_SMALL)

    cloned = clone(scorer)

    assert cloned.model == "owafrs"
    assert cloned.similarity == "linear"
    assert cloned.engine == "blockwise"
    assert cloned.block_size == 2
    assert cloned.ub_owa_method_name == "linear"
    assert cloned.lb_owa_method_name == "harmonic"
    assert not hasattr(cloned, "positive_region_")
    assert cloned.fit_score(X_SMALL, Y_SMALL).shape == (len(Y_SMALL),)


def test_positive_region_scorer_forwards_blockwise_execution_options():
    """Scorer users can request blockwise approximation through public params."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        backend="numpy",
        **MODEL_SCORER_CONFIGS["vqrs"],
    )

    scores = scorer.fit_score(X_SMALL, Y_SMALL)
    dense_scores = compute_positive_region(
        X_SMALL,
        Y_SMALL,
        model="vqrs",
        similarity="linear",
        **MODEL_SCORER_CONFIGS["vqrs"],
    )

    np.testing.assert_allclose(scores, dense_scores)
    assert scorer.result_.engine == "blockwise"
    assert scorer.result_.block_size == 2
    assert scorer.result_.used_blockwise is True
    assert scorer.result_.backend == "numpy"


def test_positive_region_scorer_accepts_nested_config():
    """Nested public configs can be used with the object-oriented scorer."""
    nested_config = normalize_flat_config_to_nested(
        {
            "type": "owafrs",
            "similarity": "linear",
            **MODEL_SCORER_CONFIGS["owafrs"],
        }
    )
    scorer = FuzzyRoughPositiveRegionScorer(model="owafrs", config=nested_config)

    scores = scorer.fit_score(X_SMALL, Y_SMALL)
    expected = compute_positive_region(X_SMALL, Y_SMALL, model="owafrs", config=nested_config)

    np.testing.assert_allclose(scores, expected)


def test_positive_region_scorer_accepts_extra_flat_params_mapping():
    """Advanced flat params can be passed through extra_params."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="owafrs",
        similarity="linear",
        extra_params={
            "ub_tnorm_name": "minimum",
            "lb_implicator_name": "lukasiewicz",
            "ub_owa_method_name": "linear",
            "lb_owa_method_name": "harmonic",
        },
    )

    assert scorer.fit_score(X_SMALL, Y_SMALL).shape == (len(Y_SMALL),)


def test_positive_region_scorer_rejects_non_mapping_extra_params():
    """extra_params is a public pass-through mapping, not an arbitrary object."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="linear",
        extra_params=[("ub_tnorm_name", "minimum")],
    )

    with pytest.raises(TypeError, match="extra_params must be a mapping"):
        scorer.fit(X_SMALL, Y_SMALL)


def test_positive_region_scorer_rejects_unfitted_access():
    """Unfitted scorer access fails with sklearn's NotFittedError."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs")

    with pytest.raises(NotFittedError):
        scorer.score_samples()

    with pytest.raises(NotFittedError):
        scorer.as_result()
