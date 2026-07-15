# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for public API behavior and metadata hardening."""

from __future__ import annotations

import importlib
import logging

import numpy as np
import pytest

from frsutils import (
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_approximations,
)


X_PHASE3 = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.8, 0.8],
        [0.9, 0.9],
    ],
    dtype=float,
)
Y_PHASE3 = np.array([0, 0, 1, 1])


def test_precomputed_similarity_does_not_claim_unknown_similarity_provenance():
    """Precomputed matrices should not inherit the public linear default as provenance."""
    similarity_matrix = build_similarity_matrix(
        X_PHASE3,
        similarity="gaussian",
        similarity_sigma=0.4,
    )

    result = compute_approximations(
        None,
        Y_PHASE3,
        model="itfrs",
        similarity_matrix=similarity_matrix,
    )

    assert result.similarity is None
    assert result.config is not None
    assert "similarity" not in result.config
    assert "similarity_tnorm" not in result.config
    assert not any(key.startswith("similarity_") for key in result.config)


@pytest.mark.parametrize(
    "similarity_kwargs",
    [
        {"similarity": "gaussian"},
        {"similarity_sigma": 0.4},
        {"config": {"similarity_tnorm": "product"}},
    ],
)
def test_precomputed_similarity_rejects_unused_similarity_configuration(similarity_kwargs):
    """Explicit similarity settings must not be silently ignored for precomputed matrices."""
    similarity_matrix = build_similarity_matrix(X_PHASE3, similarity="linear")

    with pytest.raises(ValueError, match="precomputed similarity_matrix"):
        compute_approximations(
            None,
            Y_PHASE3,
            model="itfrs",
            similarity_matrix=similarity_matrix,
            **similarity_kwargs,
        )


def test_dense_approximation_path_calls_lower_and_upper_once(monkeypatch):
    """Dense public computation should derive boundary and positive region from cached arrays."""
    approximations_api = importlib.import_module("frsutils.api.approximations")
    calls = {"lower": 0, "upper": 0, "boundary": 0, "positive_region": 0}

    class CountingModel:
        """Minimal model double that records approximation method calls."""

        def lower_approximation(self):
            """Return deterministic lower values and record the call."""
            calls["lower"] += 1
            return np.array([0.1, 0.2, 0.3, 0.4])

        def upper_approximation(self):
            """Return deterministic upper values and record the call."""
            calls["upper"] += 1
            return np.array([0.5, 0.6, 0.7, 0.8])

        def boundary_region(self):
            """Record an unexpected direct boundary computation."""
            calls["boundary"] += 1
            raise AssertionError("boundary_region should not be called")

        def positive_region(self):
            """Record an unexpected direct positive-region computation."""
            calls["positive_region"] += 1
            raise AssertionError("positive_region should not be called")

    def build_counting_model(*args, **kwargs):
        """Return the counting model double for dense public computation."""
        return CountingModel()

    monkeypatch.setattr(
        approximations_api,
        "build_fuzzy_rough_model",
        build_counting_model,
    )

    result = compute_approximations(
        None,
        Y_PHASE3,
        model="itfrs",
        similarity_matrix=np.eye(len(Y_PHASE3)),
    )

    assert calls == {"lower": 1, "upper": 1, "boundary": 0, "positive_region": 0}
    np.testing.assert_allclose(result.boundary, result.upper - result.lower)
    np.testing.assert_allclose(result.positive_region, result.lower)
    assert result.positive_region is not result.lower


def test_score_samples_rejects_unseen_x_instead_of_ignoring_it():
    """The scorer should fail clearly when asked to score unseen samples."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    scorer.fit(X_PHASE3, Y_PHASE3)

    with pytest.raises(ValueError, match="Scoring unseen samples is not supported"):
        scorer.score_samples(X_PHASE3[:2])


def test_extra_params_rejects_duplicate_explicit_constructor_parameter():
    """extra_params should not override a named scorer constructor parameter."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="linear",
        ub_tnorm_p=2.0,
        extra_params={"ub_tnorm_p": 3.0},
    )

    with pytest.raises(ValueError, match="ub_tnorm_p.*explicit scorer constructor"):
        scorer.fit(X_PHASE3, Y_PHASE3)


def test_extra_params_rejects_alias_for_explicit_constructor_parameter():
    """Alias canonicalization should not bypass duplicate extra-parameter detection."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="vqrs",
        similarity="linear",
        lb_fuzzy_quantifier_alpha=0.2,
        extra_params={"lb_alpha": 0.3},
    )

    with pytest.raises(
        ValueError,
        match="lb_fuzzy_quantifier_alpha.*explicit scorer constructor",
    ):
        scorer.fit(X_PHASE3, Y_PHASE3)


def test_extra_params_still_accepts_non_constructor_flat_parameter():
    """extra_params should remain usable for flat contract keys without named slots."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="linear",
        extra_params={"logger": logging.getLogger("frsutils.phase3-test")},
    )

    scores = scorer.fit_score(X_PHASE3, Y_PHASE3)

    assert scores.shape == (len(Y_PHASE3),)
