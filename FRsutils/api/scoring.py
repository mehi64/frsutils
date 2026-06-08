"""
@file scoring.py
@brief Scikit-learn-friendly public scorers for fuzzy-rough approximation outputs.

This module exposes lightweight estimator-style helpers above the task-oriented
approximation API. The first public scorer focuses on positive-region scores,
which are useful for downstream packages such as standalone fuzzy-rough sampling
libraries and for end users who want a reusable fitted object.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FuzzyRoughPositiveRegionScorer       Fit/reuse positive-region approximation scores

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: consumes only FRsutils.api task-level approximation helpers
# - Adapter Pattern: keeps flat public params and forwards them to compute_approximations
# - Estimator Pattern: follows sklearn-style fit/get_params/set_params behavior
# - Boundary Validation: validates scorer usage at the public API edge
# - Dependency Inversion: downstream packages can depend on this scorer, not internals
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api import FuzzyRoughPositiveRegionScorer
#
# scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
# scores = scorer.fit_score(X, y)
# result = scorer.as_result()
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from FRsutils.api.approximations import compute_approximations
from FRsutils.api.results import FuzzyRoughApproximationResult


class FuzzyRoughPositiveRegionScorer(BaseEstimator):
    """
    @brief Reusable public estimator for fuzzy-rough positive-region scores.

    The scorer wraps `compute_approximations(...)` and stores the resulting
    lower/upper/boundary/positive-region arrays as fitted attributes. It is meant
    for users and downstream packages that want a stable object-oriented API
    without importing FRsutils internals.

    @param model: Fuzzy-rough model alias, e.g. "itfrs", "owafrs", or "vqrs".
    @param similarity: Optional similarity alias for matrix construction.
    @param similarity_sigma: Optional Gaussian similarity sigma parameter.
    @param similarity_tnorm: Optional t-norm alias for feature-level aggregation.
    @param ub_tnorm_name: Optional upper-approximation t-norm alias.
    @param lb_implicator_name: Optional lower-approximation implicator alias.
    @param ub_owa_method_name: Optional OWAFRS upper OWA method alias.
    @param lb_owa_method_name: Optional OWAFRS lower OWA method alias.
    @param ub_owa_method_base: Optional OWAFRS upper OWA base parameter.
    @param lb_owa_method_base: Optional OWAFRS lower OWA base parameter.
    @param lb_fuzzy_quantifier_name: Optional VQRS lower quantifier alias.
    @param ub_fuzzy_quantifier_name: Optional VQRS upper quantifier alias.
    @param lb_fuzzy_quantifier_alpha: Optional lower quantifier alpha.
    @param lb_fuzzy_quantifier_beta: Optional lower quantifier beta.
    @param ub_fuzzy_quantifier_alpha: Optional upper quantifier alpha.
    @param ub_fuzzy_quantifier_beta: Optional upper quantifier beta.
    @param similarity_matrix: Optional precomputed similarity matrix.
    @param config: Optional flat or nested FRsutils config mapping.
    @param return_similarity_matrix: If True, store the similarity matrix in result_.
    @param extra_params: Optional mapping for advanced flat parameters not yet exposed.
    """

    def __init__(
        self,
        model: str = "itfrs",
        similarity: Optional[str] = None,
        similarity_sigma: Optional[float] = None,
        similarity_tnorm: Optional[str] = None,
        ub_tnorm_name: Optional[str] = None,
        lb_implicator_name: Optional[str] = None,
        ub_owa_method_name: Optional[str] = None,
        lb_owa_method_name: Optional[str] = None,
        ub_owa_method_base: Optional[float] = None,
        lb_owa_method_base: Optional[float] = None,
        lb_fuzzy_quantifier_name: Optional[str] = None,
        ub_fuzzy_quantifier_name: Optional[str] = None,
        lb_fuzzy_quantifier_alpha: Optional[float] = None,
        lb_fuzzy_quantifier_beta: Optional[float] = None,
        ub_fuzzy_quantifier_alpha: Optional[float] = None,
        ub_fuzzy_quantifier_beta: Optional[float] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        config: Optional[Mapping[str, Any]] = None,
        return_similarity_matrix: bool = False,
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model = model
        self.similarity = similarity
        self.similarity_sigma = similarity_sigma
        self.similarity_tnorm = similarity_tnorm
        self.ub_tnorm_name = ub_tnorm_name
        self.lb_implicator_name = lb_implicator_name
        self.ub_owa_method_name = ub_owa_method_name
        self.lb_owa_method_name = lb_owa_method_name
        self.ub_owa_method_base = ub_owa_method_base
        self.lb_owa_method_base = lb_owa_method_base
        self.lb_fuzzy_quantifier_name = lb_fuzzy_quantifier_name
        self.ub_fuzzy_quantifier_name = ub_fuzzy_quantifier_name
        self.lb_fuzzy_quantifier_alpha = lb_fuzzy_quantifier_alpha
        self.lb_fuzzy_quantifier_beta = lb_fuzzy_quantifier_beta
        self.ub_fuzzy_quantifier_alpha = ub_fuzzy_quantifier_alpha
        self.ub_fuzzy_quantifier_beta = ub_fuzzy_quantifier_beta
        self.similarity_matrix = similarity_matrix
        self.config = config
        self.return_similarity_matrix = return_similarity_matrix
        self.extra_params = extra_params

    def _flat_config(self) -> Dict[str, Any]:
        """
        @brief Build flat public config from explicit scorer parameters.

        @return: Flat configuration dictionary with None values removed.
        """
        params: Dict[str, Any] = {
            "similarity_sigma": self.similarity_sigma,
            "similarity_tnorm": self.similarity_tnorm,
            "ub_tnorm_name": self.ub_tnorm_name,
            "lb_implicator_name": self.lb_implicator_name,
            "ub_owa_method_name": self.ub_owa_method_name,
            "lb_owa_method_name": self.lb_owa_method_name,
            "ub_owa_method_base": self.ub_owa_method_base,
            "lb_owa_method_base": self.lb_owa_method_base,
            "lb_fuzzy_quantifier_name": self.lb_fuzzy_quantifier_name,
            "ub_fuzzy_quantifier_name": self.ub_fuzzy_quantifier_name,
            "lb_fuzzy_quantifier_alpha": self.lb_fuzzy_quantifier_alpha,
            "lb_fuzzy_quantifier_beta": self.lb_fuzzy_quantifier_beta,
            "ub_fuzzy_quantifier_alpha": self.ub_fuzzy_quantifier_alpha,
            "ub_fuzzy_quantifier_beta": self.ub_fuzzy_quantifier_beta,
        }
        if self.extra_params is not None:
            if not isinstance(self.extra_params, Mapping):
                raise TypeError("extra_params must be a mapping when provided.")
            params.update(dict(self.extra_params))
        return {key: value for key, value in params.items() if value is not None}

    def fit(self, X: Optional[np.ndarray], y: np.ndarray):
        """
        @brief Fit the scorer and cache fuzzy-rough approximation outputs.

        @param X: Input feature matrix, or None when a similarity matrix is available.
        @param y: Label vector aligned with X or similarity_matrix.
        @return: self.
        """
        result = compute_approximations(
            X,
            y,
            model=self.model,
            similarity=self.similarity,
            similarity_matrix=self.similarity_matrix,
            config=self.config,
            return_similarity_matrix=self.return_similarity_matrix,
            **self._flat_config(),
        )

        self.result_: FuzzyRoughApproximationResult = result
        self.positive_region_ = result.positive_region
        self.lower_ = result.lower
        self.upper_ = result.upper
        self.boundary_ = result.boundary
        self.n_samples_in_ = len(np.asarray(y))
        return self

    def fit_score(self, X: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
        """
        @brief Fit the scorer and return positive-region scores directly.

        @param X: Input feature matrix, or None when a similarity matrix is available.
        @param y: Label vector aligned with X or similarity_matrix.
        @return: One-dimensional positive-region score array.
        """
        return self.fit(X, y).positive_region_

    def score_samples(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        @brief Return fitted training positive-region scores.

        The current FRsutils fuzzy-rough approximation models compute scores for
        the fitted similarity matrix. Therefore this method intentionally returns
        the cached training scores and does not yet score unseen samples.

        @param X: Ignored placeholder for sklearn-like call sites.
        @return: Fitted positive-region score array.
        @raises NotFittedError: If fit has not been called.
        """
        check_is_fitted(self, "positive_region_")
        return self.positive_region_

    def as_result(self) -> FuzzyRoughApproximationResult:
        """
        @brief Return the fitted approximation result object.

        @return: Fitted FuzzyRoughApproximationResult.
        @raises NotFittedError: If fit has not been called.
        """
        check_is_fitted(self, "result_")
        return self.result_


__all__ = ["FuzzyRoughPositiveRegionScorer"]
