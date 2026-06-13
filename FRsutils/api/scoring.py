# SPDX-License-Identifier: BSD-3-Clause
"""Scoring helpers for fuzzy-rough approximation outputs.

This module belongs to the stable public API layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from FRsutils.api.approximations import compute_approximations
from FRsutils.api.results import FuzzyRoughApproximationResult


class FuzzyRoughPositiveRegionScorer(BaseEstimator):
    """Reusable public estimator for fuzzy-rough positive-region scores.
    
    The scorer wraps `compute_approximations(...)` and stores the resulting
    lower/upper/boundary/positive-region arrays as fitted attributes. It is meant
    for users and downstream packages that want a stable object-oriented API
    without importing FRsutils internals.
    
    Parameters
    ----------
    model : object
        Fuzzy-rough model alias, e.g. "itfrs", "owafrs", or "vqrs".
    similarity : object
        Optional similarity alias for matrix construction.
    similarity_sigma : object
        Optional Gaussian similarity sigma parameter.
    similarity_tnorm : object
        Optional t-norm alias for feature-level aggregation.
    ub_tnorm_name : object
        Optional upper-approximation t-norm alias.
    lb_implicator_name : object
        Optional lower-approximation implicator alias.
    ub_owa_method_name : object
        Optional OWAFRS upper OWA method alias.
    lb_owa_method_name : object
        Optional OWAFRS lower OWA method alias.
    ub_owa_method_base : object
        Optional OWAFRS upper OWA base parameter.
    lb_owa_method_base : object
        Optional OWAFRS lower OWA base parameter.
    lb_fuzzy_quantifier_name : object
        Optional VQRS lower quantifier alias.
    ub_fuzzy_quantifier_name : object
        Optional VQRS upper quantifier alias.
    lb_fuzzy_quantifier_alpha : object
        Optional lower quantifier alpha.
    lb_fuzzy_quantifier_beta : object
        Optional lower quantifier beta.
    ub_fuzzy_quantifier_alpha : object
        Optional upper quantifier alpha.
    ub_fuzzy_quantifier_beta : object
        Optional upper quantifier beta.
    similarity_matrix : object
        Optional precomputed similarity matrix.
    config : object
        Optional flat or nested FRsutils config mapping.
    return_similarity_matrix : object
        If True, store the similarity matrix in result_.
    engine : object
        Approximation execution engine forwarded to compute_approximations.
    block_size : object
        Block size forwarded when engine="blockwise".
    backend : object
        Backend alias forwarded for blockwise similarity-block execution.
    extra_params : object
        Optional mapping for advanced flat parameters not yet exposed.
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
        engine: str = "dense",
        block_size: int = 1024,
        backend: str = "numpy",
        extra_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Initialize the FuzzyRoughPositiveRegionScorer instance."""
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
        self.engine = engine
        self.block_size = block_size
        self.backend = backend
        self.extra_params = extra_params

    def _flat_config(self) -> Dict[str, Any]:
        """Build flat public config from explicit scorer parameters.
                
                Returns
                -------
                Dict[str, Any]
                    Flat configuration dictionary with None values removed.
                
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
        """Fit the scorer and cache fuzzy-rough approximation outputs.
                
                Parameters
                ----------
                X : Optional[np.ndarray]
                    Input feature matrix, or None when a similarity matrix is available.
                y : np.ndarray
                    Label vector aligned with X or similarity_matrix.
                
                Returns
                -------
                object
                    self.
                
        """
        result = compute_approximations(
            X,
            y,
            model=self.model,
            similarity=self.similarity,
            similarity_matrix=self.similarity_matrix,
            config=self.config,
            return_similarity_matrix=self.return_similarity_matrix,
            engine=self.engine,
            block_size=self.block_size,
            backend=self.backend,
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
        """Fit the scorer and return positive-region scores directly.
                
                Parameters
                ----------
                X : Optional[np.ndarray]
                    Input feature matrix, or None when a similarity matrix is available.
                y : np.ndarray
                    Label vector aligned with X or similarity_matrix.
                
                Returns
                -------
                np.ndarray
                    One-dimensional positive-region score array.
                
        """
        return self.fit(X, y).positive_region_

    def score_samples(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return fitted training positive-region scores.
                
                The current FRsutils fuzzy-rough approximation models compute scores for
                the fitted similarity matrix. Therefore this method intentionally returns
                the cached training scores and does not yet score unseen samples.
                
                Parameters
                ----------
                X : Optional[np.ndarray]
                    Ignored placeholder for sklearn-like call sites.
                
                Returns
                -------
                np.ndarray
                    Fitted positive-region score array.
                
                Raises
                ------
                NotFittedError
                    If fit has not been called.
                
        """
        check_is_fitted(self, "positive_region_")
        return self.positive_region_

    def as_result(self) -> FuzzyRoughApproximationResult:
        """Return the fitted approximation result object.
                
                Returns
                -------
                FuzzyRoughApproximationResult
                    Fitted FuzzyRoughApproximationResult.
                
                Raises
                ------
                NotFittedError
                    If fit has not been called.
                
        """
        check_is_fitted(self, "result_")
        return self.result_


__all__ = ["FuzzyRoughPositiveRegionScorer"]
