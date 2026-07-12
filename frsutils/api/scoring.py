# SPDX-License-Identifier: BSD-3-Clause
"""Scikit-learn-style scoring helpers for fuzzy-rough approximations.

This module provides the public estimator wrapper used to compute and cache
fuzzy-rough positive-region scores without importing internal core modules.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .approximations import compute_approximations
from .results import FuzzyRoughApproximationResult

class FuzzyRoughPositiveRegionScorer(BaseEstimator):
    """Estimate fuzzy-rough positive-region scores for fitted samples.

    The scorer wraps :func:`frsutils.compute_approximations` and stores the
    resulting lower, upper, boundary, and positive-region arrays as fitted
    attributes. It follows scikit-learn estimator conventions for cloning,
    ``get_params``, and ``set_params``.

    Parameters
    ----------
    model : str, default="itfrs"
        Fuzzy-rough model alias, such as ``"itfrs"``, ``"vqrs"``, or
        ``"owafrs"``.
    similarity : str or None, default=None
        Similarity alias used when constructing a similarity matrix from ``X``.
    similarity_matrix : ndarray of shape (n_samples, n_samples) or None, default=None
        Optional precomputed similarity matrix.
    config : Mapping or None, default=None
        Optional flat or nested frsutils configuration.
    engine : {"dense", "blockwise"}, default="dense"
        Approximation execution engine.
    block_size : int, default=1024
        Block size used by ``engine="blockwise"``.
    backend : str, default="numpy"
        Backend alias for blockwise similarity-block execution.
    extra_params : Mapping or None, default=None
        Optional flat parameters not represented by explicit constructor
        arguments.
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
        """Return flat public config with ``None`` values removed."""
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
        X : ndarray of shape (n_samples, n_features) or None
            Feature matrix, or ``None`` when ``similarity_matrix`` is provided.
        y : array-like of shape (n_samples,)
            Label vector aligned with ``X`` or ``similarity_matrix``.

        Returns
        -------
        self : FuzzyRoughPositiveRegionScorer
            Fitted scorer.
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
        """Fit the scorer and return positive-region scores.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or None
            Feature matrix, or ``None`` when ``similarity_matrix`` is provided.
        y : array-like of shape (n_samples,)
            Label vector aligned with ``X`` or ``similarity_matrix``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Positive-region scores for the fitted samples.
        """
        return self.fit(X, y).positive_region_

    def score_samples(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Return cached positive-region scores for fitted samples.

        The current public scorer computes scores for the fitted similarity
        matrix. The optional ``X`` argument is accepted only for sklearn-like call
        sites and is not used to score unseen samples.

        Parameters
        ----------
        X : ndarray or None, default=None
            Ignored placeholder for sklearn-like call sites.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Fitted positive-region score array.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        check_is_fitted(self, "positive_region_")
        return self.positive_region_

    def as_result(self) -> FuzzyRoughApproximationResult:
        """Return the fitted public approximation result object.

        Returns
        -------
        result : FuzzyRoughApproximationResult
            Cached approximation result from the last ``fit`` call.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called.
        """
        check_is_fitted(self, "result_")
        return self.result_

__all__ = ["FuzzyRoughPositiveRegionScorer"]
