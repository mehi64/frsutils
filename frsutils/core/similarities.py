# SPDX-License-Identifier: BSD-3-Clause
"""Similarity functions and dense similarity-matrix construction utilities.

This module belongs to the core fuzzy-rough computation layer.
"""

from numbers import Real
from typing import Callable, Optional, Dict, Any
import numpy as np
from abc import abstractmethod
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from frsutils.utils.init_helpers import normalize_flat_config_to_nested
from frsutils.core.tnorms import TNorm


class Similarity(RegistryFactoryMixin):
    """Abstract base class for all similarity functions.
    
    Provides a unified interface and registry for defining similarity measures.
    Dense public calls remain NumPy-based, while `compute_backend` gives
    similarity engines a formula-level backend hook.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """Compute scalar or pairwise feature-level similarity values.
                
                Parameters
                ----------
                x : array-like
                    Scalar or pairwise-compatible array.
                y : array-like
                    Scalar or pairwise-compatible array.
                
                Returns
                -------
                float or np.ndarray
                    Scalar similarity or a two-dimensional pairwise array.

                Raises
                ------
                ValueError
                    If the broadcast difference is one- or three-dimensional.
                
        """
        diff = np.asarray(x) - np.asarray(y)
        if diff.ndim == 0:
            return float(self._compute(np.asarray(diff).reshape(1, 1))[0, 0])
        if diff.ndim != 2:
            raise ValueError(
                "Similarity calls require scalar inputs or a 2D pairwise "
                "difference matrix."
            )
        return self._compute(diff)

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """Optional parameter validation for subclass-specific settings."""
        raise NotImplementedError("all subclasses must implement validate_params")

    def _validate_diff(self, diff: np.ndarray):
        """Ensure the input is a 2D NumPy array of pairwise differences."""
        if not isinstance(diff, np.ndarray):
            raise TypeError("Input 'diff' must be a NumPy array.")
        self._validate_backend_diff(diff)

    def _validate_backend_diff(self, diff: Any) -> None:
        """Validate a NumPy/CuPy-like pairwise difference matrix.
                
                Parameters
                ----------
                diff : Any
                    Backend array with an `ndim` attribute.
                
                Raises
                ------
                ValueError
                    If diff is not two-dimensional.
                
        """
        if not hasattr(diff, "ndim") or diff.ndim != 2:
            raise ValueError("Expected a 2D pairwise difference matrix.")

    @abstractmethod
    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute the similarity given NumPy pairwise differences.
                
                Parameters
                ----------
                diff : np.ndarray
                    Pairwise difference matrix (n, n)
                
                Returns
                -------
                np.ndarray
                    Similarity values.
                
        """
        raise NotImplementedError("all subclasses must implement _compute()")

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Compute similarity from pairwise differences using an array namespace.
                
                Subclasses with backend-safe vectorized formulas override this method.
                The default implementation keeps NumPy behavior for legacy subclasses and
                fails explicitly for non-NumPy namespaces.
                
                Parameters
                ----------
                diff : Any
                    NumPy/CuPy-like pairwise difference matrix.
                xp : Any
                    Array namespace, usually `numpy` or `cupy`.
                
                Returns
                -------
                object
                    Backend array of similarity values.
                
                Raises
                ------
                NotImplementedError
                    If the subclass has no backend formula.
                
        """
        self._validate_backend_diff(diff)
        if xp is np:
            return self._compute(np.asarray(diff))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a backend-aware similarity formula."
        )


@Similarity.register("linear")
class LinearSimilarity(Similarity):
    """Linear similarity function: sim = max(0, 1 - |x - y|) for normalized inputs."""

    def __init__(self):
        """Initialize the LinearSimilarity instance."""
        self.validate_params()

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute similarity values for validated inputs."""
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Backend-aware linear similarity formula."""
        self._validate_backend_diff(diff)
        return xp.maximum(0.0, 1.0 - xp.abs(diff))

    @classmethod
    def validate_params(cls, **kwargs):
        """This class does not need parameter validation."""
        pass

    def _get_params(self) -> dict:
        """No parameters."""
        return {}


@Similarity.register("gaussian", "gauss")
class GaussianSimilarity(Similarity):
    """Gaussian similarity: sim = exp(-(x - y)^2 / (2 * sigma^2)).
    
    Parameters
    ----------
    sigma : object
        Standard deviation for the Gaussian kernel.
    """

    def __init__(self, sigma: float = 0.1):
        """Initialize the GaussianSimilarity instance."""
        self.validate_params(sigma=sigma)
        self.sigma = sigma

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute similarity values for validated inputs."""
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Backend-aware Gaussian similarity formula."""
        self._validate_backend_diff(diff)
        return xp.exp(-((diff ** 2) / (2.0 * self.sigma ** 2)))

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {"sigma": self.sigma}

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        sigma = kwargs.get("sigma")
        if (
            sigma is None
            or isinstance(sigma, (bool, np.bool_))
            or not isinstance(sigma, Real)
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError(
                "Parameter 'sigma' must be provided as a finite positive number "
                "and must not be boolean."
            )


def calculate_similarity_matrix(
    X: np.ndarray,
    similarity_func: Similarity,
    tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """Compute a dense pairwise similarity matrix from input features.
        
        Parameters
        ----------
        X : np.ndarray
            Normalized input matrix of shape (n_samples, n_features).
        similarity_func : Similarity
            Instance of a Similarity subclass.
        tnorm : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Binary T-norm operator (e.g. min, product).
        
        Returns
        -------
        np.ndarray
            Similarity matrix (n_samples, n_samples).
        
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array")
    try:
        finite_values = np.isfinite(X).all()
    except TypeError as exc:
        raise ValueError("X must contain only finite numeric values.") from exc
    if not finite_values:
        raise ValueError("X must contain only finite numeric values.")

    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros((0, 0))

    sim_matrix = np.ones((n_samples, n_samples), dtype=np.float64)

    for k in range(n_features):
        col = X[:, k].reshape(-1, 1)
        sim_k = similarity_func(col, col.T)
        sim_matrix = tnorm(sim_matrix, sim_k)

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def build_similarity_matrix(X: np.ndarray, config: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
    """Build a pairwise similarity matrix from input features and config.
        
        Parameters
        ----------
        X : np.ndarray
            Normalized input matrix of shape (n_samples, n_features).
        config : Optional[Dict[str, Any]]
            Optional nested config. If omitted, kwargs are normalized.
        kwargs : object
            Flat config according to the naming standard.
        
        Returns
        -------
        np.ndarray
            Pairwise similarity matrix (n x n).
        
    """
    nested = config if isinstance(config, dict) else normalize_flat_config_to_nested(kwargs)

    sim_cfg = nested.get("similarity", {}) if isinstance(nested, dict) else {}
    tnorm_cfg = nested.get("similarity_tnorm", {}) if isinstance(nested, dict) else {}

    similarity_type = sim_cfg.get("name") or kwargs.get("similarity") or "gaussian"
    similarity_params = dict(sim_cfg.get("params") if isinstance(sim_cfg.get("params"), dict) else {})
    if str(similarity_type).lower() in {"gaussian", "gauss"} and "sigma" in kwargs and "sigma" not in similarity_params:
        # Backward-compatible legacy flat alias used by older tests/examples.
        similarity_params["sigma"] = kwargs["sigma"]

    tnorm_type = tnorm_cfg.get("name") or kwargs.get("similarity_tnorm") or "minimum"
    tnorm_params = tnorm_cfg.get("params") if isinstance(tnorm_cfg.get("params"), dict) else {}

    similarity_func = Similarity.create(similarity_type, **similarity_params)
    tnorm_func = TNorm.create(tnorm_type, **tnorm_params)

    return calculate_similarity_matrix(X, similarity_func, tnorm_func)


# CosineSimilarity
# ExponentialSimilarity
# YagerSimilarity
# HammingSimilarity
# DiceSimilarity
# JaccardSimilarity
# TverskySimilarity
