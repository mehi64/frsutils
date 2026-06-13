# SPDX-License-Identifier: BSD-3-Clause
"""Similarity functions and dense similarity-matrix construction utilities.

This module belongs to the core fuzzy-rough computation layer.
"""

from typing import Callable, Optional, Dict, Any
import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested
from FRsutils.core.tnorms import TNorm


class Similarity(RegistryFactoryMixin):
    """
    @brief Abstract base class for all similarity functions.

    Provides a unified interface and registry for defining similarity measures.
    Dense public calls remain NumPy-based, while `compute_backend` gives
    similarity engines a formula-level backend hook.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        @brief Compute similarity between two vectors or pairwise feature arrays.

        @param x: Feature vector or pairwise-compatible NumPy array.
        @param y: Feature vector or pairwise-compatible NumPy array.
        @return: Similarity score or matrix.
        """
        diff = x - y
        self._validate_diff(diff)

        if diff.ndim == 0:
            return self._compute(np.array([[diff]]))[0, 0]
        if diff.ndim == 1:
            diff = diff[:, None] - diff[None, :]
        return self._compute(diff)

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation for subclass-specific settings.
        """
        raise NotImplementedError("all subclasses must implement validate_params")

    def _validate_diff(self, diff: np.ndarray):
        """
        @brief Ensure the input is a 2D NumPy array of pairwise differences.
        """
        if not isinstance(diff, np.ndarray):
            raise TypeError("Input 'diff' must be a NumPy array.")
        self._validate_backend_diff(diff)

    def _validate_backend_diff(self, diff: Any) -> None:
        """
        @brief Validate a NumPy/CuPy-like pairwise difference matrix.

        @param diff: Backend array with an `ndim` attribute.
        @raises ValueError: If diff is not two-dimensional.
        """
        if not hasattr(diff, "ndim") or diff.ndim != 2:
            raise ValueError("Expected a 2D pairwise difference matrix.")

    @abstractmethod
    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """
        @brief Compute the similarity given NumPy pairwise differences.

        @param diff: Pairwise difference matrix (n, n)
        @return: Similarity values.
        """
        raise NotImplementedError("all subclasses must implement _compute()")

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """
        @brief Compute similarity from pairwise differences using an array namespace.

        Subclasses with backend-safe vectorized formulas override this method.
        The default implementation keeps NumPy behavior for legacy subclasses and
        fails explicitly for non-NumPy namespaces.

        @param diff: NumPy/CuPy-like pairwise difference matrix.
        @param xp: Array namespace, usually `numpy` or `cupy`.
        @return: Backend array of similarity values.
        @raises NotImplementedError: If the subclass has no backend formula.
        """
        self._validate_backend_diff(diff)
        if xp is np:
            return self._compute(np.asarray(diff))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a backend-aware similarity formula."
        )


@Similarity.register("linear")
class LinearSimilarity(Similarity):
    """
    @brief Linear similarity function: sim = max(0, 1 - |x - y|) for normalized inputs.
    """

    def __init__(self):
        self.validate_params()

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """@brief Backend-aware linear similarity formula."""
        self._validate_backend_diff(diff)
        return xp.maximum(0.0, 1.0 - xp.abs(diff))

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation.
        """
        pass

    def _get_params(self) -> dict:
        """
        @brief No parameters.
        """
        return {}


@Similarity.register("gaussian", "gauss")
class GaussianSimilarity(Similarity):
    """
    @brief Gaussian similarity: sim = exp(-(x - y)^2 / (2 * sigma^2)).

    @param sigma: Standard deviation for the Gaussian kernel.
    """

    def __init__(self, sigma: float = 0.1):
        self.validate_params(sigma=sigma)
        self.sigma = sigma

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """@brief Backend-aware Gaussian similarity formula."""
        self._validate_backend_diff(diff)
        return xp.exp(-((diff ** 2) / (2.0 * self.sigma ** 2)))

    def _get_params(self) -> dict:
        return {"sigma": self.sigma}

    @classmethod
    def validate_params(cls, **kwargs):
        sigma = kwargs.get("sigma")
        if sigma is None or not isinstance(sigma, (float, int)) or sigma <= 0:
            raise ValueError("Parameter 'sigma' must be provided and be a positive number.")


def calculate_similarity_matrix(
    X: np.ndarray,
    similarity_func: Similarity,
    tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    @brief Compute a dense pairwise similarity matrix from input features.

    @param X: Normalized input matrix of shape (n_samples, n_features).
    @param similarity_func: Instance of a Similarity subclass.
    @param tnorm: Binary T-norm operator (e.g. min, product).
    @return: Similarity matrix (n_samples, n_samples).
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array")
    if X.size == 0:
        return np.zeros((0, 0))
    n_samples, n_features = X.shape
    sim_matrix = np.ones((n_samples, n_samples), dtype=np.float64)

    for k in range(n_features):
        col = X[:, k].reshape(-1, 1)
        sim_k = similarity_func(col, col.T)
        sim_matrix = tnorm(sim_matrix, sim_k)

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def build_similarity_matrix(X: np.ndarray, config: Optional[Dict[str, Any]] = None, **kwargs) -> np.ndarray:
    """
    @brief Build a pairwise similarity matrix from input features and config.

    @param X: Normalized input matrix of shape (n_samples, n_features).
    @param config: Optional nested config. If omitted, kwargs are normalized.
    @param kwargs: Flat config according to the naming standard.
    @return: Pairwise similarity matrix (n x n).
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
