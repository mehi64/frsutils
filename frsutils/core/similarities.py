# SPDX-License-Identifier: BSD-3-Clause
"""Similarity functions and dense similarity-matrix construction utilities."""

from abc import abstractmethod
from numbers import Real
from typing import Any, Callable, Mapping

import numpy as np

from frsutils.core.tnorms import TNorm
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from frsutils.utils.init_helpers import normalize_flat_config_to_nested


class Similarity(RegistryFactoryMixin):
    """Abstract base class for registered feature-level similarity functions.

    Dense calls use NumPy. Backend-aware subclasses expose equivalent formulas
    through :meth:`compute_backend` for similarity-engine execution.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float | np.ndarray:
        """Compute scalar or pairwise feature-level similarities.

        Parameters
        ----------
        x, y : array-like
            Scalar inputs or broadcast-compatible arrays whose difference is a
            two-dimensional pairwise matrix.

        Returns
        -------
        float or ndarray
            Scalar similarity or a two-dimensional pairwise similarity array.

        Raises
        ------
        ValueError
            If the broadcast difference is not scalar or two-dimensional.
        """
        diff = np.asarray(x) - np.asarray(y)
        if diff.ndim == 0:
            return float(self._compute(diff.reshape(1, 1))[0, 0])
        if diff.ndim != 2:
            raise ValueError(
                "Similarity calls require scalar inputs or a 2D pairwise "
                "difference matrix."
            )
        return self._compute(diff)

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate subclass-specific constructor parameters."""
        raise NotImplementedError("all subclasses must implement validate_params")

    def _validate_diff(self, diff: np.ndarray) -> None:
        """Validate a NumPy pairwise-difference matrix."""
        if not isinstance(diff, np.ndarray):
            raise TypeError("Input 'diff' must be a NumPy array.")
        self._validate_backend_diff(diff)

    def _validate_backend_diff(self, diff: Any) -> None:
        """Validate a NumPy/CuPy-like pairwise-difference matrix.

        Parameters
        ----------
        diff : array-like
            Backend array exposing an ``ndim`` attribute.

        Raises
        ------
        ValueError
            If ``diff`` is not two-dimensional.
        """
        if not hasattr(diff, "ndim") or diff.ndim != 2:
            raise ValueError("Expected a 2D pairwise difference matrix.")

    @abstractmethod
    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute NumPy similarities from validated pairwise differences.

        Parameters
        ----------
        diff : ndarray of shape (n_rows, n_columns)
            Pairwise feature differences.

        Returns
        -------
        ndarray of shape (n_rows, n_columns)
            Pairwise feature similarities.
        """
        raise NotImplementedError("all subclasses must implement _compute()")

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Compute similarities using a NumPy-compatible array namespace.

        Parameters
        ----------
        diff : array-like of shape (n_rows, n_columns)
            Pairwise feature differences.
        xp : module, default=numpy
            Array namespace, normally NumPy or CuPy.

        Returns
        -------
        array-like
            Pairwise similarities owned by ``xp``.

        Raises
        ------
        NotImplementedError
            If a subclass has no non-NumPy backend formula.
        """
        self._validate_backend_diff(diff)
        if xp is np:
            return self._compute(np.asarray(diff))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a backend-aware "
            "similarity formula."
        )


@Similarity.register("linear")
class LinearSimilarity(Similarity):
    """Linear similarity ``max(0, 1 - abs(x - y))``."""

    def __init__(self):
        """Initialize a parameterless linear similarity."""
        self.validate_params()

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute linear similarities for NumPy pairwise differences."""
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Compute linear similarities using the supplied array namespace."""
        self._validate_backend_diff(diff)
        return xp.maximum(0.0, 1.0 - xp.abs(diff))

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate the parameterless constructor contract."""

    def _get_params(self) -> dict:
        """Return the parameterless serialization payload."""
        return {}


@Similarity.register("gaussian", "gauss")
class GaussianSimilarity(Similarity):
    """Gaussian similarity ``exp(-diff**2 / (2 * sigma**2))``.

    Parameters
    ----------
    sigma : float, default=0.1
        Finite positive Gaussian scale parameter.
    """

    def __init__(self, sigma: float = 0.1):
        """Initialize Gaussian similarity with a validated scale."""
        self.validate_params(sigma=sigma)
        self.sigma = sigma

    def _compute(self, diff: np.ndarray) -> np.ndarray:
        """Compute Gaussian similarities for NumPy pairwise differences."""
        self._validate_diff(diff)
        return self.compute_backend(diff, xp=np)

    def compute_backend(self, diff: Any, *, xp: Any = np):
        """Compute Gaussian similarities using the supplied array namespace."""
        self._validate_backend_diff(diff)
        return xp.exp(-((diff**2) / (2.0 * self.sigma**2)))

    def _get_params(self) -> dict:
        """Return the Gaussian scale for serialization."""
        return {"sigma": self.sigma}

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate the Gaussian scale parameter.

        Raises
        ------
        ValueError
            If ``sigma`` is boolean, non-numeric, non-finite, or not positive.
        """
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
    tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Compute a dense pairwise similarity matrix from feature data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Finite numeric feature matrix.
    similarity_func : Similarity
        Feature-level similarity component.
    tnorm : callable
        T-norm used to aggregate feature-level similarities.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Dense pairwise similarity matrix with a unit diagonal.

    Raises
    ------
    ValueError
        If ``X`` is not a finite two-dimensional NumPy array.
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
    for feature_index in range(n_features):
        column = X[:, feature_index].reshape(-1, 1)
        feature_similarity = similarity_func(column, column.T)
        sim_matrix = tnorm(sim_matrix, feature_similarity)

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix


def build_similarity_matrix(
    X: np.ndarray,
    config: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Build a dense pairwise similarity matrix from configuration.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Finite numeric feature matrix.
    config : mapping or None, default=None
        Optional nested FRsutils configuration. When omitted, flat keyword
        arguments are normalized into the nested internal representation.
    **kwargs : object
        Flat similarity and similarity-T-norm parameters.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Dense pairwise similarity matrix.
    """
    nested = config if isinstance(config, Mapping) else normalize_flat_config_to_nested(kwargs)
    similarity_config = nested.get("similarity", {})
    tnorm_config = nested.get("similarity_tnorm", {})

    similarity_type = similarity_config.get("name") or kwargs.get("similarity") or "gaussian"
    similarity_params = dict(
        similarity_config.get("params")
        if isinstance(similarity_config.get("params"), Mapping)
        else {}
    )
    if (
        str(similarity_type).lower() in {"gaussian", "gauss"}
        and "sigma" in kwargs
        and "sigma" not in similarity_params
    ):
        similarity_params["sigma"] = kwargs["sigma"]

    tnorm_type = tnorm_config.get("name") or kwargs.get("similarity_tnorm") or "minimum"
    tnorm_params = (
        tnorm_config.get("params")
        if isinstance(tnorm_config.get("params"), Mapping)
        else {}
    )

    similarity_func = Similarity.create(similarity_type, **similarity_params)
    tnorm_func = TNorm.create(tnorm_type, **tnorm_params)
    return calculate_similarity_matrix(X, similarity_func, tnorm_func)
