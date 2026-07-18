# SPDX-License-Identifier: BSD-3-Clause
"""Base abstractions and shared validation for fuzzy-rough models."""

from abc import abstractmethod
from typing import Any

import numpy as np

from frsutils.utils.base_component_with_logger import BaseComponentWithLogger
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class FuzzyRoughModel(RegistryFactoryMixin, BaseComponentWithLogger):
    """Abstract base class for dense fuzzy-rough approximation models.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Finite fuzzy-relation matrix with values in ``[0, 1]``. Entry
        ``similarity_matrix[i, j]`` represents ``R(x_i, x_j)``.
    labels : ndarray of shape (n_samples,)
        One-dimensional label vector aligned with the matrix rows and columns.
    logger : logging.Logger or None, default=None
        Optional logger used by the component base class.

    Notes
    -----
    The relation orientation is ``rows_are_queries``: row ``i`` is aggregated
    to produce the approximation value returned for sample ``x_i`` and column
    ``j`` contributes evidence from sample ``x_j``. Symmetry and a unit
    diagonal are not required. Each concrete model excludes self-comparisons
    according to its current model contract.
    """

    relation_orientation = "rows_are_queries"

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        logger=None,
    ):
        """Initialize shared dense-model data after validating its contract."""
        BaseComponentWithLogger.__init__(self, logger)
        self.validate_params_base(
            similarity_matrix=similarity_matrix,
            labels=labels,
        )
        self.similarity_matrix = similarity_matrix
        self.labels = labels

    @abstractmethod
    def lower_approximation(self) -> np.ndarray:
        """Compute lower-approximation values for all samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Lower-approximation values in row order.
        """
        raise NotImplementedError("lower_approximation is not implemented")

    @abstractmethod
    def upper_approximation(self) -> np.ndarray:
        """Compute upper-approximation values for all samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Upper-approximation values in row order.
        """
        raise NotImplementedError("upper_approximation is not implemented")

    def signed_boundary(self) -> np.ndarray:
        """Compute the un-clipped signed boundary as upper minus lower.

        Returns
        -------
        ndarray of shape (n_samples,)
            Signed difference between upper and lower approximations.

        Notes
        -----
        Values may be negative when custom component choices do not guarantee
        ``lower <= upper``.
        """
        return self.upper_approximation() - self.lower_approximation()

    def boundary_region(self) -> np.ndarray:
        """Return signed-boundary values through the legacy method name.

        Returns
        -------
        ndarray of shape (n_samples,)
            The same values returned by :meth:`signed_boundary`.
        """
        return self.signed_boundary()

    def positive_region(self) -> np.ndarray:
        """Return positive-region scores using the current lower-score contract.

        Returns
        -------
        ndarray of shape (n_samples,)
            Lower-approximation values.
        """
        return self.lower_approximation()

    @abstractmethod
    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the model configuration and optionally its data.

        Parameters
        ----------
        include_data : bool, default=False
            Include the similarity matrix and labels when True.

        Returns
        -------
        dict
            Serialized model representation.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        data: dict,
        similarity_matrix=None,
        labels=None,
        logger=None,
    ):
        """Reconstruct a model from serialized component configuration.

        Parameters
        ----------
        data : dict
            Serialized model representation.
        similarity_matrix : array-like or None, default=None
            Optional data override for the serialized matrix.
        labels : array-like or None, default=None
            Optional data override for serialized labels.
        logger : logging.Logger or None, default=None
            Optional logger for the reconstructed model.

        Returns
        -------
        FuzzyRoughModel
            Reconstructed concrete model instance.
        """
        raise NotImplementedError

    @classmethod
    def validate_params_base(cls, **kwargs: Any) -> None:
        """Validate the shared dense-model matrix and label contract.

        Parameters
        ----------
        **kwargs : object
            Must contain ``similarity_matrix`` and ``labels``.

        Raises
        ------
        ValueError
            If the matrix or labels violate the dense core-model contract.
        """
        similarity_matrix = kwargs.get("similarity_matrix")
        labels = kwargs.get("labels")

        if similarity_matrix is None or labels is None:
            raise ValueError("similarity_matrix and labels must be provided.")

        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2:
            raise ValueError("similarity_matrix must be a 2D NumPy array.")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square.")
        if similarity_matrix.shape[0] < 2:
            raise ValueError(
                "Fuzzy-rough approximation models require at least two samples."
            )
        try:
            finite_values = np.isfinite(similarity_matrix).all()
        except TypeError as exc:
            raise ValueError(
                "similarity_matrix must contain numeric finite values."
            ) from exc
        if not finite_values:
            raise ValueError("similarity_matrix must contain only finite values.")
        if not ((0.0 <= similarity_matrix).all() and (similarity_matrix <= 1.0).all()):
            raise ValueError("All similarity values must be in the range [0.0, 1.0].")
        if not isinstance(labels, np.ndarray) or labels.ndim != 1:
            raise ValueError("labels must be a 1D array-like vector.")
        if len(labels) != similarity_matrix.shape[0]:
            raise ValueError("Length of labels must match similarity_matrix size.")

    @classmethod
    @abstractmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict):
        """Construct a concrete model from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like or None, default=None
            Optional dense similarity matrix.
        labels : array-like or None, default=None
            Optional label vector.
        **config : dict
            Model-specific flat or nested component configuration.

        Returns
        -------
        FuzzyRoughModel
            Configured concrete model instance.
        """
        raise NotImplementedError("Subclasses must implement from_config().")
