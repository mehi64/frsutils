# SPDX-License-Identifier: BSD-3-Clause
"""Base abstractions for fuzzy-rough approximation models.

This module belongs to the core fuzzy-rough computation layer.
"""

from abc import abstractmethod
import numpy as np
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from frsutils.utils.base_component_with_logger import BaseComponentWithLogger


class FuzzyRoughModel(RegistryFactoryMixin, BaseComponentWithLogger):
    """Abstract base class for fuzzy-rough approximation models.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Finite fuzzy-relation matrix with values in the interval [0, 1].
    labels : ndarray of shape (n_samples,)
        One-dimensional label vector aligned with the matrix.

    Notes
    -----
    Core model construction requires at least two samples. The low-level model
    contract does not require symmetry or a unit diagonal, which allows direct
    use of finite fuzzy relations as well as conventional similarity matrices.
    Each model excludes self-comparisons explicitly where its definition
    requires it.
    """

    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray,
                 logger=None):
        
        """Initialize the FuzzyRoughModel instance."""
        BaseComponentWithLogger.__init__(self, logger)
        self.validate_params_base(similarity_matrix=similarity_matrix, 
                                  labels=labels)
        
        self.similarity_matrix = similarity_matrix
        self.labels = labels

    @abstractmethod
    def lower_approximation(self) -> np.ndarray:
        """Abstract method to compute lower approximation.
                
                Returns
                -------
                np.ndarray
                    Array of lower approximation values.
                
        """
        raise NotImplementedError("lower_approximation is not implemented")

    @abstractmethod
    def upper_approximation(self) -> np.ndarray:
        """Abstract method to compute upper approximation.
                
                Returns
                -------
                np.ndarray
                    Array of upper approximation values.
                
        """
        raise NotImplementedError("upper_approximation is not implemented")


    def boundary_region(self) -> np.ndarray:
        """Compute the boundary region (upper - lower).
                
                Returns
                -------
                np.ndarray
                    Difference of upper and lower approximation arrays.
                
        """
        return self.upper_approximation() - self.lower_approximation()

    def positive_region(self) -> np.ndarray:
        """Return the positive region (same as lower approx).
                
                Returns
                -------
                np.ndarray
                    Lower approximation values.
                
        """
        return self.lower_approximation()

    @abstractmethod
    def to_dict(self, include_data: bool = False) -> dict:
        """Parameters
                ----------
                include_data : bool
                    If True, include similarity matrix and labels in output.
                
                Returns
                -------
                dict
                    Serialized dictionary representation of the model.
                
        """
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None):
        """Parameters
                ----------
                data : dict
                    Serialized dict.
                similarity_matrix : object
                    Optional matrix override.
                labels : object
                    Optional label override.
                logger : object
                    Optional logger override.
                
                Returns
                -------
                object
                    Reconstructed model instance.
                
        """
        raise NotImplementedError       

    @classmethod
    def validate_params_base(cls, **kwargs):
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
        """Alternate constructor from flat config dict.
                
                Parameters
                ----------
                config : dict
                    Config dictionary.
                similarity_matrix : object
                    Optional matrix.
                labels : object
                    Optional labels.
                logger : object
                    Optional logger.
                
                Returns
                -------
                object
                    Model instance.
                
        """
        raise NotImplementedError("Subclasses must implement from_config().")
