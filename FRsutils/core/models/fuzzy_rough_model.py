"""
@file base_fuzzy_rough_model.py
@brief Base class for fuzzy rough set approximation models.

Defines the abstract contract that all fuzzy rough models must implement,
including lower and upper approximations, boundary and positive regions.

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Template Method Pattern: abstract lower_approximation() and upper_approximation()
# - SRP: Handles only shape/format validation and contract definition
# - LSP: All subclasses can safely extend this without changing client behavior
# - Fail-Fast Validation: Input validation at init
##############################################
"""

from abc import abstractmethod
import numpy as np
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from FRsutils.utils.base_component_with_logger import BaseComponentWithLogger


class FuzzyRoughModel(RegistryFactoryMixin, BaseComponentWithLogger):
    """
    @brief Abstract base class for fuzzy-rough approximation models.

    @param similarity_matrix: Symmetric (n x n) similarity matrix in [0, 1]
    @param labels: Label vector of length n
    """

    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray,
                 logger=None):
        
        BaseComponentWithLogger.__init__(self, logger)
        self.validate_params_base(similarity_matrix=similarity_matrix, 
                                  labels=labels)
        
        self.similarity_matrix = similarity_matrix
        self.labels = labels

    @abstractmethod
    def lower_approximation(self) -> np.ndarray:
        """
        @brief Abstract method to compute lower approximation.

        @return: Array of lower approximation values.
        """
        raise NotImplementedError("lower_approximation is not implemented")

    @abstractmethod
    def upper_approximation(self) -> np.ndarray:
        """
        @brief Abstract method to compute upper approximation.

        @return: Array of upper approximation values.
        """
        raise NotImplementedError("upper_approximation is not implemented")


    def boundary_region(self) -> np.ndarray:
        """
        @brief Compute the boundary region (upper - lower).

        @return: Difference of upper and lower approximation arrays.
        """
        return self.upper_approximation() - self.lower_approximation()

    def positive_region(self) -> np.ndarray:
        """
        @brief Return the positive region (same as lower approx).

        @return: Lower approximation values.
        """
        return self.lower_approximation()

    @abstractmethod
    def to_dict(self, include_data: bool = False) -> dict:
        """
        @param include_data: If True, include similarity matrix and labels in output.
        @return: Serialized dictionary representation of the model.
        """
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None):
        """
        @param data: Serialized dict.
        @param similarity_matrix: Optional matrix override.
        @param labels: Optional label override.
        @param logger: Optional logger override.
        @return: Reconstructed model instance.
        """
        raise NotImplementedError       

    @classmethod
    def validate_params_base(cls, **kwargs):
        """
        @brief Validates similarity matrix and labels
        """
        similarity_matrix = kwargs.get("similarity_matrix")
        labels = kwargs.get("labels")

        if ((similarity_matrix is None) or (labels is None)):
            raise ValueError("similarity_matrix and labels must be provided.")
        
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2:
            raise ValueError("similarity_matrix must be a 2D NumPy array.")
        if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square.")
        if not ((0.0 <= similarity_matrix).all() and (similarity_matrix <= 1.0).all()):
            raise ValueError("All similarity values must be in the range [0.0, 1.0].")
        if len(labels) != similarity_matrix.shape[0]:
            raise ValueError("Length of labels must match similarity_matrix size.")

    @classmethod
    @abstractmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict):
        """
        @brief Alternate constructor from flat config dict.

        @param config: Config dictionary.
        @param similarity_matrix: Optional matrix.
        @param labels: Optional labels.
        @param logger: Optional logger.
        @return: Model instance.
        """
        raise NotImplementedError("Subclasses must implement from_config().")
