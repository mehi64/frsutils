"""
@file itfrs.py
@brief Implementation of the IT2 Fuzzy Rough Set (ITFRS) approximation model.

Provides a concrete implementation of the lower and upper approximations
using a fuzzy implicator and a T-norm operator over a similarity matrix.

##############################################
# ✅ Quick Summary of Features
# - ITFRS model for fuzzy rough approximation
# - Pluggable architecture for T-norm and Implicator
# - Lower and upper approximation computation
# - Class introspection and serialization support
# - Logger injection via BaseComponentWithLogger

# ✅ Summary Table of Design Principles
# - Strategy Pattern: Uses user-defined T-norm and Implicator strategies
# - Template Method: Inherits abstract methods from BaseFuzzyRoughModel
# - Adapter Pattern: Provides to_dict/from_dict for serialization
# - Clean Code: SRP, DRY, LSP, docstring documentation, fail-fast checks
##############################################
"""

from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp
import numpy as np


@FuzzyRoughModel.register("itfrs")
class ITFRS(FuzzyRoughModel):
    """
    @brief Implicator-TNorm Fuzzy Rough Set approximation model.

    @param similarity_matrix: Precomputed similarity matrix (n x n)
    @param labels: Array of class labels for each instance
    @param tnorm: T-norm operator (object from TNorm)
    @param implicator: Fuzzy implicator operator (object from Implicator)
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 ub_tnorm: tn.TNorm, 
                 lb_implicator: imp.Implicator,
                 logger=None):
        super().__init__(similarity_matrix,
                          labels,
                          logger=logger)
        
        self.validate_params(ub_tnorm=ub_tnorm, 
                             lb_implicator=lb_implicator)

        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator
        
        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def lower_approximation(self) -> np.ndarray:
        """
        @brief Compute the lower approximation using the implicator.

        @return: Lower approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 1.0)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self) -> np.ndarray:
        """
        @brief Compute the upper approximation using the T-norm.

        @return: Upper approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        return np.max(tnorm_vals, axis=1)

    def to_dict(self, include_data: bool = False) -> dict:
        """
        @brief Serialize the ITFRS model to a dictionary.

        @param include_data: If True, include similarity_matrix and labels in the output.

        @return: Dictionary representation of the model.
        """
        data = {
            "type": "itfrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data


    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "ITFRS":
        """
        @brief Reconstruct an ITFRS model from a serialized dictionary.

        @param data: Serialized dictionary (from to_dict)
        @param similarity_matrix: Optional matrix to override or fill in if not in data
        @param labels: Optional label vector to override or fill in if not in data
        @param logger: Optional logger

        @return: ITFRS instance
        """
        
        # Rebuild operators
        tnorm = tn.TNorm.from_dict(data["ub_tnorm"])
        implicator = imp.Implicator.from_dict(data["lb_implicator"])

        # Use matrix and labels from dict if present, else fallback to args
        sim = np.array(data["similarity_matrix"]) if "similarity_matrix" in data else similarity_matrix
        lbl = np.array(data["labels"]) if "labels" in data else labels

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in data or as arguments.")

        return cls(sim, lbl, tnorm, implicator, logger=logger)

    def describe_params_detailed(self) -> dict:
        """
        @brief Describe internal T-norm and implicator parameters.

        @return: Dictionary describing parameters of components.
        """
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed()
        }
    
    def _get_params(self) -> dict:
        """
        @brief Describe internal T-norm and implicator parameters.

        @return: Dictionary containing T-norm and implicator used in itfrs.
        """
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief validation hook.

        @param kwargs
        """
        
        tnrm = kwargs.get("ub_tnorm")
        if tnrm is None or not isinstance(tnrm, tn.TNorm):
            raise ValueError("Parameter 'tnorm' must be provided and be an instance of derived classes from TNorm.")

        impli = kwargs.get("lb_implicator")
        if impli is None or not isinstance(impli, imp.Implicator):
            raise ValueError("Parameter 'implicator' must be provided and be an instance of derived classes from Implicator.")


    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "ITFRS":
        """
        @brief Create an ITFRS instance from a configuration dictionary.

        @param config: Serialized config dict (can include tnorm, implicator, and optionally data)
        @param similarity_matrix: Optional override for similarity matrix
        @param labels: Optional override for label vector
        @return: ITFRS instance
        """

        # Load operators from dict or registry
        ub_tnorm = config.get("ub_tnorm")
        if isinstance(ub_tnorm, dict):
            ub_tnorm = tn.TNorm.from_dict(ub_tnorm)
        elif ub_tnorm is None:
            tn_name =config.get("ub_tnorm_name")
            # tn_name =config["ub_tnorm_name"]
            ub_tnorm = tn.TNorm.create(tn_name, **config)

        lb_implicator = config.get("lb_implicator")
        if isinstance(lb_implicator, dict):
            lb_implicator = imp.Implicator.from_dict(lb_implicator)
        elif lb_implicator is None:
            imp_name = config.get("lb_implicator_name")
            lb_implicator = imp.Implicator.create(imp_name, **config)

        # Handle matrix and labels
        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either as arguments or in the config dictionary.")

        logger = config.get("logger", None)
        return cls(sim, lbl, ub_tnorm, lb_implicator, logger=logger)
