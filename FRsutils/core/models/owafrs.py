"""
@file owafrs.py
@brief OWAFRS (Ordered Weighted Averaging Fuzzy Rough Set) model implementation.

Extends fuzzy rough approximation using OWA weights for more flexible decision regions.

##############################################
# ✅ Quick Summary of Features
# - OWA-based lower and upper approximations
# - Linear weighting strategy for aggregation
# - Support for vectorized similarity and label matrices
# - Pluggable architecture for T-norms and implicators

# ✅ Design Patterns & Principles Used
# - Strategy: Delegates to provided T-norm and Implicator strategies
# - Template Method: Extends abstract base for fuzzy rough models
# - Adapter: `to_dict()` / `from_dict()` for serialization
# - Clean Code: SRP, fail-fast checks, structured docs, LSP
##############################################
"""

from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
import FRsutils.core.tnorms as tn
import FRsutils.core.owa_weights as owa_weights
import FRsutils.core.implicators as imp
from FRsutils.utils.logger.logger_util import get_logger
import numpy as np


@FuzzyRoughModel.register("owafrs")
class OWAFRS(FuzzyRoughModel):
    """
    @brief Ordered Weighted Averaging Fuzzy Rough Sets (OWAFRS) approximation model.
    """
    def __init__(self,
                 similarity_matrix: np.ndarray,
                 labels: np.ndarray,
                 ub_tnorm: tn.TNorm,
                 lb_implicator: imp.Implicator,
                 ub_owa_method: owa_weights.OWAWeights,
                 lb_owa_method: owa_weights.OWAWeights,
                 logger=None):
        
        super().__init__(similarity_matrix,
                          labels,
                          logger=logger)
        self.validate_params(ub_tnorm=ub_tnorm, 
                             lb_implicator=lb_implicator,
                             ub_owa_method=ub_owa_method,
                             lb_owa_method=lb_owa_method)

        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator

        self.ub_owa_method = ub_owa_method
        self.lb_owa_method = lb_owa_method

        n = len(labels)

        # n-1 because the same instance is not included in calculations
        self.lb_owa_weights = lb_owa_method.weights(n-1, order='asc')
        self.ub_owa_weights = ub_owa_method.weights(n-1, order='desc')

        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def lower_approximation(self) -> np.ndarray:
        """
        @brief Compute the lower approximation using the owa.

        @return: Lower approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        
        # to omitt the same instance from calculations, the
        # main diagonal is set to 0.0. then each row is sorted in descending order
        # then the last row is sliced because the last row always contains 0.0.
        # finally, the matrix is multiplied by the owa weights.
        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.lb_owa_weights)


    def upper_approximation(self) -> np.ndarray:
        """
        @brief Compute the upper approximation using the T-norm.

        @return: Upper approximation array (n,)
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        
        # to omitt the same instance from calculations, the
        # main diagonal is set to 0.0. then each row is sorted in descending order
        # then the last row is sliced because the last row always contains 0.0.
        # finally, the matrix is multiplied by the owa weights.
        np.fill_diagonal(tnorm_vals, 0.0)
        sorted_matrix = np.sort(tnorm_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.ub_owa_weights)

    def to_dict(self, include_data: bool = False) -> dict:
        """
        @brief Serialize the OWAFRS model to a dictionary.

        @param include_data: If True, include similarity_matrix and labels in the output.

        @return: Dictionary representation of the model.
        """
        data = {
            "type": "owafrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict(),
            "ub_owa_method": self.ub_owa_method.to_dict(),
            "lb_owa_method": self.lb_owa_method.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data
    
    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "OWAFRS":
        """
        @brief Reconstruct an OWAFRS model from a serialized dictionary.

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

        ub_owa_method = owa_weights.OWAWeights.from_dict(data["ub_owa_method"])
        lb_owa_method = owa_weights.OWAWeights.from_dict(data["lb_owa_method"])
        
        return cls(sim, lbl, tnorm, implicator, ub_owa_method, lb_owa_method, logger=logger)

   
    def describe_params_detailed(self) -> dict:
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed(),
            "ub_owa_method": self.ub_owa_method.describe_params_detailed(),
            "lb_owa_method": self.lb_owa_method.describe_params_detailed()
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

        ub_owa_method = kwargs.get("ub_owa_method")
        if ub_owa_method is None or not isinstance(ub_owa_method, owa_weights.OWAWeights):
            raise ValueError("Parameter 'ub_owa_method' must be provided and be an instance of derived classes from OWAWeights.")
        
        lb_owa_method = kwargs.get("lb_owa_method")
        if lb_owa_method is None or not isinstance(lb_owa_method, owa_weights.OWAWeights):
            raise ValueError("Parameter 'lb_owa_method' must be provided and be an instance of derived classes from OWAWeights.")

    def _get_params(self) -> dict:
        """
        @brief Describe internal parameters.

        @return: Dictionary containing T-norm, implicator and owa weights used in owafrs.
        """
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "ub_owa_method":self.ub_owa_method,
            "lb_owa_method":self.lb_owa_method,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "OWAFRS":
        """
        @brief Create an OWAFRS instance from a configuration dictionary.

        @param config: Serialized config dict (can include tnorm, implicator, owa weighhts for upper and lower, and optionally data)
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

        lb_owa_method = config.get("lb_owa_method")
        if isinstance(lb_owa_method, dict):
            lb_owa_method = owa_weights.OWAWeights.from_dict(lb_owa_method)
        elif lb_owa_method is None:
            lb_owa_name = config.get("lb_owa_method_name")
            lb_owa_method = owa_weights.OWAWeights.create(lb_owa_name, **config)

        ub_owa_method = config.get("ub_owa_method")
        if isinstance(ub_owa_method, dict):
            ub_owa_method = owa_weights.OWAWeights.from_dict(ub_owa_method)
        elif ub_owa_method is None:
            ub_owa_name = config.get("ub_owa_method_name")
            ub_owa_method = owa_weights.OWAWeights.create(ub_owa_name, **config)

        # Handle matrix and labels
        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either as arguments or in the config dictionary.")

        logger = config.get("logger", None)


        return cls(sim, lbl, ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method, logger=logger)

