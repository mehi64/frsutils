"""
@file vqrs.py
@brief VQRS (Vaguely Quantified Rough Set) model implementation.

Supports both direct construction and lazy instantiation via config or dictionary.

##############################################
# ✅ Summary of Clean Code and Design Patterns
# - Strategy Pattern: Fuzzy quantifier is configurable
# - Adapter Pattern: Implements to_dict/from_dict serialization
# - Template Method: Inherits from FuzzyRoughModel abstract interface
# - Logger Injection: Supports injected or default logger
# - Fail-Fast Validation: Ensures correct quantifier types
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# Direct instantiation
from FRsutils.core.models.vqrs import VQRS
from FRsutils.core.fuzzy_quantifiers import LinearFuzzyQuantifier
import numpy as np

sim_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
labels = np.array([1, 1])
fq = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
vqrs = VQRS(sim_matrix, labels, fq, fq)

vqrs.lower_approximation()
vqrs.upper_approximation()

# From config
config = {
    "lb_fuzzy_quantifier": {"type": "linear", "alpha": 0.1, "beta": 0.6},
    "ub_fuzzy_quantifier": {"type": "linear", "alpha": 0.2, "beta": 1.0}
}
vqrs2 = VQRS.from_config(config, similarity_matrix=sim_matrix, labels=labels)

# Serialization
model_dict = vqrs.to_dict(include_data=True)
vqrs_restored = VQRS.from_dict(model_dict)
"""


import numpy as np
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel



@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """
    @brief VQRS model for fuzzy rough approximation using fuzzy quantifiers.

    @param similarity_matrix: Pairwise similarity matrix (n x n)
    @param labels: Corresponding label vector (n,)
    @param fuzzy_quantifier_lower: FuzzyQuantifier instance for lower approx
    @param fuzzy_quantifier_upper: FuzzyQuantifier instance for upper approx
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 lb_fuzzy_quantifier: FuzzyQuantifier,
                 ub_fuzzy_quantifier: FuzzyQuantifier,
                 logger=None):
        super().__init__(similarity_matrix, 
                         labels, 
                         logger=logger)
        
        self.validate_params(
            lb_fuzzy_quantifier=lb_fuzzy_quantifier,
            ub_fuzzy_quantifier=ub_fuzzy_quantifier
        )

        self.lb_fuzzy_quantifier = lb_fuzzy_quantifier
        self.ub_fuzzy_quantifier = ub_fuzzy_quantifier
        self.tnorm = tn.MinTNorm()

        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def _interim_calculations(self) -> np.ndarray:
        """
        @brief R_Ay which is then feed into upper and lower quantifiers
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # to ommit the same instace from calculations, we set the main diagonal to 0.0
        # later on the sum operator ignores that.
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)

        # since the similarity matrix has the main diagonal 1.0, the sum needs to cbe reduced by 1.0
        # so we reduce
        denominator = np.sum(self.similarity_matrix, axis=1) - 1.0

        interim = numerator / denominator
        return interim

    def lower_approximation(self) -> np.ndarray:
        """
        @brief Compute the lower approximation using the fuzzy quantifier.

        @return: Lower approximation array (n,)
        """
        return self.lb_fuzzy_quantifier(self._interim_calculations())

    def upper_approximation(self) -> np.ndarray:
        """
        @brief Compute the upper approximation using the fuzzy quantifier.

        @return: Upper approximation array (n,)
        """
        return self.ub_fuzzy_quantifier(self._interim_calculations())

    def to_dict(self, include_data: bool = False) -> dict:
        """
        @brief Serialize the VQS model to a dictionary.

        @param include_data: If True, include similarity_matrix and labels in the output.

        @return: Dictionary representation of the model.
        """
        data = {
            "type": "vqrs",
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.to_dict(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "VQRS":
        """
        @brief Reconstruct an VQRS model from a serialized dictionary.

        @param data: Serialized dictionary (from to_dict)
        @param similarity_matrix: Optional matrix to override or fill in if not in data
        @param labels: Optional label vector to override or fill in if not in data
        @param logger: Optional logger

        @return: VQRS instance
        """
        fq_lower = FuzzyQuantifier.from_dict(data["lb_fuzzy_quantifier"])
        fq_upper = FuzzyQuantifier.from_dict(data["ub_fuzzy_quantifier"])

        # Use matrix and labels from dict if present, else fallback to args
        sim = np.array(data["similarity_matrix"]) if "similarity_matrix" in data else similarity_matrix
        lbl = np.array(data["labels"]) if "labels" in data else labels

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in data or as arguments.")

        return cls(sim, lbl, fq_lower, fq_upper, logger=logger)

    def _get_params(self) -> dict:
        """
        @brief Describe internal lower and upper fuzzy_quantifier parameters.

        @return: Dictionary containing lower and upper fuzzy_quantifier used in vqrs.
        """
        return {
            "tnorm": self.tnorm,
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier,
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "VQRS":
        """
        @brief Create a VQRS instance from a configuration dictionary.

        @param config: Serialized config dict (can include tnorm, fuzzy quantifiers, and optionally data)
        @param similarity_matrix: Optional override for similarity matrix
        @param labels: Optional override for label vector
        @return: VQRS instance
        """
        # Load operators from dict or registry
        ub_fuzzy_quantifier = config.get("ub_fuzzy_quantifier")
        if isinstance(ub_fuzzy_quantifier, dict):
            ub_fuzzy_quantifier = FuzzyQuantifier.from_dict(ub_fuzzy_quantifier)
        elif ub_fuzzy_quantifier is None:
            ub_fuzzy_quantifier_name = config.get("ub_fuzzy_quantifier_name")
            ub_fuzzy_quantifier = FuzzyQuantifier.create(ub_fuzzy_quantifier_name, namespace="ub", **config)

        lb_fuzzy_quantifier = config.get("lb_fuzzy_quantifier")
        if isinstance(lb_fuzzy_quantifier, dict):
            lb_fuzzy_quantifier = FuzzyQuantifier.from_dict(lb_fuzzy_quantifier)
        elif lb_fuzzy_quantifier is None:
            lb_fuzzy_quantifier_name = config.get("lb_fuzzy_quantifier_name")
            lb_fuzzy_quantifier = FuzzyQuantifier.create(lb_fuzzy_quantifier_name, namespace="lb", **config)

        # Handle matrix and labels
        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in config or as arguments.")

        logger = config.get("logger", None)
        return cls(sim, lbl, lb_fuzzy_quantifier, ub_fuzzy_quantifier, logger=logger)
    
    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief validation hook.

        @param kwargs
        """
        fq_l = kwargs.get("lb_fuzzy_quantifier")
        fq_u = kwargs.get("ub_fuzzy_quantifier")

        if fq_l is None or not isinstance(fq_l, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_lower must be a valid FuzzyQuantifier instance.")
        
        if fq_u is None or not isinstance(fq_u, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_upper must be a valid FuzzyQuantifier instance.")

    def describe_params_detailed(self) -> dict:
        return {
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.describe_params_detailed(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.describe_params_detailed()
        }
