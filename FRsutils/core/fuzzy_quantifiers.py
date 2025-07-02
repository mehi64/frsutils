
"""
@file fuzzy_quantifiers.py
@brief Framework for parameterized fuzzy quantifiers used in fuzzy logic systems.

Supports registration, instantiation via alias, and computation of linear and quadratic fuzzy quantifiers.

##############################################
# ✅ Summary of Design Principles & Clean Code
# - Registry Pattern: supports alias-based instantiation via decorators
# - Factory Method: `create()` builds objects from name + params
# - Strategy Pattern: each subclass (Linear, Quadratic) defines specific quantifier logic
# - Adapter Pattern: `to_dict()` / `from_dict()` for serialization
# - Fail-Fast Validation: type and range checks in validate_params
# - LSP: all quantifiers are substitutable via the same base interface
# - SRP: each class handles only its quantification logic
##############################################

@example
>>> fq = FuzzyQuantifier.create("linear", alpha=0.2, beta=0.6)
>>> fq(np.array([0.1, 0.3, 0.7]))
array([0. , 0.25, 1. ])
"""

import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from FRsutils.utils.validation_utils.validation_utils import validate_range_0_1

class FuzzyQuantifier(RegistryFactoryMixin):
    """
    @brief Abstract base class for fuzzy quantifiers.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        @brief Computes the fuzzy membership value(s) for input.

        @param x: Scalar or array of values in [0, 1]
        @return: Scalar or array of membership degrees
        """
        raise NotImplementedError("all subclasses must implement __call__")


    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief Validates the alpha and beta parameters.

        Ensures:
        - Both are floats or ints
        - 0 <= alpha < beta <= 1

        @throws ValueError on invalid parameters
        """
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")

        if alpha is None or not isinstance(alpha, (float, int)):
            raise ValueError(f"Missing or invalid parameter: {alpha}. It must be provided and be an int or a float number")
        if beta is None or not isinstance(beta, (float, int)):
            raise ValueError(f"Missing or invalid parameter: {beta}. It must be provided and be an int or a float number")
        if not (0 <= alpha < beta <= 1):
            raise ValueError("Require 0 <= alpha < beta <= 1")

    def _get_params(self) -> dict:
        """
        @brief Returns parameters for serialization

        @return: Dict with 'alpha' and 'beta'
        """
        return {"alpha": self.alpha,
                "beta": self.beta,
                "validate_inputs" : self.validate_inputs}


    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyQuantifier":
        """
        @brief Instantiates a quantifier from dictionary config.

        @param data: Dictionary with 'type', 'alpha', and 'beta'
        @return: Constructed FuzzyQuantifier object
        """
        q_type = data["type"]
        return cls.create(q_type, alpha=data["alpha"], beta=data["beta"])


@FuzzyQuantifier.register("linear")
class LinearFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Linear fuzzy quantifier: piecewise linear membership.

    Defined as:
    Q(x) = 0            if x <= alpha  
           1            if x >= beta  
           (x - alpha)/(beta - alpha) otherwise
    """

    def __init__(self, 
                 alpha: float, 
                 beta: float,
                 validate_inputs: bool = True):
        """
        @brief constructs a linear fuzzy quantifier

        @param alpha: Threshold for the left edge of the membership function
        @param beta: Threshold for the right edge of the membership function
        @param validate_inputs: Whether to validate inputs per call 
        of the fuzzy quantifier(default: True)

        NOTE: validate_inputs checks the correctness of input data (np.ndarray) and
        it is different than validate_params() which checks the correctness of the
        parameters (alpha and beta).
        
        @return: nothing
        """
        self.validate_params(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        self.validate_inputs = validate_inputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if self.validate_inputs:
            validate_range_0_1(x)

        return np.where(x <= self.alpha, 0.0,
                        np.where(x >= self.beta, 1.0,
                                 (x - self.alpha) / (self.beta - self.alpha)))

    def to_dict(self) -> dict:
        """
        @brief Serialize instance to dict

        @return: Dict with type, alpha, and beta
        """
        return {"type": 'linear',
                "alpha": self.alpha,
                "beta": self.beta}


@FuzzyQuantifier.register("quadratic", "quad")
class QuadraticFuzzyQuantifier(FuzzyQuantifier):
    """
    @brief Quadratic fuzzy quantifier: smooth parabolic membership.

    Defined as:
    Q(x) =  0                                     if x <= alpha  
            2*((x-alpha)/(beta-alpha))^2          if alpha < x <= mid  
            1 - 2*((x-beta)/(beta-alpha))^2       if mid < x <= beta  
            1                                     if x > beta
    """

    def __init__(self, 
                 alpha: float, 
                 beta: float,
                 validate_inputs: bool = True):
        """
        @brief constructs a linear fuzzy quantifier

        @param alpha: Threshold for the left edge of the membership function
        @param beta: Threshold for the right edge of the membership function
        @param validate_inputs: Whether to validate inputs per call 
        of the fuzzy quantifier(default: True)
        
        NOTE: validate_inputs checks the correctness of input data (np.ndarray) and
        it is different than validate_params() which checks the correctness of the
        parameters (alpha and beta).
        
        @return: nothing
        """
        self.validate_params(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        self.validate_inputs = validate_inputs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)

        if self.validate_inputs:
             validate_range_0_1(x)

        mid = (self.alpha + self.beta) / 2.0
        denom = (self.beta - self.alpha) ** 2.0

        result = np.zeros_like(x)
        mask2 = (self.alpha < x) & (x <= mid)
        mask3 = (mid < x) & (x <= self.beta)
        mask4 = self.beta < x

        result[mask2] = 2.0 * ((x[mask2] - self.alpha) ** 2.0) / denom
        result[mask3] = 1.0 - (2.0 * ((x[mask3] - self.beta) ** 2.0) / denom)
        result[mask4] = 1.0
        return result

    def to_dict(self) -> dict:
        """
        @brief Serialize instance to dict

        @return: Dict with type, alpha, and beta
        """
        return {"type": 'quadratic',
                "alpha": self.alpha,
                "beta": self.beta}