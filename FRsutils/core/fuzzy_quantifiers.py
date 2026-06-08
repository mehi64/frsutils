
"""
@file fuzzy_quantifiers.py
@brief Framework for parameterized fuzzy quantifiers used in fuzzy logic systems.

Supports registration, instantiation via alias, computation of linear and
quadratic fuzzy quantifiers, and backend-aware formula hooks for NumPy/CuPy-like
array namespaces.

##############################################
# ✅ Summary of Design Principles & Clean Code
# - Registry Pattern: supports alias-based instantiation via decorators
# - Factory Method: `create()` builds objects from name + params
# - Strategy Pattern: each subclass defines a specific quantifier formula
# - Adapter Pattern: `to_dict()` / `from_dict()` for serialization
# - Fail-Fast Validation: type and range checks in validate_params
# - Backend Boundary: component owns its NumPy/CuPy formula
##############################################

@example
>>> fq = FuzzyQuantifier.create("linear", alpha=0.2, beta=0.6)
>>> fq(np.array([0.1, 0.3, 0.7]))
array([0. , 0.25, 1. ])
"""

from typing import Any
import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin
from FRsutils.utils.validation_utils.validation_utils import validate_range_0_1


class FuzzyQuantifier(RegistryFactoryMixin):
    """@brief Abstract base class for fuzzy quantifiers."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        @brief Compute fuzzy membership value(s) for NumPy input.
        """
        return self.compute_backend(np.asarray(x), xp=np, validate_inputs=self.validate_inputs)

    @abstractmethod
    def compute_backend(self, x: Any, *, xp: Any = np, validate_inputs: bool = True):
        """
        @brief Compute fuzzy membership value(s) using a backend array namespace.
        """
        raise NotImplementedError("all subclasses must implement compute_backend")

    def _validate_backend_input(self, x: Any, *, xp: Any = np, validate_inputs: bool = True) -> None:
        """
        @brief Validate quantifier input range for NumPy/CuPy-like arrays.
        """
        if not validate_inputs:
            return
        if xp is np:
            validate_range_0_1(np.asarray(x))
            return
        out_of_range = xp.any((x < 0.0) | (x > 1.0))
        if bool(xp.asnumpy(out_of_range)):
            raise ValueError("Fuzzy quantifier inputs must be in range [0.0, 1.0].")

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief Validate alpha and beta parameters.
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
        @brief Returns parameters for serialization.
        """
        return {"alpha": self.alpha, "beta": self.beta, "validate_inputs": self.validate_inputs}

    @classmethod
    def from_dict(cls, data: dict) -> "FuzzyQuantifier":
        """
        @brief Instantiate a quantifier from dictionary config.
        """
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")

        if "name" in data and "params" in data:
            params = data.get("params") or {}
            if not isinstance(params, dict):
                raise TypeError("data['params'] must be a dict")
            return cls.create(data["name"], **params)

        if "type" in data and "params" in data:
            params = data.get("params") or {}
            if not isinstance(params, dict):
                raise TypeError("data['params'] must be a dict")
            name = data.get("name") or data.get("type")
            return cls.create(name, **params)

        if "type" in data and "alpha" in data and "beta" in data:
            return cls.create(
                data["type"],
                alpha=data["alpha"],
                beta=data["beta"],
                validate_inputs=data.get("validate_inputs", True),
            )

        raise ValueError("Unsupported quantifier dictionary format. Expected keys like (name, params) or (type, alpha, beta).")


@FuzzyQuantifier.register("linear")
class LinearFuzzyQuantifier(FuzzyQuantifier):
    """@brief Linear piecewise fuzzy quantifier."""

    def __init__(self, alpha: float, beta: float, validate_inputs: bool = True):
        self.validate_params(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        self.validate_inputs = validate_inputs

    def compute_backend(self, x: Any, *, xp: Any = np, validate_inputs: bool = True):
        """@brief Backend-aware linear fuzzy quantifier formula."""
        self._validate_backend_input(x, xp=xp, validate_inputs=validate_inputs)
        return xp.where(
            x <= self.alpha,
            0.0,
            xp.where(x >= self.beta, 1.0, (x - self.alpha) / (self.beta - self.alpha)),
        )

    def to_dict(self) -> dict:
        """@brief Serialize instance to a standardized dict."""
        return {"type": self.name, "name": self.name, "params": self._get_params(), "alpha": self.alpha, "beta": self.beta}


@FuzzyQuantifier.register("quadratic", "quad")
class QuadraticFuzzyQuantifier(FuzzyQuantifier):
    """@brief Quadratic smooth fuzzy quantifier."""

    def __init__(self, alpha: float, beta: float, validate_inputs: bool = True):
        self.validate_params(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta
        self.validate_inputs = validate_inputs

    def compute_backend(self, x: Any, *, xp: Any = np, validate_inputs: bool = True):
        """@brief Backend-aware quadratic fuzzy quantifier formula."""
        self._validate_backend_input(x, xp=xp, validate_inputs=validate_inputs)
        mid = (self.alpha + self.beta) / 2.0
        denom = (self.beta - self.alpha) ** 2.0
        lower_curve = 2.0 * ((x - self.alpha) ** 2.0) / denom
        upper_curve = 1.0 - (2.0 * ((x - self.beta) ** 2.0) / denom)
        return xp.where(
            x <= self.alpha,
            0.0,
            xp.where(x <= mid, lower_curve, xp.where(x <= self.beta, upper_curve, 1.0)),
        )

    def to_dict(self) -> dict:
        """@brief Serialize instance to dict."""
        return {"type": "quadratic", "name": self.name, "params": self._get_params(), "alpha": self.alpha, "beta": self.beta}
