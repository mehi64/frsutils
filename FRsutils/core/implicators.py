"""
@file implicators.py
@brief Fuzzy implicators framework with backend-aware vectorized formulas.

Provides an extensible, class-based system for computing fuzzy logic implicators.
Supports registration, creation, serialization, parameter validation, NumPy public
calls, and formula-level backend hooks for future NumPy/CuPy approximation paths.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# register(*names)                     Register implicator with aliases
# create(name, **kwargs)               Instantiate implicator from registry
# list_available()                     Returns registered implicators
# to_dict() / from_dict()              Serialization / deserialization
# validate_params()                    Validates constructor parameters
# __call__()                           NumPy-compatible scalar/vector/matrix call
# compute_backend(a, b, xp=...)        Backend-aware element-wise formula

# ✅ Summary Table of Design Patterns
# Category              Name                Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern        Factory Method      Implicator.create(name, **kwargs)
# Design Pattern        Registry Pattern    Implicator._registry and register()
# Design Pattern        Template Method     Base class defines call/validation flow
# Design Pattern        Strategy Pattern    Each subclass defines its own formula
# Design Pattern        Decorator           @Implicator.register('name', ...)
# Backend Boundary      NumPy/CuPy formulas are owned by components, not engines
##############################################
"""

from typing import Any, Union
import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class Implicator(RegistryFactoryMixin):
    """
    @brief Abstract base class for fuzzy implicators.

    Dense public calls remain NumPy-compatible. `compute_backend` exposes the
    same formulas as vectorized backend operations so approximation engines do
    not need to mirror implicator logic.
    """

    def __call__(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        @brief Apply the implicator to scalar or NumPy array inputs.
        """
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        if a_arr.shape != b_arr.shape:
            raise ValueError(f"Incompatible shapes: {a_arr.shape} and {b_arr.shape}")

        if np.isscalar(a) and np.isscalar(b):
            return self._compute_scalar(float(a), float(b))

        return self.compute_backend(a_arr, b_arr, xp=np, validate_inputs=True)

    def _validate_backend_pair(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True) -> None:
        """
        @brief Validate backend array pair shape and optionally value range.
        """
        if getattr(a, "shape", None) != getattr(b, "shape", None):
            raise ValueError(f"Incompatible shapes: {getattr(a, 'shape', None)} and {getattr(b, 'shape', None)}")
        if not validate_inputs:
            return

        out_of_range = xp.any((a < 0.0) | (a > 1.0) | (b < 0.0) | (b > 1.0))
        if xp is np:
            has_out_of_range = bool(out_of_range)
        else:
            has_out_of_range = bool(xp.asnumpy(out_of_range))
        if has_out_of_range:
            raise ValueError("Implicator inputs must be in range [0.0, 1.0].")

    @abstractmethod
    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        """
        @brief Apply the implicator to two backend arrays element-wise.
        """
        raise NotImplementedError("all subclasses must implement compute_backend")

    @abstractmethod
    def _compute_scalar(self, a: float, b: float) -> float:
        """
        @brief Perform implicator operation on a pair of scalars.
        """
        raise NotImplementedError("all subclasses must implement _compute_scalar")

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """@brief Optional parameter validation hook for subclasses."""
        raise NotImplementedError("all subclasses must implement validate_params")

    @abstractmethod
    def _get_params(self) -> dict:
        """@brief Returns a dictionary of the implicator's parameters."""
        raise NotImplementedError("all derived classes need to implement _get_params")


@Implicator.register("lukasiewicz", "luk")
class LukasiewiczImplicator(Implicator):
    """@brief Lukasiewicz implicator: I(a, b) = min(1, 1 - a + b)."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.minimum(1.0, 1.0 - a + b)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("goedel")
class GoedelImplicator(Implicator):
    """@brief Gödel implicator: I(a, b) = 1 if a <= b; else b."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.where(a <= b, 1.0, b)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("kleenedienes", "kleene", "kd")
class KleeneDienesImplicator(Implicator):
    """@brief Kleene-Dienes implicator: I(a, b) = max(1 - a, b)."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.maximum(1.0 - a, b)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("reichenbach")
class ReichenbachImplicator(Implicator):
    """@brief Reichenbach implicator: I(a, b) = 1 - a + (a * b)."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return 1.0 - a + a * b

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("goguen", "product")
class GoguenImplicator(Implicator):
    """@brief Goguen implicator: I(a, b) = 1 if a <= b; b / a otherwise."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        safe_a = xp.where(a > 0.0, a, 1.0)
        ratio = b / safe_a
        return xp.where(a <= b, 1.0, xp.where(a > 0.0, ratio, 1.0))

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("rescher")
class RescherImplicator(Implicator):
    """@brief Rescher implicator: I(a, b) = 1 if a <= b; 0 otherwise."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.where(a <= b, 1.0, 0.0)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("yager")
class YagerImplicator(Implicator):
    """@brief Yager implicator: I(a, b) = b^a if a > 0 or b > 0; 1 otherwise."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.where((a == 0.0) & (b == 0.0), 1.0, b ** a)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("weber")
class WeberImplicator(Implicator):
    """@brief Weber implicator: I(a, b) = b if a == 1; 1 if a < 1."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.where(a == 1.0, b, 1.0)

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@Implicator.register("fodor")
class FodorImplicator(Implicator):
    """@brief Fodor implicator: I(a, b) = max(1 - a, b) if a > b; 1 otherwise."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np, validate_inputs: bool = True):
        self._validate_backend_pair(a, b, xp=xp, validate_inputs=validate_inputs)
        return xp.where(a <= b, 1.0, xp.maximum(1.0 - a, b))

    def _compute_scalar(self, a: float, b: float) -> float:
        return float(self.compute_backend(np.asarray(a), np.asarray(b), xp=np))

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}
