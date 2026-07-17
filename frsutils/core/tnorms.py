# SPDX-License-Identifier: BSD-3-Clause
"""Backend-aware fuzzy T-norm operators for fuzzy-rough computations.

This module belongs to the core fuzzy-rough computation layer.
"""

from numbers import Real
from typing import Any
import numpy as np
from abc import abstractmethod
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class TNorm(RegistryFactoryMixin):
    """Abstract base class for all T-norms.
    
    Provides registration, factory instantiation, serialization, and support for
    scalar/vector/matrix input handling. Dense public behavior remains NumPy-based,
    while `compute_backend` and `reduce_backend` provide formula-level hooks for
    similarity/approximation engines.
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply the T-norm to two NumPy arrays or matrices element-wise."""
        return self.compute_backend(a, b, xp=np)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """Reduce a NumPy array using the T-norm."""
        return self.reduce_backend(arr, xp=np)

    @abstractmethod
    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Apply the T-norm to two backend arrays element-wise."""
        raise NotImplementedError("all subclasses must implement compute_backend")

    @abstractmethod
    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce a backend array using the T-norm."""
        raise NotImplementedError("all subclasses must implement reduce_backend")

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """Optional parameter validation hook for subclasses."""
        raise NotImplementedError("all subclasses must implement validate_params")

    @abstractmethod
    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        raise NotImplementedError("all derived classes need to implement _get_params")


@TNorm.register('minimum', 'min', 'goedel', 'standardintersection')
class MinTNorm(TNorm):
    """Minimum T-norm: min(a, b)."""

    def __init__(self):
        """Initialize the MinTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return xp.minimum(a, b)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        return xp.min(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register('product', 'prod', 'algebraic')
class ProductTNorm(TNorm):
    """Product T-norm: a * b."""

    def __init__(self):
        """Initialize the ProductTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return a * b

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        return xp.prod(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register('lukasiewicz', 'luk', 'bounded', 'boundeddifference')
class LukasiewiczTNorm(TNorm):
    """Łukasiewicz T-norm: max(0, a + b - 1)."""

    def __init__(self):
        """Initialize the LukasiewiczTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return xp.maximum(0.0, a + b - 1.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register("drastic", "drasticproduct")
class DrasticProductTNorm(TNorm):
    """Drastic Product T-norm. 
        T(a, b) = a if b == 1, b if a == 1, and; 0 otherwise."""

    def __init__(self):
        """Initialize the DrasticProductTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return xp.where(b == 1.0, a, xp.where(a == 1.0, b, 0.0))

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register("einstein", "einsteinproduct")
class EinsteinProductTNorm(TNorm):
    """Einstein Product T-norm.
        T(a, b) = (a * b) / (2 - a - b + a * b)"""

    def __init__(self):
        """Initialize the EinsteinProductTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        denom = 2.0 - (a + b - a * b)
        safe_denom = xp.where(denom != 0.0, denom, 1.0)
        return xp.where(denom != 0.0, (a * b) / safe_denom, 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass


@TNorm.register("hamacher", "hamacherproduct")
class HamacherProductTNorm(TNorm):
    """Hamacher Product T-norm.
        T(a, b) = (a * b) / (a + b - a * b), returning 0 when a == b == 0."""

    def __init__(self):
        """Initialize the HamacherProductTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        denom = a + b - a * b
        safe_denom = xp.where(denom != 0.0, denom, 1.0)
        return xp.where(denom != 0.0, (a * b) / safe_denom, 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register("nilpotent", "nilpotentminimum")
class NilpotentMinimumTNorm(TNorm):
    """Nilpotent Minimum T-norm.
        T(a, b) = min(a, b) if a + b > 1, and 0 otherwise."""

    def __init__(self):
        """Initialize the NilpotentMinimumTNorm instance."""
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return xp.where((a + b) > 1.0, xp.minimum(a, b), 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {}


@TNorm.register('yager', 'yg')
class YagerTNorm(TNorm):
    """Yager T-norm: 1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p)). NOTE: (p>0)"""

    def __init__(self, p: float = 2.0):
        """Initialize the YagerTNorm instance."""
        self.validate_params(p=p)
        self.p = p

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """Compute the component formula using the provided array backend."""
        return 1.0 - xp.minimum(
            1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
        )

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """Reduce backend arrays using this aggregation operator."""
        return 1.0 - xp.minimum(
            1.0, xp.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
        )

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        p = kwargs.get('p')
        if p is None:
            raise ValueError("Missing required parameter: p")
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, Real):
            raise ValueError("Parameter 'p' must be a finite positive real number")
        if not np.isfinite(p) or p <= 0:
            raise ValueError("Parameter 'p' must be a finite positive real number")

    def _get_params(self) -> dict:
        """Return constructor parameters for serialization."""
        return {"p": self.p}
