# SPDX-License-Identifier: BSD-3-Clause
"""Backend-aware fuzzy T-norm operators for fuzzy-rough computations.

This module belongs to the core fuzzy-rough computation layer.
"""

from typing import Any
import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class TNorm(RegistryFactoryMixin):
    """
    @brief Abstract base class for all T-norms.

    Provides registration, factory instantiation, serialization, and support for
    scalar/vector/matrix input handling. Dense public behavior remains NumPy-based,
    while `compute_backend` and `reduce_backend` provide formula-level hooks for
    similarity/approximation engines.
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        @brief Apply the T-norm to two NumPy arrays or matrices element-wise.
        """
        return self.compute_backend(a, b, xp=np)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """
        @brief Reduce a NumPy array using the T-norm.
        """
        return self.reduce_backend(arr, xp=np)

    @abstractmethod
    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        """
        @brief Apply the T-norm to two backend arrays element-wise.
        """
        raise NotImplementedError("all subclasses must implement compute_backend")

    @abstractmethod
    def reduce_backend(self, arr: Any, *, xp: Any = np):
        """
        @brief Reduce a backend array using the T-norm.
        """
        raise NotImplementedError("all subclasses must implement reduce_backend")

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation hook for subclasses.
        """
        raise NotImplementedError("all subclasses must implement validate_params")

    @abstractmethod
    def _get_params(self) -> dict:
        raise NotImplementedError("all derived classes need to implement _get_params")


@TNorm.register('minimum', 'min', 'goedel', 'standardintersection')
class MinTNorm(TNorm):
    """@brief Minimum T-norm: min(a, b)."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return xp.minimum(a, b)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        return xp.min(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register('product', 'prod', 'algebraic')
class ProductTNorm(TNorm):
    """@brief Product T-norm: a * b."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return a * b

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        return xp.prod(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register('lukasiewicz', 'luk', 'bounded', 'boundeddifference')
class LukasiewiczTNorm(TNorm):
    """@brief Łukasiewicz T-norm: max(0, a + b - 1)."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return xp.maximum(0.0, a + b - 1.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register("drastic", "drasticproduct")
class DrasticProductTNorm(TNorm):
    """@brief Drastic Product T-norm."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return xp.where(b == 1.0, a, xp.where(a == 1.0, b, 0.0))

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register("einstein", "einsteinproduct")
class EinsteinProductTNorm(TNorm):
    """@brief Einstein Product T-norm."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        denom = 2.0 - (a + b - a * b)
        safe_denom = xp.where(denom != 0.0, denom, 1.0)
        return xp.where(denom != 0.0, (a * b) / safe_denom, 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        pass


@TNorm.register("hamacher", "hamacherproduct")
class HamacherProductTNorm(TNorm):
    """@brief Hamacher Product T-norm."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        denom = a + b - a * b
        safe_denom = xp.where(denom != 0.0, denom, 1.0)
        return xp.where(denom != 0.0, (a * b) / safe_denom, 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register("nilpotent", "nilpotentminimum")
class NilpotentMinimumTNorm(TNorm):
    """@brief Nilpotent Minimum T-norm."""

    def __init__(self):
        self.validate_params()

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return xp.where((a + b) > 1.0, xp.minimum(a, b), 0.0)

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        result = arr[0]
        for x in arr[1:]:
            result = self.compute_backend(result, x, xp=xp)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        pass

    def _get_params(self) -> dict:
        return {}


@TNorm.register('yager', 'yg')
class YagerTNorm(TNorm):
    """
    @brief Yager T-norm: 1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p)).
    """

    def __init__(self, p: float = 2.0):
        self.validate_params(p=p)
        self.p = p

    def compute_backend(self, a: Any, b: Any, *, xp: Any = np):
        return 1.0 - xp.minimum(
            1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
        )

    def reduce_backend(self, arr: Any, *, xp: Any = np):
        return 1.0 - xp.minimum(
            1.0, xp.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
        )

    @classmethod
    def validate_params(cls, **kwargs):
        p = kwargs.get('p')
        if p is None:
            raise ValueError("Missing required parameter: p")
        if not isinstance(p, (int, float)):
            raise ValueError("Parameter 'p' must be a float or int")
        if p <= 0:
            raise ValueError("Parameter 'p' must be greater than 0")

    def _get_params(self) -> dict:
        return {"p": self.p}
