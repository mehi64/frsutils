"""
@file tnorms.py
@brief Fuzzy T-norms Framework

Provides a pluggable architecture for defining T-norm operators used in fuzzy rough set theory.
Implements factory registration, serialization, validation, and multi-input support.

##############################################
# ✅ Quick Summary of Features
# Feature				Description
# ----------------------------------------------------------------------------------
# register(*names)		Register T-norm with aliases
# create(name, **kwargs)	Instantiate T-norm from registry
# list_available()		Returns registered T-norms
# to_dict() / from_dict()	Serialization / deserialization
# help()				Returns class-level documentation
# validate_params()		Validates constructor parameters
# name				Returns lowercase class name
# get_params()			Introspect parameter structure and values
# __call__()				Handles scalar, vector, matrix application
# reduce()				Aggregation operation

# ✅ Summary Table of Design Patterns
# Category				Name			Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern		Factory Method		TNorm.create(name, **kwargs)
# Design Pattern		Registry Pattern	TNorm._registry and register()
# Design Pattern		Template Method		Defines abstract __call__ and reduce methods
# Design Pattern		Strategy Pattern	Each subclass defines its logic
# Design Pattern		Decorator		    @TNorm.register(...)
# Design Pattern		Adapter			    Serialization via to_dict/from_dict
# Architecture		    Pluggable			New T-norms extend base class via registration
# Clean Code			SRP, DRY, LSP, Fail-Fast, Reflection
##############################################
"""

import numpy as np
from abc import abstractmethod
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin

class TNorm(RegistryFactoryMixin):
    """
    @brief Abstract base class for all T-norms.

    Provides registration, factory instantiation, serialization, and support for
    scalar/vector/matrix input handling. Subclasses must define `_call__` and `reduce`.
    """
    
    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        @brief Apply the T-norm to two arrays or matrices element-wise.

        @param a: First input array.
        @param b: Second input array.
        @return: Element-wise result of the T-norm.
        """
        raise NotImplementedError("all subclasses must implement __call__")

    @abstractmethod
    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """
        @brief Reduce an array using the T-norm.

        @param arr: 2D array of shape (n_samples, n_features).
        @return: Reduced array along axis=0.
        """
        raise NotImplementedError("all subclasses must implement reduce")

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """
        @brief Optional parameter validation hook for subclasses.
        
        @param kwargs: Parameters to validate.
        """
        raise NotImplementedError("all subclasses must implement validate_params")
    
    @abstractmethod
    def _get_params(self)-> dict:
        raise NotImplementedError("all derived calsses need to implemet _get_params")


@TNorm.register('minimum', 'min', 'goedel', 'standardintersection')
class MinTNorm(TNorm):
    """
    @brief Minimum T-norm: min(a, b)
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.min(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}


@TNorm.register('product', 'prod', 'algebraic')
class ProductTNorm(TNorm):
    """
    @brief Product T-norm: a * b
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.prod(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

@TNorm.register('lukasiewicz', 'luk', 'bounded', 'boundeddifference')
class LukasiewiczTNorm(TNorm):
    """
    @brief Łukasiewicz T-norm: max(0, a + b - 1)
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a + b - 1.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = np.maximum(0.0, result + x - 1.0)
        return result
    
    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

@TNorm.register("drastic", "drasticproduct")
class DrasticProductTNorm(TNorm):
    """
    @brief Drastic Product T-norm:
    - a if b == 1
    - b if a == 1
    - 0 otherwise
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(b == 1.0, a, np.where(a == 1.0, b, 0.0))

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = np.where(x == 1.0, result, np.where(result == 1.0, x, 0.0))
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

@TNorm.register("einstein", "einsteinproduct")
class EinsteinProductTNorm(TNorm):
    """
    @brief Einstein Product T-norm:
    T(a, b) = (a * b) / (2 - (a + b - a * b))
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = 2 - (a + b - a * b)
        return np.where(denom != 0, (a * b) / denom, 0.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = (result * x) / (2 - (result + x - result * x))
        return result
    
    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

@TNorm.register("hamacher", "hamacherproduct")
class HamacherProductTNorm(TNorm):
    """
    @brief Hamacher Product T-norm:
    T(a,b) = 0 if a = b = 0, else ab / (a + b - ab)
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = a + b - a * b
            result = np.divide(a * b, denom, out=np.zeros_like(a), where=denom != 0)
        return result


    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            denom = result + x - result * x
            result = np.where(denom != 0, (result * x) / denom, 0.0)

        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

@TNorm.register("nilpotent", "nilpotentminimum")
class NilpotentMinimumTNorm(TNorm):
    """
    @brief Nilpotent Minimum T-norm:
    T(a, b) = min(a, b) if (a + b) > 1 else 0
    """
    def __init__(self):
        self.validate_params()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where((a + b) > 1.0, np.minimum(a, b), 0.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = np.where((result + x) > 1.0, np.minimum(result, x), 0.0)
        return result

    @classmethod
    def validate_params(cls, **kwargs):
        """
        @brief This class does not need parameter validation
        """
        pass

    def _get_params(self)-> dict:
        """
        @brief no parameters
        """
        return {}

@TNorm.register('yager', 'yg')
class YagerTNorm(TNorm):
    """
    @brief Yager T-norm: 
    1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p))

    @param p: Exponent parameter that controls the shape (default = 2.0).
    """
    def __init__(self, p: float):
        self.validate_params(p=p)
        self.p = p

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Compute Yager T-norm element-wise
        return 1.0 - np.minimum(
            1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
        )

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        # Reduce across axis using Yager logic
        return 1.0 - np.minimum(
            1.0, np.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
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