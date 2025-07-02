"""
@file implicators.py
@brief Fuzzy Implicators Framework

Provides an extensible, class-based system for computing fuzzy logic implicators.
Supports registration, creation, serialization, and parameter validation.

##############################################
# ✅ Quick Summary of Features
# Feature				        Description
# ----------------------------------------------------------------------------------
# register(*names)		        Register implicator with aliases
# create(name, **kwargs)	    Instantiate implicator from registry
# list_available()		        Returns registered implicators
# to_dict() / from_dict()	    Serialization / deserialization
# help()			        Returns class-level documentation
# validate_params()		        Validates constructor parameters
# describe_params_detailed()	Returns parameter types and values
# __call__()			    Smart dispatcher for scalar, vector, or matrix

# ✅ Summary Table of Design Patterns
# Category			Name		    Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern	Factory Method		Implicator.create(name, **kwargs)
# Design Pattern	Registry Pattern	Implicator._registry and register()
# Design Pattern	Template Method		Defines abstract methods in base class
# Design Pattern	Decorator	    @Implicator.register('name', ...)
# Design Pattern	Strategy Pattern	Each subclass defines its own logic
# Design Pattern	Adapter		    type-based to_dict/from_dict handling
# Architecture	    Pluggable		No modification needed for extension
# Clean Code		SRP, DRY, LSP, Fail-Fast, Docstring Documentation
##############################################
"""

import numpy as np
from abc import abstractmethod
from typing import Union
from FRsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin

class Implicator(RegistryFactoryMixin):
    """
    @brief Abstract base class for fuzzy implicators.

    Provides registration, creation, validation, and serialization logic.
    All concrete implicators must implement `_compute_elementwise`.
    """

    def __call__(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        @brief Apply the implicator to scalar or NumPy array inputs.

        Automatically handles scalar, vector, or matrix inputs.

        @param a: Scalar or NumPy array.
        @param b: Scalar or NumPy array with same shape.
        @return: Scalar or array of implicator results.
        """
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        if a_arr.shape != b_arr.shape:
            raise ValueError(f"Incompatible shapes: {a_arr.shape} and {b_arr.shape}")

        if np.isscalar(a) and np.isscalar(b):
            return self._compute_scalar(float(a), float(b))

        vec_func = np.vectorize(self._compute_scalar)
        return vec_func(a_arr, b_arr)

    @abstractmethod
    def _compute_scalar(self, a: float, b: float) -> float:
        """
        @brief Perform implicator operation on a pair of scalars.

        Subclasses implement this with their specific logic.

        @param a: Scalar float input.
        @param b: Scalar float input.
        @return: Implicator value.
        """
        raise NotImplementedError("all subclasses must implement _compute_scalar")
    

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
        """
        @brief Returns a dictionary of the implicator's parameters.
        @return: Parameter dictionary.
        """
        raise NotImplementedError("all derived calsses need to implemet _get_params")



#region <Non-parameterized implicators>

@Implicator.register("lukasiewicz","luk")
class LukasiewiczImplicator(Implicator):
    """
    @brief Lukasiewicz implicator: I(a, b) = min(1, 1 - a + b)
    """
    def __init__(self):
        self.validate_params()
        
    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Implicator inputs must be in range [0.0, 1.0].")
        return min(1.0, 1.0 - a + b)

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

@Implicator.register("goedel")
class GoedelImplicator(Implicator):
    """
    @brief Gödel implicator: I(a, b) = 1 if a <= b; else b
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Implicator inputs must be in range [0.0, 1.0].")
        return 1.0 if a <= b else b
    
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

@Implicator.register("kleenedienes", "kleene", "kd")
class KleeneDienesImplicator(Implicator):
    """
    @brief Kleene-Dienes implicator: I(a, b) = max(1 - a, b)
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return max(1.0 - a, b)
    
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


@Implicator.register("reichenbach")
class ReichenbachImplicator(Implicator):
    """
    @brief Reichenbach implicator: I(a, b) = 1 - a + (a * b)
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return 1.0 - a + a * b
    
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


@Implicator.register("goguen", "product")
class GoguenImplicator(Implicator):
    """
    @brief Goguen implicator: I(a, b) = 1 if a <= b; b / a otherwise
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return 1.0 if a <= b else b / a if a > 0 else 1.0
    
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

@Implicator.register("rescher")
class RescherImplicator(Implicator):
    """
    @brief Rescher implicator: I(a, b) = 1 if a <= b; 0.0 otherwise
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        return 1.0 if a <= b else 0.0
    
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

@Implicator.register("yager")
class YagerImplicator(Implicator):
    """
    @brief Yager implicator: I(a, b) = b^a if a > 0 or b > 0; 1 otherwise
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        
        if (a==b==0):
            return 1.0
        return b**a
    
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

@Implicator.register("weber")
class WeberImplicator(Implicator):
    """
    @brief Weber implicator: I(a, b) = b if a == 1 ; 1 if a<1
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        
        if a == 1.0:
            return b
        if a < 1.0:
            return 1.0
        else:
            raise ValueError("must not happen")

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

@Implicator.register("fodor")
class FodorImplicator(Implicator):
    """
    @brief Fodor implicator: I(a, b) = max(1 - a, b) if a > b; 1 otherwise
    """
    def __init__(self):
        self.validate_params()

    def _compute_scalar(self, a: float, b: float) -> float:
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("Inputs must be in range [0.0, 1.0].")
        
        return 1.0 if a <= b else max(1.0 - a, b)
    
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

# @Implicator.register("gaines")
# class GainesImplicator(Implicator):
#     """
#     @brief Gaines implicator: I(a, b) = 1 if a <= b; else b / a
#     """
#     def __init__(self):
#         self.validate_params()

#     def _compute_scalar(self, a: float, b: float) -> float:
#         if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
#             raise ValueError("Inputs must be in range [0.0, 1.0].")
#         if a <= b:
#             return 1.0
#         # a>b and a!=0
#         elif a > 0:
#             return b / a
#         # a>b and a=0
#         else:
#             return 0.0
        
    # @classmethod
    # def validate_params(cls, **kwargs):
    #     """
    #     @brief This class does not need parameter validation
    #     """
    #     pass

    # def _get_params(self)-> dict:
    #     """
    #     @brief no parameters
    #     """
    #     return {}


#endregion

#region<Parameterized implicators>






# @Implicator.register("yager")
# class YagerParametricImplicator(Implicator):
#     # we need __init__ because needed parameters for yager implicator gotten
#     # be checking the signature of this function. Moreover, it is called when
#     # create function is called
#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def _compute_scalar(self, a: float, b: float) -> float:
#         if not (0 <= a <= 1 and 0 <= b <= 1):
#             raise ValueError("Inputs must be in range [0, 1].")
#         return min(1.0, (1 - (a ** self.p) + (b ** self.p)) ** (1.0 / self.p))

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")
#         if p is None:
#             raise ValueError("Missing required parameter: p")
#         if not isinstance(p, (int, float)) or p <= 0:
#             raise ValueError("Parameter 'p' must be a positive number.")

# @Implicator.register("weber")
# class WeberImplicator(Implicator):
#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def _compute_scalar(self, a: float, b: float) -> float:
#         denom = (a ** self.p + (1 - a) ** self.p)
#         return min(1.0, b ** self.p / denom if denom != 0 else 1.0)

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")
#         if p is None:
#             raise ValueError("Missing required parameter: p")
#         if not isinstance(p, (int, float)) or p <= 0:
#             raise ValueError("Parameter 'p' must be a positive number.")

# @Implicator.register("frank")
# class FrankImplicator(Implicator):
#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def _compute_scalar(self, a: float, b: float) -> float:
#         if self.p == 1:
#             return 1.0 - a + a * b
#         num = (self.p ** b - 1) * (1 - self.p ** a)
#         denom = self.p - 1
#         result = 1 + num / denom
#         return np.clip(np.log(result) / np.log(self.p), 0, 1)

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")
#         if p is None or not isinstance(p, (int, float)) or p <= 0 or p == 1:
#             raise ValueError("Parameter 'p' must be > 0 and != 1")

# @Implicator.register("sugeno-weber", "sw")
# class SugenoWeberImplicator(Implicator):
#     def __init__(self, p: float = 1.0):
#         self.p = p

#     def _compute_scalar(self, a: float, b: float) -> float:
#         denom = 1 + self.p * (1 - a) * b
#         return min(1.0, (b - a + a * b) / denom if denom != 0 else 1.0)

#     @classmethod
#     def validate_params(cls, **kwargs):
#         p = kwargs.get("p")
#         if p is None or not isinstance(p, (int, float)):
#             raise ValueError("Parameter 'p' must be a number")

#endregion