"""
T-norms
Provides an extensible and optimized framework to compute T-norms
"""

import numpy as np
from abc import ABC, abstractmethod


class TNorm(ABC):
    """
    Abstract base class for all T-norms.

    Provides a standard interface for pairwise and reduction operations.
    
    """

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Apply the T-norm to two arrays element-wise.

        @param a: First input array.
        @param b: Second input array.
        @return: Element-wise result of the T-norm.
        """
        pass

    @abstractmethod
    def reduce(self, arr: np.ndarray) -> np.ndarray:
        """
        Reduce a single array using the T-norm.

        @param arr: Array of shape (n_samples, n_samples).
        @return: Reduced value for each row/column on axis=1.
        """
        pass


class MinTNorm(TNorm):
    """
    Minimum T-norm: min(a, b)
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.min(arr, axis=0)


class ProductTNorm(TNorm):
    """
    Product T-norm: a * b
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        return np.prod(arr, axis=0)



class LukasiewiczTNorm(TNorm):
    """
    Łukasiewicz T-norm: max(0, a + b - 1)
    """

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, a + b - 1.0)

    def reduce(self, arr: np.ndarray) -> np.ndarray:
        result = arr[0]
        for x in arr[1:]:
            result = max(0.0, result + x - 1.0)
        return result


# TODO: uncomment and test this TNorm
# class YagerTNorm(TNorm):
#     """
#     Yager T-norm: 1 - min(1, [(1 - a)^p + (1 - b)^p]^(1/p))

#     @param p: The exponent parameter (default: 2.0)
#     """

#     def __init__(self, p: float = 2.0):
#         self.p = p

#     def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
#         return 1.0 - np.minimum(
#             1.0, ((1.0 - a) ** self.p + (1.0 - b) ** self.p) ** (1.0 / self.p)
#         )

#     def reduce(self, arr: np.ndarray) -> np.ndarray:
#         return 1.0 - np.minimum(
#             1.0, np.sum((1.0 - arr) ** self.p, axis=0) ** (1.0 / self.p)
#         )

# example usage
# tnorm = YagerTNorm(p=3.0)
# tnorm = ProductTNorm()
# result_pairwise = tnorm(arr1, arr2)
# result_reduced = tnorm.reduce(arr2d)