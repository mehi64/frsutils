"""
Similarities

Provides an extensible and optimized framework to compute similarity matrices
with pluggable similarity functions (using inheritance)

@author: Mehran Amiri
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
# from pydantic import BaseModel, Field
from typing import Literal, Optional

# ------------------------------------------------------------------------------
# Similarity Function Base & Implementations
# ------------------------------------------------------------------------------

class SimilarityFunction(ABC):
    """
    Abstract base class for scalar similarity functions.
    """
    @abstractmethod
    def compute(self, diff: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, diff: np.ndarray) -> np.ndarray:
        return self.compute(diff)


class LinearSimilarity(SimilarityFunction):
    """
    Linear similarity: sim = max(0, 1 - |v1 - v2|)
    """
    def compute(self, diff: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(diff))


class GaussianSimilarity(SimilarityFunction):
    """
    Gaussian similarity: sim = exp(-diff^2 / (2 * sigma^2))

    @param sigma: Standard deviation for the Gaussian kernel (must be > 0)
    """
    def __init__(self, sigma: float = 0.1):
        # if not (0 < sigma <= 0.5):
        #     raise ValueError("sigma must be in the range (0, 0.5]")
        self.sigma = sigma

    def compute(self, diff: np.ndarray) -> np.ndarray:
       result = np.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))
       return result

# ------------------------------------------------------------------------------
# Similarity Matrix Computation
# ------------------------------------------------------------------------------

def calculate_similarity_matrix(
    X: np.ndarray,
    similarity_func: SimilarityFunction,
    tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Compute the pairwise similarity matrix for samples using a vectorized
    similarity function and T-norm operator.

    @param X: Normalized input matrix of shape (n_samples, n_features)
    @param similarity_func: Instance of SimilarityFunction subclass
    @param tnorm: Callable T-norm function (e.g., min, product, Yager)
    @return: Similarity matrix of shape (n_samples, n_samples)
    """
    n_samples, n_features = X.shape
    sim_matrix = np.ones((n_samples, n_samples), dtype=np.float64)

    for k in range(n_features):
        col = X[:, k].reshape(-1, 1)  # Shape (n, 1)
        diff = col - col.T            # Shape (n, n)
        sim_k = similarity_func(diff) # Compute similarity
        sim_matrix = tnorm(sim_matrix, sim_k)  # Apply T-norm

    np.fill_diagonal(sim_matrix, 1.0)
    return sim_matrix