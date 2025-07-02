# import numpy as np
# from abc import ABC, abstractmethod
# from typing import Type, Dict, List, Callable
# import inspect

# # -----------------------------------------------
# # Similarity Function Framework
# # -----------------------------------------------

# def _filter_args(cls, kwargs: dict) -> dict:
#     sig = inspect.signature(cls.__init__)
#     return {k: v for k, v in kwargs.items() if k in sig.parameters}


# class SimilarityFunction(ABC):
#     _registry: Dict[str, Type['SimilarityFunction']] = {}
#     _aliases: Dict[Type['SimilarityFunction'], List[str]] = {}

#     @classmethod
#     def register(cls, *names: str):
#         def decorator(subclass: Type['SimilarityFunction']):
#             if not names:
#                 raise ValueError("At least one name must be provided for registration.")
#             cls._aliases[subclass] = list(map(str.lower, names))
#             for name in names:
#                 key = name.lower()
#                 if key in cls._registry:
#                     raise ValueError(f"SimilarityFunction alias '{key}' already registered.")
#                 cls._registry[key] = subclass
#             return subclass
#         return decorator

#     @classmethod
#     def create(cls, name: str, strict: bool = False, **kwargs) -> 'SimilarityFunction':
#         name = name.lower()
#         if name not in cls._registry:
#             raise ValueError(f"Unknown similarity function: {name}")
#         sim_cls = cls._registry[name]
#         sim_cls.validate_params(**kwargs)
#         ctor_args = _filter_args(sim_cls, kwargs)
#         if strict:
#             unused = set(kwargs) - set(ctor_args)
#             if unused:
#                 raise ValueError(f"Unused parameters in strict mode: {unused}")
#         return sim_cls(**ctor_args)

#     @classmethod
#     def list_available(cls) -> Dict[str, List[str]]:
#         return {names[0]: names for names in cls._aliases.values()}

#     @classmethod
#     def validate_params(cls, **kwargs):
#         pass

#     def to_dict(self) -> dict:
#         return {
#             "type": self.__class__.__name__.replace("Similarity", "").lower(),
#             **self._get_params()
#         }

#     @classmethod
#     def from_dict(cls, data: dict) -> 'SimilarityFunction':
#         data = data.copy()
#         name = data.pop("type")
#         return cls.create(name, **data)

#     def help(self) -> str:
#         return self.__class__.__doc__.strip() if self.__class__.__doc__ else "No documentation available."

#     def _get_params(self) -> dict:
#         return {}

#     @abstractmethod
#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         pass

#     def __call__(self, diff: np.ndarray) -> np.ndarray:
#         return self.compute(diff)

# # -----------------------------------------------
# # Similarity Implementations
# # -----------------------------------------------

# @SimilarityFunction.register("linear")
# class LinearSimilarity(SimilarityFunction):
#     """Linear similarity: sim = max(0, 1 - |x - y|)"""
#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         return np.maximum(0.0, 1.0 - np.abs(diff))

# @SimilarityFunction.register("gaussian", "gauss")
# class GaussianSimilarity(SimilarityFunction):
#     """Gaussian similarity: sim = exp(-diff^2 / (2 * sigma^2))"""
#     def __init__(self, sigma: float = 0.1):
#         self.sigma = sigma

#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         return np.exp(-(diff ** 2) / (2.0 * self.sigma ** 2))

#     def _get_params(self) -> dict:
#         return {"sigma": self.sigma}

#     @classmethod
#     def validate_params(cls, **kwargs):
#         sigma = kwargs.get("sigma")
#         if sigma is None or not isinstance(sigma, (float, int)) or sigma <= 0:
#             raise ValueError("Parameter 'sigma' must be a positive number.")

# @SimilarityFunction.register("custom-log")
# class LogSimilarity(SimilarityFunction):
#     """Logarithmic similarity: sim = 1 / (1 + log(1 + |diff|))"""
#     def compute(self, diff: np.ndarray) -> np.ndarray:
#         return 1.0 / (1.0 + np.log1p(np.abs(diff)))

# # -----------------------------------------------
# # Similarity Matrix Computation
# # -----------------------------------------------

# def calculate_similarity_matrix(
#     X: np.ndarray,
#     similarity_func: SimilarityFunction,
#     tnorm: Callable[[np.ndarray, np.ndarray], np.ndarray]
# ) -> np.ndarray:
#     n_samples, n_features = X.shape
#     sim_matrix = np.ones((n_samples, n_samples), dtype=np.float64)

#     for k in range(n_features):
#         col = X[:, k].reshape(-1, 1)
#         diff = col - col.T
#         sim_k = similarity_func(diff)
#         sim_matrix = tnorm(sim_matrix, sim_k)

#     np.fill_diagonal(sim_matrix, 1.0)
#     return sim_matrix

# # -----------------------------------------------
# # Demo
# # -----------------------------------------------

# if __name__ == "__main__":
#     # Sample input matrix
#     X = np.array([
#         [0.1, 0.5],
#         [0.2, 0.7],
#         [0.4, 0.6]
#     ])

#     # Choose similarity function
#     sim_func = SimilarityFunction.create("gaussian", sigma=0.2)

#     # Choose T-norm
#     def product_tnorm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#         return a * b

#     # Compute similarity matrix
#     sim_matrix = calculate_similarity_matrix(X, sim_func, product_tnorm)

#     print("Similarity Matrix:")
#     print(sim_matrix)

#     # Serialization example
#     sim_dict = sim_func.to_dict()
#     print("\nSerialized:", sim_dict)

#     # Reconstruction from dict
#     sim_func_reconstructed = SimilarityFunction.from_dict(sim_dict)
#     print("Reconstructed:", sim_func_reconstructed.help())
