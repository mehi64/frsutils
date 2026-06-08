"""
@file backends.py
@brief Lightweight array-backend helpers for similarity-engine execution modes.

This module resolves array execution backends for FRsutils blockwise similarity
computation. NumPy remains the default and dependency-free path. CuPy is an
optional GPU backend used only when explicitly requested, so normal FRsutils
imports and NumPy workflows do not require CUDA/CuPy to be installed.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# ArrayBackend                         Small value object describing an array backend
# build_array_backend                  Resolve and validate a backend alias
# is_numpy_backend                     Predicate for NumPy-backed execution
# is_cupy_backend                      Predicate for CuPy-backed execution
# to_numpy                             Convert backend arrays back to NumPy arrays

# ✅ Design Patterns & Clean Code Notes
# - Adapter Pattern: hides concrete NumPy/CuPy namespaces behind a small object
# - Boundary Validation: validates backend names at construction boundaries
# - Optional Dependency Boundary: imports CuPy only when explicitly requested
# - Conservative Extension: backend='numpy' and backend='auto' remain CPU-stable
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.core.backends import build_array_backend
#
# backend = build_array_backend("numpy")
# X = backend.asarray([[0.0], [1.0]], dtype=float)
#
# gpu_backend = build_array_backend("cupy")  # requires optional CuPy install
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    """
    @brief Small immutable descriptor for array-backend behavior.

    @param name: Canonical backend name, e.g. "numpy" or "cupy".
    @param xp: Array namespace module.
    """

    name: str
    xp: Any

    def asarray(self, value: Any, dtype: Any = None):
        """
        @brief Convert a value into an array using this backend namespace.

        @param value: Candidate array-like value.
        @param dtype: Optional dtype passed through to the backend.
        @return: Backend array.
        """
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = np.float64):
        """
        @brief Allocate a zero array using this backend namespace.

        @param shape: Target array shape.
        @param dtype: Target dtype.
        @return: Backend zero array.
        """
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = np.float64):
        """
        @brief Allocate a ones array using this backend namespace.

        @param shape: Target array shape.
        @param dtype: Target dtype.
        @return: Backend ones array.
        """
        return self.xp.ones(shape, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        """
        @brief Convert a backend array to a NumPy array.

        @param value: Backend or NumPy array-like value.
        @return: NumPy array.
        """
        if self.name == "cupy":
            return self.xp.asnumpy(value)
        return np.asarray(value)


def _build_cupy_backend() -> ArrayBackend:
    """
    @brief Import CuPy lazily and return a backend descriptor.

    @return: CuPy ArrayBackend descriptor.
    @raises ImportError: If CuPy is not installed in the current environment.
    """
    try:
        import cupy as cp  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "backend='cupy' requires the optional CuPy package. Install a CUDA-compatible "
            "CuPy build, e.g. `pip install cupy-cuda12x`, or use backend='numpy'."
        ) from exc
    return ArrayBackend(name="cupy", xp=cp)


def build_array_backend(backend: str = "numpy") -> ArrayBackend:
    """
    @brief Resolve a supported array backend alias.

    NumPy and `auto` resolve to NumPy to preserve the stable CPU behavior from
    earlier phases. CuPy is opt-in through `backend="cupy"` or `backend="cuda"`.

    @param backend: Backend alias: "numpy", "auto", "cupy", or "cuda".
    @return: ArrayBackend descriptor.
    @raises TypeError: If backend is not a string.
    @raises ValueError: If backend is unknown.
    @raises ImportError: If CuPy is requested but not installed.
    """
    if not isinstance(backend, str) or not backend.strip():
        raise TypeError("backend must be a non-empty string.")

    normalized = backend.strip().lower()
    if normalized in {"numpy", "np", "auto"}:
        return ArrayBackend(name="numpy", xp=np)
    if normalized in {"cupy", "cp", "cuda", "gpu"}:
        return _build_cupy_backend()

    raise ValueError("Unsupported backend. Use backend='numpy', backend='auto', or backend='cupy'.")


def is_numpy_backend(backend: ArrayBackend) -> bool:
    """
    @brief Return True when the resolved backend is NumPy.

    @param backend: ArrayBackend descriptor.
    @return: True for NumPy-backed execution.
    """
    return isinstance(backend, ArrayBackend) and backend.name == "numpy"


def is_cupy_backend(backend: ArrayBackend) -> bool:
    """
    @brief Return True when the resolved backend is CuPy.

    @param backend: ArrayBackend descriptor.
    @return: True for CuPy-backed execution.
    """
    return isinstance(backend, ArrayBackend) and backend.name == "cupy"


def to_numpy(value: Any, backend: ArrayBackend) -> np.ndarray:
    """
    @brief Convert a backend array to NumPy using a backend descriptor.

    @param value: Backend array-like value.
    @param backend: ArrayBackend descriptor.
    @return: NumPy array.
    """
    if not isinstance(backend, ArrayBackend):
        raise TypeError("backend must be an ArrayBackend instance.")
    return backend.to_numpy(value)


__all__ = [
    "ArrayBackend",
    "build_array_backend",
    "is_cupy_backend",
    "is_numpy_backend",
    "to_numpy",
]
