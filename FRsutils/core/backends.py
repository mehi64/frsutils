# SPDX-License-Identifier: BSD-3-Clause
"""Backend resolution utilities for NumPy and optional GPU execution.

This module belongs to the core fuzzy-rough computation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayBackend:
    """Small immutable descriptor for array-backend behavior.
    
    Parameters
    ----------
    name : object
        Canonical backend name, e.g. "numpy" or "cupy".
    xp : object
        Array namespace module.
    """

    name: str
    xp: Any

    def asarray(self, value: Any, dtype: Any = None):
        """Convert a value into an array using this backend namespace.
                
                Parameters
                ----------
                value : Any
                    Candidate array-like value.
                dtype : Any
                    Optional dtype passed through to the backend.
                
                Returns
                -------
                object
                    Backend array.
                
        """
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = np.float64):
        """Allocate a zero array using this backend namespace.
                
                Parameters
                ----------
                shape : Any
                    Target array shape.
                dtype : Any
                    Target dtype.
                
                Returns
                -------
                object
                    Backend zero array.
                
        """
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = np.float64):
        """Allocate a ones array using this backend namespace.
                
                Parameters
                ----------
                shape : Any
                    Target array shape.
                dtype : Any
                    Target dtype.
                
                Returns
                -------
                object
                    Backend ones array.
                
        """
        return self.xp.ones(shape, dtype=dtype)

    def to_numpy(self, value: Any) -> np.ndarray:
        """Convert a backend array to a NumPy array.
                
                Parameters
                ----------
                value : Any
                    Backend or NumPy array-like value.
                
                Returns
                -------
                np.ndarray
                    NumPy array.
                
        """
        if self.name == "cupy":
            return self.xp.asnumpy(value)
        return np.asarray(value)


def _build_cupy_backend() -> ArrayBackend:
    """Import CuPy lazily and return a backend descriptor.
        
        Returns
        -------
        ArrayBackend
            CuPy ArrayBackend descriptor.
        
        Raises
        ------
        ImportError
            If CuPy is not installed in the current environment.
        
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
    """Resolve a supported array backend alias.
        
        NumPy and `auto` resolve to NumPy to preserve the stable CPU behavior from
        earlier phases. CuPy is opt-in through `backend="cupy"` or `backend="cuda"`.
        
        Parameters
        ----------
        backend : str
            Backend alias: "numpy", "auto", "cupy", or "cuda".
        
        Returns
        -------
        ArrayBackend
            ArrayBackend descriptor.
        
        Raises
        ------
        TypeError
            If backend is not a string.
        ValueError
            If backend is unknown.
        ImportError
            If CuPy is requested but not installed.
        
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
    """Return True when the resolved backend is NumPy.
        
        Parameters
        ----------
        backend : ArrayBackend
            ArrayBackend descriptor.
        
        Returns
        -------
        bool
            True for NumPy-backed execution.
        
    """
    return isinstance(backend, ArrayBackend) and backend.name == "numpy"


def is_cupy_backend(backend: ArrayBackend) -> bool:
    """Return True when the resolved backend is CuPy.
        
        Parameters
        ----------
        backend : ArrayBackend
            ArrayBackend descriptor.
        
        Returns
        -------
        bool
            True for CuPy-backed execution.
        
    """
    return isinstance(backend, ArrayBackend) and backend.name == "cupy"


def to_numpy(value: Any, backend: ArrayBackend) -> np.ndarray:
    """Convert a backend array to NumPy using a backend descriptor.
        
        Parameters
        ----------
        value : Any
            Backend array-like value.
        backend : ArrayBackend
            ArrayBackend descriptor.
        
        Returns
        -------
        np.ndarray
            NumPy array.
        
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
