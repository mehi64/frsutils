# FRsutils/utils/math_utils/backend.py

USE_GPU = True  # 👈 Global toggle to force CPU mode

try:
    import cupy as cp
    if not USE_GPU:
        raise ImportError("GPU disabled by flag")
    xp = cp
    IS_GPU = True
except ImportError:
    import numpy as np
    xp = np
    IS_GPU = False

def to_backend_array(arr):
    """
    @brief Converts a NumPy array to CuPy if GPU is enabled, otherwise returns unchanged.
    """
    if IS_GPU and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr

def to_numpy_array(arr):
    """
    @brief Converts a CuPy array to NumPy if GPU is active, otherwise returns unchanged.
    """
    if IS_GPU and isinstance(arr, xp.ndarray):
        return cp.asnumpy(arr)
    return arr

def is_backend_array(arr):
    """
    @brief Checks whether an array is from the active backend (NumPy or CuPy).
    """
    return isinstance(arr, xp.ndarray)

def zeros_like(arr):
    """
    @brief Returns a zero-initialized array with the same shape and dtype as input.
    """
    return xp.zeros_like(arr)

def empty_like(arr):
    """
    @brief Returns an uninitialized array with the same shape and dtype as input.
    """
    return xp.empty_like(arr)

def array(data, dtype=None):
    """
    @brief Constructs a backend array from list/tuple/other arrays.
    """
    return xp.array(data, dtype=dtype) if dtype else xp.array(data)

