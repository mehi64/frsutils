import numpy as np

def _owa_suprimum_weights_linear(n: int):
    """Generates linear OWA weights (normalized):
    n: length of the weights array
    used for suprimum operator smoothing
    {1, 2, 3, . . ., n} : wi = 2(n − i + 1)/n(n + 1)"""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be an integer number >= 1")
        # return np.array([])
    if n == 1:
        return np.array([1.0])
    weights = np.arange(n, 0, -1)
    denominator = n * (n + 1) / 2.0
    if denominator == 0 or not np.isfinite(denominator):
         # Fallback for very large n where calculation might overflow/be zero
        # return np.full(n, 1.0 / n)
        raise ValueError("Denominator is zero or non-finite, cannot compute weights.")

    val = weights / denominator
    sum_vals = np.sum(val)
    assert np.isclose(sum_vals, 1.0)
    return val

def _owa_infimum_weights_linear(n: int):
    """Generates OWA weights for infimum operators:
    n: length of the weights array
    used for infimum operator smoothing
    {1, 2, 3, . . ., n} : wi = 2i/n(n + 1)"""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n <= 0:
        raise ValueError("n must be an integer number >= 1")
        # return np.array([])
    if n == 1:
        return np.array([1.0])
    weights = np.arange(1, n+1, 1)
    denominator = n * (n + 1) / 2.0
    if denominator == 0 or not np.isfinite(denominator):
         # Fallback for very large n where calculation might overflow/be zero
        # return np.full(n, 1.0 / n)
        raise ValueError("divided by 0.0 error or some infinite values in denimonator calculations")
    val = weights / denominator
    sum_vals = np.sum(val)
    assert np.isclose(sum_vals, 1.0)
    return val