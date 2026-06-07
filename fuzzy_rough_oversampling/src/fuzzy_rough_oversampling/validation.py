"""
@file validation.py
@brief Validation helpers for fuzzy-rough oversampling algorithms.

This module contains local validation helpers needed by the standalone
oversampling package. These helpers intentionally avoid importing FRsutils
internal validation modules so the package remains coupled only to FRsutils.api.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# validate_ranking_strategy_choice     Validate supported ranking strategy names
# compatible_dataset_with_fuzzy_rough  Validate normalized tabular X/y inputs

# ✅ Design Patterns & Clean Code Notes
# - SRP: validation utilities are separate from estimator logic
# - Fail Fast: invalid inputs raise early with actionable errors
# - Dependency Inversion: no dependency on FRsutils internal utility paths
##############################################
"""

from __future__ import annotations

import numpy as np

ALLOWED_RANKING_STRATEGIES = {"pos", "lower", "upper"}


def validate_ranking_strategy_choice(name: str) -> str:
    """
    @brief Validate a ranking strategy choice.

    @param name: Ranking strategy name.
    @return: Normalized ranking strategy name.
    @raises TypeError: If name is not a string.
    @raises ValueError: If name is not supported.
    """
    if not isinstance(name, str):
        raise TypeError("ranking_strategy must be a string.")
    normalized = name.strip().lower()
    if normalized not in ALLOWED_RANKING_STRATEGIES:
        raise ValueError(
            f"Invalid value '{name}' for ranking_strategy. "
            f"Allowed values are: {sorted(ALLOWED_RANKING_STRATEGIES)}."
        )
    return normalized


def compatible_dataset_with_fuzzy_rough(X: np.ndarray, y: np.ndarray) -> None:
    """
    @brief Validate the dataset assumptions used by fuzzy-rough computations.

    @param X: 2D NumPy feature matrix with float values in [0, 1].
    @param y: 1D NumPy target array aligned with X rows.
    @raises TypeError: If X/y types are incompatible.
    @raises ValueError: If shapes or value ranges are invalid.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if not np.issubdtype(X.dtype, np.floating):
        raise TypeError("X elements must be of float type.")
    if np.any(X < 0.0) or np.any(X > 1.0):
        raise ValueError("All elements in X must be in the range [0.0, 1.0].")

    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy ndarray.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(y) != X.shape[0]:
        raise ValueError("Length of y must be equal to the first dimension of X.")


__all__ = [
    "ALLOWED_RANKING_STRATEGIES",
    "compatible_dataset_with_fuzzy_rough",
    "validate_ranking_strategy_choice",
]
