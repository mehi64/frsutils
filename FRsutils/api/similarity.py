"""
@file similarity.py
@brief Public similarity API for downstream FRsutils consumers.

This module exposes stable similarity-matrix entry points that external
packages should use. It intentionally hides lower-level implementation details
and keeps downstream packages independent from internal module organization.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# Similarity                           Public registry base for similarity functions
# list_similarities                    Inspect registered similarity aliases
# calculate_similarity_matrix          Low-level pairwise matrix calculation helper
# build_similarity_matrix              Validated public similarity matrix builder

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable import path for public similarity operations
# - Strategy Pattern: similarity functions and t-norms remain pluggable internally
# - Registry Pattern: similarity aliases are still resolved by the core registry
# - Boundary Validation: validates user/downstream inputs at the public API edge
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api.similarity import build_similarity_matrix
#
# sim = build_similarity_matrix(
#     X,
#     similarity="gaussian",
#     similarity_sigma=0.5,
#     similarity_tnorm="minimum",
# )
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from FRsutils.core.similarities import (
    Similarity,
    build_similarity_matrix as _core_build_similarity_matrix,
    calculate_similarity_matrix as _core_calculate_similarity_matrix,
)


def _as_2d_feature_matrix(X: Any) -> np.ndarray:
    """
    @brief Convert and validate a public feature-matrix input.

    @param X: Candidate feature matrix.
    @return: 2D NumPy array view/copy.
    @raises ValueError: If X is missing or not two-dimensional.
    """
    if X is None:
        raise ValueError("X must be provided when building a similarity matrix.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like feature matrix.")
    return X_array


def list_similarities():
    """
    @brief List registered public similarity aliases.

    @return: Registry mapping from primary similarity names to aliases.
    """
    return Similarity.list_available()


def calculate_similarity_matrix(X: Any, similarity_func: Similarity, tnorm) -> np.ndarray:
    """
    @brief Compute a pairwise similarity matrix using already-built components.

    @param X: Normalized 2D feature matrix.
    @param similarity_func: Similarity component instance.
    @param tnorm: Binary T-norm component instance/callable.
    @return: Pairwise similarity matrix.
    """
    return _core_calculate_similarity_matrix(_as_2d_feature_matrix(X), similarity_func, tnorm)


def build_similarity_matrix(
    X: Any,
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> np.ndarray:
    """
    @brief Build a pairwise similarity matrix from flat or nested public config.

    This is the stable public entry point for users and downstream packages.
    It accepts:
    - flat sklearn-style params, e.g. `similarity="gaussian"`
    - nested FRsutils config, e.g. `{"similarity": {"name": ...}}`

    @param X: Normalized 2D feature matrix.
    @param config: Optional flat or nested FRsutils config mapping.
    @param flat_config: Additional flat configuration values.
    @return: Pairwise similarity matrix.
    @raises TypeError: If config is provided but is not mapping-like.
    @raises ValueError: If X is not a 2D matrix.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    X_array = _as_2d_feature_matrix(X)
    config_dict = dict(config) if config is not None else None
    return _core_build_similarity_matrix(X_array, config=config_dict, **flat_config)


__all__ = [
    "Similarity",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
    "list_similarities",
]
