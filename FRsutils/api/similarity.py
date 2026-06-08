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
# build_similarity_engine              Build dense/blockwise similarity engines
# backend="cupy"                       Optional GPU block computation for blockwise engines

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable import path for public similarity operations
# - Strategy Pattern: similarity functions and t-norms remain pluggable internally
# - Registry Pattern: similarity aliases are still resolved by the core registry
# - Boundary Validation: validates user/downstream inputs at the public API edge
# - Conservative Extension: engine support is additive and keeps dense behavior stable
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
from FRsutils.core.similarity_engine import (
    BaseSimilarityEngine,
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    SimilarityBlock,
    build_similarity_engine as _core_build_similarity_engine,
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


def build_similarity_engine(
    X: Any,
    *,
    engine: str = "dense",
    block_size: int = 1024,
    config: Optional[Mapping[str, Any]] = None,
    backend: str = "numpy",
    **flat_config: Any,
) -> BaseSimilarityEngine:
    """
    @brief Build a dense or blockwise similarity engine from public inputs.

    The engine abstraction is additive. Existing callers should keep using
    `build_similarity_matrix`; blockwise approximation code can consume the
    returned engine directly. backend="cupy" accelerates similarity-block
    calculation when CuPy is installed and a CUDA device is available.

    @param X: Normalized 2D feature matrix.
    @param engine: Engine alias, currently "dense" or "blockwise".
    @param block_size: Positive block size for blockwise engines.
    @param config: Optional flat or nested FRsutils config mapping.
    @param backend: Backend alias. Use "numpy"/"auto" or explicit optional "cupy".
    @param flat_config: Additional flat configuration values.
    @return: Similarity engine instance.
    @raises TypeError: If config is not mapping-like.
    @raises ValueError: If X is not a 2D matrix or engine/backend is unsupported.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    X_array = _as_2d_feature_matrix(X)
    config_dict = dict(config) if config is not None else None
    return _core_build_similarity_engine(
        X_array,
        engine=engine,
        block_size=block_size,
        config=config_dict,
        backend=backend,
        **flat_config,
    )


__all__ = [
    "Similarity",
    "SimilarityBlock",
    "BaseSimilarityEngine",
    "BlockwiseSimilarityEngine",
    "DenseSimilarityEngine",
    "build_similarity_engine",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
    "list_similarities",
]
