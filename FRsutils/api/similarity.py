# SPDX-License-Identifier: BSD-3-Clause
"""Public similarity-matrix construction API for FRsutils.

This module belongs to the stable public API layer.
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
    """Convert and validate a public feature-matrix input.
        
        Parameters
        ----------
        X : Any
            Candidate feature matrix.
        
        Returns
        -------
        np.ndarray
            2D NumPy array view/copy.
        
        Raises
        ------
        ValueError
            If X is missing or not two-dimensional.
        
    """
    if X is None:
        raise ValueError("X must be provided when building a similarity matrix.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like feature matrix.")
    return X_array


def list_similarities():
    """List registered public similarity aliases.
    
    Returns
    -------
    object
        Registry mapping from primary similarity names to aliases.
    """
    return Similarity.list_available()


def calculate_similarity_matrix(X: Any, similarity_func: Similarity, tnorm) -> np.ndarray:
    """Compute a pairwise similarity matrix using already-built components.
        
        Parameters
        ----------
        X : Any
            Normalized 2D feature matrix.
        similarity_func : Similarity
            Similarity component instance.
        tnorm : object
            Binary T-norm component instance/callable.
        
        Returns
        -------
        np.ndarray
            Pairwise similarity matrix.
        
    """
    return _core_calculate_similarity_matrix(_as_2d_feature_matrix(X), similarity_func, tnorm)


def build_similarity_matrix(
    X: Any,
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> np.ndarray:
    """Build a pairwise similarity matrix from flat or nested public config.
        
        This is the stable public entry point for users and downstream packages.
        It accepts:
        - flat sklearn-style params, e.g. `similarity="gaussian"`
        - nested FRsutils config, e.g. `{"similarity": {"name": ...}}`
        
        Parameters
        ----------
        X : Any
            Normalized 2D feature matrix.
        config : Optional[Mapping[str, Any]]
            Optional flat or nested FRsutils config mapping.
        flat_config : Any
            Additional flat configuration values.
        
        Returns
        -------
        np.ndarray
            Pairwise similarity matrix.
        
        Raises
        ------
        TypeError
            If config is provided but is not mapping-like.
        ValueError
            If X is not a 2D matrix.
        
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
    """Build a dense or blockwise similarity engine from public inputs.
        
        The engine abstraction is additive. Existing callers should keep using
        `build_similarity_matrix`; blockwise approximation code can consume the
        returned engine directly. backend="cupy" accelerates similarity-block
        calculation when CuPy is installed and a CUDA device is available.
        
        Parameters
        ----------
        X : Any
            Normalized 2D feature matrix.
        engine : str
            Engine alias, currently "dense" or "blockwise".
        block_size : int
            Positive block size for blockwise engines.
        config : Optional[Mapping[str, Any]]
            Optional flat or nested FRsutils config mapping.
        backend : str
            Backend alias. Use "numpy"/"auto" or explicit optional "cupy".
        flat_config : Any
            Additional flat configuration values.
        
        Returns
        -------
        BaseSimilarityEngine
            Similarity engine instance.
        
        Raises
        ------
        TypeError
            If config is not mapping-like.
        ValueError
            If X is not a 2D matrix or engine/backend is unsupported.
        
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
