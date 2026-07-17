# SPDX-License-Identifier: BSD-3-Clause
"""Public similarity-matrix construction API for frsutils.

This module belongs to the stable public API layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from frsutils.core.similarities import (
    Similarity,
    calculate_similarity_matrix as _core_calculate_similarity_matrix,
)
from frsutils.core.similarity_engine import (
    BaseSimilarityEngine,
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    SimilarityBlock,
    build_similarity_components as _core_build_similarity_components,
    build_similarity_engine as _core_build_similarity_engine,
)

from .config import prepare_flat_public_config

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
        If X is missing, not two-dimensional, or contains non-finite values.
    """
    if X is None:
        raise ValueError("X must be provided when building a similarity matrix.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like feature matrix.")
    if not np.isfinite(X_array).all():
        raise ValueError("X must contain only finite numeric values.")
    return X_array

def _prepare_public_similarity_config(
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return validated flat similarity config with stable public defaults.

    Parameters
    ----------
    config : Mapping[str, Any] or None
        Optional flat public similarity configuration mapping.
    flat_config : Mapping[str, Any]
        Additional flat public similarity configuration values.

    Returns
    -------
    Dict[str, Any]
        Canonical flat similarity configuration with stable defaults filled.

    Raises
    ------
    ValueError
        If nested, unknown, or out-of-scope configuration is supplied.
    """
    explicit: Dict[str, Any] = dict(config or {})
    explicit.update(dict(flat_config))
    explicit = prepare_flat_public_config(explicit, scope="similarity")

    effective: Dict[str, Any] = {
        "similarity": "linear",
        "similarity_tnorm": "minimum",
    }
    effective.update(explicit)
    return effective


def list_similarities() -> Dict[str, list[str]]:
    """List registered public similarity aliases.

    Returns
    -------
    Dict[str, list[str]]
        Mapping from primary similarity names to registered aliases.
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
    """Build a pairwise similarity matrix from flat public configuration.

    This is the stable public entry point for users and downstream packages.
    Component parameters follow the documented selector/prefix naming contract.

    Parameters
    ----------
    X : Any
        Normalized 2D feature matrix.
    config : Optional[Mapping[str, Any]]
        Optional flat public similarity configuration mapping.
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
        If X is not a finite 2D matrix or flat configuration is nested,
        unknown, or incompatible with the selected similarity component alias.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    X_array = _as_2d_feature_matrix(X)
    effective_config = _prepare_public_similarity_config(config, flat_config)
    similarity_func, tnorm_func, _ = _core_build_similarity_components(
        **effective_config,
    )
    return _core_calculate_similarity_matrix(X_array, similarity_func, tnorm_func)

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
        Optional flat public similarity configuration mapping.
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
        If X is not a finite 2D matrix, engine/backend is unsupported, or flat
        configuration is nested, unknown, or incompatible with the selected
        similarity component alias.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    X_array = _as_2d_feature_matrix(X)
    effective_config = _prepare_public_similarity_config(config, flat_config)
    return _core_build_similarity_engine(
        X_array,
        engine=engine,
        block_size=block_size,
        backend=backend,
        **effective_config,
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
