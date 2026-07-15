# SPDX-License-Identifier: BSD-3-Clause
"""Task-oriented public APIs for fuzzy-rough approximations.

This module is the stable public boundary for dense and blockwise
approximation computation. Public result arrays are returned as NumPy arrays,
even when optional backend-aware blockwise internals use CuPy.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from .config import (
    build_default_flat_config,
    prepare_flat_public_config,
    resolve_public_model_type,
    select_flat_public_config,
)
from .models import build_fuzzy_rough_model
from .results import FuzzyRoughApproximationResult
from .similarity import build_similarity_engine, build_similarity_matrix
from frsutils.core.approximation_engines import (
    compute_itfrs_blockwise,
    compute_owafrs_blockwise,
    compute_vqrs_blockwise,
)

def _as_public_labels(y: Any) -> np.ndarray:
    """Return public labels as a validated one-dimensional NumPy array.

    Parameters
    ----------
    y : Any
        Candidate class-label vector supplied to the public approximation API.

    Returns
    -------
    np.ndarray
        One-dimensional label array.

    Raises
    ------
    ValueError
        If labels are missing or not one-dimensional.
    """
    if y is None:
        raise ValueError("y/labels must be provided as a 1D label vector.")

    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D label vector.")
    return labels

def _validate_x_label_alignment(X: Optional[Any], labels: np.ndarray) -> None:
    """Validate sample-count alignment when X is supplied.

    Parameters
    ----------
    X : Optional[Any]
        Optional feature matrix.
    labels : np.ndarray
        One-dimensional label vector.

    Raises
    ------
    ValueError
        If X is two-dimensional and its sample count does not match labels.
    """
    if X is None:
        return

    X_array = np.asarray(X)
    if X_array.ndim == 2 and X_array.shape[0] != len(labels):
        raise ValueError("Length of labels must match X sample count.")

def _as_validated_similarity_matrix(
    similarity_matrix: Optional[Any],
    labels: np.ndarray,
) -> Optional[np.ndarray]:
    """Validate an optional precomputed similarity matrix.

    Parameters
    ----------
    similarity_matrix : Optional[Any]
        Optional pairwise similarity matrix.
    labels : np.ndarray
        One-dimensional label vector used to check matrix size.

    Returns
    -------
    Optional[np.ndarray]
        Float NumPy matrix when supplied, otherwise None.

    Raises
    ------
    ValueError
        If the matrix is not square or is not aligned with labels.
    """
    if similarity_matrix is None:
        return None

    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be a 2D square matrix.")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix must be square.")
    if sim.shape[0] != len(labels):
        raise ValueError("similarity_matrix size must match the length of y.")
    return sim

def _prepare_effective_config(
    *,
    model: str,
    similarity: Optional[str],
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge flat public config with authoritative approximation defaults.

    Parameters
    ----------
    model : str
        Resolved fuzzy-rough model alias.
    similarity : str or None
        Optional explicit public similarity alias.
    config : Mapping[str, Any] or None
        Optional flat public approximation configuration mapping.
    flat_config : Mapping[str, Any]
        Additional flat public keyword parameters.

    Returns
    -------
    Dict[str, Any]
        Canonical effective flat approximation configuration.

    Raises
    ------
    TypeError
        If ``config`` is not mapping-like.
    ValueError
        If nested, unknown, out-of-scope, or model-incompatible parameters are
        supplied.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    explicit_flat: Dict[str, Any] = dict(config or {})
    explicit_flat.update(dict(flat_config))
    if similarity is not None:
        explicit_flat["similarity"] = similarity
    explicit_flat = prepare_flat_public_config(
        explicit_flat,
        model=model,
        scope="approximation",
    )

    effective = build_default_flat_config(model, similarity)
    effective.update(explicit_flat)
    effective["type"] = model
    if similarity is not None:
        effective["similarity"] = similarity
    return effective


def _normalize_execution_engine(engine: str) -> str:
    """Normalize the approximation execution-engine alias.

    Parameters
    ----------
    engine : str
        Public execution-engine alias.

    Returns
    -------
    str
        Canonical alias, either "dense" or "blockwise".

    Raises
    ------
    TypeError
        If engine is not a non-empty string.
    ValueError
        If engine is unknown.
    """
    if not isinstance(engine, str) or not engine.strip():
        raise TypeError("engine must be a non-empty string.")

    normalized = engine.strip().lower()
    if normalized in {"dense", "full", "matrix"}:
        return "dense"
    if normalized in {"blockwise", "chunkwise", "blocked"}:
        return "blockwise"
    raise ValueError("Unknown approximation engine. Use engine='dense' or engine='blockwise'.")

def _similarity_name_from_config(
    effective_config: Mapping[str, Any],
) -> Optional[str]:
    """Return the similarity alias from effective flat configuration."""
    similarity_name = effective_config.get("similarity")
    return similarity_name if isinstance(similarity_name, str) else None


def _explicit_similarity_config(
    *,
    similarity: Optional[str],
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return explicitly supplied similarity parameters from public inputs."""
    explicit_config: Dict[str, Any] = dict(config or {})
    explicit_config.update(dict(flat_config))
    if similarity is not None:
        explicit_config["similarity"] = similarity
    return select_flat_public_config(explicit_config, scope="similarity")


def _without_similarity_config(
    effective_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return an effective config snapshot without unused similarity settings."""
    similarity_config = select_flat_public_config(
        effective_config,
        scope="similarity",
    )
    return {
        key: value
        for key, value in effective_config.items()
        if key not in similarity_config
    }


def _compute_dense_approximations(
    *,
    X: Optional[np.ndarray],
    labels: np.ndarray,
    model_alias: str,
    similarity_matrix: Optional[np.ndarray],
    effective_config: Mapping[str, Any],
    return_similarity_matrix: bool,
) -> FuzzyRoughApproximationResult:
    """Compute approximations through the existing dense model path.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Optional feature matrix used when no precomputed matrix is supplied.
    labels : np.ndarray
        Label vector.
    model_alias : str
        Normalized fuzzy-rough model alias.
    similarity_matrix : Optional[np.ndarray]
        Optional precomputed dense pairwise matrix.
    effective_config : Mapping[str, Any]
        Flat config snapshot.
    return_similarity_matrix : bool
        Whether to include the dense matrix in the result.

    Returns
    -------
    FuzzyRoughApproximationResult
        Public approximation result object.
    """
    similarity_config = select_flat_public_config(
        effective_config,
        scope="similarity",
    )
    model_config = select_flat_public_config(
        effective_config,
        scope="model",
    )

    sim = similarity_matrix
    if sim is None:
        if X is None:
            raise ValueError("X must be provided when similarity_matrix is not supplied.")
        sim = build_similarity_matrix(np.asarray(X), **similarity_config)
    else:
        sim = np.asarray(similarity_matrix)

    fr_model = build_fuzzy_rough_model(
        model_alias,
        similarity_matrix=sim,
        labels=labels,
        config=model_config,
    )

    lower = np.asarray(fr_model.lower_approximation())
    upper = np.asarray(fr_model.upper_approximation())
    boundary = upper - lower
    positive_region = lower.copy()

    return FuzzyRoughApproximationResult(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
        model=model_alias,
        similarity=_similarity_name_from_config(effective_config),
        similarity_matrix=sim if return_similarity_matrix else None,
        config=dict(effective_config),
        engine="dense",
        backend="numpy",
        block_size=None,
        used_blockwise=False,
        used_gpu_similarity_blocks=False,
        used_gpu_approximation_accumulators=False,
    )

def _compute_blockwise_approximations(
    *,
    X: Optional[np.ndarray],
    labels: np.ndarray,
    model_alias: str,
    similarity_matrix: Optional[np.ndarray],
    effective_config: Mapping[str, Any],
    return_similarity_matrix: bool,
    block_size: int,
    backend: str,
) -> FuzzyRoughApproximationResult:
    """Compute exact blockwise approximations for supported models.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Feature matrix required for blockwise similarity generation.
    labels : np.ndarray
        Label vector.
    model_alias : str
        Normalized model alias. Blockwise supports ITFRS, VQRS, and OWAFRS.
    similarity_matrix : Optional[np.ndarray]
        Must be None for blockwise execution.
    effective_config : Mapping[str, Any]
        Flat config snapshot.
    return_similarity_matrix : bool
        Whether to materialize and return the matrix for inspection.
    block_size : int
        Positive block size passed to the similarity engine.
    backend : str
        Array backend alias. NumPy/auto are stable; CuPy is optional and explicit.

    Returns
    -------
    FuzzyRoughApproximationResult
        Public approximation result object.
    """
    if model_alias not in {"itfrs", "vqrs", "owafrs"}:
        raise NotImplementedError(
            "Blockwise approximations currently support only model='itfrs', model='vqrs', or model='owafrs'."
        )
    if similarity_matrix is not None:
        raise ValueError("engine='blockwise' requires X and does not accept precomputed similarity_matrix.")
    if X is None:
        raise ValueError("X must be provided when engine='blockwise'.")

    similarity_config = select_flat_public_config(
        effective_config,
        scope="similarity",
    )
    similarity_engine = build_similarity_engine(
        np.asarray(X),
        engine="blockwise",
        block_size=block_size,
        backend=backend,
        **similarity_config,
    )

    if model_alias == "itfrs":
        blockwise = compute_itfrs_blockwise(similarity_engine, labels, config=effective_config)
    elif model_alias == "vqrs":
        blockwise = compute_vqrs_blockwise(similarity_engine, labels, config=effective_config)
    else:
        blockwise = compute_owafrs_blockwise(similarity_engine, labels, config=effective_config)
    similarity_matrix_for_result = similarity_engine.to_dense() if return_similarity_matrix else None
    resolved_backend = getattr(getattr(similarity_engine, "backend", None), "name", "numpy")

    return FuzzyRoughApproximationResult(
        lower=np.asarray(blockwise.lower),
        upper=np.asarray(blockwise.upper),
        boundary=np.asarray(blockwise.boundary),
        positive_region=np.asarray(blockwise.positive_region),
        model=model_alias,
        similarity=_similarity_name_from_config(effective_config),
        similarity_matrix=similarity_matrix_for_result,
        config=dict(effective_config),
        engine="blockwise",
        backend=resolved_backend,
        block_size=block_size,
        used_blockwise=True,
        used_gpu_similarity_blocks=resolved_backend == "cupy",
        used_gpu_approximation_accumulators=bool(
            getattr(blockwise, "used_gpu_approximation_accumulators", False)
        ),
    )

def compute_approximations(
    X: Optional[np.ndarray],
    y: np.ndarray,
    *,
    model: Optional[str] = None,
    similarity: Optional[str] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    config: Optional[Mapping[str, Any]] = None,
    return_similarity_matrix: bool = False,
    engine: str = "dense",
    block_size: int = 1024,
    backend: str = "numpy",
    **flat_config: Any,
) -> FuzzyRoughApproximationResult:
    """Compute fuzzy-rough lower, upper, boundary, and positive-region values.

    Dense execution uses direct model classes with materialized similarity
    matrices. ITFRS, VQRS, and OWAFRS direct classes are dense NumPy reference
    paths. Blockwise execution computes similarity blocks and model-specific
    reductions through the approximation-engine layer. Optional CuPy support
    is internal to blockwise execution; returned public arrays are NumPy
    arrays for downstream scientific Python compatibility. For OWAFRS, CuPy
    support currently means GPU-backed similarity blocks, not GPU-resident OWA
    aggregation buffers.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or None
        Input feature matrix. Required unless ``similarity_matrix`` is provided.
    y : array-like of shape (n_samples,)
        Label vector aligned with ``X`` or ``similarity_matrix``.
    model : str or None, default=None
        Optional fuzzy-rough model alias. When omitted, the model is resolved
        from config and otherwise defaults to "itfrs".
    similarity : str or None, default=None
        Optional similarity alias for matrix construction.
    similarity_matrix : ndarray of shape (n_samples, n_samples) or None, default=None
        Optional precomputed pairwise similarity matrix.
    config : Mapping[str, Any] or None, default=None
        Optional flat public approximation configuration mapping.
    return_similarity_matrix : bool, default=False
        If True, include the matrix in the result object.
    engine : {"dense", "blockwise"}, default="dense"
        Approximation execution engine, "dense" or "blockwise".
    block_size : int, default=1024
        Positive block size used by engine="blockwise".
    backend : str, default="numpy"
        Array backend alias. Use "numpy"/"auto" or explicit optional "cupy".
    flat_config : Any
        Additional flat sklearn-style model/similarity parameters.

    Returns
    -------
    result : FuzzyRoughApproximationResult
        Named approximation arrays and execution metadata.

    Raises
    ------
    TypeError
        If a public config mapping or selector has an invalid type.
    ValueError
        If inputs are misaligned, execution settings are incompatible, nested
        config is supplied, or flat configuration is unknown or incompatible
        with the selected model/component alias.
    """
    model_alias = resolve_public_model_type(
        model_type=model,
        config=config,
        flat_config=flat_config,
    )
    labels = _as_public_labels(y)
    _validate_x_label_alignment(X, labels)
    validated_similarity_matrix = _as_validated_similarity_matrix(similarity_matrix, labels)
    execution_engine = _normalize_execution_engine(engine)

    if execution_engine == "blockwise" and validated_similarity_matrix is not None:
        raise ValueError(
            "engine='blockwise' requires X and does not accept precomputed "
            "similarity_matrix."
        )

    if validated_similarity_matrix is not None:
        explicit_similarity = _explicit_similarity_config(
            similarity=similarity,
            config=config,
            flat_config=flat_config,
        )
        if explicit_similarity:
            offending = sorted(explicit_similarity)[0]
            raise ValueError(
                f"Similarity parameter '{offending}' cannot be supplied with "
                "a precomputed similarity_matrix because it would not be used."
            )

    effective_config = _prepare_effective_config(
        model=model_alias,
        similarity=similarity,
        config=config,
        flat_config=flat_config,
    )

    if validated_similarity_matrix is not None:
        effective_config = _without_similarity_config(effective_config)

    if execution_engine == "blockwise":
        return _compute_blockwise_approximations(
            X=X,
            labels=labels,
            model_alias=model_alias,
            similarity_matrix=validated_similarity_matrix,
            effective_config=effective_config,
            return_similarity_matrix=return_similarity_matrix,
            block_size=block_size,
            backend=backend,
        )

    return _compute_dense_approximations(
        X=X,
        labels=labels,
        model_alias=model_alias,
        similarity_matrix=validated_similarity_matrix,
        effective_config=effective_config,
        return_similarity_matrix=return_similarity_matrix,
    )

def compute_lower_approximation(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Compute only the lower approximation values.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Input feature matrix, or None when similarity_matrix is provided in kwargs.
    y : np.ndarray
        Label vector.
    kwargs : Any
        Parameters forwarded to compute_approximations.

    Returns
    -------
    np.ndarray
        Lower approximation array.
    """
    return compute_approximations(X, y, **kwargs).lower

def compute_upper_approximation(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Compute only the upper approximation values.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Input feature matrix, or None when similarity_matrix is provided in kwargs.
    y : np.ndarray
        Label vector.
    kwargs : Any
        Parameters forwarded to compute_approximations.

    Returns
    -------
    np.ndarray
        Upper approximation array.
    """
    return compute_approximations(X, y, **kwargs).upper

def compute_boundary_region(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Compute only the boundary-region values.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Input feature matrix, or None when similarity_matrix is provided in kwargs.
    y : np.ndarray
        Label vector.
    kwargs : Any
        Parameters forwarded to compute_approximations.

    Returns
    -------
    np.ndarray
        Boundary-region array.
    """
    return compute_approximations(X, y, **kwargs).boundary

def compute_positive_region(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Compute only the positive-region values.

    Parameters
    ----------
    X : Optional[np.ndarray]
        Input feature matrix, or None when similarity_matrix is provided in kwargs.
    y : np.ndarray
        Label vector.
    kwargs : Any
        Parameters forwarded to compute_approximations.

    Returns
    -------
    np.ndarray
        Positive-region score array.
    """
    return compute_approximations(X, y, **kwargs).positive_region

__all__ = [
    "compute_approximations",
    "compute_boundary_region",
    "compute_lower_approximation",
    "compute_positive_region",
    "compute_upper_approximation",
]
