"""
@file approximations.py
@brief Task-oriented public APIs for fuzzy-rough approximations.

This module provides the user-facing approximation API that sits above the
lower-level public model and similarity builders. It is intended for two user
classes:

1) End users who want direct lower/upper/boundary/positive-region outputs.
2) Downstream packages that need a stable API but may precompute similarity
   matrices or pass model-specific flat parameters.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# compute_approximations               Compute lower/upper/boundary/positive region outputs
# compute_lower_approximation          Convenience wrapper for lower approximation
# compute_upper_approximation          Convenience wrapper for upper approximation
# compute_boundary_region              Convenience wrapper for boundary region
# compute_positive_region              Convenience wrapper for positive-region scores

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable task API above internal model/similarity builders
# - Adapter Pattern: accepts flat sklearn-style params or nested config
# - Dependency Inversion: downstream packages depend on FRsutils.api only
# - DRY: single compute_approximations implementation powers all wrappers
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api import compute_approximations, compute_positive_region
#
# result = compute_approximations(
#     X,
#     y,
#     model="itfrs",
#     similarity="linear",
#     ub_tnorm_name="minimum",
#     lb_implicator_name="lukasiewicz",
# )
# positive_scores = compute_positive_region(X, y, model="itfrs")
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import numpy as np

from FRsutils.api.models import build_fuzzy_rough_model
from FRsutils.api.results import FuzzyRoughApproximationResult
from FRsutils.api.similarity import build_similarity_matrix


_DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "type": "itfrs",
    "similarity": "linear",
    "similarity_tnorm": "minimum",
    "ub_tnorm_name": "minimum",
    "ub_tnorm_p": 2.0,
    "lb_implicator_name": "lukasiewicz",
    "ub_owa_method_name": "linear",
    "lb_owa_method_name": "linear",
    "ub_owa_method_base": 2.0,
    "lb_owa_method_base": 2.0,
    "lb_fuzzy_quantifier_name": "linear",
    "ub_fuzzy_quantifier_name": "linear",
    "lb_fuzzy_quantifier_alpha": 0.1,
    "lb_fuzzy_quantifier_beta": 0.6,
    "ub_fuzzy_quantifier_alpha": 0.1,
    "ub_fuzzy_quantifier_beta": 0.6,
}


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """
    @brief Return True when config looks like FRsutils internal nested config.

    @param config: Candidate config mapping.
    @return: True if fuzzy-rough nested sections are present.
    """
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _default_flat_config(model: str, similarity: Optional[str]) -> Dict[str, Any]:
    """
    @brief Build default flat config for public approximation APIs.

    @param model: Public fuzzy-rough model alias.
    @param similarity: Optional public similarity alias.
    @return: Flat configuration dictionary with safe defaults.
    """
    cfg = dict(_DEFAULT_MODEL_CONFIG)
    cfg["type"] = model
    if similarity is not None:
        cfg["similarity"] = similarity
    return cfg


def _default_nested_config(model: str, similarity: Optional[str], config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    @brief Fill missing required pieces in a nested config without overwriting explicit values.

    @param model: Public fuzzy-rough model alias.
    @param similarity: Optional public similarity alias.
    @param config: User-provided nested config.
    @return: Defensive copy with minimal defaults filled in.
    """
    nested = deepcopy(dict(config))
    nested.setdefault("similarity", {})
    nested.setdefault("similarity_tnorm", {})
    nested.setdefault("fr_model", {})

    nested["similarity"].setdefault("name", similarity or "linear")
    nested["similarity"].setdefault("params", {})
    nested["similarity_tnorm"].setdefault("name", "minimum")
    nested["similarity_tnorm"].setdefault("params", {})

    fr_cfg = nested["fr_model"]
    fr_cfg.setdefault("type", model)
    fr_cfg.setdefault("ub_tnorm", {"name": "minimum", "params": {}})
    fr_cfg.setdefault("lb_implicator", {"name": "lukasiewicz", "params": {}})
    fr_cfg.setdefault("ub_owa_method", {"name": "linear", "params": {"base": 2.0}})
    fr_cfg.setdefault("lb_owa_method", {"name": "linear", "params": {"base": 2.0}})
    fr_cfg.setdefault("lb_fuzzy_quantifier", {"name": "linear", "params": {"alpha": 0.1, "beta": 0.6}})
    fr_cfg.setdefault("ub_fuzzy_quantifier", {"name": "linear", "params": {"alpha": 0.1, "beta": 0.6}})
    return nested


def _prepare_effective_config(
    *,
    model: str,
    similarity: Optional[str],
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    @brief Merge user config with public defaults for approximation computation.

    @param model: Public fuzzy-rough model alias.
    @param similarity: Optional public similarity alias.
    @param config: Optional flat or nested config mapping.
    @param flat_config: Additional flat kwargs.
    @return: Effective flat or nested config.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    if config is not None and _is_nested_frs_config(config):
        nested = _default_nested_config(model, similarity, config)
        # Flat kwargs are intentionally not merged into nested config because a
        # nested config should be authoritative at the public API boundary.
        if flat_config:
            raise ValueError("Do not mix nested config with extra flat keyword parameters.")
        return nested

    effective = _default_flat_config(model, similarity)
    if config is not None:
        effective.update(dict(config))
    effective.update(dict(flat_config))
    effective["type"] = model
    if similarity is not None:
        effective["similarity"] = similarity
    return effective


def compute_approximations(
    X: Optional[np.ndarray],
    y: np.ndarray,
    *,
    model: str = "itfrs",
    similarity: Optional[str] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    config: Optional[Mapping[str, Any]] = None,
    return_similarity_matrix: bool = False,
    **flat_config: Any,
) -> FuzzyRoughApproximationResult:
    """
    @brief Compute fuzzy-rough lower, upper, boundary, and positive-region values.

    @param X: Input feature matrix. Required unless similarity_matrix is provided.
    @param y: Label vector aligned with X and/or similarity_matrix.
    @param model: Fuzzy-rough model alias, e.g. "itfrs", "owafrs", or "vqrs".
    @param similarity: Optional similarity alias for matrix construction.
    @param similarity_matrix: Optional precomputed pairwise similarity matrix.
    @param config: Optional flat or nested FRsutils config mapping.
    @param return_similarity_matrix: If True, include the matrix in the result object.
    @param flat_config: Additional flat sklearn-style model/similarity parameters.
    @return: FuzzyRoughApproximationResult with named approximation arrays.
    @raises ValueError: If neither X nor similarity_matrix is provided.
    """
    if not isinstance(model, str) or not model.strip():
        raise TypeError("model must be a non-empty string.")

    model_alias = model.strip().lower()
    labels = np.asarray(y)
    effective_config = _prepare_effective_config(
        model=model_alias,
        similarity=similarity,
        config=config,
        flat_config=flat_config,
    )

    sim = similarity_matrix
    if sim is None:
        if X is None:
            raise ValueError("X must be provided when similarity_matrix is not supplied.")
        if _is_nested_frs_config(effective_config):
            sim = build_similarity_matrix(np.asarray(X), config=effective_config)
        else:
            sim = build_similarity_matrix(np.asarray(X), **effective_config)
    else:
        sim = np.asarray(similarity_matrix)

    fr_model = build_fuzzy_rough_model(
        model_alias,
        similarity_matrix=sim,
        labels=labels,
        config=effective_config,
    )

    lower = np.asarray(fr_model.lower_approximation())
    upper = np.asarray(fr_model.upper_approximation())
    boundary = np.asarray(fr_model.boundary_region())
    positive_region = np.asarray(fr_model.positive_region())

    similarity_name = None
    if isinstance(effective_config.get("similarity"), Mapping):
        similarity_name = effective_config.get("similarity", {}).get("name")
    else:
        similarity_name = effective_config.get("similarity")

    return FuzzyRoughApproximationResult(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
        model=model_alias,
        similarity=similarity_name,
        similarity_matrix=sim if return_similarity_matrix else None,
        config=dict(effective_config),
    )


def compute_lower_approximation(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    @brief Compute only the lower approximation values.

    @param X: Input feature matrix, or None when similarity_matrix is provided in kwargs.
    @param y: Label vector.
    @param kwargs: Parameters forwarded to compute_approximations.
    @return: Lower approximation array.
    """
    return compute_approximations(X, y, **kwargs).lower


def compute_upper_approximation(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    @brief Compute only the upper approximation values.

    @param X: Input feature matrix, or None when similarity_matrix is provided in kwargs.
    @param y: Label vector.
    @param kwargs: Parameters forwarded to compute_approximations.
    @return: Upper approximation array.
    """
    return compute_approximations(X, y, **kwargs).upper


def compute_boundary_region(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    @brief Compute only the boundary-region values.

    @param X: Input feature matrix, or None when similarity_matrix is provided in kwargs.
    @param y: Label vector.
    @param kwargs: Parameters forwarded to compute_approximations.
    @return: Boundary-region array.
    """
    return compute_approximations(X, y, **kwargs).boundary


def compute_positive_region(X: Optional[np.ndarray], y: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    @brief Compute only the positive-region values.

    @param X: Input feature matrix, or None when similarity_matrix is provided in kwargs.
    @param y: Label vector.
    @param kwargs: Parameters forwarded to compute_approximations.
    @return: Positive-region score array.
    """
    return compute_approximations(X, y, **kwargs).positive_region


__all__ = [
    "compute_approximations",
    "compute_boundary_region",
    "compute_lower_approximation",
    "compute_positive_region",
    "compute_upper_approximation",
]
