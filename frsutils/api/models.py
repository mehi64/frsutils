# SPDX-License-Identifier: BSD-3-Clause
"""Public builders for fuzzy-rough model objects.

This module belongs to the stable public API layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel

# Import concrete model modules/classes so their @FuzzyRoughModel.register(...)
# decorators run before public registry lookup is used by downstream packages.
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.models.owafrs import OWAFRS
from frsutils.core.models.vqrs import VQRS
from frsutils.utils.init_helpers import normalize_flat_config_to_nested

from .config import (
    build_default_flat_config,
    prepare_flat_public_config,
    resolve_public_model_type,
)

def _as_similarity_matrix(similarity_matrix: Any) -> np.ndarray:
    """Convert and validate a public similarity-matrix input.

    Parameters
    ----------
    similarity_matrix : Any
        Candidate square pairwise similarity matrix.

    Returns
    -------
    np.ndarray
        2D NumPy array.

    Raises
    ------
    ValueError
        If the matrix is missing, non-square, not 2D, contains non-finite
        values, or represents fewer than two samples.
    """
    if similarity_matrix is None:
        raise ValueError("similarity_matrix must be provided.")

    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be a 2D array.")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix must be square.")
    if sim.shape[0] < 2:
        raise ValueError(
            "Fuzzy-rough approximation models require at least two samples."
        )
    if not np.isfinite(sim).all():
        raise ValueError("similarity_matrix must contain only finite values.")
    return sim

def _as_labels(labels: Any, *, expected_length: int) -> np.ndarray:
    """Convert and validate labels aligned with a similarity matrix.

    Parameters
    ----------
    labels : Any
        Candidate label vector.
    expected_length : int
        Required label length.

    Returns
    -------
    np.ndarray
        1D NumPy label array.

    Raises
    ------
    ValueError
        If labels are missing or misaligned.
    """
    if labels is None:
        raise ValueError("labels must be provided.")

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array-like vector.")
    if len(labels_array) != expected_length:
        raise ValueError("Length of labels must match similarity_matrix size.")
    return labels_array

def get_fuzzy_rough_model_class(model_type: str):
    """Resolve a registered fuzzy-rough model class by public alias.

    Parameters
    ----------
    model_type : str
        Registered model alias, e.g. "itfrs", "owafrs", or "vqrs".

    Returns
    -------
    object
        Registered model class.

    Raises
    ------
    TypeError
        If model_type is not a non-empty string.
    ValueError
        If no model is registered for the alias.
    """
    if not isinstance(model_type, str) or not model_type.strip():
        raise TypeError("model_type must be a non-empty string.")
    return FuzzyRoughModel.get_class(model_type.strip().lower())

def list_fuzzy_rough_models() -> Dict[str, list[str]]:
    """List available fuzzy-rough model aliases.

    Returns
    -------
    Dict[str, list[str]]
        Mapping from primary alias to all aliases registered for each model.
    """
    return FuzzyRoughModel.list_available()

def build_fuzzy_rough_model(
    model_type: Optional[str] = None,
    *,
    similarity_matrix: Any,
    labels: Any,
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> FuzzyRoughModel:
    """Build a registered fuzzy-rough model from flat public configuration.

    The model type is resolved from the explicit argument, then flat config, and
    otherwise defaults to ITFRS. Conflicting explicit sources are rejected.
    Normalized nested component configuration remains an internal frsutils
    implementation detail.

    Parameters
    ----------
    model_type : str or None, default=None
        Optional explicit model alias. Must agree with ``type`` in flat config
        when both are provided.
    similarity_matrix : Any
        Pairwise similarity matrix used by the model.
    labels : Any
        Label vector aligned with the similarity matrix.
    config : Mapping[str, Any] or None, default=None
        Optional flat public model configuration mapping.
    flat_config : Any
        Additional flat model configuration values.

    Returns
    -------
    FuzzyRoughModel
        Constructed fuzzy-rough model instance.

    Raises
    ------
    TypeError
        If config is not mapping-like.
    ValueError
        If nested config, an out-of-scope parameter, a conflicting model type,
        or invalid matrix/labels are provided. Fuzzy-rough model construction
        requires at least two aligned samples.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    sim = _as_similarity_matrix(similarity_matrix)
    labels_array = _as_labels(labels, expected_length=sim.shape[0])

    resolved_type = resolve_public_model_type(
        model_type=model_type,
        config=config,
        flat_config=flat_config,
    )

    external_config: Dict[str, Any] = dict(config or {})
    external_config.update(flat_config)
    explicit_config = prepare_flat_public_config(
        external_config,
        model=resolved_type,
        scope="model",
    )

    constructor_config = build_default_flat_config(resolved_type)
    constructor_config = prepare_flat_public_config(
        {
            key: value
            for key, value in constructor_config.items()
            if key not in {"similarity", "similarity_tnorm"}
        },
        model=resolved_type,
        scope="model",
    )
    constructor_config.update(explicit_config)
    constructor_config["type"] = resolved_type
    nested_config = normalize_flat_config_to_nested(constructor_config)

    model_cls = get_fuzzy_rough_model_class(resolved_type)

    # Dense model constructors already consume this private normalized config.
    constructor_config["_nested_config"] = nested_config

    return model_cls.from_config(
        similarity_matrix=sim,
        labels=labels_array,
        **constructor_config,
    )

__all__ = [
    "FuzzyRoughModel",
    "ITFRS",
    "OWAFRS",
    "VQRS",
    "build_fuzzy_rough_model",
    "get_fuzzy_rough_model_class",
    "list_fuzzy_rough_models",
]
