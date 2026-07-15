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

from .config import validate_flat_public_config

def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when config already looks like frsutils internal nested config.

    Parameters
    ----------
    config : Mapping[str, Any]
        Candidate configuration mapping.

    Returns
    -------
    bool
        True if the mapping contains nested fuzzy-rough config sections.
    """
    if isinstance(config.get("fr_model"), Mapping):
        return True
    if isinstance(config.get("similarity"), Mapping):
        return True
    if isinstance(config.get("similarity_tnorm"), Mapping):
        return True
    return False

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
        If the matrix is missing, non-square, or not 2D.
    """
    if similarity_matrix is None:
        raise ValueError("similarity_matrix must be provided.")

    sim = np.asarray(similarity_matrix, dtype=float)
    if sim.ndim != 2:
        raise ValueError("similarity_matrix must be a 2D array.")
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix must be square.")
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

def _resolve_model_type(
    *,
    model_type: Optional[str],
    external_config: Mapping[str, Any],
    nested_config: Mapping[str, Any],
) -> str:
    """Resolve the model alias from explicit, flat, or nested config sources.

    Parameters
    ----------
    model_type : Optional[str]
        Explicit model alias from the public positional/keyword arg.
    external_config : Mapping[str, Any]
        Flat-or-mixed public config snapshot.
    nested_config : Mapping[str, Any]
        Nested frsutils config snapshot.

    Returns
    -------
    str
        Normalized model alias.

    Raises
    ------
    ValueError
        If aliases conflict or no alias is available.
    """
    nested_type = None
    fr_cfg = nested_config.get("fr_model", {}) if isinstance(nested_config, Mapping) else {}
    if isinstance(fr_cfg, Mapping):
        nested_type = fr_cfg.get("type")

    flat_type = external_config.get("type")
    candidates = [value for value in (model_type, nested_type, flat_type) if value is not None]
    normalized = []
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("A fuzzy-rough model type must be a non-empty string.")
        normalized.append(value.strip().lower())

    if not normalized:
        raise ValueError("A fuzzy-rough model type must be provided via model_type or config['type'].")

    if len(set(normalized)) > 1:
        raise ValueError(f"Conflicting fuzzy-rough model types were provided: {sorted(set(normalized))}.")

    return normalized[0]

def build_fuzzy_rough_model(
    model_type: Optional[str] = None,
    *,
    similarity_matrix: Any,
    labels: Any,
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> FuzzyRoughModel:
    """Build a registered fuzzy-rough model from flat or nested config.

    This is the recommended public construction point for downstream packages.
    It accepts:
    - flat sklearn-style params, e.g. `type="itfrs"`, `ub_tnorm_name="minimum"`
    - nested frsutils config, e.g. `{"fr_model": {"type": "itfrs", ...}}`

    Parameters
    ----------
    model_type : Optional[str]
        Optional explicit model alias. Must agree with config when both are provided.
    similarity_matrix : Any
        Pairwise similarity matrix used by the model.
    labels : Any
        Label vector aligned with the similarity matrix.
    config : Optional[Mapping[str, Any]]
        Optional flat or nested configuration mapping.
    flat_config : Any
        Additional flat configuration values.

    Returns
    -------
    FuzzyRoughModel
        Constructed fuzzy-rough model instance.

    Raises
    ------
    TypeError
        If config is not mapping-like.
    ValueError
        If the model type or matrix/labels are invalid.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    sim = _as_similarity_matrix(similarity_matrix)
    labels_array = _as_labels(labels, expected_length=sim.shape[0])

    external_config: Dict[str, Any] = dict(config or {})
    external_config.update(flat_config)

    if _is_nested_frs_config(external_config):
        nested_config = dict(external_config)
        fr_cfg = nested_config.get("fr_model", {})
        if not isinstance(fr_cfg, Mapping):
            raise TypeError("nested config section 'fr_model' must be a mapping.")
    else:
        if model_type is not None:
            external_config["type"] = model_type
        nested_config = normalize_flat_config_to_nested(external_config)

    resolved_type = _resolve_model_type(
        model_type=model_type,
        external_config=external_config,
        nested_config=nested_config,
    )
    if config is not None and _is_nested_frs_config(config):
        validate_flat_public_config(flat_config, model=resolved_type)
    else:
        validate_flat_public_config(external_config, model=resolved_type)
    model_cls = get_fuzzy_rough_model_class(resolved_type)

    # Keep the original flat/mixed config for backwards-compatible `from_config`
    # paths, and pass nested config through the private key already used
    # internally by frsutils model constructors.
    constructor_config = dict(external_config)
    constructor_config["type"] = resolved_type
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
