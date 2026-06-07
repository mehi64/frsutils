"""
@file models.py
@brief Public fuzzy-rough model API for downstream FRsutils consumers.

This module exposes model construction and registry lookup through a stable API.
External packages such as a standalone `frsmote` package should use this module
instead of importing directly from deep internal model paths.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FuzzyRoughModel                      Public model registry base class
# ITFRS / OWAFRS / VQRS                Registered fuzzy-rough model classes
# get_fuzzy_rough_model_class          Resolve a registered model class by alias
# list_fuzzy_rough_models              List registered model aliases
# build_fuzzy_rough_model              Build a model from flat or nested config

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable public surface for model construction
# - Registry Pattern: resolves fuzzy-rough model implementations by alias
# - Factory Method: build_fuzzy_rough_model constructs model instances
# - Adapter Pattern: accepts flat sklearn-style config or nested internal config
# - Dependency Inversion: downstream packages depend on this API, not internals
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api.models import build_fuzzy_rough_model
#
# model = build_fuzzy_rough_model(
#     "itfrs",
#     similarity_matrix=sim,
#     labels=y,
#     ub_tnorm_name="minimum",
#     lb_implicator_name="lukasiewicz",
# )
# pos = model.positive_region()
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel

# Import concrete model modules/classes so their @FuzzyRoughModel.register(...)
# decorators run before public registry lookup is used by downstream packages.
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models.vqrs import VQRS
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """
    @brief Return True when config already looks like FRsutils internal nested config.

    @param config: Candidate configuration mapping.
    @return: True if the mapping contains nested fuzzy-rough config sections.

    Notes:
    - Flat configs may legitimately contain keys such as `similarity="gaussian"`.
      Therefore, a key alone is not enough to classify the config as nested.
    """
    if isinstance(config.get("fr_model"), Mapping):
        return True
    if isinstance(config.get("similarity"), Mapping):
        return True
    if isinstance(config.get("similarity_tnorm"), Mapping):
        return True
    return False


def get_fuzzy_rough_model_class(model_type: str):
    """
    @brief Resolve a registered fuzzy-rough model class by public alias.

    @param model_type: Registered model alias, e.g. "itfrs", "owafrs", or "vqrs".
    @return: Registered model class.
    @raises TypeError: If model_type is not a non-empty string.
    @raises ValueError: If no model is registered for the alias.
    """
    if not isinstance(model_type, str) or not model_type.strip():
        raise TypeError("model_type must be a non-empty string.")
    return FuzzyRoughModel.get_class(model_type.strip().lower())


def list_fuzzy_rough_models() -> Dict[str, list[str]]:
    """
    @brief List available fuzzy-rough model aliases.

    @return: Mapping from primary alias to all aliases registered for each model.
    """
    return FuzzyRoughModel.list_available()


def build_fuzzy_rough_model(
    model_type: Optional[str] = None,
    *,
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> FuzzyRoughModel:
    """
    @brief Build a registered fuzzy-rough model from flat or nested config.

    This is the recommended public construction point for downstream packages.
    It accepts:
    - flat sklearn-style params, e.g. `type="itfrs"`, `ub_tnorm_name="minimum"`
    - nested FRsutils config, e.g. `{"fr_model": {"type": "itfrs", ...}}`

    @param model_type: Optional explicit model alias. Overrides config when provided.
    @param similarity_matrix: Pairwise similarity matrix used by the model.
    @param labels: Label vector aligned with the similarity matrix.
    @param config: Optional flat or nested configuration mapping.
    @param flat_config: Additional flat configuration values.
    @return: Constructed fuzzy-rough model instance.
    @raises TypeError: If config is not mapping-like.
    @raises ValueError: If the model type cannot be resolved.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    external_config: Dict[str, Any] = dict(config or {})
    external_config.update(flat_config)

    if _is_nested_frs_config(external_config):
        nested_config = dict(external_config)
        fr_cfg = nested_config.get("fr_model", {})
        if not isinstance(fr_cfg, Mapping):
            raise TypeError("nested config section 'fr_model' must be a mapping.")
        resolved_type = model_type or fr_cfg.get("type") or external_config.get("type")
    else:
        if model_type is not None:
            external_config["type"] = model_type
        nested_config = normalize_flat_config_to_nested(external_config)
        resolved_type = model_type or nested_config.get("fr_model", {}).get("type") or external_config.get("type")

    if not isinstance(resolved_type, str) or not resolved_type.strip():
        raise ValueError("A fuzzy-rough model type must be provided via model_type or config['type'].")

    model_cls = get_fuzzy_rough_model_class(resolved_type)

    # Keep the original flat config for backwards-compatible `from_config` paths,
    # and pass nested config through the private key already used internally by
    # FRsutils model constructors.
    constructor_config = dict(external_config)
    constructor_config["type"] = resolved_type.strip().lower()
    constructor_config["_nested_config"] = nested_config

    return model_cls.from_config(
        similarity_matrix=similarity_matrix,
        labels=labels,
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
