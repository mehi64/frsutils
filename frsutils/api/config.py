# SPDX-License-Identifier: BSD-3-Clause
"""Flat public configuration defaults, routing, and parameter validation.

This module centralizes the flat parameter contract used by public frsutils API
endpoints while keeping normalized nested component configuration internal.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional, Set

from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from frsutils.core.implicators import Implicator
from frsutils.core.models.vqrs_components import build_default_vqrs_flat_config
from frsutils.core.owa_weights import OWAWeights
from frsutils.core.similarities import Similarity
from frsutils.core.tnorms import TNorm


_COMPONENT_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "similarity": {
        "selector": "similarity",
        "prefix": "similarity",
        "registry": Similarity,
        "default": "linear",
        "direct_object_key": None,
    },
    "similarity_tnorm": {
        "selector": "similarity_tnorm",
        "prefix": "similarity_tnorm",
        "registry": TNorm,
        "default": "minimum",
        "direct_object_key": None,
    },
    "ub_tnorm": {
        "selector": "ub_tnorm_name",
        "prefix": "ub_tnorm",
        "registry": TNorm,
        "default": "minimum",
        "direct_object_key": "ub_tnorm",
    },
    "lb_implicator": {
        "selector": "lb_implicator_name",
        "prefix": "lb_implicator",
        "registry": Implicator,
        "default": "lukasiewicz",
        "direct_object_key": "lb_implicator",
    },
    "ub_owa_method": {
        "selector": "ub_owa_method_name",
        "prefix": "ub_owa_method",
        "registry": OWAWeights,
        "default": "linear",
        "direct_object_key": "ub_owa_method",
    },
    "lb_owa_method": {
        "selector": "lb_owa_method_name",
        "prefix": "lb_owa_method",
        "registry": OWAWeights,
        "default": "linear",
        "direct_object_key": "lb_owa_method",
    },
    "lb_fuzzy_quantifier": {
        "selector": "lb_fuzzy_quantifier_name",
        "prefix": "lb_fuzzy_quantifier",
        "registry": FuzzyQuantifier,
        "default": "linear",
        "direct_object_key": "lb_fuzzy_quantifier",
    },
    "ub_fuzzy_quantifier": {
        "selector": "ub_fuzzy_quantifier_name",
        "prefix": "ub_fuzzy_quantifier",
        "registry": FuzzyQuantifier,
        "default": "linear",
        "direct_object_key": "ub_fuzzy_quantifier",
    },
}

_MODEL_COMPONENTS: Dict[str, Set[str]] = {
    "itfrs": {"ub_tnorm", "lb_implicator"},
    "owafrs": {"ub_tnorm", "lb_implicator", "ub_owa_method", "lb_owa_method"},
    "vqrs": {"lb_fuzzy_quantifier", "ub_fuzzy_quantifier"},
}

_SCOPE_COMPONENTS: Dict[str, Set[str]] = {
    "approximation": set(_COMPONENT_CONTRACTS),
    "model": set(_COMPONENT_CONTRACTS) - {"similarity", "similarity_tnorm"},
    "similarity": {"similarity", "similarity_tnorm"},
}

_SCOPE_GENERAL_KEYS: Dict[str, Set[str]] = {
    "approximation": {"type", "logger"},
    "model": {"type", "logger"},
    "similarity": set(),
}

_LEGACY_KEY_ALIASES: Dict[str, str] = {
    "similarity_name": "similarity",
    "similarity_tnorm_name": "similarity_tnorm",
    "gaussian_similarity_sigma": "similarity_sigma",
    "lb_alpha": "lb_fuzzy_quantifier_alpha",
    "lb_beta": "lb_fuzzy_quantifier_beta",
    "ub_alpha": "ub_fuzzy_quantifier_alpha",
    "ub_beta": "ub_fuzzy_quantifier_beta",
    "sigma": "similarity_sigma",
}

_INTERNAL_NESTED_SECTION_KEYS = {"fr_model", "oversampler"}
_DEFAULT_MODEL_TYPE = "itfrs"

_DEFAULT_BASE_CONFIG: Dict[str, Any] = {
    "similarity": "linear",
    "similarity_tnorm": "minimum",
}

_DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "itfrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    },
    "owafrs": {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
    },
    "vqrs": build_default_vqrs_flat_config(),
}


def _normalize_scope(scope: str) -> str:
    """Return a canonical public validation scope."""
    if not isinstance(scope, str) or not scope.strip():
        raise TypeError("scope must be a non-empty string.")
    normalized = scope.strip().lower()
    if normalized not in _SCOPE_COMPONENTS:
        raise ValueError(
            "Unknown public configuration scope. Use 'approximation', "
            "'model', or 'similarity'."
        )
    return normalized


def _normalize_model_type(value: Any) -> str:
    """Normalize one model alias or raise for an invalid explicit value."""
    if not isinstance(value, str) or not value.strip():
        raise TypeError("A fuzzy-rough model type must be a non-empty string.")
    return value.strip().lower()


def _reject_internal_nested_config(config: Mapping[str, Any]) -> None:
    """Reject normalized nested configuration at a public API boundary."""
    has_nested_section = any(key in config for key in _INTERNAL_NESTED_SECTION_KEYS)
    has_nested_similarity = isinstance(config.get("similarity"), Mapping)
    has_nested_similarity_tnorm = isinstance(config.get("similarity_tnorm"), Mapping)
    if has_nested_section or has_nested_similarity or has_nested_similarity_tnorm:
        raise ValueError(
            "Nested configuration is internal to frsutils and is not accepted by "
            "the public API. Pass flat parameters using the documented selector "
            "and prefix naming contract."
        )


def canonicalize_flat_public_config(flat_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a defensive flat config copy with legacy keys canonicalized.

    Parameters
    ----------
    flat_config : Mapping[str, Any]
        Flat public configuration mapping.

    Returns
    -------
    Dict[str, Any]
        Canonical flat configuration. Canonical names take precedence over
        backward-compatible aliases when both are supplied.

    Raises
    ------
    TypeError
        If ``flat_config`` is not mapping-like.
    ValueError
        If internal nested configuration is supplied.
    """
    if not isinstance(flat_config, Mapping):
        raise TypeError("flat_config must be a mapping (dict-like).")

    _reject_internal_nested_config(flat_config)
    canonical: Dict[str, Any] = {}
    canonical_keys = set(flat_config) - set(_LEGACY_KEY_ALIASES)
    for key, value in flat_config.items():
        canonical_key = _LEGACY_KEY_ALIASES.get(key, key)
        if key in _LEGACY_KEY_ALIASES and canonical_key in canonical_keys:
            continue
        canonical[canonical_key] = value
    return canonical


def _model_types_from_mapping(config: Mapping[str, Any]) -> list[Any]:
    """Return model-type values explicitly present in a flat config mapping."""
    canonical = canonicalize_flat_public_config(config)
    return [canonical["type"]] if canonical.get("type") is not None else []


def resolve_public_model_type(
    *,
    model_type: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    flat_config: Optional[Mapping[str, Any]] = None,
) -> str:
    """Resolve a model alias from explicit flat public configuration sources.

    Parameters
    ----------
    model_type : str or None, default=None
        Explicit public model argument.
    config : Mapping[str, Any] or None, default=None
        Optional flat public configuration mapping.
    flat_config : Mapping[str, Any] or None, default=None
        Additional flat public keyword parameters.

    Returns
    -------
    str
        Normalized model alias. The default is ``"itfrs"`` when no source
        supplies a model type.

    Raises
    ------
    TypeError
        If a config is not mapping-like or an explicit model value is invalid.
    ValueError
        If nested config is supplied or explicit sources conflict.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")
    if flat_config is not None and not isinstance(flat_config, Mapping):
        raise TypeError("flat_config must be a mapping when provided.")

    candidates = []
    if model_type is not None:
        candidates.append(model_type)
    if config is not None:
        candidates.extend(_model_types_from_mapping(config))
    if flat_config is not None:
        candidates.extend(_model_types_from_mapping(flat_config))

    if not candidates:
        return _DEFAULT_MODEL_TYPE

    normalized = [_normalize_model_type(value) for value in candidates]
    unique_types = set(normalized)
    if len(unique_types) > 1:
        raise ValueError(
            "Conflicting fuzzy-rough model types were provided: "
            f"{sorted(unique_types)}."
        )
    return normalized[0]


def build_default_flat_config(
    model: str,
    similarity: Optional[str] = None,
) -> Dict[str, Any]:
    """Build authoritative flat defaults for a public approximation request.

    Parameters
    ----------
    model : str
        Resolved fuzzy-rough model alias.
    similarity : str or None, default=None
        Optional similarity alias overriding the public linear default.

    Returns
    -------
    Dict[str, Any]
        Flat public configuration containing base and model defaults.
    """
    model_alias = _normalize_model_type(model)
    config = dict(_DEFAULT_BASE_CONFIG)
    config.update(_DEFAULT_MODEL_CONFIGS.get(model_alias, {}))
    config["type"] = model_alias
    if similarity is not None:
        config["similarity"] = similarity
    return config


def _component_for_key(key: str) -> Optional[str]:
    """Return the most specific component contract associated with a flat key."""
    for component_name, contract in _COMPONENT_CONTRACTS.items():
        if key == contract["selector"] or key == contract["direct_object_key"]:
            return component_name

    ordered_components = sorted(
        _COMPONENT_CONTRACTS.items(),
        key=lambda item: len(item[1]["prefix"]),
        reverse=True,
    )
    for component_name, contract in ordered_components:
        if key.startswith(f"{contract['prefix']}_"):
            return component_name
    return None


def _accepted_constructor_params(registry: Any, alias: str) -> Set[str]:
    """Return named constructor parameters accepted by a registered alias."""
    target_cls = registry.get_class(alias)
    signature = inspect.signature(target_cls.__init__)
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }


def _component_explicit_keys(
    flat_config: Mapping[str, Any],
    component_name: str,
) -> Set[str]:
    """Return explicit canonical keys belonging to one component contract."""
    return {
        key
        for key in flat_config
        if _component_for_key(key) == component_name
    }


def validate_flat_public_config(
    flat_config: Mapping[str, Any],
    *,
    model: Optional[str] = None,
    scope: str = "approximation",
) -> None:
    """Validate flat names for one public API endpoint scope.

    Parameters
    ----------
    flat_config : Mapping[str, Any]
        Explicit flat configuration supplied at a public API boundary.
    model : str or None, default=None
        Optional model alias used to reject parameters for components that the
        selected model does not consume.
    scope : {"approximation", "model", "similarity"}, default="approximation"
        Public endpoint contract against which parameter names are validated.

    Raises
    ------
    TypeError
        If ``flat_config`` is not mapping-like.
    ValueError
        If nested configuration is supplied, a parameter name is unknown or
        out of scope, or a component parameter is incompatible with the selected
        model or registered component alias.
    """
    normalized_scope = _normalize_scope(scope)
    explicit = canonicalize_flat_public_config(flat_config)
    allowed_components = _SCOPE_COMPONENTS[normalized_scope]
    allowed_general_keys = _SCOPE_GENERAL_KEYS[normalized_scope]
    normalized_model = (
        model.strip().lower()
        if isinstance(model, str) and model.strip()
        else None
    )

    for key in explicit:
        if key in allowed_general_keys:
            continue
        component_name = _component_for_key(key)
        if component_name is None:
            raise ValueError(
                f"Unknown flat configuration parameter '{key}'. "
                "See docs/user/configuration.md for the public naming contract."
            )
        if component_name not in allowed_components:
            raise ValueError(
                f"Flat parameter '{key}' is not accepted by the "
                f"{normalized_scope} public API."
            )

    if normalized_model in _MODEL_COMPONENTS:
        allowed_model_components = _MODEL_COMPONENTS[normalized_model]
        for component_name in allowed_components:
            if component_name in {"similarity", "similarity_tnorm"}:
                continue
            component_keys = _component_explicit_keys(explicit, component_name)
            if component_keys and component_name not in allowed_model_components:
                offending = sorted(component_keys)[0]
                raise ValueError(
                    f"Flat parameter '{offending}' is not used by "
                    f"model='{normalized_model}'."
                )

    for component_name in allowed_components:
        contract = _COMPONENT_CONTRACTS[component_name]
        component_keys = _component_explicit_keys(explicit, component_name)
        if not component_keys:
            continue

        selector = contract["selector"]
        prefix = contract["prefix"]
        direct_object_key = contract["direct_object_key"]

        if direct_object_key in component_keys and len(component_keys) > 1:
            raise ValueError(
                f"Do not mix direct component '{direct_object_key}' with flat "
                f"'{prefix}_*' parameters."
            )
        if direct_object_key in component_keys:
            continue

        alias = explicit.get(selector, contract["default"])
        if not isinstance(alias, str) or not alias.strip():
            raise TypeError(f"{selector} must be a non-empty string alias.")
        alias = alias.strip().lower()

        accepted_params = _accepted_constructor_params(contract["registry"], alias)
        parameter_keys = component_keys - {selector}
        for key in parameter_keys:
            parameter_name = key[len(prefix) + 1 :]
            if parameter_name not in accepted_params:
                expected_keys = sorted(
                    f"{prefix}_{name}" for name in accepted_params
                )
                expected_text = (
                    ", ".join(expected_keys)
                    if expected_keys
                    else "no component parameters"
                )
                raise ValueError(
                    f"Flat parameter '{key}' is not supported for "
                    f"{selector}='{alias}'. Expected {expected_text}."
                )


def prepare_flat_public_config(
    flat_config: Mapping[str, Any],
    *,
    model: Optional[str] = None,
    scope: str = "approximation",
) -> Dict[str, Any]:
    """Canonicalize and validate flat configuration for a public endpoint.

    Parameters
    ----------
    flat_config : Mapping[str, Any]
        Public flat configuration mapping.
    model : str or None, default=None
        Optional resolved model alias for model-specific validation.
    scope : {"approximation", "model", "similarity"}, default="approximation"
        Public endpoint contract.

    Returns
    -------
    Dict[str, Any]
        Defensive canonical flat configuration copy.
    """
    canonical = canonicalize_flat_public_config(flat_config)
    validate_flat_public_config(canonical, model=model, scope=scope)
    return canonical


def select_flat_public_config(
    flat_config: Mapping[str, Any],
    *,
    scope: str,
) -> Dict[str, Any]:
    """Select flat parameters consumed by one public endpoint scope.

    Parameters
    ----------
    flat_config : Mapping[str, Any]
        Canonical or alias-bearing flat configuration mapping.
    scope : {"approximation", "model", "similarity"}
        Target endpoint scope.

    Returns
    -------
    Dict[str, Any]
        Flat configuration subset routed to the selected endpoint.
    """
    normalized_scope = _normalize_scope(scope)
    canonical = canonicalize_flat_public_config(flat_config)
    allowed_components = _SCOPE_COMPONENTS[normalized_scope]
    allowed_general_keys = _SCOPE_GENERAL_KEYS[normalized_scope]

    selected: Dict[str, Any] = {}
    for key, value in canonical.items():
        component_name = _component_for_key(key)
        if key in allowed_general_keys or component_name in allowed_components:
            selected[key] = value
    return selected


__all__ = [
    "build_default_flat_config",
    "canonicalize_flat_public_config",
    "prepare_flat_public_config",
    "resolve_public_model_type",
    "select_flat_public_config",
    "validate_flat_public_config",
]
