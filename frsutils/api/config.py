# SPDX-License-Identifier: BSD-3-Clause
"""Configuration helpers and flat public-parameter validation for frsutils.

This module defines the stable naming contract used to route flat public
parameters to registered similarity, T-norm, implicator, OWA, and fuzzy
quantifier components.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional, Set

from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from frsutils.core.implicators import Implicator
from frsutils.core.owa_weights import OWAWeights
from frsutils.core.similarities import Similarity
from frsutils.core.tnorms import TNorm
from frsutils.utils.init_helpers import (
    apply_config_aliases,
    extract_prefixed_params,
    normalize_flat_config_to_nested,
)


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

_GENERAL_FLAT_KEYS: Set[str] = {
    "type",
    "sampling_strategy",
    "instance_ranking_strategy",
    "sampling_ratio",
    "k_neighbors",
    "random_state",
    "bias_interpolation",
    "logger",
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


def _canonicalize_explicit_keys(flat_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return explicitly supplied flat keys mapped to canonical public names."""
    canonical: Dict[str, Any] = {}
    for key, value in flat_config.items():
        canonical_key = _LEGACY_KEY_ALIASES.get(key, key)
        if canonical_key not in canonical or key == canonical_key:
            canonical[canonical_key] = value
    return canonical


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


def _component_explicit_keys(flat_config: Mapping[str, Any], component_name: str) -> Set[str]:
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
) -> None:
    """Validate flat public configuration names against registered components.

    Parameters
    ----------
    flat_config : Mapping[str, Any]
        Explicit flat configuration supplied at a public API boundary.
    model : str or None, default=None
        Optional model alias used to reject parameters for components that the
        selected model does not consume.

    Raises
    ------
    TypeError
        If ``flat_config`` is not mapping-like.
    ValueError
        If a parameter name is unknown, is routed to the wrong model, is mixed
        with a direct component object, or is unsupported by the selected
        registered component.
    """
    if not isinstance(flat_config, Mapping):
        raise TypeError("flat_config must be a mapping (dict-like).")

    explicit = _canonicalize_explicit_keys(flat_config)
    normalized_model = model.strip().lower() if isinstance(model, str) and model.strip() else None

    for key in explicit:
        if key in _GENERAL_FLAT_KEYS:
            continue
        if _component_for_key(key) is None:
            raise ValueError(
                f"Unknown flat configuration parameter '{key}'. "
                "See docs/user/configuration.md for the public naming contract."
            )

    if normalized_model in _MODEL_COMPONENTS:
        allowed_model_components = _MODEL_COMPONENTS[normalized_model]
        for component_name in _COMPONENT_CONTRACTS:
            if component_name in {"similarity", "similarity_tnorm"}:
                continue
            component_keys = _component_explicit_keys(explicit, component_name)
            if component_keys and component_name not in allowed_model_components:
                offending = sorted(component_keys)[0]
                raise ValueError(
                    f"Flat parameter '{offending}' is not used by model='{normalized_model}'."
                )

    for component_name, contract in _COMPONENT_CONTRACTS.items():
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
                expected_keys = sorted(f"{prefix}_{name}" for name in accepted_params)
                expected_text = ", ".join(expected_keys) if expected_keys else "no component parameters"
                raise ValueError(
                    f"Flat parameter '{key}' is not supported for {selector}='{alias}'. "
                    f"Expected {expected_text}."
                )


__all__ = [
    "apply_config_aliases",
    "extract_prefixed_params",
    "normalize_flat_config_to_nested",
    "validate_flat_public_config",
]
