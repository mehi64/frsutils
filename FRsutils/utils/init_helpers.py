"""
@file init_helpers.py
@brief Helper utilities for initializing FRsutils components.

This module provides two small-but-critical helpers used across the project:

1) **assign_allowed_kwargs**
   - A lightweight schema-based assignment helper (legacy utility).

2) **normalize_flat_config_to_nested**
   - Converts a *scikit-learn friendly* flat parameter dictionary into an **internal nested config**.
   - The nested config isolates parameters by component (similarity, t-norm, implicator, etc.)
     to prevent collisions and to simplify object construction.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# assign_allowed_kwargs                Assign & validate kwargs via a small schema
# normalize_flat_config_to_nested      Flat -> nested config normalization
# apply_config_aliases                 Backward-compatible alias mapping
# extract_prefixed_params              Extract component params by prefix

# ✅ Design Patterns & Clean Code Notes
# - Adapter: Flat sklearn params -> internal nested config
# - SRP: This module contains only init/config helpers
# - Fail-fast: Input validation and safe defaults
# - Backward compatibility: Optional alias mapping for legacy configs

##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# External (flat) config for sklearn / GridSearchCV
# config = {
#     "type": "owafrs",
#     "similarity": "gaussian",
#     "similarity_sigma": 0.5,
#     "similarity_tnorm": "minimum",
#     "ub_tnorm_name": "product",
#     "lb_implicator_name": "lukasiewicz",
#     "ub_owa_method_name": "linear",
#     "lb_owa_method_name": "linear",
#     "k_neighbors": 3,
# }
# nested = normalize_flat_config_to_nested(config)
#
# nested["similarity"] == {"name": "gaussian", "params": {"sigma": 0.5}}

"""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Set


# -----------------------------------------------------------------------------
# Legacy helper
# -----------------------------------------------------------------------------

def assign_allowed_kwargs(instance, kwargs: dict, schema: dict):
    """
    @brief Assigns validated kwargs to instance attributes based on a schema.

    @param instance: The object to assign attributes to (usually 'self').
    @param kwargs: Dictionary of keyword arguments to extract from.
    @param schema: Dictionary of {key: spec} where spec may include:
        - 'type': 'float', 'int', 'str', 'bool' (required)
        - 'required': bool (default False)
        - 'default': default value if missing and not required
        - 'range': (min, max) for floats or ints
        - 'allowed': set of allowed values (for strings)

    @throws ValueError, TypeError if validation fails.
    """
    for key, spec in schema.items():
        # Get value: first from kwargs, then default if available
        value = kwargs.get(key, spec.get('default', None))

        if spec.get('required', False) and key not in kwargs:
            raise ValueError(f"Missing required parameter '{key}'.")

        if value is None:
            continue  # skip assigning if optional and not provided

        # Type checking
        expected_type = spec['type']
        if expected_type == 'float' and not isinstance(value, float):
            raise TypeError(f"Parameter '{key}' must be a float.")
        elif expected_type == 'int' and not isinstance(value, int):
            raise TypeError(f"Parameter '{key}' must be an int.")
        elif expected_type == 'str' and not isinstance(value, str):
            raise TypeError(f"Parameter '{key}' must be a string.")
        elif expected_type == 'bool' and not isinstance(value, bool):
            raise TypeError(f"Parameter '{key}' must be a bool.")

        # Range checking
        if 'range' in spec:
            lo, hi = spec['range']
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(f"Parameter '{key}' must be in range [{lo}, {hi}].")

        # Allowed values for enums (e.g., strategy type)
        if 'allowed' in spec and value not in spec['allowed']:
            raise ValueError(f"Parameter '{key}' must be one of {sorted(spec['allowed'])}.")

        # Assignment
        setattr(instance, key + '_name', value)


# -----------------------------------------------------------------------------
# Flat -> nested normalization
# -----------------------------------------------------------------------------

_ALIAS_MAP: Dict[str, str] = {
    # Similarity legacy aliases
    "similarity_name": "similarity",
    "similarity_tnorm_name": "similarity_tnorm",
    "gaussian_similarity_sigma": "similarity_sigma",

    # VQRS legacy aliases (older convention)
    "lb_alpha": "lb_fuzzy_quantifier_alpha",
    "lb_beta": "lb_fuzzy_quantifier_beta",
    "ub_alpha": "ub_fuzzy_quantifier_alpha",
    "ub_beta": "ub_fuzzy_quantifier_beta",
}


def apply_config_aliases(flat_config: MutableMapping[str, Any], *, explicit_keys: Optional[Set[str]] = None) -> None:
    """ 
    @brief Apply backward-compatible alias mapping in-place.

    @param flat_config: Mutable dict-like config.

    Notes:
    - Only fills missing new keys; does not overwrite explicitly provided new keys.
    """
    for legacy_key, new_key in _ALIAS_MAP.items():
        if legacy_key not in flat_config:
            continue

        # If the new key is missing, always fill it.
        if new_key not in flat_config:
            flat_config[new_key] = flat_config[legacy_key]
            continue

        # If we know which keys were explicitly provided by the caller,
        # allow legacy keys to override defaults (but NOT explicit new keys).
        if explicit_keys is not None:
            legacy_is_explicit = legacy_key in explicit_keys
            new_is_explicit = new_key in explicit_keys
            if legacy_is_explicit and not new_is_explicit:
                flat_config[new_key] = flat_config[legacy_key]
                continue

        # Otherwise, only fill if the new key is None.
        if flat_config.get(new_key) is None:
            flat_config[new_key] = flat_config[legacy_key]


def extract_prefixed_params(flat_config: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """
    @brief Extract component-specific params based on the naming standard.

    According to the flat naming standard, component parameters are encoded as:
        <prefix>_<param_name>

    Example:
        prefix="similarity" extracts:
            similarity_sigma -> {"sigma": value}

    @param flat_config: Flat config mapping.
    @param prefix: Component prefix.
    @return: Dict of extracted parameters without the prefix.
    """
    pfx = f"{prefix}_"
    out: Dict[str, Any] = {}
    for k, v in flat_config.items():
        if k.startswith(pfx):
            out[k[len(pfx):]] = v
    return out


def _component_spec_from_flat(
    flat_config: Mapping[str, Any],
    *,
    selector_key: str,
    prefix: str,
    explicit_obj_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """ 
    @brief Build a component spec from flat config.

    @param flat_config: Flat input mapping.
    @param selector_key: Key that contains component name (e.g., "ub_tnorm_name").
    @param prefix: Prefix for component parameters (e.g., "ub_tnorm").
    @param explicit_obj_key: Optional key that may contain a dict/instance (e.g., "ub_tnorm").
    @return: {"name": <str>, "params": <dict>} or an existing dict if provided.
    """
    if explicit_obj_key and explicit_obj_key in flat_config:
        obj = flat_config.get(explicit_obj_key)
        if obj is not None:
            # Could be instance or dict; consumers will handle.
            if isinstance(obj, dict):
                return dict(obj)
            return {"__instance__": obj}

    name = flat_config.get(selector_key)
    if name is None:
        return None

    params = extract_prefixed_params(flat_config, prefix)
    # Ensure selector itself does not leak into params.
    params.pop("name", None)
    return {"name": name, "params": params}


def normalize_flat_config_to_nested(
    flat_config: Mapping[str, Any],
    *,
    apply_aliases: bool = True,
    explicit_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    @brief Convert a flat sklearn-friendly config to an internal nested config.

    The returned nested structure isolates parameters by component to avoid naming collisions.

    @param flat_config: Flat mapping (e.g., estimator.get_params() / GridSearchCV params).
    @param apply_aliases: If True, supports a small set of legacy aliases.
    @return: Nested config dictionary.
    """
    if not isinstance(flat_config, Mapping):
        raise TypeError("flat_config must be a mapping (dict-like).")

    flat: Dict[str, Any] = dict(flat_config)
    if apply_aliases:
        apply_config_aliases(flat, explicit_keys=explicit_keys)

    oversampler_keys = {
        "sampling_strategy",
        "instance_ranking_strategy",
        "sampling_ratio",
        "k_neighbors",
        "random_state",
        "bias_interpolation",
    }

    nested: Dict[str, Any] = {
        "oversampler": {k: flat.get(k) for k in oversampler_keys if k in flat},
        "similarity": {
            "name": flat.get("similarity"),
            "params": extract_prefixed_params(flat, "similarity"),
        },
        "similarity_tnorm": {
            "name": flat.get("similarity_tnorm"),
            "params": extract_prefixed_params(flat, "similarity_tnorm"),
        },
        "fr_model": {
            "type": flat.get("type"),
        },
    }

    # Do not pass selector keys as component parameters.
    nested["similarity"]["params"].pop("name", None)
    nested["similarity_tnorm"]["params"].pop("name", None)

    # Remove accidental overlaps: similarity_tnorm_* should not be inside similarity.params
    nested["similarity"]["params"].pop("tnorm", None)

    # Model components (ITFRS/OWAFRS/VQRS)
    fr = nested["fr_model"]

    # ITFRS
    spec = _component_spec_from_flat(flat, selector_key="ub_tnorm_name", prefix="ub_tnorm", explicit_obj_key="ub_tnorm")
    if spec is not None:
        fr["ub_tnorm"] = spec

    spec = _component_spec_from_flat(flat, selector_key="lb_implicator_name", prefix="lb_implicator", explicit_obj_key="lb_implicator")
    if spec is not None:
        fr["lb_implicator"] = spec

    # OWAFRS
    spec = _component_spec_from_flat(flat, selector_key="ub_owa_method_name", prefix="ub_owa_method", explicit_obj_key="ub_owa_method")
    if spec is not None:
        fr["ub_owa_method"] = spec

    spec = _component_spec_from_flat(flat, selector_key="lb_owa_method_name", prefix="lb_owa_method", explicit_obj_key="lb_owa_method")
    if spec is not None:
        fr["lb_owa_method"] = spec

    # VQRS
    spec = _component_spec_from_flat(
        flat,
        selector_key="lb_fuzzy_quantifier_name",
        prefix="lb_fuzzy_quantifier",
        explicit_obj_key="lb_fuzzy_quantifier",
    )
    if spec is not None:
        fr["lb_fuzzy_quantifier"] = spec

    spec = _component_spec_from_flat(
        flat,
        selector_key="ub_fuzzy_quantifier_name",
        prefix="ub_fuzzy_quantifier",
        explicit_obj_key="ub_fuzzy_quantifier",
    )
    if spec is not None:
        fr["ub_fuzzy_quantifier"] = spec

    return nested
