# SPDX-License-Identifier: BSD-3-Clause
"""Shared component construction utilities for VQRS models.

This module keeps fuzzy-quantifier construction consistent between the dense
VQRS reference model and backend-aware blockwise approximation engines.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.tnorms import TNorm
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


_DEFAULT_LB_FUZZY_QUANTIFIER = {
    "name": "linear",
    "params": {"alpha": 0.1, "beta": 0.6},
}
_DEFAULT_UB_FUZZY_QUANTIFIER = {
    "name": "linear",
    "params": {"alpha": 0.1, "beta": 0.6},
}


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when a mapping already uses FRsutils nested sections."""
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _as_vqrs_nested_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool,
) -> Mapping[str, Any]:
    """Normalize flat, nested, or private dense-model VQRS config.

    Parameters
    ----------
    config : Mapping or None
        Flat VQRS config, nested FRsutils config, or a mapping containing the
        private ``_nested_config`` key used by public builders.
    require_explicit_components : bool
        If True, missing lower or upper fuzzy quantifier specs are reported by
        the caller instead of being filled with default components.

    Returns
    -------
    Mapping[str, Any]
        Nested FRsutils configuration.
    """
    if config is None:
        if require_explicit_components:
            return {"fr_model": {"type": "vqrs"}}
        return {
            "fr_model": {
                "type": "vqrs",
                "lb_fuzzy_quantifier": dict(_DEFAULT_LB_FUZZY_QUANTIFIER),
                "ub_fuzzy_quantifier": dict(_DEFAULT_UB_FUZZY_QUANTIFIER),
            }
        }

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    nested = config.get("_nested_config")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("_nested_config must be a mapping when provided.")
        return nested

    if _is_nested_frs_config(config):
        return config

    return normalize_flat_config_to_nested(dict(config))


def build_vqrs_components_from_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool = False,
) -> Tuple[FuzzyQuantifier, FuzzyQuantifier, TNorm]:
    """Build VQRS lower and upper fuzzy quantifiers from config.

    Parameters
    ----------
    config : Mapping or None
        Flat VQRS config, nested FRsutils config, or a mapping containing the
        private ``_nested_config`` key used by public builders.
    require_explicit_components : bool, default=False
        If True, missing lower or upper fuzzy quantifier specs raise
        ``ValueError``. If False, the VQRS default linear quantifiers are used.

    Returns
    -------
    lb_fuzzy_quantifier : FuzzyQuantifier
        Lower-approximation fuzzy quantifier instance.
    ub_fuzzy_quantifier : FuzzyQuantifier
        Upper-approximation fuzzy quantifier instance.
    tnorm : TNorm
        Minimum T-norm used by the dense VQRS reference formula.
    """
    nested = _as_vqrs_nested_config(
        config,
        require_explicit_components=require_explicit_components,
    )
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    lb_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("lb_fuzzy_quantifier"))
    ub_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("ub_fuzzy_quantifier"))

    if lb_fuzzy_quantifier is None:
        if require_explicit_components:
            raise ValueError("lb_fuzzy_quantifier_name must be provided for VQRS config.")
        lb_fuzzy_quantifier = FuzzyQuantifier.create(
            _DEFAULT_LB_FUZZY_QUANTIFIER["name"],
            **_DEFAULT_LB_FUZZY_QUANTIFIER["params"],
        )
    if ub_fuzzy_quantifier is None:
        if require_explicit_components:
            raise ValueError("ub_fuzzy_quantifier_name must be provided for VQRS config.")
        ub_fuzzy_quantifier = FuzzyQuantifier.create(
            _DEFAULT_UB_FUZZY_QUANTIFIER["name"],
            **_DEFAULT_UB_FUZZY_QUANTIFIER["params"],
        )

    tnorm = TNorm.create("minimum")
    return lb_fuzzy_quantifier, ub_fuzzy_quantifier, tnorm
