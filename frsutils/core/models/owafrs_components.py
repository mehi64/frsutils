# SPDX-License-Identifier: BSD-3-Clause
"""Shared component construction utilities for OWAFRS models.

This module keeps OWAFRS T-norm, implicator, and OWA-weight construction
consistent between the dense NumPy reference model and blockwise approximation
engines.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from frsutils.core.implicators import Implicator
from frsutils.core.owa_weights import OWAWeights
from frsutils.core.tnorms import TNorm
from frsutils.utils.init_helpers import normalize_flat_config_to_nested


_DEFAULT_UB_TNORM = {"name": "minimum", "params": {}}
_DEFAULT_LB_IMPLICATOR = {"name": "lukasiewicz", "params": {}}
_DEFAULT_UB_OWA_METHOD = {"name": "linear", "params": {}}
_DEFAULT_LB_OWA_METHOD = {"name": "linear", "params": {}}


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when a mapping already uses frsutils nested sections."""
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _copy_with_owafrs_legacy_aliases(config: Mapping[str, Any]) -> dict:
    """Return a flat OWAFRS config with supported legacy aliases expanded."""
    flat = dict(config)

    if "p" in flat and "ub_tnorm_p" not in flat:
        flat["ub_tnorm_p"] = flat["p"]

    if "base" in flat:
        if "ub_owa_method_base" not in flat:
            flat["ub_owa_method_base"] = flat["base"]
        if "lb_owa_method_base" not in flat:
            flat["lb_owa_method_base"] = flat["base"]

    return flat


def _as_owafrs_nested_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool,
) -> Mapping[str, Any]:
    """Normalize flat, nested, or private dense-model OWAFRS config.

    Parameters
    ----------
    config : Mapping or None
        Flat OWAFRS config, nested frsutils config, or a mapping containing the
        private ``_nested_config`` key used by public builders.
    require_explicit_components : bool
        If True, missing component specs are reported by the caller instead of
        being filled with OWAFRS defaults.

    Returns
    -------
    Mapping[str, Any]
        Nested frsutils configuration.
    """
    if config is None:
        if require_explicit_components:
            return {"fr_model": {"type": "owafrs"}}
        return {
            "fr_model": {
                "type": "owafrs",
                "ub_tnorm": dict(_DEFAULT_UB_TNORM),
                "lb_implicator": dict(_DEFAULT_LB_IMPLICATOR),
                "ub_owa_method": dict(_DEFAULT_UB_OWA_METHOD),
                "lb_owa_method": dict(_DEFAULT_LB_OWA_METHOD),
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

    flat = _copy_with_owafrs_legacy_aliases(config)
    return normalize_flat_config_to_nested(flat)


def build_owafrs_components_from_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool = False,
) -> Tuple[TNorm, Implicator, OWAWeights, OWAWeights]:
    """Build OWAFRS T-norm, implicator, and OWA strategies from config.

    Parameters
    ----------
    config : Mapping or None
        Flat OWAFRS config, nested frsutils config, or a mapping containing the
        private ``_nested_config`` key used by public builders.
    require_explicit_components : bool, default=False
        If True, missing OWAFRS component specs raise ``ValueError``. If False,
        the OWAFRS defaults are used.

    Returns
    -------
    ub_tnorm : TNorm
        Upper-approximation T-norm instance.
    lb_implicator : Implicator
        Lower-approximation implicator instance.
    ub_owa_method : OWAWeights
        OWA weighting strategy for the upper approximation.
    lb_owa_method : OWAWeights
        OWA weighting strategy for the lower approximation.
    """
    nested = _as_owafrs_nested_config(
        config,
        require_explicit_components=require_explicit_components,
    )
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    ub_tnorm = TNorm.create_from_spec(fr_cfg.get("ub_tnorm"))
    lb_implicator = Implicator.create_from_spec(fr_cfg.get("lb_implicator"))
    ub_owa_method = OWAWeights.create_from_spec(fr_cfg.get("ub_owa_method"))
    lb_owa_method = OWAWeights.create_from_spec(fr_cfg.get("lb_owa_method"))

    if ub_tnorm is None:
        if require_explicit_components:
            raise ValueError("ub_tnorm_name must be provided for OWAFRS config.")
        ub_tnorm = TNorm.create(_DEFAULT_UB_TNORM["name"])
    if lb_implicator is None:
        if require_explicit_components:
            raise ValueError("lb_implicator_name must be provided for OWAFRS config.")
        lb_implicator = Implicator.create(_DEFAULT_LB_IMPLICATOR["name"])
    if ub_owa_method is None:
        if require_explicit_components:
            raise ValueError("ub_owa_method_name must be provided for OWAFRS config.")
        ub_owa_method = OWAWeights.create(_DEFAULT_UB_OWA_METHOD["name"])
    if lb_owa_method is None:
        if require_explicit_components:
            raise ValueError("lb_owa_method_name must be provided for OWAFRS config.")
        lb_owa_method = OWAWeights.create(_DEFAULT_LB_OWA_METHOD["name"])

    return ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method
