# SPDX-License-Identifier: BSD-3-Clause
"""Shared component construction utilities for ITFRS models.

This module keeps ITFRS operator construction consistent between the dense
reference model and backend-aware blockwise approximation engines.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from FRsutils.core.implicators import Implicator
from FRsutils.core.tnorms import TNorm
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when a mapping already uses FRsutils nested sections."""
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _copy_with_itfrs_legacy_aliases(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flat config copy with ITFRS-only legacy aliases applied."""
    flat = dict(config)
    if "ub_tnorm_p" not in flat and "p" in flat and flat.get("ub_tnorm_name") is not None:
        flat["ub_tnorm_p"] = flat["p"]
    return flat


def _as_itfrs_nested_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool,
) -> Mapping[str, Any]:
    """Normalize flat, nested, or private dense-model ITFRS config.

    Parameters
    ----------
    config : Mapping or None
        Flat config, nested FRsutils config, or a mapping containing the private
        ``_nested_config`` key used by public model builders.
    require_explicit_components : bool
        If True, missing ITFRS operators are reported by the caller instead of
        being filled with default components.

    Returns
    -------
    Mapping[str, Any]
        Nested FRsutils configuration.
    """
    if config is None:
        if require_explicit_components:
            return {"fr_model": {"type": "itfrs"}}
        return normalize_flat_config_to_nested(
            {"type": "itfrs", "ub_tnorm_name": "minimum", "lb_implicator_name": "lukasiewicz"}
        )

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    nested = config.get("_nested_config")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("_nested_config must be a mapping when provided.")
        return nested

    if _is_nested_frs_config(config):
        return config

    flat = _copy_with_itfrs_legacy_aliases(config)
    return normalize_flat_config_to_nested(flat)


def build_itfrs_components_from_config(
    config: Optional[Mapping[str, Any]],
    *,
    require_explicit_components: bool = False,
) -> Tuple[TNorm, Implicator]:
    """Build ITFRS upper T-norm and lower implicator from config.

    Parameters
    ----------
    config : Mapping or None
        Flat ITFRS config, nested FRsutils config, or a mapping containing the
        private ``_nested_config`` key used by public builders.
    require_explicit_components : bool, default=False
        If True, missing ``ub_tnorm`` or ``lb_implicator`` specs raise
        ``ValueError``. If False, the ITFRS defaults ``minimum`` and
        ``lukasiewicz`` are used.

    Returns
    -------
    ub_tnorm : TNorm
        Upper-approximation T-norm instance.
    lb_implicator : Implicator
        Lower-approximation implicator instance.
    """
    nested = _as_itfrs_nested_config(
        config,
        require_explicit_components=require_explicit_components,
    )
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    ub_tnorm = TNorm.create_from_spec(fr_cfg.get("ub_tnorm"))
    lb_implicator = Implicator.create_from_spec(fr_cfg.get("lb_implicator"))

    if ub_tnorm is None:
        if require_explicit_components:
            raise ValueError("ub_tnorm_name must be provided for ITFRS config.")
        ub_tnorm = TNorm.create("minimum")
    if lb_implicator is None:
        if require_explicit_components:
            raise ValueError("lb_implicator_name must be provided for ITFRS config.")
        lb_implicator = Implicator.create("lukasiewicz")

    return ub_tnorm, lb_implicator
