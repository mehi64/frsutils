"""
@file config.py
@brief Public configuration helpers for downstream FRsutils consumers.

This module is a small facade over FRsutils internal configuration utilities.
External packages should import config normalization helpers from here instead
of depending directly on `FRsutils.utils.init_helpers`.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# normalize_flat_config_to_nested      Convert flat sklearn params into nested config
# apply_config_aliases                 Apply supported backwards-compatible aliases
# extract_prefixed_params              Extract component params by prefix

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable public wrapper over internal helper module
# - Adapter Pattern: supports flat external params and nested internal config
# - SRP: only configuration helpers are exposed here
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api.config import normalize_flat_config_to_nested
#
# nested = normalize_flat_config_to_nested({
#     "type": "itfrs",
#     "similarity": "gaussian",
#     "similarity_sigma": 0.5,
#     "ub_tnorm_name": "minimum",
#     "lb_implicator_name": "lukasiewicz",
# })
"""

from FRsutils.utils.init_helpers import (
    apply_config_aliases,
    extract_prefixed_params,
    normalize_flat_config_to_nested,
)

__all__ = [
    "apply_config_aliases",
    "extract_prefixed_params",
    "normalize_flat_config_to_nested",
]
