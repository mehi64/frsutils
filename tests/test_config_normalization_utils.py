# SPDX-License-Identifier: BSD-3-Clause
"""Tests for flat-to-nested configuration normalization."""

import pytest

from FRsutils.utils.init_helpers import normalize_flat_config_to_nested

def test_normalize_supports_legacy_gaussian_similarity_sigma_alias():
    flat = {
        "similarity": "gaussian",
        "gaussian_similarity_sigma": 0.5,
        "similarity_tnorm": "minimum",
        "type": "itfrs",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    }

    nested = normalize_flat_config_to_nested(flat)
    assert nested["similarity"]["name"] == "gaussian"
    assert nested["similarity"]["params"]["sigma"] == 0.5


def test_normalize_builds_fr_model_component_specs():
    flat = {
        "type": "owafrs",
        "ub_tnorm_name": "yager",
        "ub_tnorm_p": 3.0,
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "exponential",
        "ub_owa_method_base": 3.0,
        "lb_owa_method_name": "linear",
    }

    nested = normalize_flat_config_to_nested(flat)
    fr = nested["fr_model"]

    assert fr["type"] == "owafrs"
    assert fr["ub_tnorm"]["name"] == "yager"
    assert fr["ub_tnorm"]["params"]["p"] == 3.0
    assert fr["ub_owa_method"]["name"] == "exponential"
    assert fr["ub_owa_method"]["params"]["base"] == 3.0
