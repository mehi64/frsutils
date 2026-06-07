"""
@file test_frsmote_registration.py
@brief Registration tests for the migrated FRSMOTE class.
"""

from fuzzy_rough_oversampling import FRSMOTE, build_oversampler, get_oversampler_class


def test_frsmote_is_registered() -> None:
    """@brief Verify that FRSMOTE can be resolved by public aliases."""
    assert get_oversampler_class("frsmote") is FRSMOTE
    assert get_oversampler_class("fuzzy_rough_smote") is FRSMOTE


def test_build_frsmote() -> None:
    """@brief Verify that the registry factory builds FRSMOTE instances."""
    sampler = build_oversampler("frsmote", random_state=42, k_neighbors=3)
    assert isinstance(sampler, FRSMOTE)
    assert sampler.get_params()["random_state"] == 42
    assert sampler.get_params()["k_neighbors"] == 3
