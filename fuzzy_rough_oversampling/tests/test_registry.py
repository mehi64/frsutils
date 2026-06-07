"""
@file test_registry.py
@brief Unit tests for oversampler registry behavior.
"""

import pytest

from fuzzy_rough_oversampling.registry import (
    build_oversampler,
    get_oversampler_class,
    list_oversamplers,
    register_oversampler,
)


def test_register_and_build_oversampler() -> None:
    """@brief Verify that future algorithms can register and be constructed."""

    @register_oversampler("dummy_phase2", aliases=("dummy_alias",))
    class DummyOversampler:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    assert get_oversampler_class("dummy_alias") is DummyOversampler
    assert list_oversamplers()["dummy_phase2"] == ["dummy_alias", "dummy_phase2"]

    instance = build_oversampler("dummy_phase2", value=7)
    assert isinstance(instance, DummyOversampler)
    assert instance.value == 7


def test_unknown_oversampler_raises_clear_error() -> None:
    """@brief Verify that unknown algorithm names fail with a clear error."""
    with pytest.raises(ValueError, match="Unknown fuzzy-rough oversampler"):
        build_oversampler("not_registered")
