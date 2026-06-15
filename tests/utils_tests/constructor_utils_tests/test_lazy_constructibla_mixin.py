# SPDX-License-Identifier: BSD-3-Clause
"""Comprehensive pytest unit tests for LazyConstructibleMixin."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------
# Robust imports
# ---------------------------------------------------------------------
# In the project, this module typically lives at:
#   frsutils/utils/constructor_utils/lazy_constructible_mixin.py
# The fallback allows running tests in environments where that package path
# is not available (e.g., a sandbox with the file at repo root).
try:
    from frsutils.utils.constructor_utils.lazy_constructible_mixin import (
        LazyConstructibleMixin,
        LifecycleState,
    )
except ImportError:  # pragma: no cover
    from lazy_constructible_mixin import LazyConstructibleMixin, LifecycleState


# ---------------------------------------------------------------------
# Test doubles (Dummy classes) to isolate mixin behavior
# ---------------------------------------------------------------------

class DummyModel:
    """Minimal class with a from_config factory for verifying forwarding.
    
    We store the last call inputs so tests can assert:
    - positional args are forwarded correctly
    - config keys/values are forwarded correctly
    """
    last_call = None

    @classmethod
    def from_config(cls, *args, **config):
        # Store inputs for assertions.
        """From config."""
        cls.last_call = {
            "args": args,
            "config": dict(config),
        }

        # Mutate kwargs locally to ensure it does NOT affect caller's stored config
        # (should be safe because **kwargs creates a new dict at call time).
        config.pop("a", None)

        # Return a simple object that represents the built instance.
        return {"built": True, "args": args, "config": dict(cls.last_call["config"])}


class DummyRegistry:
    """Minimal registry with get_class(type_str) -> class.
    
    Contract expected by the mixin:
    registry.get_class(config["type"]) -> class with from_config(...)
    """
    call_count = 0
    last_type = None

    @staticmethod
    def get_class(type_str: str):
        """Return class values for tests or public helpers."""
        DummyRegistry.call_count += 1
        DummyRegistry.last_type = type_str
        if type_str != "dummy":
            raise KeyError(f"Unknown type: {type_str}")
        return DummyModel


class DummyComponent(LazyConstructibleMixin):
    """Concrete subclass to make LazyConstructibleMixin instantiable.
    
    This class only exists for unit tests. It implements `_finalize_object` and
    records whether/when it was called, and what `_lazy_object` looked like then.
    """

    def __init__(self):
        # Note: mixin has no __init__. We add this only for test observability.
        """Initialize the DummyComponent instance."""
        self.finalize_called = 0
        self.finalize_seen_lazy_object = "NOT_CALLED"

    def _finalize_object(self):
        """Test hook: record finalize call count and the value of `_lazy_object`."""
        self.finalize_called += 1
        self.finalize_seen_lazy_object = getattr(self, "_lazy_object", None)


# ---------------------------------------------------------------------
# Lifecycle / state-machine tests
# ---------------------------------------------------------------------

def test_default_state_when_state_attr_missing_is_unconfigured():
    """When `_state` is not set, the mixin must behave as UNCONFIGURED.
    
    The implementation uses:
    getattr(self, "_state", LifecycleState.UNCONFIGURED)
    """
    obj = DummyComponent()

    # No internal state attribute initially
    assert not hasattr(obj, "_state")

    # Properties must still work safely
    assert obj.state_enum == LifecycleState.UNCONFIGURED
    assert obj.state_str == LifecycleState.UNCONFIGURED.name
    assert obj.is_built is False


def test_set_state_allows_only_unconfigured_to_configured_to_built():
    """Valid transition path: UNCONFIGURED -> CONFIGURED -> BUILT."""
    obj = DummyComponent()

    obj._set_state(LifecycleState.CONFIGURED)
    assert obj.state_enum == LifecycleState.CONFIGURED

    obj._set_state(LifecycleState.BUILT)
    assert obj.state_enum == LifecycleState.BUILT
    assert obj.is_built is True


@pytest.mark.parametrize(
    "start_state, target_state",
    [
        (LifecycleState.UNCONFIGURED, LifecycleState.BUILT),     # skipping CONFIGURED
        (LifecycleState.BUILT, LifecycleState.BUILT),            # same state
    ],
)
def test_set_state_rejects_invalid_transitions(start_state, target_state):
    """`_set_state` must reject illegal transitions based on _ALLOWED_TRANSITIONS."""
    obj = DummyComponent()
    obj._state = start_state

    with pytest.raises(RuntimeError, match=r"Invalid state transition"):
        obj._set_state(target_state)


# ---------------------------------------------------------------------
# configure(...) tests
# ---------------------------------------------------------------------

def test_configure_rejects_empty_config():
    """configure() must raise ValueError if config is empty."""
    obj = DummyComponent()

    with pytest.raises(ValueError, match=r"config cannot be empty"):
        obj.configure(model_registry=DummyRegistry)  # no **config provided


def test_configure_sets_fields_and_transitions_to_configured():
    """configure() stores config/registry, resets lazy object, and moves to CONFIGURED."""
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1, b="x")

    assert obj.state_enum == LifecycleState.CONFIGURED
    assert obj.state_str == LifecycleState.CONFIGURED.name
    assert obj._model_registry is DummyRegistry
    assert obj._object_config == {"type": "dummy", "a": 1, "b": "x"}
    assert obj._lazy_object is None
    assert obj.is_built is False


def test_configure_stores_copy_not_reference():
    """configure() must store a copy of input config so external dict mutations don't leak."""
    obj = DummyComponent()

    user_cfg = {"type": "dummy", "a": 1}
    obj.configure(model_registry=DummyRegistry, **user_cfg)

    # Mutate external dict after configure
    user_cfg["a"] = 999
    user_cfg["new_key"] = "leak?"

    assert obj._object_config == {"type": "dummy", "a": 1}


def test_configure_twice_replaces_config_and_keeps_configured_state():
    """Current mixin contract allows reconfiguration while CONFIGURED."""
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1)
    obj.configure(model_registry=DummyRegistry, type="dummy", a=2)

    assert obj.state_enum == LifecycleState.CONFIGURED
    assert obj._object_config == {"type": "dummy", "a": 2}
    assert obj._lazy_object is None


def test_configure_after_build_resets_to_configured():
    """Current mixin contract allows reconfiguration after build."""
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1)
    obj.build()

    obj.configure(model_registry=DummyRegistry, type="dummy", a=2)

    assert obj.state_enum == LifecycleState.CONFIGURED
    assert obj.is_built is False
    assert obj._object_config == {"type": "dummy", "a": 2}
    assert obj._lazy_object is None


# ---------------------------------------------------------------------
# build(*args) tests
# ---------------------------------------------------------------------

def test_build_requires_configured_state():
    """build() must raise RuntimeError unless current state is CONFIGURED."""
    obj = DummyComponent()

    with pytest.raises(RuntimeError, match=r"not configured|already built"):
        obj.build()


def test_build_constructs_lazy_object_when_registry_and_type_present_and_forwards_args():
    """If registry exists and config contains 'type', build() must:
    
    - resolve class via registry.get_class(config["type"])
    - call cls.from_config(*args, **config)
    - store the resulting object in `_lazy_object`
    - call `_finalize_object`
    - set state to BUILT
    """
    # Reset observation state
    DummyRegistry.call_count = 0
    DummyRegistry.last_type = None
    DummyModel.last_call = None

    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1, b="x")

    sim = "SIM_MATRIX"
    y = "LABELS"
    obj.build(sim, y)

    # lifecycle
    assert obj.state_enum == LifecycleState.BUILT
    assert obj.is_built is True

    # registry usage: configure validates once and build resolves once.
    assert DummyRegistry.call_count == 2
    assert DummyRegistry.last_type == "dummy"

    # from_config call must receive positional args and full config
    assert DummyModel.last_call is not None
    assert DummyModel.last_call["args"] == (sim, y)
    assert DummyModel.last_call["config"] == {"type": "dummy", "a": 1, "b": "x"}

    # lazy object created and visible to finalize hook
    assert obj._lazy_object is not None
    assert obj._lazy_object["built"] is True
    assert obj.finalize_called == 1
    assert obj.finalize_seen_lazy_object is obj._lazy_object

    # caller config stored inside object must remain intact (even if from_config mutates its kwargs)
    assert obj._object_config == {"type": "dummy", "a": 1, "b": "x"}


def test_build_does_not_construct_lazy_object_when_registry_is_none_but_still_builds_lifecycle():
    """If model_registry is None, build() must skip construction but still:
    
    - call finalize hook
    - transition to BUILT
    - allow lazy_object accessor to return None (valid)
    """
    obj = DummyComponent()
    obj.configure(model_registry=None, type="dummy", a=1)

    obj.build("SIM", "Y")

    assert obj.state_enum == LifecycleState.BUILT
    assert obj._lazy_object is None
    assert obj.finalize_called == 1

    # built lifecycle allows accessing lazy_object; it should return None safely
    assert obj.lazy_object is None


def test_configure_with_registry_requires_type_key():
    """Current validation requires a type key when a registry is provided."""
    obj = DummyComponent()

    with pytest.raises(ValueError, match=r"type"):
        obj.configure(model_registry=DummyRegistry, a=1, b=2)


def test_build_twice_is_not_allowed_by_precondition():
    """After the first build, the object is BUILT.
    
    Subsequent build() calls must raise because state != CONFIGURED.
    """
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1)

    obj.build()
    assert obj.state_enum == LifecycleState.BUILT

    with pytest.raises(RuntimeError, match=r"not configured|already built"):
        obj.build()


# ---------------------------------------------------------------------
# lazy_object property tests
# ---------------------------------------------------------------------

def test_lazy_object_raises_before_build_even_if_configured():
    """Accessing lazy_object before build() must raise RuntimeError."""
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1)

    with pytest.raises(RuntimeError, match=r"has not been built yet"):
        _ = obj.lazy_object


def test_lazy_object_returns_built_object_after_build():
    """After build(), lazy_object must return `_lazy_object` (may be None if not constructed).
    
    Here we force construction, so it must return the built dict.
    """
    obj = DummyComponent()
    obj.configure(model_registry=DummyRegistry, type="dummy", a=1)
    obj.build("SIM", "Y")

    assert obj.lazy_object is obj._lazy_object
    assert obj.lazy_object["built"] is True
