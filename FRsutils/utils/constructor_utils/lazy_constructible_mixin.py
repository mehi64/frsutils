"""
@file lazy_constructible_mixin.py
@brief Lifecycle mixin for lazily-constructed, config-driven components.

This mixin implements a small, explicit lifecycle:
`UNCONFIGURED -> CONFIGURED -> BUILT`, enabling:
- scikit-learn friendly cloning (store configuration; delay heavy build)
- reproducible reconstruction via `from_config`

In FRsutils, estimators keep their **flat** config in `_object_config` for
Pipeline/GridSearch compatibility, while an optional internal `_nested_config`
may be attached to avoid parameter-name collisions during subcomponent builds.

##############################################
# ✅ Quick Summary of Features
# - configure(**config) stores configuration and validates
# - build(*args) constructs the internal object (via model_registry)
# - lifecycle enforcement via LifecycleState
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# mixin_user.configure(model_registry=FuzzyRoughModel, **flat_config)
# mixin_user.build(similarity_matrix, y)
"""
from enum import Enum, auto
from abc import ABC, abstractmethod

class LifecycleState(Enum):
    UNCONFIGURED = auto()
    CONFIGURED = auto()
    BUILT = auto()

class LazyConstructibleMixin(ABC):
    _ALLOWED_TRANSITIONS = {
        LifecycleState.UNCONFIGURED: [LifecycleState.CONFIGURED],
        LifecycleState.CONFIGURED: [LifecycleState.CONFIGURED, LifecycleState.BUILT],
        LifecycleState.BUILT: [LifecycleState.CONFIGURED],
    }

    def _set_state(self, new_state):
        current = getattr(self, "_state", LifecycleState.UNCONFIGURED)
        if new_state not in self._ALLOWED_TRANSITIONS.get(current, []):
            raise RuntimeError(f"Invalid state transition: {current.name} → {new_state.name}")
        self._state = new_state

    def _validate_config(self, *, model_registry=None, **config):
        """
        @brief Generic, lightweight validation for clone-friendly config.
        """
        if model_registry is None:
            return

        if not hasattr(model_registry, "get_class") or not callable(getattr(model_registry, "get_class")):
            raise TypeError("`model_registry` must implement a callable get_class(type: str) method.")

        if "type" not in config:
            raise ValueError("When `model_registry` is provided, config must include a 'type' key.")

        if not isinstance(config["type"], str) or not config["type"].strip():
            raise ValueError("Config key 'type' must be a non-empty string.")

        try:
            model_registry.get_class(config["type"])
        except Exception as exc:
            raise ValueError(
                f"Unknown type '{config['type']}'. The provided model_registry could not resolve it."
            ) from exc

    def configure(self, *, model_registry=None, **config):
        if not config:
            raise ValueError("No configuration was provided to `configure()`. The config cannot be empty.")

        if hasattr(self, "_validate_config"):
            self._validate_config(model_registry=model_registry, **config)

        self._object_config = dict(config)
        self._model_registry = model_registry
        self._lazy_object = None

        self._set_state(LifecycleState.CONFIGURED)

    def build(self, *args):
        if getattr(self, "_state", LifecycleState.UNCONFIGURED) != LifecycleState.CONFIGURED:
            raise RuntimeError("Either Object is not configured or is already built.")

        # NOTE:
        # - `_object_config` is intentionally *flat* (scikit-learn friendly).
        # - `_nested_config` (if present) is an internal representation used to build
        #   sub-components without name collisions.
        config = dict(self._object_config)

        nested = getattr(self, "_nested_config", None)
        if isinstance(nested, dict):
            # Keep this key private/internal. Model builders may prefer it.
            config["_nested_config"] = nested

        if self._model_registry and 'type' in config:
            cls = self._model_registry.get_class(config['type'])
            self._lazy_object = cls.from_config(*args, **config)

        self._finalize_object()
        self._set_state(LifecycleState.BUILT)

    @abstractmethod
    def _finalize_object(self):
        raise NotImplementedError("Subclasses must implement _finalize_object().")

    @property
    def is_built(self) -> bool:
        return getattr(self, "_state", LifecycleState.UNCONFIGURED) == LifecycleState.BUILT

    @property
    def state_str(self) -> str:
        return getattr(self, "_state", LifecycleState.UNCONFIGURED).name

    @property
    def state_enum(self) -> LifecycleState:
        return getattr(self, "_state", LifecycleState.UNCONFIGURED)

    @property
    def lazy_object(self):
        if getattr(self, "_state", LifecycleState.UNCONFIGURED) != LifecycleState.BUILT:
            raise RuntimeError("Object has not been built yet. Call build() first.")
        return self._lazy_object