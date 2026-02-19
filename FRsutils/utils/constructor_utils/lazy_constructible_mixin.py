"""
@file lazy_constructible_mixin.py
@brief Unified mixin for managing lazy construction and initialization of objects.

This mixin enables:
- Supplying full config via `.configure(...)`
- Deferred execution of `.build()`
- Safe, introspectable lifecycle state via `.is_built`
- Compatibility with scikit-learn and imbalanced-learn, when combined with compliant oversamplers

##############################################
# ✅ Design Patterns & Clean Code
# - Factory Method: `.empty()`
# - Separation of Concerns: clear role for each method
# - Lifecycle Tracking: `LifecycleState` enum
# - Open/Closed: supports component injection via registry
##############################################
"""

from enum import Enum, auto
from abc import ABC, abstractmethod

class LifecycleState(Enum):
    # A shell of the object created but not configured and not built
    UNCONFIGURED = auto()
    
    # A configuration of the object to build it later, is
    # stored inside the object but still not built
    CONFIGURED = auto()
    
    # The object is built and ready for use
    BUILT = auto()

class LazyConstructibleMixin(ABC):
    """
    @brief Mixin for deferred object and subcomponent lifecycle management.
    Meant to be combined with sklearn/imbalanced-learn compatible oversamplers.
    
    Lifecycle State Management:
    This mixin tracks object readiness through 3 states using `LifecycleState`:

    - UNCONFIGURED:
        The object has been instantiated but no configuration has been provided.

        NOTICE: This state is a dummy state and is not stored. When an object of a
        class that inherits form LazyConstructibleMixin is instanciated with (),
        thae object is just a shell and has not state property. Therefore,
        self._state is cecked. if it is not present, then the state is UNCONFIGURED.
        if some cases when an object is partially initalized, some have self._state 
        and then the state is written inside that.

    - CONFIGURED:
        The object has received configuration (via `.configure(...)`) and is
        ready for initialization but not yet fully built.

    - BUILT:
        The object has been fully build (via `.build(...)`) and is
        now ready for use (e.g., fit, predict, transform, resample).

    State transitions are strictly enforced:
        UNCONFIGURED → CONFIGURED → BUILT

    Invalid transitions raise RuntimeError.
    
    This mixin does not have __init__. When instantiating, the __init__ method of class object is called
    which silently pass. It does nothing. However, this class has an abstract method. Therefore, this 
    class can not be instantiated directly.
    """

    _ALLOWED_TRANSITIONS = {
        LifecycleState.UNCONFIGURED: [LifecycleState.CONFIGURED],
        LifecycleState.CONFIGURED: [LifecycleState.BUILT],
        LifecycleState.BUILT: []
    }

    def _set_state(self, new_state):
        """
        @brief Safely updates the object's internal lifecycle state.

        Enforces allowed state transitions based on `_ALLOWED_TRANSITIONS`.
        Prevents illegal transitions such as skipping configuration or re-initializing.

        @param new_state: The target `LifecycleState` to transition into.

        @raises RuntimeError: If the transition is not permitted from the current state.
        """
        current = getattr(self, "_state", LifecycleState.UNCONFIGURED)
        if new_state not in self._ALLOWED_TRANSITIONS.get(current, []):
            raise RuntimeError(f"Invalid state transition: {current.name} → {new_state.name}")
        self._state = new_state

    def configure(self, *, model_registry=None, **config):
        """
        @brief Stores configuration and optional registry for deferred construction.

        This method captures the full configuration for the object (including any
        subcomponent such as a fuzzy rough model) and stores it internally.
        It also optionally receives a registry to use later for dynamic instantiation
        based on type strings in the configuration.

        This supports:
        - Lazy instantiation of components via deferred `.initialize()`
        - Registry/Factory Pattern using `model_registry.get_class(type)`
        - Open/Closed Principle: new models can be added without changing base logic

        Example usage:
        >>> configure(model_registry=FuzzyRoughModel, type="itfrs", similarity="gaussian")
        
        @param model_registry: Registry or factory class used to resolve models.
            Must implement `get_class(type: str)` returning a class object.
            Example: `FuzzyRoughModel.get_class("itfrs") -> ITFRS`.
        @param config: Configuration dictionary containing all hyperparameters
            including a required "type" key to identify the component class.

        @raises RuntimeError: If the state transition is invalid.
        @raises ValueError: If the config dictionary is empty.
        """
        
        if not config:
            raise ValueError("No configuration was provided to `configure()`. The config cannot be empty.")

        self._object_config = dict(config)
        self._model_registry = model_registry
        self._lazy_object = None # This is the object will be created later
        self._set_state(LifecycleState.CONFIGURED)

# TODO: update the use of args in all classes functions in the project in docstrings
    def build(self, *args):
        """
        @brief Finalizes the object and (optionally) constructs its lazy subcomponent.

        This method completes the lifecycle transition CONFIGURED → BUILT.

        If a `model_registry` was provided in `configure(...)` and the stored configuration
        contains a `'type'` key, the corresponding class will be resolved via:
            `cls = model_registry.get_class(config["type"])`
        and then instantiated using:
            `cls.from_config(*args, **config)`

        Why do we need `*args`?
        `configure(**config)` is intentionally limited to lightweight, pipeline-friendly
        hyperparameters (numbers/strings/strategies) that should be cloneable and tunable
        by scikit-learn / imbalanced-learn.

        Some inputs required to build the internal fuzzy-rough model are only available
        at runtime (typically during `fit` / `fit_resample`) and are often large objects
        (e.g., numpy arrays). These should NOT be stored inside `config` to keep the
        object cloneable and grid-search friendly.

        Therefore, `*args` is used to pass runtime-only objects positionally to
        `from_config(*args, **config)`.

        Typical `args` in this project (common convention):
        - args[0]: similarity_matrix  (e.g., shape [n_samples, n_samples])
        - args[1]: labels / target vector `y` (optional depending on the model)

        What SHOULD be passed via `args`:
        - Runtime data needed for construction of the internal model, usually:
        `(similarity_matrix,)` or `(similarity_matrix, y)`.

        What should NOT be passed via `args`:
        - Hyperparameters such as t-norm / implicator choices, quantifier parameters,
        OWA strategy, alpha/beta parameters, etc. These must be provided via
        `configure(**config)`.
        - The registry and the `'type'` selector (also provided via `configure(...)`).
        - Generally, do not pass the raw feature matrix `X` here unless the concrete
        `from_config` implementation explicitly expects it.

        Example:
        >>> obj.configure(
        ...     model_registry=FuzzyRoughModel,
        ...     type="itfrs",
        ...     lb_tnorm="minimum",
        ...     ub_implicator="goguen",
        ... )
        >>> obj.build(similarity_matrix, y)
        # forwards to: ITFRS.from_config(similarity_matrix, y, type="itfrs", lb_tnorm=..., ub_implicator=...)

        @param args: Runtime objects forwarded positionally to the resolved class'
                    `from_config(*args, **config)` method. Commonly
                    `(similarity_matrix,)` or `(similarity_matrix, y)`.
        @raises RuntimeError: If the object is not in `LifecycleState.CONFIGURED`.
        """
        if getattr(self, "_state", LifecycleState.UNCONFIGURED) != LifecycleState.CONFIGURED:
            raise RuntimeError("Either Object is not configured or is already built.")
        # TODO: Do we need dict here? isn't it already dict?
        config = dict(self._object_config)

        # build _lazy_object if model_registry and 'type' are present
        if self._model_registry and 'type' in config:
            cls = self._model_registry.get_class(config['type'])
            # We need args param again in addition to config, because args stores other data types than config.
            # For more information see the docstring of this function
            self._lazy_object = cls.from_config(*args, **config)
        
        self._finalize_object() 
        self._set_state(LifecycleState.BUILT)


    @abstractmethod
    def _finalize_object(self):
        """
        @brief Hook for subclasses to finalize any internal setup (assign attributes etc.)
        after .configure() and before .build() is marked complete.

        Called automatically inside `build(...)`.

        @raises RuntimeError: If the transition is not permitted from the current state.
        """
        raise NotImplementedError("Subclasses must implement _finalize_object().")


    @property
    def is_built(self) -> bool:
        """
        @brief Returns True if object has been configured and initialized.
        """
        return getattr(self, "_state", LifecycleState.UNCONFIGURED) == LifecycleState.BUILT

    @property
    def state_str(self) -> str:
        """@brief Lifecycle state as string name (e.g. 'UNCONFIGURED')."""
        return getattr(self, "_state", LifecycleState.UNCONFIGURED).name

    @property
    def state_enum(self) -> LifecycleState:
        """@brief Lifecycle state as enum (preferred for logic checks)."""
        return getattr(self, "_state", LifecycleState.UNCONFIGURED)

    
    @property
    def lazy_object(self):
        """
        @brief Accessor for the built/ready model.
        @return The model instance created by build(...).
        @raises RuntimeError If the object has not been built yet.
        """
        if getattr(self, "_state", LifecycleState.UNCONFIGURED) != LifecycleState.BUILT:
            raise RuntimeError("Object has not been built yet. Call build() first.")
        return self._lazy_object

