"""
@file lazy_constructible_mixin.py
@brief Unified mixin for managing lazy construction and initialization of objects.

This mixin enables:
- Supplying full config via `.configure(...)`
- Deferred execution of `.build()`
- Safe, introspectable lifecycle state via `.is_ready`
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
        """
        
        if not config:
            raise ValueError("No configuration was provided to `configure()`. The config cannot be empty.")

        self._object_config = dict(config)
        self._model_registry = model_registry
        self._lazy_object = None # This is the object will be created later
        self._set_state(LifecycleState.CONFIGURED)
    
    def build(self, *args, **kwargs):
        """
        @brief Initializes the object. Builds internal component if needed.

        @param args: Passed to subcomponent's from_config()
        @param kwargs: Any runtime values for initialization.
        """
        if getattr(self, "_state", LifecycleState.UNCONFIGURED) != LifecycleState.CONFIGURED:
            raise RuntimeError("Object must be configured before build().")

        config = dict(self._object_config)

        # 🧠 Always build _lazy_object if model_registry and 'type' are present
        if self._model_registry and 'type' in config:
            cls = self._model_registry.get_class(config['type'])
            self._lazy_object = cls.from_config(*args, **config)

        # ⚠️ Make sure _build_from_config doesn't overwrite _lazy_object = None!
        # self._build_from_config(**config)

        self._set_state(LifecycleState.BUILT)


    @abstractmethod
    def _build_from_config(self, **config):
        """
        @brief Subclass-specific initialization logic from stored config.
        NOTE: THIS FUNCTION SHOULD BE IMPLEMENTED BY SUBCLASSES AND IS NOT OPTIONAL.

        This method should be implemented by subclasses to unpack and assign
        values from the configuration dictionary stored during `.configure(...)`.

        It is called internally by `.initialize()` after any lazy components
        (e.g., registry-based models) have been instantiated.

        Separating this logic allows:
        - Clean separation of concerns (framework lifecycle vs. subclass logic)
        - Easy subclass overrides without breaking initialization flow
        - Support for plug-in architectures and registry-based components
        - Easier unit testing of configuration unpacking in isolation

        @param config: The same dictionary originally passed to `.configure(...)`.
        It typically includes model hyperparameters, fuzzy settings, and other options.
        """
        raise NotImplementedError("Subclasses must implement _initialize_from_config.")

    def ensure_build(self, *args, **kwargs):
        """
        @brief Initializes object on demand if not yet initialized.

        @param args: Forwarded to initialize()
        @param kwargs: Forwarded to initialize()
        """
        # if the state is not initialized, initialize it
        state = getattr(self, "_state", LifecycleState.UNCONFIGURED)
        if state != LifecycleState.BUILT:
            self.build(*args, **kwargs)


    @property
    def is_built(self) -> bool:
        """
        @brief Returns True if object has been configured and initialized.
        """
        return getattr(self, "_state", LifecycleState.UNCONFIGURED) == LifecycleState.BUILT

    @property
    def state(self) -> str:
        return getattr(self, "_state", LifecycleState.UNCONFIGURED).name
    
    @property
    def lazy_object(self):
        """
        @brief Accessor for the built/ready model.
        @return: The model instance created by initialize
        @raises RuntimeError: If self.state != LifecycleState.READY
        """
        if self.state != LifecycleState.BUILT:
            raise RuntimeError("Object has not been built yet. Call ensure_ready() first.")
        return self._lazy_object
