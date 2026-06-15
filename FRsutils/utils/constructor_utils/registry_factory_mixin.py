# SPDX-License-Identifier: BSD-3-Clause
"""Registry-backed factory mixin for named component construction.

This module provides shared utility behavior used by frsutils components.
"""

import inspect
from typing import Type, Dict, List, Any
from abc import ABC, abstractmethod

class RegistryFactoryMixin(ABC):
    """Mixin class for pluggable component registration and instantiation.
    
    Includes registry management, parameter filtering, dynamic factory creation,
    serialization, and runtime introspection utilities.
    """
    
    def __init_subclass__(cls, **kwargs):
        """Automatically initializes per-subclass registry.
        
        Ensures that each subclass maintains its own `_registry` and `_aliases`.
        This is because some tnorms and implicators can have the same name, e.g. yager, luk.
        """

        # __init_subclass__ is a special method in Python, automatically called whenever
        # a class is subclassed. It's like a constructor—but for classes, not instances.
        # You override it when you want to customize how subclasses 
        # are constructed.
        # 
        # Every class that inherits from RegistryFactoryMixin 
        # (e.g. TNorm, Implicator, SimilarityFunction) automatically 
        # gets its own _registry and _aliases dictionaries because 
        # Python calls this method at the time of class definition, 
        # not at runtime.

        # print(f"{cls.__name__} __init_subclass__ is called")
        super().__init_subclass__(**kwargs)
        
        # a mapping between names and classes. it stores all registered classes
        cls._registry: Dict[str, Type] = {}
        
        # a mapping between classes and several aliases for each class.
        # for example, Yager could beregistered with 'yager', 'yg', 'yager_implicator', etc.    
        cls._aliases: Dict[Type, List[str]] = {}

    @classmethod
    def register(cls, *names: str):
        """Class decorator to register a subclass with one or more aliases.
                
                Registers the class in the global registry and stores all aliases.
                
                Parameters
                ----------
                names : str
                    One or more alias names for the subclass.
                
                Returns
                -------
                object
                    Class decorator.
                
        """
        def decorator(subclass: Type):
            if not names:
                raise ValueError("At least one name must be provided for registration.")
            cls._aliases[subclass] = list(map(str.lower, names))
            for name in names:
                key = name.lower()
                if key in cls._registry:
                    raise ValueError(f"Alias '{key}' is already registered in {cls.__name__}.")
                cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, strict: bool = False, namespace: str = None, **kwargs) -> Any:
        """Instantiates a subclass by alias name with optional flat namespaced parameters.
                
                This supports scikit-learn compatible flat parameter dictionaries:
                - FuzzyQuantifier.create("linear", namespace="lb", **config)
                will extract lb_alpha and lb_beta and pass them as alpha, beta.
                
                Parameters
                ----------
                name : str
                    Alias of the registered subclass.
                strict : bool
                    If True, raises an error if extra unused kwargs exist.
                namespace : str
                    Optional prefix for parameter isolation (e.g. 'lb').
                kwargs : object
                    Full flat config dict.
                
                Returns
                -------
                Any
                    Instantiated subclass.
                
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown alias: {name}")
        
        target_cls = cls._registry[name]

        # Namespace filtering if needed
        if namespace:
            prefix = f"{namespace}_"
            filtered = {
                k[len(prefix):]: v for k, v in kwargs.items()
                if k.startswith(prefix)
            }
        else:
            filtered = kwargs

        # parameter validation is done inside __init__ 
        # all classes inherited from RegistryFactoryMixin must have __init__
        # and call validate_params in there.
        
        # ctor_args means constructor arguments
        # filter out unused parameters
        ctor_args = cls._filter_args(target_cls, filtered)

        if strict:
            unused = set(filtered) - set(ctor_args)
            if unused:
                raise ValueError(f"[{cls.__name__}] Unused parameters: {unused}")

        return target_cls(**ctor_args)

    @classmethod
    def create_from_spec(cls, spec_or_obj: Any, *, strict: bool = False) -> Any:
        """Instantiate a registered component from a nested spec or an existing instance.
                
                This helper enables the project-wide convention for **internal nested configs**:
                {"name": <alias>, "params": {...}}
                It also accepts several backward-compatible forms:
                - direct instances of the registry base class
                - {"__instance__": <obj>} (internal marker produced by the normalizer)
                - objects serialized by `to_dict()` ("name" + "params")
                - legacy compact dicts such as {"type": <alias>, ...} (no "params" key)
                
                Parameters
                ----------
                spec_or_obj : Any
                    Nested spec dict or instance.
                strict : bool
                    If True, errors on unused parameters when calling `create()`.
                
                Returns
                -------
                Any
                    Instantiated component or the input instance.
                
        """
        if spec_or_obj is None:
            return None

        # Internal marker: pass-through instance.
        if isinstance(spec_or_obj, dict) and '__instance__' in spec_or_obj:
            return spec_or_obj['__instance__']

        # Already an instance of this registry.
        if isinstance(spec_or_obj, cls):
            return spec_or_obj

        if isinstance(spec_or_obj, dict):
            # Preferred internal spec format
            if 'name' in spec_or_obj:
                name = spec_or_obj.get('name')
                params = spec_or_obj.get('params') or {}
                if not isinstance(params, dict):
                    raise TypeError("spec['params'] must be a dict")
                return cls.create(name, strict=strict, **params)

            # Legacy compact format: {type: <alias>, ...}
            if 'type' in spec_or_obj and 'params' not in spec_or_obj:
                name = spec_or_obj.get('type')
                params = {k: v for k, v in spec_or_obj.items() if k not in {'type'}}
                return cls.create(name, strict=strict, **params)

        raise TypeError(f'Unsupported component spec/object: {type(spec_or_obj)}')



    @classmethod
    def list_available(cls) -> Dict[str, List[str]]:
        """Lists all registered subclasses and their aliases.
                
                Returns
                -------
                Dict[str, List[str]]
                    Dictionary mapping primary alias to all aliases.
                
        """
        return {names[0]: names for _, names in cls._aliases.items()}

    @staticmethod
    def _filter_args(cls, kwargs: dict) -> dict:
        """Filters kwargs to only include those accepted by a class constructor.
                
                Inspects the constructor signature and removes any extraneous keyword arguments.
                
                Parameters
                ----------
                cls : object
                    Target class.
                kwargs : dict
                    Full dictionary of keyword arguments.
                
                Returns
                -------
                dict
                    Filtered dictionary of valid constructor arguments.
                
        """
        sig = inspect.signature(cls.__init__)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    def describe_params_detailed(self) -> dict:
        """Returns a dictionary describing the current instance's parameters.
                
                Uses reflection to enumerate the constructor parameters and their current values.
                
                Returns
                -------
                dict
                    Dictionary mapping parameter names to their type and value.
                
        """
        sig = inspect.signature(self.__init__)
        return {
            name: {"type": type(getattr(self, name)).__name__, "value": getattr(self, name)}
            for name in sig.parameters if name != "self" and hasattr(self, name)
        }

    @abstractmethod
    def _get_params(self)-> dict:
        raise NotImplementedError("All derived classes must implement _get_params.")


    def to_dict(self) -> dict:
        """Serializes the instance to a dictionary.
                
                Returns
                -------
                dict
                    Dictionary with "type" and "params" fields.
                
        """
        return {"type": self.__class__.__name__, "name": self.name, "params": self._get_params()}

    @classmethod
    def from_dict(cls, data: dict) -> Any:
        """Deserializes an instance from a dictionary.
                
                Uses the type key to instantiate the correct registered subclass.
                
                Parameters
                ----------
                data : dict
                    Dictionary with "type" and optionally "params".
                
                Returns
                -------
                Any
                    Instantiated object.
                
        """
        return cls.create(data["name"], **data.get("params", {}))

    def help(self) -> str:
        """Returns the class-level docstring.
                
                Useful for introspection and documentation tools.
                
                Returns
                -------
                str
                    String representation of the docstring or fallback text.
                
        """
        return inspect.getdoc(self.__class__) or "No documentation available."

    @classmethod
    @abstractmethod
    def validate_params(cls, **kwargs):
        """parameter validation hook for subclasses.
        
        all subclasses must implement validate_params
        
        Parameters
        ----------
        kwargs : object
            Parameters to validate.
        """
        raise NotImplementedError("all subclasses must implement validate_params")
    
    @property
    def name(self) -> str:
        """Returns the registered name of the class (lowercased, with known suffix removed).
                
                Strips one of the known suffixes like 'TNorm', 'Implicator', etc., based on class name.
                
                Returns
                -------
                str
                    Cleaned lowercase name.
                
        """
        name = self.__class__.__name__
        suffixes = ["TNorm", "Implicator", "Similarity", "SimilarityFunction", 
                    "FuzzyQuantifier", "Model", "Function", "Strategy"]  # Expand as needed

        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break  # Stop after first match

        return name.lower()
    
    @classmethod
    def get_registered_name(cls, instance: Any) -> str:
        """Get the primary registered alias for a given instance.
                
                Parameters
                ----------
                instance : Any
                    An instance of a registered subclass.
                
                Returns
                -------
                str
                    Primary alias string.
                
        """
        for klass, aliases in cls._aliases.items():
            if isinstance(instance, klass):
                return aliases[0]
        raise ValueError(f"No registered alias found for instance of type {type(instance).__name__}")
    
    @classmethod
    def get_class(cls, name: str) -> Type:
        """Returns the registered class associated with a given alias.
                
                Parameters
                ----------
                name : str
                    The alias name (case-insensitive).
                
                Returns
                -------
                Type
                    The class object registered under that alias.
                
                Raises
                ------
                ValueError
                    If alias is not registered.
                
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unknown alias: {name}")
        return cls._registry[name]
