"""
@file base_allpurpose_fuzzy_rough_oversampler.py
@brief Abstract base class for oversampling using Fuzzy Rough Sets.
"""

from abc import ABC, abstractmethod
from imblearn.over_sampling.base import BaseOverSampler
import warnings
import FRsutils.utils.validation_utils.validation_utils as valutil
from FRsutils.utils.constructor_utils.lazy_constructible_mixin import LazyConstructibleMixin
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel



class BaseAllPurposeFuzzyRoughOversampler(LazyConstructibleMixin, ABC, BaseOverSampler):
    """
    @brief Abstract base class for oversampling using Fuzzy Rough Sets.
    """

    def __init__(self, **kwargs):
        """
        @brief Initializes the fuzzy rough oversampler with safe defaults.

        Even if kwargs is empty (e.g., cloned estimators in sklearn), we must pass a valid
        `sampling_strategy` to BaseOverSampler and keep sampler-level parameters consistent.

        Sampler-level parameters handled here:
        - sampling_strategy (imblearn BaseOverSampler)
        - instance_ranking_strategy (FRsutils logic)
        - sampling_ratio (FRsutils logic)

        @param kwargs Dictionary of sampler + fuzzy-rough model configuration parameters.
        """
        # ---- sampler-level defaults (must never be None) ----
        sampling_strategy = kwargs.get("sampling_strategy", "auto")
        instance_ranking_strategy = kwargs.get("instance_ranking_strategy", "pos")
        sampling_ratio = kwargs.get("sampling_ratio", None)
        type = kwargs.get("type", "itfrs")

        # BaseOverSampler expects a valid sampling_strategy (default is typically "auto")
        super().__init__(sampling_strategy=sampling_strategy)

        # Keep sampler-level attributes always defined (so methods can safely access them)
        if isinstance(instance_ranking_strategy, str):
            self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(instance_ranking_strategy)
        if isinstance(type, str):
            self.type = type
        # if isinstance(instance_ranking_strategy, dict):
            self.instance_ranking_strategy = instance_ranking_strategy
        # else:
        #     raise TypeError("instance_ranking_strategy must be a string or a dict.")

        self.sampling_ratio = sampling_ratio
        
   
    def _get_target_classes(self):
        if self.instance_ranking_strategy == 'pos':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            return list(self.instance_ranking_strategy.keys())
        else:
            warnings.warn(f"Unsupported strategy: {self.instance_ranking_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]

    def _get_num_samples(self, class_label):
        if self.sampling_ratio == None:
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
            return max(0, target_count - self.target_stats_[class_label])
        
        # each class can have its own sampling ration injected by a dictionary
        elif isinstance(self.sampling_ratio, dict):
            aa = self.target_stats_[class_label] * self.sampling_ratio[str(class_label)]
        
        elif isinstance(self.sampling_ratio, int) or isinstance(self.sampling_ratio, float):
            aa = self.target_stats_[class_label] * self.sampling_ratio
        
        target_count = round(aa) + 1
        return target_count
        # else:
        #     warnings.warn(f"Fallback to 'auto' for strategy '{self.instance_ranking_strategy}'")
        #     majority_class = max(self.target_stats_, key=self.target_stats_.get)
        #     target_count = self.target_stats_[majority_class]

    def _validate_config(self, *, model_registry=None, **config):
        """
        @brief Validates sampler configuration during configure() (fail-fast).

        - Runs generic registry/type checks from LazyConstructibleMixin.
        - Then enforces sampler-specific parameter checks via `_check_params(...)`.

        @param model_registry: Optional registry used to resolve model types.
        @param config: Clone-friendly configuration (numbers/strings/flags).
        @raises ValueError/TypeError: If configuration is invalid.
        """
        super()._validate_config(model_registry=model_registry, **config)

        # sampler-specific validation (mandatory for concrete samplers)
        self._check_params(model_registry=model_registry, **config)    

    def configure(self, *, model_registry=None, **config):
        """
        @brief Stores config for lazy build, while applying sampler-level params immediately.

        Keeps BaseOverSampler behavior consistent with stored config, especially when configure()
        is called after __init__ (e.g., via set_params in GridSearchCV).

        @param model_registry: Optional registry for fuzzy-rough models.
        @param config: Flat configuration dictionary.
        """
        # Apply sampler-level params immediately (if present)
        if "sampling_strategy" in config:
            self.sampling_strategy = config["sampling_strategy"]

        if "instance_ranking_strategy" in config:
            v = config["instance_ranking_strategy"]
            if isinstance(v, str):
                self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(v)
            elif isinstance(v, dict):
                self.instance_ranking_strategy = v
            else:
                raise TypeError("instance_ranking_strategy must be a string or a dict.")

        if "sampling_ratio" in config:
            self.sampling_ratio = config["sampling_ratio"]

        return super().configure(model_registry=model_registry, **config)

    def get_params(self, deep=True):
        """
        @brief sklearn-compatible parameter export.

        Must return stable parameters for clone/GridSearch even when UNCONFIGURED.
        We expose:
        - sampler-level attributes (sampling_strategy, instance_ranking_strategy, sampling_ratio)
        - stored config (_object_config) if present

        If deep=True and any parameter value has get_params(), it is expanded using `key__subkey`.

        @param deep: Whether to return nested estimator parameters.
        @return dict of parameters.
        """
        params = {}

        # sampler-level always available (thanks to defaults in __init__)
        params["sampling_strategy"] = getattr(self, "sampling_strategy", "auto")
        params["instance_ranking_strategy"] = getattr(self, "instance_ranking_strategy", "pos")
        params["sampling_ratio"] = getattr(self, "sampling_ratio", None)

        # add stored config if available
        if hasattr(self, "_object_config") and isinstance(self._object_config, dict):
            params.update(self._object_config)

        # ensure sampler-level values reflect attributes (not stale config)
        params["sampling_strategy"] = getattr(self, "sampling_strategy", params.get("sampling_strategy", "auto"))
        params["instance_ranking_strategy"] = getattr(
            self, "instance_ranking_strategy", params.get("instance_ranking_strategy", "pos")
        )
        params["sampling_ratio"] = getattr(self, "sampling_ratio", params.get("sampling_ratio", None))

        if not deep:
            return dict(params)

        # expand nested estimators if any
        deep_params = {}
        for k, v in list(params.items()):
            if hasattr(v, "get_params") and callable(getattr(v, "get_params")):
                for sub_k, sub_v in v.get_params(deep=True).items():
                    deep_params[f"{k}__{sub_k}"] = sub_v
        params.update(deep_params)
        return params

    @abstractmethod
    def _check_params(self, **kwargs):
        raise NotImplementedError("_check_params must be implemented")

    @abstractmethod
    def fit_resample(self, X, y):
        raise NotImplementedError("fit_resample must be implemented")
