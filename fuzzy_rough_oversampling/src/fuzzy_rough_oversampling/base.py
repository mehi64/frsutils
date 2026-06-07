"""
@file base.py
@brief Shared base classes for fuzzy-rough oversampling algorithms.

This module contains the reusable estimator/configuration layer used by concrete
fuzzy-rough oversamplers such as FRSMOTE and future algorithms such as FRADASYN.
The classes keep an external flat parameter interface for sklearn/GridSearchCV,
while internally building a nested FRsutils configuration through the public
`FRsutils.api` facade.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# LifecycleState                       Small lifecycle enum for lazy model building
# FuzzyRoughOversamplerMixin           Shared flat fuzzy-rough parameter extraction
# BaseAllPurposeFuzzyRoughOversampler  Common sklearn-compatible oversampler config
# BaseSoloFuzzyRoughOversampler        Base for algorithms using one FR model directly

# ✅ Design Patterns & Clean Code Notes
# - Template Method: concrete algorithms implement _fit_resample and sample hooks
# - Adapter Pattern: flat sklearn params -> nested FRsutils config
# - Facade Pattern: FRsutils functionality is consumed via _frsutils.py
# - Dependency Inversion: this package depends on FRsutils public API, not internals
# - SRP: estimator lifecycle/config code is separated from concrete algorithms
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# class FRSMOTE(BaseSoloFuzzyRoughOversampler):
#     def _fit_resample(self, X, y):
#         ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum, auto
from typing import Any, Dict
import warnings

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_X_y

from fuzzy_rough_oversampling._frsutils import (
    build_fuzzy_rough_model,
    build_similarity_matrix,
    get_fuzzy_rough_model_class,
    normalize_flat_config_to_nested,
)
from fuzzy_rough_oversampling.validation import (
    compatible_dataset_with_fuzzy_rough,
    validate_ranking_strategy_choice,
)


class LifecycleState(Enum):
    """
    @brief Lifecycle states used for lazy fuzzy-rough model construction.
    """

    UNCONFIGURED = auto()
    CONFIGURED = auto()
    BUILT = auto()


class FuzzyRoughOversamplerMixin:
    """
    @brief Lightweight mixin for shared fuzzy-rough oversampler behavior.

    The mixin provides a small helper for collecting flat fuzzy-rough engine
    parameters from an estimator. Concrete algorithms may reuse it when they need
    to forward only fuzzy-rough-specific params into FRsutils config/model APIs.
    """

    _FR_ENGINE_PARAM_PREFIXES = (
        "similarity_",
        "ub_",
        "lb_",
    )

    def build_fuzzy_rough_params(self) -> Dict[str, Any]:
        """
        @brief Return flat fuzzy-rough engine params from estimator attributes.

        @return: Mapping of flat params that should be passed to FRsutils public
            config/model APIs.
        """
        params: Dict[str, Any] = {}
        for attr_name, value in vars(self).items():
            if attr_name.startswith("_"):
                continue
            if attr_name in {"type", "similarity", "similarity_tnorm"}:
                params[attr_name] = value
                continue
            if attr_name.startswith(self._FR_ENGINE_PARAM_PREFIXES):
                params[attr_name] = value
        return params


class BaseAllPurposeFuzzyRoughOversampler(
    FuzzyRoughOversamplerMixin,
    ABC,
    BaseOverSampler,
):
    """
    @brief Shared sklearn-compatible base for fuzzy-rough oversamplers.

    This class owns the flat estimator configuration, nested FRsutils config, and
    common sampling-ratio helpers. It intentionally lives in the standalone
    oversampling package because it depends on imbalanced-learn and represents
    application-layer behavior, not FRsutils core math.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "sampling_strategy": "auto",
        "instance_ranking_strategy": "pos",
        "sampling_ratio": None,
        "type": "itfrs",
    }

    _ALLOWED_TRANSITIONS = {
        LifecycleState.UNCONFIGURED: {LifecycleState.CONFIGURED},
        LifecycleState.CONFIGURED: {LifecycleState.CONFIGURED, LifecycleState.BUILT},
        LifecycleState.BUILT: {LifecycleState.CONFIGURED},
    }

    @classmethod
    def _collect_default_config(cls) -> Dict[str, Any]:
        """
        @brief Collect DEFAULT_CONFIG dictionaries along the class hierarchy.

        @return: Merged default configuration from base classes to subclass.
        """
        merged: Dict[str, Any] = {}
        for klass in reversed(cls.mro()):
            default_config = getattr(klass, "DEFAULT_CONFIG", None)
            if isinstance(default_config, dict):
                merged.update(default_config)
        return merged

    def __init__(self, **kwargs: Any) -> None:
        """
        @brief Initialize the oversampler with flat sklearn-compatible params.

        @param kwargs: Flat estimator parameters. Concrete classes may call
            configure() after initialization to validate non-default user params.
        """
        cfg = self._collect_default_config()
        cfg.update(kwargs)

        super().__init__(sampling_strategy=cfg.get("sampling_strategy", "auto"))

        self._state = LifecycleState.UNCONFIGURED
        self._object_config: Dict[str, Any] = {}
        self._nested_config = normalize_flat_config_to_nested(
            cfg,
            explicit_keys=set(kwargs.keys()),
        )
        self._lazy_object = None

        self.sampling_strategy = cfg.get("sampling_strategy", "auto")
        self.instance_ranking_strategy = self._normalize_instance_ranking_strategy(
            cfg.get("instance_ranking_strategy", "pos")
        )
        self.sampling_ratio = cfg.get("sampling_ratio", None)
        self.type = self._normalize_model_type(cfg.get("type", "itfrs"))

    @staticmethod
    def _normalize_model_type(model_type: Any) -> str:
        """
        @brief Validate and normalize a fuzzy-rough model type value.

        @param model_type: Candidate model type.
        @return: Normalized model type string.
        """
        if not isinstance(model_type, str) or not model_type.strip():
            raise TypeError("type must be a non-empty string.")
        return model_type.strip().lower()

    @staticmethod
    def _normalize_instance_ranking_strategy(strategy: Any) -> Any:
        """
        @brief Validate a ranking strategy value.

        @param strategy: String strategy or per-class dictionary.
        @return: Validated strategy.
        """
        if isinstance(strategy, str):
            return validate_ranking_strategy_choice(strategy)
        if isinstance(strategy, dict):
            return strategy
        raise TypeError("instance_ranking_strategy must be a string or a dict.")

    def _set_state(self, new_state: LifecycleState) -> None:
        """
        @brief Move the estimator lifecycle state forward safely.

        @param new_state: Target lifecycle state.
        @raises RuntimeError: If the transition is not allowed.
        """
        current = getattr(self, "_state", LifecycleState.UNCONFIGURED)
        if new_state not in self._ALLOWED_TRANSITIONS.get(current, set()):
            raise RuntimeError(f"Invalid state transition: {current.name} → {new_state.name}")
        self._state = new_state

    def _validate_config(self, **config: Any) -> None:
        """
        @brief Validate common fuzzy-rough model configuration.

        @param config: Flat estimator configuration.
        """
        model_type = self._normalize_model_type(config.get("type", "itfrs"))
        get_fuzzy_rough_model_class(model_type)
        self._check_params(**config)

    def configure(self, **config: Any):
        """
        @brief Store flat config and regenerate nested FRsutils config.

        @param config: Flat sklearn-friendly parameters.
        @return: self
        """
        explicit_keys = config.pop("_explicit_keys", None)
        if explicit_keys is None:
            explicit_keys = set(config.keys())

        merged = self._collect_default_config()
        merged.update(config)

        self._validate_config(**merged)
        self._nested_config = normalize_flat_config_to_nested(
            merged,
            explicit_keys=set(explicit_keys),
        )

        self.sampling_strategy = merged.get("sampling_strategy", self.sampling_strategy)
        self.sampling_ratio = merged.get("sampling_ratio", self.sampling_ratio)
        self.instance_ranking_strategy = self._normalize_instance_ranking_strategy(
            merged.get("instance_ranking_strategy", self.instance_ranking_strategy)
        )
        self.type = self._normalize_model_type(merged.get("type", self.type))

        self._object_config = dict(merged)
        self._lazy_object = None
        self._set_state(LifecycleState.CONFIGURED)
        return self

    def build(self, similarity_matrix: np.ndarray, labels: np.ndarray) -> None:
        """
        @brief Build the underlying fuzzy-rough model through FRsutils public API.

        @param similarity_matrix: Pairwise similarity matrix.
        @param labels: Label vector aligned with the similarity matrix.
        """
        if self.state_enum != LifecycleState.CONFIGURED:
            raise RuntimeError("Estimator must be configured before build().")

        self._lazy_object = build_fuzzy_rough_model(
            similarity_matrix=similarity_matrix,
            labels=labels,
            config=getattr(self, "_nested_config", None),
            **dict(getattr(self, "_object_config", {})),
        )
        self._finalize_object()
        self._set_state(LifecycleState.BUILT)

    def set_params(self, **params: Any):
        """
        @brief scikit-learn compatible flat parameter update.

        @param params: Flat estimator parameters, e.g. from GridSearchCV.
        @return: self
        """
        if not params:
            return self

        base_config = dict(getattr(self, "_object_config", {}))
        if not base_config:
            base_config = self._collect_default_config()
        base_config.update(params)
        self.configure(_explicit_keys=set(params.keys()), **base_config)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        @brief Return flat sklearn-compatible estimator parameters.

        @param deep: If True, include nested estimator params when any value has
            get_params. This mirrors sklearn conventions.
        @return: Flat parameter dictionary.
        """
        params: Dict[str, Any] = self._collect_default_config()

        if isinstance(getattr(self, "_object_config", None), dict):
            params.update(self._object_config)

        params["sampling_strategy"] = getattr(
            self,
            "sampling_strategy",
            params.get("sampling_strategy", "auto"),
        )
        params["instance_ranking_strategy"] = getattr(
            self,
            "instance_ranking_strategy",
            params.get("instance_ranking_strategy", "pos"),
        )
        params["sampling_ratio"] = getattr(
            self,
            "sampling_ratio",
            params.get("sampling_ratio", None),
        )
        params["type"] = getattr(self, "type", params.get("type", "itfrs"))

        if not deep:
            return dict(params)

        deep_params: Dict[str, Any] = {}
        for key, value in list(params.items()):
            if hasattr(value, "get_params") and callable(getattr(value, "get_params")):
                for sub_key, sub_value in value.get_params(deep=True).items():
                    deep_params[f"{key}__{sub_key}"] = sub_value
        params.update(deep_params)
        return params

    def _get_target_classes(self):
        """
        @brief Determine which classes should be oversampled.

        @return: Iterable of target class labels.
        """
        if self.instance_ranking_strategy == "pos":
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        if isinstance(self.instance_ranking_strategy, dict):
            return list(self.instance_ranking_strategy.keys())

        warnings.warn(
            f"Unsupported strategy: {self.instance_ranking_strategy}. Falling back to 'pos'."
        )
        majority_class = max(self.target_stats_, key=self.target_stats_.get)
        return [cls for cls in self.classes_ if cls != majority_class]

    def _get_num_samples(self, class_label: Any) -> int:
        """
        @brief Compute how many synthetic samples should be generated for a class.

        @param class_label: Class to oversample.
        @return: Number of synthetic samples to generate.
        """
        if self.sampling_ratio is None:
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
            return max(0, target_count - self.target_stats_[class_label])

        if isinstance(self.sampling_ratio, dict):
            key = class_label if class_label in self.sampling_ratio else str(class_label)
            if key not in self.sampling_ratio:
                raise ValueError(
                    f"sampling_ratio dict does not contain a ratio for class '{class_label}'."
                )
            ratio = self.sampling_ratio[key]
            if not isinstance(ratio, (int, float)):
                raise TypeError("sampling_ratio per-class values must be numeric.")
            raw_count = self.target_stats_[class_label] * ratio
        elif isinstance(self.sampling_ratio, (int, float)):
            raw_count = self.target_stats_[class_label] * self.sampling_ratio
        else:
            raise TypeError("sampling_ratio must be None, a number, or a dict.")

        return int(round(raw_count)) + 1

    @property
    def is_built(self) -> bool:
        """
        @brief Return True when the lazy fuzzy-rough model has been built.
        """
        return self.state_enum == LifecycleState.BUILT

    @property
    def state_enum(self) -> LifecycleState:
        """
        @brief Return the current lifecycle state.
        """
        return getattr(self, "_state", LifecycleState.UNCONFIGURED)

    @property
    def state_str(self) -> str:
        """
        @brief Return the current lifecycle state name.
        """
        return self.state_enum.name

    @property
    def lazy_object(self):
        """
        @brief Return the built fuzzy-rough model.

        @raises RuntimeError: If build() has not been called.
        """
        if self.state_enum != LifecycleState.BUILT:
            raise RuntimeError("Object has not been built yet. Call build() first.")
        return self._lazy_object

    @abstractmethod
    def _check_params(self, **kwargs: Any) -> None:
        """@brief Validate concrete algorithm parameters."""
        raise NotImplementedError("_check_params must be implemented")

    @abstractmethod
    def _finalize_object(self) -> None:
        """@brief Cache concrete attributes after the object is built."""
        raise NotImplementedError("_finalize_object must be implemented")

    @abstractmethod
    def fit_resample(self, X, y):
        """@brief Fit the estimator and return resampled X/y."""
        raise NotImplementedError("fit_resample must be implemented")


class BaseSoloFuzzyRoughOversampler(BaseAllPurposeFuzzyRoughOversampler):
    """
    @brief Base for oversamplers guided directly by one fuzzy-rough model.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "similarity": "linear",
        "similarity_name": None,
        "similarity_sigma": 0.1,
        "gaussian_similarity_sigma": None,
        "similarity_tnorm": "minimum",
        "similarity_tnorm_name": None,
        "similarity_tnorm_p": 2.0,
        "k_neighbors": 5,
        "bias_interpolation": False,
        "random_state": None,
    }

    def __init__(self, **kwargs: Any) -> None:
        """@brief Initialize a solo fuzzy-rough oversampler."""
        super().__init__(**kwargs)

    def fit(self, X, y):
        """
        @brief Validate data, build similarity matrix, and build fuzzy-rough model.

        @param X: Feature matrix normalized to [0, 1].
        @param y: Target labels.
        @return: self
        """
        compatible_dataset_with_fuzzy_rough(X, y)
        X, y = check_X_y(X, y, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        if self.state_enum == LifecycleState.UNCONFIGURED:
            self.configure(**self._collect_default_config())

        if not isinstance(getattr(self, "_object_config", None), dict) or not self._object_config:
            raise RuntimeError(
                "Estimator is missing _object_config. Ensure configure()/set_params() succeeded."
            )

        self._validate_config(**self._object_config)

        # Rebuild per fit call so repeated fit_resample calls on the same estimator
        # use the current X/y rather than a stale similarity/model object.
        if self.state_enum == LifecycleState.BUILT:
            self._set_state(LifecycleState.CONFIGURED)

        similarity_matrix = build_similarity_matrix(
            X,
            config=getattr(self, "_nested_config", None),
            **self._object_config,
        )
        self.build(similarity_matrix, y)
        return self

    @property
    def positive_region(self):
        """
        @brief Return the positive-region score for each fitted instance.
        """
        return self.lazy_object.lower_approximation()

    @abstractmethod
    def _check_params(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def _fit_resample(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label):
        raise NotImplementedError

    @abstractmethod
    def _generate_new_samples(self, *args: Any, **kwargs: Any):
        raise NotImplementedError
