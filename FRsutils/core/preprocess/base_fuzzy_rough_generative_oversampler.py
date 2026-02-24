"""
@file base_fuzzy_rough_generative_oversampler.py
@brief Abstract base class for fuzzy-rough oversamplers that use an external generator.

This module defines a base API for oversamplers that:
- Follow the imbalanced-learn oversampler contract
- Use fuzzy-rough models for guidance/ranking
- Rely on a *generator* component (e.g., GAN/Diffusion/other) for sample synthesis

The external estimator parameters remain **flat** for scikit-learn compatibility.
Internally, a nested config representation is built and stored in `_nested_config`
(see `normalize_flat_config_to_nested`).

##############################################
# ✅ Quick Summary of Features
# - sklearn/imbalanced-learn compatible base class
# - Lazy config/build lifecycle via LazyConstructibleMixin
# - Internal nested config derived from flat params
# - Hooks for building generator and performing fit_resample

# ✅ Design Patterns
# - Template Method: subclasses implement _build_generator and _fit_resample
# - Adapter: flat params -> nested config
# - SRP: separates generator construction from oversampling logic
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# class MyGenerativeFRSampler(BaseFuzzyRoughGenerativeOversampler):
#     def _check_params(self, **kwargs):
#         ...
#     def _build_generator(self, X, y):
#         ...
#     def _fit_resample(self, X, y):
#         ...
#
# sampler = MyGenerativeFRSampler(type="itfrs", similarity="gaussian", similarity_sigma=0.5)
# X_res, y_res = sampler.fit_resample(X, y)

"""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from typing import Any, Dict

import numpy as np
from sklearn.utils import check_X_y

from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from FRsutils.core.preprocess.base_allpurpose_fuzzy_rough_oversampler import BaseAllPurposeFuzzyRoughOversampler
from FRsutils.utils.constructor_utils.lazy_constructible_mixin import LifecycleState
from FRsutils.utils.fuzzy_rough_dataset_validation_utils import compatible_dataset_with_FuzzyRough


class BaseFuzzyRoughGenerativeOversampler(BaseAllPurposeFuzzyRoughOversampler):
    """ 
    @brief Base class for fuzzy-rough oversamplers that rely on an external generator.

    Subclasses are expected to:
    - validate parameters in `_check_params`
    - build generator artifacts in `_build_generator`
    - implement oversampling logic in `_fit_resample`
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "random_state": None,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        """ 
        @brief Fit basic dataset stats and (optionally) generator.

        @param X: feature matrix (expected normalized to [0,1] for fuzzy-rough similarity)
        @param y: target labels
        @return: self
        """
        compatible_dataset_with_FuzzyRough(X, y)
        X, y = check_X_y(X, y, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        if self.state_enum == LifecycleState.UNCONFIGURED:
            defaults = self._collect_default_config()
            self.configure(model_registry=FuzzyRoughModel, **defaults)

        # Generator can be built here or lazily in fit_resample
        return self

    def fit_resample(self, X, y):
        self.fit(X, y)
        return self._fit_resample(X, y)

    def _finalize_object(self):
        """ 
        @brief Finalize cached attributes from the flat config.

        Subclasses may extend but should keep this sklearn-friendly.
        """
        cfg = dict(getattr(self, "_object_config", {}))
        defaults = self._collect_default_config()

        self.random_state = cfg.get("random_state", defaults.get("random_state"))
        self.sampling_strategy = cfg.get("sampling_strategy", defaults.get("sampling_strategy", "auto"))
        self.instance_ranking_strategy = cfg.get(
            "instance_ranking_strategy", defaults.get("instance_ranking_strategy", "pos")
        )
        self.sampling_ratio = cfg.get("sampling_ratio", defaults.get("sampling_ratio"))
        self.type = cfg.get("type", defaults.get("type", "itfrs"))

    @abstractmethod
    def _check_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_generator(self, X, y):
        """ 
        @brief Build/initialize an external generator.

        @param X: feature matrix
        @param y: labels
        """
        raise NotImplementedError

    @abstractmethod
    def _fit_resample(self, X, y):
        """ 
        @brief Perform the oversampling.

        @param X: feature matrix
        @param y: labels
        @return: (X_resampled, y_resampled)
        """
        raise NotImplementedError
