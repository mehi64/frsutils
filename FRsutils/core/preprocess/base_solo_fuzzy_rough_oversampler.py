""" 
@file base_solo_fuzzy_rough_oversampler.py
@brief Base class for fuzzy rough oversamplers that don’t rely on external generators.
...
"""
from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from typing import Any, Dict
import numpy as np
from sklearn.utils import check_X_y

from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from FRsutils.core.preprocess.base_allpurpose_fuzzy_rough_oversampler import BaseAllPurposeFuzzyRoughOversampler
from FRsutils.core.similarities import build_similarity_matrix
from FRsutils.utils.constructor_utils.lazy_constructible_mixin import LifecycleState
from FRsutils.utils.fuzzy_rough_dataset_validation_utils import compatible_dataset_with_FuzzyRough

class BaseSoloFuzzyRoughOversampler(BaseAllPurposeFuzzyRoughOversampler):
    DEFAULT_CONFIG: Dict[str, Any] = {
        "similarity": "linear",
        "similarity_tnorm": "minimum",
        "k_neighbors": 5,
        "bias_interpolation": False,
        "random_state": None,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        compatible_dataset_with_FuzzyRough(X, y)
        X, y = check_X_y(X, y, accept_sparse=False)

        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        if self.state_enum == LifecycleState.UNCONFIGURED:
            defaults = self._collect_default_config()
            registry = getattr(self, "_model_registry", None) or FuzzyRoughModel
            self.configure(model_registry=registry, **defaults)

        if not hasattr(self, "_object_config") or not getattr(self, "_object_config"):
            raise RuntimeError("Estimator is missing _object_config. Ensure configure()/set_params() succeeded.")

        registry = getattr(self, "_model_registry", None) or FuzzyRoughModel
        self._model_registry = registry
        self._validate_config(model_registry=registry, **self._object_config)

        if not self.is_built:
            similarity_matrix = build_similarity_matrix(X, **self._object_config)
            self.build(similarity_matrix, y)

        return self

    @property
    def positive_region(self):
        return self.lazy_object.lower_approximation()

    def set_params(self, **params):
        if not params:
            return self
        base_config = dict(getattr(self, "_object_config", {}))
        base_config.update(params)
        registry = getattr(self, "_model_registry", None) or FuzzyRoughModel
        self.configure(model_registry=registry, **base_config)
        return self

    @abstractmethod
    def _check_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _fit_resample(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label):
        raise NotImplementedError

    @abstractmethod
    def _generate_new_samples(self, *args, **kwargs):
        raise NotImplementedError