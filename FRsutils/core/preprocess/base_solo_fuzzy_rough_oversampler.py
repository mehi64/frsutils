"""
@file base_solo_fuzzy_rough_oversampler.py
@brief Base class for fuzzy rough oversamplers that don’t rely on external generators.
"""

import numpy as np
from collections import Counter
from sklearn.utils import check_X_y
from abc import abstractmethod
from FRsutils.core.preprocess.base_allpurpose_fuzzy_rough_oversampler import BaseAllPurposeFuzzyRoughOversampler
from FRsutils.utils.fuzzy_rough_dataset_validation_utils import compatible_dataset_with_FuzzyRough
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from FRsutils.core.similarities import build_similarity_matrix

class BaseSoloFuzzyRoughOversampler(BaseAllPurposeFuzzyRoughOversampler):
    def __init__(self, **kwargs):
        """
        @brief Initializes solo fuzzy rough oversampler with fuzzy config and SMOTE-related settings.
        @param kwargs Dictionary of hyperparameters including k_neighbors, bias_interpolation, etc.
        """
        self.k_neighbors = kwargs.get("k_neighbors", 5)
        self.bias_interpolation = kwargs.get("bias_interpolation", False)
        self.random_state = kwargs.get("random_state", None)
        
        super().__init__(**kwargs)

    def fit(self, X, y):
        """
        @brief Validates the input dataset, computes similarity matrix, and builds the fuzzy-rough model.

        @param X Normalized feature matrix (2D np.ndarray).
        @param y Target class labels (1D np.ndarray).

        @return self
        """
        compatible_dataset_with_FuzzyRough(X, y)
        self._check_params()

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        config = self.get_params(deep=False)

        if not self.is_built:
            if self.state == "UNCONFIGURED":
                self.configure(**config, model_registry=FuzzyRoughModel)

            similarity_matrix = build_similarity_matrix(X, **self._object_config)
            self.ensure_build(similarity_matrix, y, **config)


        return self

    @property
    def positive_region(self):
        return self._lazy_object.lower_approximation()

    def get_params(self, deep=True):
        return {
            "k_neighbors": self.k_neighbors,
            "bias_interpolation": self.bias_interpolation,
            "random_state": self.random_state,
            "sampling_strategy": self.sampling_strategy,
            "instance_ranking_strategy": self.instance_ranking_strategy,
            **(self._object_config if hasattr(self, "_object_config") else {})
        }

    
    def set_params(self, **params):
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                self._object_config[key] = val
        return self

    def _build_from_config(self, **config):
        for k, v in config.items():
            setattr(self, k, v)
    
    @abstractmethod
    def _check_params(self): pass

    @abstractmethod
    def _fit_resample(self, X, y): pass

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label): pass

    @abstractmethod
    def _generate_new_samples(self): pass
