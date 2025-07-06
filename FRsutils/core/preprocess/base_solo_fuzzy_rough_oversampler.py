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
        super().__init__(**kwargs)

        # k_neighbors = kwargs.get("k_neighbors")
        # bias_interpolation = kwargs.get("bias_interpolation", False)
        # random_state = kwargs.get("random_state", None)

        # if k_neighbors is None:
        #     raise ValueError("`k_neighbors` must be provided when instantiation")
        # if bias_interpolation is None:
        #     raise ValueError("`bias_interpolation` must be provided when instantiation")
        # if random_state is None:
        #     raise ValueError("`random_state` must be provided when instantiation")
        
        # self.k_neighbors = k_neighbors
        # self.bias_interpolation = bias_interpolation
        # self.random_state = random_state

        

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
            self.build(similarity_matrix, y)


        return self

    @property
    def positive_region(self):
        return self._lazy_object.lower_approximation()

    def get_params(self, deep=True):

        if self.state == 'UNCONFIGURED': return {}

        return {
            **(self._object_config if hasattr(self, "_object_config") else {})
        }

    
    def set_params(self, **params):
        if self.state == 'UNCONFIGURED':
            self.configure(model_registry=FuzzyRoughModel, **params)
        return self


    
    @abstractmethod
    def _check_params(self, **kwargs): 
        raise NotImplementedError("_check_params must be implemented")

    @abstractmethod
    def _fit_resample(self, X, y):
        raise NotImplementedError("_fit_resample must be implemented")

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label): 
        raise NotImplementedError("_prepare_minority_samples must be implemented")

    @abstractmethod
    def _generate_new_samples(self): 
        raise NotImplementedError("_generate_new_samples must be implemented")
