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
        @brief Initializes the fuzzy rough oversampler config via kwargs.

        @param sampling_strategy Oversampling strategy for imbalanced-learn.
        @param kwargs Dictionary of fuzzy-rough model configuration parameters.
        """
        sampling_strategy = kwargs.get('sampling_strategy', 'auto')
        super().__init__(sampling_strategy=sampling_strategy)
        
        self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(
            kwargs.get('instance_ranking_strategy', 'pos')
        )

        self._lazy_model_registry = FuzzyRoughModel
        
        if kwargs:
            # ✅ Apply all config values and prepare for lazy build
            self.configure(model_registry=self._lazy_model_registry, **kwargs)
        
   
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
        if self.instance_ranking_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            target_count = max(self.target_stats_[class_label], self.instance_ranking_strategy[class_label])
        else:
            warnings.warn(f"Fallback to 'auto' for strategy '{self.instance_ranking_strategy}'")
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]

        return max(0, target_count - self.target_stats_[class_label])

    @abstractmethod
    def _check_params(self):
        pass

    @abstractmethod
    def fit_resample(self, X, y):
        pass
