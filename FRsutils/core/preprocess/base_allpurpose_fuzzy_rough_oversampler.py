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
        sampling_strategy = kwargs.get('sampling_strategy')
        instance_ranking_strategy = kwargs.get('instance_ranking_strategy')

        # if sampling_strategy is None:
        #     raise ValueError("`sampling_strategy` must be provided when instantiation")

        super().__init__(sampling_strategy=sampling_strategy)

        # if instance_ranking_strategy is None:
        #     raise ValueError("`instance_ranking_strategy` must be provided when instantiation")

        # # TODO: check and refine validation utils
        # self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(
        #     instance_ranking_strategy
        # )

        # if kwargs:
        #     # Apply all config values and prepare for lazy build
        #     self.configure(model_registry=FuzzyRoughModel, **kwargs)
        
   
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

        

    @abstractmethod
    def _check_params(self, **kwargs):
        raise NotImplementedError("_check_params must be implemented")

    @abstractmethod
    def fit_resample(self, X, y):
        raise NotImplementedError("fit_resample must be implemented")
