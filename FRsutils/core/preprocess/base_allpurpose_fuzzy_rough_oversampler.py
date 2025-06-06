import warnings
from imblearn.over_sampling.base import BaseOverSampler
from abc import ABC, abstractmethod
# from FRsutils.utils.init_helpers import assign_allowed_kwargs
# from FRsutils.utils.validation_utils import get_fr_model_param_schema

from FRsutils.utils.constructor_utils.fuzzy_rough_lazy_buildable_mixin import FuzzyRoughLazyBuildableMixin

import FRsutils.utils.validation_utils as valutil 

class BaseAllPurposeFuzzyRoughOversampler(FuzzyRoughLazyBuildableMixin, ABC, BaseOverSampler):
    """
    @brief Abstract base class for oversampling using Fuzzy Rough Sets.

    @details This base class is intended to be inherited by resamplers that either:
     - Use fuzzy-rough set theory directly for ranking or selecting instances.
     - Combine fuzzy-rough logic with generative models (e.g., VAE, GAN).
     - Use fuzzy-rough sets as a preprocessing step for other resamplers.

    It should not be used directly.

    @warning Do not instantiate or use this class directly. Use one of its concrete subclasses instead.
    """

    def __init__(self,
                 fr_model_type='ITFRS',
                 lb_implicator_type='reichenbach',
                 ub_tnorm_type='product',
                 owa_weighting_strategy_type='linear',
                 fuzzy_quantifier_type='quadratic',
                 alpha_lower=0.1,
                 beta_lower=0.6,
                 alpha_upper=0.2,
                 beta_upper=1.0,
                 similarity_type='gaussian',
                 gaussian_similarity_sigma=0.2,
                 similarity_tnorm_type='minimum',
                 instance_ranking_strategy='pos',
                 sampling_strategy='auto'):

        super().__init__(sampling_strategy=sampling_strategy)

        
        self._initialize_fr_config(fr_model_type=fr_model_type,
                                    lb_implicator_type=lb_implicator_type,
                                    ub_tnorm_type=ub_tnorm_type,
                                    owa_weighting_strategy_type=owa_weighting_strategy_type,
                                    fuzzy_quantifier_type=fuzzy_quantifier_type,
                                    alpha_lower=alpha_lower,
                                    beta_lower=beta_lower,
                                    alpha_upper=alpha_upper,
                                    beta_upper=beta_upper,
                                    similarity_type=similarity_type,
                                    gaussian_similarity_sigma=gaussian_similarity_sigma,
                                    similarity_tnorm_type=similarity_tnorm_type)

        self.instance_ranking_strategy = valutil.validate_ranking_strategy_choice(instance_ranking_strategy)
        
        
# _get_target_classes and _get_num_samples remain the same as in FRSMOTE
    def _get_target_classes(self):
        """Determine which classes to oversample based on sampling_strategy."""
        
        if self.instance_ranking_strategy == 'pos':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            return list(self.instance_ranking_strategy.keys())
        # Add more strategy handling if needed (float, list, callable)
        else:
            warnings.warn(f"Unsupported sampling_strategy: {self.instance_ranking_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]

    def _get_num_samples(self, class_label):
        """Determine number of samples to generate for a class."""
        if self.instance_ranking_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.instance_ranking_strategy, dict):
            # Ensure target count is not less than current count
            target_count = max(self.target_stats_[class_label], self.instance_ranking_strategy[class_label])
        else: # Default to balancing against majority if strategy is unclear
             warnings.warn(f"Interpreting sampling_strategy '{self.instance_ranking_strategy}' as 'auto'.")
             majority_class = max(self.target_stats_, key=self.target_stats_.get)
             target_count = self.target_stats_[majority_class]

        current_count = self.target_stats_[class_label]
        return max(0, target_count - current_count)
    
        
    @abstractmethod
    def _check_params(self):
        """
        checks correctness of parameters specific to this object.
        Each derived class must implements its own
        """
        raise NotImplementedError("Subclasses must implement _check_params.")
        
    @abstractmethod 
    def fit_resample(self, X, y):
        """
        @brief Fits the resampler to the data and returns the resampled data.
        all classes should implement this method
        
        @param X The input data.
        @param y The target labels.
        @return A tuple of the resampled data and labels.
        """
        raise NotImplementedError("Subclasses must implement fit_resample.")
 