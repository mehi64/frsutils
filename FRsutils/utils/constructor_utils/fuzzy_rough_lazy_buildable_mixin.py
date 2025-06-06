
"""
@file
@brief Mixin class that adds lazy building logic for fuzzy-rough models based on configuration.
"""

from abc import ABC
import FRsutils.utils.validation_utils as vutils
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.utils.constructor_utils.similarity_builder import build_similarity
from FRsutils.utils.constructor_utils.fr_model_builder import build_fuzzy_rough_model
from FRsutils.core.similarities import calculate_similarity_matrix

class FuzzyRoughLazyBuildableMixin(ABC):
    """
    @brief Mixin for fuzzy-rough based oversamplers or estimators to support lazy model building.

    @details This mixin handles:
     - Storing configuration params
     - Validating parameters
     - Lazily building the fuzzy-rough model and similarity matrix only when needed
    """

    def _initialize_fr_config(  self,
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
                                similarity_tnorm_type='minimum'):
        
        self.fr_model_type = vutils.validate_fr_model_choice(fr_model_type)
        self.lb_implicator_type = vutils.validate_implicator_choice(lb_implicator_type)
        self.ub_tnorm_type = vutils.validate_tnorm_choice(ub_tnorm_type)
        
        self.owa_weighting_strategy_type = vutils.validate_owa_weighting_strategy_choice(owa_weighting_strategy_type)
        self.fuzzy_quantifier_type = vutils.validate_fuzzy_quantifier_choice(fuzzy_quantifier_type)
        
        self.alpha_lower = vutils.validate_range_0_1(alpha_lower, name='alpha_lower')
        self.beta_lower = vutils.validate_range_0_1(beta_lower, name='beta_lower')

        self.alpha_upper = vutils.validate_range_0_1(alpha_upper, name='alpha_upper')
        self.beta_upper = vutils.validate_range_0_1(beta_upper, name='beta_upper')
        
        if (alpha_lower >= beta_lower):
            raise ValueError("alpha_lower must be < beta_lower")
       
        if (alpha_upper >= beta_upper):
            raise ValueError("alpha_upper must be < beta_upper")
       
        self.similarity_type = vutils.validate_similarity_choice(similarity_type)
        self.similarity_tnorm_type = vutils.validate_tnorm_choice(similarity_tnorm_type)
        self.gaussian_similarity_sigma = vutils.validate_range_0_1(alpha_upper, name='gaussian_similarity_sigma')
        if (gaussian_similarity_sigma > 0.5):
            raise ValueError("gaussian_similarity_sigma is in range [0, 0.5]")
       
        self._is_built = False

    def _build_internal_objects(self, X, y):
        """
        @brief Constructs the similarity function, and similarity t-norm, and similarity matrix with 
        them. Plus the fuzzy rough model instance.

        @details Should be called before using self.fr_model. Sets _is_built = True.
        """
        similarity_func = build_similarity(self.similarity_type, self.gaussian_similarity_sigma)
        similarity_tnorm = build_tnorm(self.similarity_tnorm_type)

        self.similarity_matrix = calculate_similarity_matrix(X, similarity_func, similarity_tnorm)

        self.fr_model = build_fuzzy_rough_model(fr_model_type=self.fr_model_type,
                                                lb_implicator_type=self.lb_implicator_type,
                                                ub_tnorm_type=self.ub_tnorm_type,
                                                owa_weighting_strategy_type=self.owa_weighting_strategy_type,
                                                fuzzy_quantifier_type=None,
                                                alpha_lower=None,
                                                beta_lower=None,
                                                alpha_upper=None,
                                                beta_upper=None,
                                                similarity_matrix=self.similarity_matrix, 
                                                labels=y)

        self.lower_app = self.fr_model.lower_approximation()
        self.upper_app = self.fr_model.upper_approximation()

        # # TODO: check this. Is that correct?what about all models?
        self.positive_region = self.lower_app
        
        self._is_built = True

    def ensure_built(self, X, y):
        """
        @brief Public method to build internal components if not already built.
        """
        if not getattr(self, '_is_built', False):
            self._build_internal_objects(X, y)