"""
@file
@brief Factory functions for building fuzzy-rough models by name.

@details These functions instantiate fuzzy-rough model objects using validated parameters.
They decouple model construction logic from the model class definitions.
"""

from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models.vqrs import VQRS
from FRsutils.core.approximations import BaseFuzzyRoughModel
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.utils.constructor_utils.function_registry import IMPLICATOR_REGISTRY, OWA_WEIGHT_REGISTRY
import FRsutils.utils.validation_utils as valutils

def build_fuzzy_rough_model(fr_model_type,
                            lb_implicator_type,
                            ub_tnorm_type,
                            owa_weighting_strategy_type,
                            fuzzy_quantifier_type,
                            alpha_lower,
                            beta_lower,
                            alpha_upper,
                            beta_upper,
                            similarity_matrix, 
                            labels) -> BaseFuzzyRoughModel:
    
   
    # build the ITFRS model (validates the parameters internally)


    if fr_model_type == 'ITFRS':
        return ITFRS(
            similarity_matrix=similarity_matrix,
            labels=labels,
            tnorm=build_tnorm(ub_tnorm_type),
            implicator=IMPLICATOR_REGISTRY[lb_implicator_type]
        )
    
    elif fr_model_type == 'OWAFRS':
        raise NotImplementedError('VQRS is not implemented yet')
        if (owa_weighting_strategy_type == 'linear'):
            owa_lb = OWA_WEIGHT_REGISTRY['linear_sup']
        return OWAFRS(
            similarity_matrix=similarity_matrix,
            labels=labels,
            tnorm=build_tnorm(ub_tnorm_type),
            implicator=IMPLICATOR_REGISTRY[lb_implicator_type],
            lower_app_weights_method=None,
            upper_app_weights_method=None
        )
    
    elif fr_model_type == 'VQRS':
        raise NotImplementedError('VQRS is not implemented yet')
        # return VQRS(
        #     similarity_matrix=similarity_matrix,
        #     labels=labels,
        #     alpha_Q_lower=fr_model_params['alpha_Q_lower'],
        #     beta_Q_lower=fr_model_params['beta_Q_lower'],
        #     alpha_Q_upper=fr_model_params['alpha_Q_upper'],
        #     beta_Q_upper=fr_model_params['beta_Q_upper']
        # )
    
    else:
        raise ValueError(f"Unsupported fuzzy rough model type: '{fr_model_type}'")
