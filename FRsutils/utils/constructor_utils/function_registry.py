"""
@file
@brief Lookup tables for function-based components: implicators, quantifiers, OWA weights.
"""

from FRsutils.core.implicators import imp_goedel, imp_kleene_dienes, imp_lukasiewicz, imp_reichenbach
from FRsutils.core.fuzzy_quantifiers import fuzzy_quantifier_linear, fuzzy_quantifier_quadratic
from FRsutils.core.owa_weights import owa_suprimum_weights_linear, owa_infimum_weights_linear

# ------------------------------
# Implicators
# ------------------------------
IMPLICATOR_REGISTRY = {
    'goedel': imp_goedel,
    'kleene_dienes': imp_kleene_dienes,
    'lukasiewicz': imp_lukasiewicz,
    'reichenbach': imp_reichenbach,
}

# ------------------------------
# Fuzzy Quantifiers
# ------------------------------
FUZZY_QUANTIFIER_REGISTRY = {
    'linear': fuzzy_quantifier_linear,
    'quad': fuzzy_quantifier_quadratic,
}

# ------------------------------
# OWA Weight Functions
# ------------------------------
OWA_WEIGHT_REGISTRY = {
    'linear_sup': owa_suprimum_weights_linear,
    'linear_inf': owa_infimum_weights_linear,
}
