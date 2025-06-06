"""
@file
@brief Factory function to build T-norm instances from string names.
"""

from FRsutils.core.tnorms import MinTNorm, ProductTNorm, LukasiewiczTNorm, TNorm
from FRsutils.utils.validation_utils import validate_tnorm_choice

def build_tnorm(name: str) -> TNorm:
    """
    @brief Instantiates a T-norm object based on its name.

    @param name Name of the T-norm ('minimum', 'product', 'lukasiewicz').

    @return An instance of a subclass of TNorm.

    @throws ValueError If the name is not recognized.
    """
    name = validate_tnorm_choice(name)

    if name == 'minimum':
        return MinTNorm()
    elif name == 'product': 
        return ProductTNorm()
    elif name == 'lukasiewicz':
        return LukasiewiczTNorm()
    else:
        raise ValueError(f"Unknown T-norm: {name}")
