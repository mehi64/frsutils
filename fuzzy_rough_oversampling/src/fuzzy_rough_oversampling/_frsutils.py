"""
@file _frsutils.py
@brief Internal bridge to the public FRsutils API.

All fuzzy-rough oversampling algorithms in this package should import FRsutils
functionality from this module instead of importing deep FRsutils internals. This
keeps the downstream package coupled only to the stable `FRsutils.api` facade.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# build_similarity_matrix              Public FRsutils similarity-matrix builder
# build_fuzzy_rough_model              Public FRsutils fuzzy-rough model factory
# get_fuzzy_rough_model_class          Public FRsutils fuzzy-rough model lookup
# normalize_flat_config_to_nested      Public FRsutils config normalization helper
# FuzzyRoughModel                      Public FRsutils fuzzy-rough model registry

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: centralizes FRsutils imports for this package
# - Dependency Inversion: algorithms depend on public FRsutils API only
# - Anti-Corruption Layer: shields algorithms from FRsutils internal path changes
##############################################
"""

from FRsutils.api import (
    FuzzyRoughModel,
    build_fuzzy_rough_model,
    build_similarity_matrix,
    get_fuzzy_rough_model_class,
    normalize_flat_config_to_nested,
)

__all__ = [
    "FuzzyRoughModel",
    "build_fuzzy_rough_model",
    "build_similarity_matrix",
    "get_fuzzy_rough_model_class",
    "normalize_flat_config_to_nested",
]
