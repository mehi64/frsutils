"""
@file __init__.py
@brief Public API surface for downstream packages built on FRsutils.

This package exposes the small, stable subset of FRsutils that external packages
such as a future standalone `frsmote` package should depend on. The goal is to
avoid coupling downstream code to deep internal paths such as
`FRsutils.utils.constructor_utils` or `FRsutils.core.preprocess`.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# build_similarity_matrix              Build pairwise fuzzy similarity matrices
# build_fuzzy_rough_model              Build registered fuzzy-rough models from config
# get_fuzzy_rough_model_class          Resolve model classes by public alias
# list_fuzzy_rough_models              Inspect available fuzzy-rough model aliases
# normalize_flat_config_to_nested      Convert sklearn-friendly flat params to nested config
# apply_config_aliases                 Apply backwards-compatible flat config aliases
# extract_prefixed_params              Extract component-specific flat params

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: provides a small stable API over internal modules
# - Registry Pattern: exposes model lookup without exposing registry internals
# - Adapter Pattern: exposes flat -> nested config normalization
# - Dependency Inversion: downstream packages depend on this API, not internals
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api import build_similarity_matrix, build_fuzzy_rough_model
#
# sim = build_similarity_matrix(X, similarity="gaussian", similarity_sigma=0.5)
# model = build_fuzzy_rough_model(
#     "itfrs",
#     similarity_matrix=sim,
#     labels=y,
#     ub_tnorm_name="minimum",
#     lb_implicator_name="lukasiewicz",
# )
# positive_region = model.positive_region()
"""

from FRsutils.api.config import (
    apply_config_aliases,
    extract_prefixed_params,
    normalize_flat_config_to_nested,
)
from FRsutils.api.models import (
    FuzzyRoughModel,
    ITFRS,
    OWAFRS,
    VQRS,
    build_fuzzy_rough_model,
    get_fuzzy_rough_model_class,
    list_fuzzy_rough_models,
)
from FRsutils.api.similarity import (
    Similarity,
    build_similarity_matrix,
    calculate_similarity_matrix,
)

__all__ = [
    "Similarity",
    "FuzzyRoughModel",
    "ITFRS",
    "OWAFRS",
    "VQRS",
    "apply_config_aliases",
    "build_fuzzy_rough_model",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
    "extract_prefixed_params",
    "get_fuzzy_rough_model_class",
    "list_fuzzy_rough_models",
    "normalize_flat_config_to_nested",
]
