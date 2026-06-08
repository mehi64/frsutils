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
# build_similarity_engine              Build dense/blockwise similarity engines
# build_fuzzy_rough_model              Build registered fuzzy-rough models from config
# compute_approximations               Compute lower/upper/boundary/positive-region outputs
# compute_positive_region              Compute positive-region scores directly
# FuzzyRoughApproximationResult        Stable public approximation result object
# FuzzyRoughPositiveRegionScorer     sklearn-style fitted positive-region scorer
# get_fuzzy_rough_model_class          Resolve model classes by public alias
# list_fuzzy_rough_models              Inspect available fuzzy-rough model aliases
# list_similarities                    Inspect available similarity aliases
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
# result = compute_approximations(X, y, model="itfrs", similarity="linear")
# positive_region = result.positive_region
#
# sim = build_similarity_matrix(X, similarity="gaussian", similarity_sigma=0.5)
# positive_region = compute_positive_region(X=None, y=y, similarity_matrix=sim)
"""

from FRsutils.api.approximations import (
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)

from FRsutils.api.config import (
    apply_config_aliases,
    extract_prefixed_params,
    normalize_flat_config_to_nested,
)
from FRsutils.api.results import FuzzyRoughApproximationResult
from FRsutils.api.scoring import FuzzyRoughPositiveRegionScorer
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
    BaseSimilarityEngine,
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    Similarity,
    SimilarityBlock,
    build_similarity_engine,
    build_similarity_matrix,
    calculate_similarity_matrix,
    list_similarities,
)

__all__ = [
    "Similarity",
    "SimilarityBlock",
    "BaseSimilarityEngine",
    "BlockwiseSimilarityEngine",
    "DenseSimilarityEngine",
    "FuzzyRoughModel",
    "FuzzyRoughApproximationResult",
    "FuzzyRoughPositiveRegionScorer",
    "ITFRS",
    "OWAFRS",
    "VQRS",
    "apply_config_aliases",
    "build_fuzzy_rough_model",
    "build_similarity_engine",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
    "compute_approximations",
    "compute_boundary_region",
    "compute_lower_approximation",
    "compute_positive_region",
    "compute_upper_approximation",
    "extract_prefixed_params",
    "get_fuzzy_rough_model_class",
    "list_fuzzy_rough_models",
    "list_similarities",
    "normalize_flat_config_to_nested",
]
