# SPDX-License-Identifier: BSD-3-Clause
"""Public API entry points for FRsutils."""

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
