# SPDX-License-Identifier: BSD-3-Clause
"""Structured public API submodule for frsutils.

The package root is the canonical user-facing import surface. This submodule
contains the same public objects in a grouped namespace for maintainers and
advanced users who prefer an explicit facade module.
"""

from .approximations import (
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_signed_boundary,
    compute_upper_approximation,
)

from .results import FuzzyRoughApproximationResult
from .scoring import FuzzyRoughPositiveRegionScorer
from .models import (
    FuzzyRoughModel,
    ITFRS,
    OWAFRS,
    VQRS,
    build_fuzzy_rough_model,
    get_fuzzy_rough_model_class,
    list_fuzzy_rough_models,
)
from .similarity import (
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
    "build_fuzzy_rough_model",
    "build_similarity_engine",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
    "compute_approximations",
    "compute_boundary_region",
    "compute_lower_approximation",
    "compute_positive_region",
    "compute_signed_boundary",
    "compute_upper_approximation",
    "get_fuzzy_rough_model_class",
    "list_fuzzy_rough_models",
    "list_similarities",
]
