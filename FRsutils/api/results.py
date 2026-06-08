"""
@file results.py
@brief Public result containers for FRsutils approximation APIs.

This module defines stable, typed return objects for the public FRsutils API.
Returning named result objects instead of positional tuples keeps user code and
external downstream packages resilient to future API growth.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FuzzyRoughApproximationResult        Result object for lower/upper/boundary/positive region outputs
# execution metadata                   Store engine/backend/block-size provenance
# GPU accumulator metadata             Record when approximation accumulators used CuPy
# as_dict                              Convert result fields into a serializable dictionary

# ✅ Design Patterns & Clean Code Notes
# - Value Object Pattern: immutable dataclass groups approximation outputs
# - Facade Support: public APIs return this stable object instead of internal models
# - Clean API: avoids fragile tuple-order dependencies for downstream packages
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api import compute_approximations
#
# result = compute_approximations(X, y, model="itfrs")
# result.lower
# result.upper
# result.positive_region
# result.engine
# result.backend
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FuzzyRoughApproximationResult:
    """
    @brief Immutable public result object for fuzzy-rough approximations.

    @param lower: Lower approximation score for each sample.
    @param upper: Upper approximation score for each sample.
    @param boundary: Boundary-region score for each sample, usually upper - lower.
    @param positive_region: Positive-region score for each sample.
    @param model: Public fuzzy-rough model alias used to compute the result.
    @param similarity: Public similarity alias used to build the matrix, if known.
    @param similarity_matrix: Optional pairwise similarity matrix.
    @param config: Optional effective flat/nested configuration snapshot.
    @param engine: Canonical execution engine used by compute_approximations.
    @param backend: Canonical resolved backend used for similarity-block execution.
    @param block_size: Block size used when engine="blockwise"; None for dense execution.
    @param used_blockwise: True when the blockwise approximation path was used.
    @param used_gpu_similarity_blocks: True when similarity blocks were computed through CuPy.
    @param used_gpu_approximation_accumulators: True when approximation reductions used CuPy accumulators.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    model: str
    similarity: Optional[str] = None
    similarity_matrix: Optional[np.ndarray] = None
    config: Optional[Dict[str, Any]] = None
    engine: str = "dense"
    backend: str = "numpy"
    block_size: Optional[int] = None
    used_blockwise: bool = False
    used_gpu_similarity_blocks: bool = False
    used_gpu_approximation_accumulators: bool = False

    def as_dict(self, *, include_similarity_matrix: bool = False) -> Dict[str, Any]:
        """
        @brief Convert this result into a plain dictionary.

        @param include_similarity_matrix: If True, include the optional similarity matrix.
        @return: Dictionary representation of the result.
        """
        data: Dict[str, Any] = {
            "lower": self.lower,
            "upper": self.upper,
            "boundary": self.boundary,
            "positive_region": self.positive_region,
            "model": self.model,
            "similarity": self.similarity,
            "config": self.config,
            "engine": self.engine,
            "backend": self.backend,
            "block_size": self.block_size,
            "used_blockwise": self.used_blockwise,
            "used_gpu_similarity_blocks": self.used_gpu_similarity_blocks,
            "used_gpu_approximation_accumulators": self.used_gpu_approximation_accumulators,
        }
        if include_similarity_matrix:
            data["similarity_matrix"] = self.similarity_matrix
        return data


__all__ = ["FuzzyRoughApproximationResult"]
