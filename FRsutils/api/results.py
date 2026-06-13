# SPDX-License-Identifier: BSD-3-Clause
"""Result containers returned by FRsutils public APIs.

This module belongs to the stable public API layer.
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
