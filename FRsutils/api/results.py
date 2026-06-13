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
    """Immutable public result object for fuzzy-rough approximations.
    
    Parameters
    ----------
    lower : object
        Lower approximation score for each sample.
    upper : object
        Upper approximation score for each sample.
    boundary : object
        Boundary-region score for each sample, usually upper - lower.
    positive_region : object
        Positive-region score for each sample.
    model : object
        Public fuzzy-rough model alias used to compute the result.
    similarity : object
        Public similarity alias used to build the matrix, if known.
    similarity_matrix : object
        Optional pairwise similarity matrix.
    config : object
        Optional effective flat/nested configuration snapshot.
    engine : object
        Canonical execution engine used by compute_approximations.
    backend : object
        Canonical resolved backend used for similarity-block execution.
    block_size : object
        Block size used when engine="blockwise"; None for dense execution.
    used_blockwise : object
        True when the blockwise approximation path was used.
    used_gpu_similarity_blocks : object
        True when similarity blocks were computed through CuPy.
    used_gpu_approximation_accumulators : object
        True when approximation reductions used CuPy accumulators.
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
        """Convert this result into a plain dictionary.
                
                Parameters
                ----------
                include_similarity_matrix : bool
                    If True, include the optional similarity matrix.
                
                Returns
                -------
                Dict[str, Any]
                    Dictionary representation of the result.
                
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
