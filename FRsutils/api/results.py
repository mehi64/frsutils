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
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    model: str
    similarity: Optional[str] = None
    similarity_matrix: Optional[np.ndarray] = None
    config: Optional[Dict[str, Any]] = None

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
        }
        if include_similarity_matrix:
            data["similarity_matrix"] = self.similarity_matrix
        return data


__all__ = ["FuzzyRoughApproximationResult"]
