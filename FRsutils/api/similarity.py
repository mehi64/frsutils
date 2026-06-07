"""
@file similarity.py
@brief Public similarity API for downstream FRsutils consumers.

This module exposes the stable similarity-matrix entry point that external
packages should use. It intentionally hides lower-level implementation details
and keeps downstream packages independent from internal module organization.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# Similarity                           Public registry base for similarity functions
# calculate_similarity_matrix          Low-level pairwise matrix calculation helper
# build_similarity_matrix              Config-driven similarity matrix builder

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: stable import path for public similarity operations
# - Strategy Pattern: similarity functions and t-norms remain pluggable internally
# - Registry Pattern: similarity aliases are still resolved by the core registry
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.api.similarity import build_similarity_matrix
#
# sim = build_similarity_matrix(
#     X,
#     similarity="gaussian",
#     similarity_sigma=0.5,
#     similarity_tnorm="minimum",
# )
"""

from FRsutils.core.similarities import (
    Similarity,
    build_similarity_matrix,
    calculate_similarity_matrix,
)

__all__ = [
    "Similarity",
    "build_similarity_matrix",
    "calculate_similarity_matrix",
]
