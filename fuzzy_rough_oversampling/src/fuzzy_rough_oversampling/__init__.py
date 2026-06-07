"""
@file __init__.py
@brief Public package API for fuzzy-rough oversampling algorithms.

This package is the standalone oversampling layer built on top of FRsutils. It
hosts FRSMOTE and is prepared for future fuzzy-rough versions of common
oversampling algorithms such as FRADASYN.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FRSMOTE                              Direct import for fuzzy-rough SMOTE
# build_oversampler                    Build a registered oversampler by public name
# get_oversampler_class                Resolve a registered oversampler class
# list_oversamplers                    Inspect registered oversampler aliases
# register_oversampler                 Register future oversampler implementations

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: exposes the package-level public API
# - Registry Pattern: decouples algorithm lookup from concrete implementations
# - Factory Method: build_oversampler constructs registered algorithms
# - Dependency Inversion: algorithms consume FRsutils public API, not internals
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from fuzzy_rough_oversampling import FRSMOTE, build_oversampler
#
# sampler = FRSMOTE(random_state=42)
# sampler2 = build_oversampler("frsmote", random_state=42)
"""

from fuzzy_rough_oversampling._version import __version__
from fuzzy_rough_oversampling.algorithms import FRSMOTE
from fuzzy_rough_oversampling.registry import (
    build_oversampler,
    get_oversampler_class,
    list_oversamplers,
    register_oversampler,
)

__all__ = [
    "__version__",
    "FRSMOTE",
    "build_oversampler",
    "get_oversampler_class",
    "list_oversamplers",
    "register_oversampler",
]
