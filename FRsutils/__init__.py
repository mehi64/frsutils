"""
@file __init__.py
@brief Top-level package exports for FRsutils.

This module keeps the historical top-level exports and exposes the new `api`
facade package. Downstream packages should prefer `FRsutils.api` for stable
imports instead of depending on deep internal paths.
"""

from . import api
from .core import tnorms, implicators, similarities
from .core.models import itfrs

__all__ = ["api", "tnorms", "implicators", "similarities", "itfrs"]
