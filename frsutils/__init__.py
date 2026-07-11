# SPDX-License-Identifier: BSD-3-Clause
"""Top-level public API for frsutils.

Import user-facing fuzzy-rough utilities directly from this package root. The
``frsutils.api`` remains available as a structured public submodule.
"""

from . import api as api
from .api import *  # noqa: F401,F403

__all__ = [*api.__all__, "api"]
