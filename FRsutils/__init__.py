# SPDX-License-Identifier: BSD-3-Clause
"""Top-level package namespace for FRsutils.

The canonical user-facing API is exposed through ``FRsutils.api`` to keep the
package root compact and avoid leaking internal implementation modules.
"""

from . import api
from .core import tnorms, implicators, similarities
from .core.models import itfrs

__all__ = ["api", "tnorms", "implicators", "similarities", "itfrs"]
