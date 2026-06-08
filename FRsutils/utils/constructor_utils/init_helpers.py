"""
@file init_helpers.py
@brief Backward-compatible import shim for configuration normalization helpers.

Older tests/examples imported init helpers from
`FRsutils.utils.constructor_utils.init_helpers`, while the canonical module is
`FRsutils.utils.init_helpers`. This shim keeps those imports working without
moving the canonical implementation.
"""

from FRsutils.utils.init_helpers import *  # noqa: F401,F403
