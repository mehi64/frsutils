"""
@file KEEL_Audit_Utility.py
@brief Backward-compatible import shim for the canonical KEEL audit utility.

The canonical implementation lives in
`FRsutils.utils.dataset_utils.KEEL.KEEL_Audit_Utility`. This module preserves
older imports used by legacy tests/examples.
"""

from FRsutils.utils.dataset_utils.KEEL.KEEL_Audit_Utility import *  # noqa: F401,F403
