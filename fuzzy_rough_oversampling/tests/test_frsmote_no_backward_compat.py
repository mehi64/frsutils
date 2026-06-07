"""
@file test_frsmote_no_backward_compat.py
@brief Hard-break import policy tests for migrated FRSMOTE.

Phase 7 intentionally does not provide backward-compatibility wrappers from
FRsutils to the standalone `fuzzy_rough_oversampling` package. Old FRSMOTE
import paths should fail so stale scripts are migrated explicitly instead of
silently depending on a temporary shim.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "FRsutils.core.preprocess.oversampling.FRSMOTE",
        "FRsutils.core.preprocess.oversampling",
        "FRsutils.core.preprocess.FRSMOTE",
    ],
)
def test_old_frsmote_import_paths_are_not_supported(module_name: str) -> None:
    """@brief Verify that old FRsutils FRSMOTE import paths are hard-broken."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
