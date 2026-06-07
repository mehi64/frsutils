"""
@file test_package_import.py
@brief Smoke tests for the standalone fuzzy-rough oversampling package.
"""

from fuzzy_rough_oversampling import FRSMOTE, __version__, list_oversamplers


def test_package_imports() -> None:
    """@brief Verify that the package-level public API imports successfully."""
    assert isinstance(__version__, str)
    assert FRSMOTE.__name__ == "FRSMOTE"
    assert "frsmote" in list_oversamplers()
