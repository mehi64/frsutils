# SPDX-License-Identifier: BSD-3-Clause
"""Repository contracts for core-module headers and docstrings."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


CORE_ROOT = Path(__file__).resolve().parents[2] / "frsutils" / "core"
CORE_MODULES = tuple(
    path
    for path in sorted(CORE_ROOT.rglob("*.py"))
    if "__pycache__" not in path.parts
)


@pytest.mark.parametrize("module_path", CORE_MODULES, ids=lambda path: str(path.relative_to(CORE_ROOT)))
def test_core_module_has_spdx_header_and_module_docstring(module_path: Path):
    """Require the project header and a module docstring in every core module."""
    source = module_path.read_text(encoding="utf-8")
    assert source.startswith("# SPDX-License-Identifier: BSD-3-Clause\n")
    assert ast.get_docstring(ast.parse(source), clean=False)


@pytest.mark.parametrize("module_path", CORE_MODULES, ids=lambda path: str(path.relative_to(CORE_ROOT)))
def test_core_definitions_have_docstrings(module_path: Path):
    """Require docstrings for every class and function defined in core code."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is None:
                missing.append(f"{node.name} at line {node.lineno}")

    assert not missing, f"Missing docstrings in {module_path}: {', '.join(missing)}"
