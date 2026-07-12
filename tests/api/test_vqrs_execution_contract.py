# SPDX-License-Identifier: BSD-3-Clause
"""Execution-contract tests for VQRS dense and blockwise paths."""

import sys
import types

import numpy as np

from frsutils import build_similarity_matrix, compute_approximations
from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from frsutils.core.models.vqrs import VQRS


class _FakeCupy(types.SimpleNamespace):
    """Small NumPy-backed stand-in for CuPy contract tests."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(value):
        """Return a NumPy representation of a fake backend array."""
        return np.asarray(value)


def _install_fake_cupy(monkeypatch):
    """Install a fake CuPy module for one test."""
    fake_cupy = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    return fake_cupy


X_VQRS_CONTRACT = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.75],
        [0.9, 0.85],
    ],
    dtype=float,
)
Y_VQRS_CONTRACT = np.array([0, 0, 1, 1])


def _lower_quantifier():
    """Build the default lower quantifier used in VQRS public defaults."""
    return FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)


def _upper_quantifier():
    """Build the default upper quantifier used in VQRS public defaults."""
    return FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)


def test_direct_vqrs_is_dense_numpy_reference_model():
    """Direct VQRS consumes a dense matrix and returns NumPy approximation arrays."""
    similarity_matrix = build_similarity_matrix(X_VQRS_CONTRACT, similarity="linear")
    model = VQRS(
        similarity_matrix,
        Y_VQRS_CONTRACT,
        lb_fuzzy_quantifier=_lower_quantifier(),
        ub_fuzzy_quantifier=_upper_quantifier(),
    )

    lower = model.lower_approximation()
    upper = model.upper_approximation()
    boundary = model.boundary_region()
    positive_region = model.positive_region()

    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert isinstance(boundary, np.ndarray)
    assert isinstance(positive_region, np.ndarray)
    np.testing.assert_allclose(boundary, upper - lower, atol=1e-12)
    np.testing.assert_allclose(positive_region, lower, atol=1e-12)


def test_public_dense_vqrs_contract_uses_numpy_result_boundary():
    """Dense public VQRS exposes NumPy arrays and no GPU/blockwise metadata."""
    result = compute_approximations(
        X_VQRS_CONTRACT,
        Y_VQRS_CONTRACT,
        model="vqrs",
        similarity="linear",
        engine="dense",
    )

    assert result.engine == "dense"
    assert result.backend == "numpy"
    assert result.used_blockwise is False
    assert result.used_gpu_similarity_blocks is False
    assert result.used_gpu_approximation_accumulators is False
    assert isinstance(result.lower, np.ndarray)
    assert isinstance(result.upper, np.ndarray)
    assert isinstance(result.boundary, np.ndarray)
    assert isinstance(result.positive_region, np.ndarray)


def test_public_blockwise_vqrs_matches_dense_without_materializing_similarity():
    """Blockwise VQRS is the public scalable path and keeps matrix output optional."""
    dense = compute_approximations(
        X_VQRS_CONTRACT,
        Y_VQRS_CONTRACT,
        model="vqrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_VQRS_CONTRACT,
        Y_VQRS_CONTRACT,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
    )

    assert blockwise.engine == "blockwise"
    assert blockwise.backend == "numpy"
    assert blockwise.used_blockwise is True
    assert blockwise.similarity_matrix is None
    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)


def test_public_cupy_blockwise_vqrs_keeps_numpy_outputs_at_api_boundary(monkeypatch):
    """CuPy-backed blockwise VQRS reports GPU internals but returns NumPy arrays."""
    _install_fake_cupy(monkeypatch)

    result = compute_approximations(
        X_VQRS_CONTRACT,
        Y_VQRS_CONTRACT,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert result.engine == "blockwise"
    assert result.backend == "cupy"
    assert result.used_blockwise is True
    assert result.used_gpu_similarity_blocks is True
    assert result.used_gpu_approximation_accumulators is True
    assert isinstance(result.lower, np.ndarray)
    assert isinstance(result.upper, np.ndarray)
    assert isinstance(result.boundary, np.ndarray)
    assert isinstance(result.positive_region, np.ndarray)
