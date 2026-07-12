# SPDX-License-Identifier: BSD-3-Clause
"""Execution-contract tests for ITFRS dense and blockwise paths."""

import sys
import types

import numpy as np

from frsutils import build_similarity_matrix, compute_approximations
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.tnorms import MinTNorm
from frsutils.core.implicators import LukasiewiczImplicator


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


X_ITFRS_CONTRACT = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.75],
        [0.9, 0.85],
    ],
    dtype=float,
)
Y_ITFRS_CONTRACT = np.array([0, 0, 1, 1])


def test_direct_itfrs_is_dense_numpy_reference_model():
    """Direct ITFRS consumes a dense matrix and returns NumPy approximation arrays."""
    similarity_matrix = build_similarity_matrix(X_ITFRS_CONTRACT, similarity="linear")
    model = ITFRS(
        similarity_matrix,
        Y_ITFRS_CONTRACT,
        ub_tnorm=MinTNorm(),
        lb_implicator=LukasiewiczImplicator(),
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


def test_public_dense_itfrs_contract_uses_numpy_result_boundary():
    """Dense public ITFRS exposes NumPy arrays and no GPU/blockwise metadata."""
    result = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
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


def test_public_blockwise_itfrs_matches_dense_without_materializing_similarity():
    """Blockwise ITFRS is the public scalable path and keeps matrix output optional."""
    dense = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
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


def test_public_cupy_blockwise_itfrs_keeps_numpy_outputs_at_api_boundary(monkeypatch):
    """CuPy-backed blockwise ITFRS reports GPU internals but returns NumPy arrays."""
    _install_fake_cupy(monkeypatch)

    result = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
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


def test_public_dense_and_blockwise_itfrs_share_component_construction():
    """Dense public ITFRS and blockwise ITFRS should resolve operators identically."""
    dense = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
        similarity="linear",
        engine="dense",
        ub_tnorm_name="yager",
        ub_tnorm_p=0.83,
        lb_implicator_name="lukasiewicz",
    )
    blockwise = compute_approximations(
        X_ITFRS_CONTRACT,
        Y_ITFRS_CONTRACT,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        ub_tnorm_name="yager",
        ub_tnorm_p=0.83,
        lb_implicator_name="lukasiewicz",
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)
