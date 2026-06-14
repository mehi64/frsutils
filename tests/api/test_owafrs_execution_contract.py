# SPDX-License-Identifier: BSD-3-Clause
"""Execution-contract tests for OWAFRS dense and blockwise paths."""

import numpy as np

from FRsutils.api import build_similarity_matrix, compute_approximations
from FRsutils.core.implicators import LukasiewiczImplicator
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.owa_weights import OWAWeights
from FRsutils.core.tnorms import MinTNorm
from tests._fake_cupy_backend import FakeCupyArray, install_fake_cupy_module


X_OWAFRS_CONTRACT = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.75],
        [0.9, 0.85],
    ],
    dtype=float,
)
Y_OWAFRS_CONTRACT = np.array([0, 0, 1, 1])


def _linear_owa():
    """Build the default linear OWA strategy used in public OWAFRS defaults."""
    return OWAWeights.create("linear")


def test_direct_owafrs_is_dense_numpy_reference_model():
    """Direct OWAFRS consumes a dense matrix and returns NumPy approximation arrays."""
    similarity_matrix = build_similarity_matrix(X_OWAFRS_CONTRACT, similarity="linear")
    model = OWAFRS(
        similarity_matrix,
        Y_OWAFRS_CONTRACT,
        ub_tnorm=MinTNorm(),
        lb_implicator=LukasiewiczImplicator(),
        ub_owa_method=_linear_owa(),
        lb_owa_method=_linear_owa(),
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


def test_public_dense_owafrs_contract_uses_numpy_result_boundary():
    """Dense public OWAFRS exposes NumPy arrays and no GPU/blockwise metadata."""
    result = compute_approximations(
        X_OWAFRS_CONTRACT,
        Y_OWAFRS_CONTRACT,
        model="owafrs",
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


def test_public_blockwise_owafrs_matches_dense_without_materializing_similarity():
    """Blockwise OWAFRS is the public scalable path and keeps matrix output optional."""
    dense = compute_approximations(
        X_OWAFRS_CONTRACT,
        Y_OWAFRS_CONTRACT,
        model="owafrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_OWAFRS_CONTRACT,
        Y_OWAFRS_CONTRACT,
        model="owafrs",
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


def test_public_cupy_blockwise_owafrs_uses_gpu_similarity_blocks_only(monkeypatch):
    """CuPy-backed OWAFRS reports GPU similarity blocks but NumPy aggregation."""
    install_fake_cupy_module(monkeypatch)

    dense = compute_approximations(
        X_OWAFRS_CONTRACT,
        Y_OWAFRS_CONTRACT,
        model="owafrs",
        similarity="linear",
        engine="dense",
    )
    result = compute_approximations(
        X_OWAFRS_CONTRACT,
        Y_OWAFRS_CONTRACT,
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert result.engine == "blockwise"
    assert result.backend == "cupy"
    assert result.used_blockwise is True
    assert result.used_gpu_similarity_blocks is True
    assert result.used_gpu_approximation_accumulators is False
    assert isinstance(result.lower, np.ndarray)
    assert isinstance(result.upper, np.ndarray)
    assert isinstance(result.boundary, np.ndarray)
    assert isinstance(result.positive_region, np.ndarray)
    assert not isinstance(result.lower, FakeCupyArray)
    assert not isinstance(result.upper, FakeCupyArray)
    assert not isinstance(result.boundary, FakeCupyArray)
    assert not isinstance(result.positive_region, FakeCupyArray)
    np.testing.assert_allclose(result.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, dense.positive_region, atol=1e-12)
