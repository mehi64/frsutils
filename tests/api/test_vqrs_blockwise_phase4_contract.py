# SPDX-License-Identifier: BSD-3-Clause
"""Phase 4 contract tests for exact blockwise VQRS approximation."""

import sys
import types

import numpy as np
import pytest

from FRsutils.api import build_similarity_matrix, compute_approximations, compute_positive_region


class _FakeCupy(types.SimpleNamespace):
    """@brief Minimal NumPy-backed CuPy stand-in for VQRS contract tests."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(value):
        """@brief Mirror cupy.asnumpy with NumPy conversion."""
        return np.asarray(value)


def _install_fake_cupy(monkeypatch):
    """@brief Install a fake CuPy module for CPU-only CI."""
    fake_cupy = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    return fake_cupy


X_BLOCKWISE = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.75, 0.8],
        [0.85, 0.75],
        [0.95, 0.9],
    ],
    dtype=float,
)
Y_BLOCKWISE = np.array([0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("block_size", [1, 2, 4, 20])
def test_blockwise_vqrs_matches_dense_vqrs_for_multiple_block_sizes(block_size):
    """@brief Exact blockwise VQRS must match dense VQRS for all result fields."""
    dense = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        block_size=block_size,
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)
    assert blockwise.model == "vqrs"
    assert blockwise.similarity == "linear"
    assert blockwise.similarity_matrix is None


def test_blockwise_vqrs_matches_dense_with_custom_quantifier_parameters():
    """@brief Blockwise VQRS must reuse lower/upper fuzzy-quantifier settings."""
    kwargs = dict(
        model="vqrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
        lb_fuzzy_quantifier_name="linear",
        lb_fuzzy_quantifier_alpha=0.0,
        lb_fuzzy_quantifier_beta=0.7,
        ub_fuzzy_quantifier_name="quadratic",
        ub_fuzzy_quantifier_alpha=0.0,
        ub_fuzzy_quantifier_beta=0.9,
    )
    dense = compute_approximations(X_BLOCKWISE, Y_BLOCKWISE, **kwargs)
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        **kwargs,
        engine="blockwise",
        block_size=2,
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)


def test_blockwise_vqrs_can_materialize_similarity_matrix_when_requested():
    """@brief return_similarity_matrix=True remains available for debugging/contract checks."""
    expected_matrix = build_similarity_matrix(X_BLOCKWISE, similarity="linear")
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
    )

    np.testing.assert_allclose(blockwise.similarity_matrix, expected_matrix, atol=1e-12)


def test_compute_positive_region_wrapper_supports_blockwise_vqrs():
    """@brief Convenience wrappers pass the VQRS blockwise execution options through."""
    dense_scores = compute_positive_region(X_BLOCKWISE, Y_BLOCKWISE, model="vqrs", similarity="linear")
    blockwise_scores = compute_positive_region(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
    )

    np.testing.assert_allclose(blockwise_scores, dense_scores, atol=1e-12)


def test_blockwise_vqrs_fake_cupy_path_uses_gpu_resident_accumulator_metadata(monkeypatch):
    """@brief VQRS marks CuPy similarity blocks and sum accumulators as GPU-backed."""
    _install_fake_cupy(monkeypatch)

    dense = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="dense",
    )
    gpu_like = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert gpu_like.backend == "cupy"
    assert gpu_like.used_blockwise is True
    assert gpu_like.used_gpu_similarity_blocks is True
    assert gpu_like.used_gpu_approximation_accumulators is True
    assert gpu_like.as_dict()["used_gpu_approximation_accumulators"] is True
    np.testing.assert_allclose(gpu_like.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(gpu_like.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(gpu_like.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(gpu_like.positive_region, dense.positive_region, atol=1e-12)


def test_blockwise_vqrs_fake_cupy_path_supports_custom_quantifier_parameters(monkeypatch):
    """@brief VQRS keeps custom fuzzy-quantifier formulas equivalent on fake CuPy."""
    _install_fake_cupy(monkeypatch)

    kwargs = dict(
        model="vqrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
        lb_fuzzy_quantifier_name="linear",
        lb_fuzzy_quantifier_alpha=0.0,
        lb_fuzzy_quantifier_beta=0.7,
        ub_fuzzy_quantifier_name="quadratic",
        ub_fuzzy_quantifier_alpha=0.0,
        ub_fuzzy_quantifier_beta=0.9,
    )
    dense = compute_approximations(X_BLOCKWISE, Y_BLOCKWISE, **kwargs)
    gpu_like = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        **kwargs,
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert gpu_like.used_gpu_approximation_accumulators is True
    np.testing.assert_allclose(gpu_like.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(gpu_like.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(gpu_like.positive_region, dense.positive_region, atol=1e-12)
