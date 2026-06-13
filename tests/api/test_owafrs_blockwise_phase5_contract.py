# SPDX-License-Identifier: BSD-3-Clause
"""Phase 5 contract tests for exact blockwise OWAFRS approximation."""

import numpy as np
import pytest

from FRsutils.api import build_similarity_matrix, compute_approximations, compute_positive_region


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
def test_blockwise_owafrs_matches_dense_owafrs_for_multiple_block_sizes(block_size):
    """@brief Exact blockwise OWAFRS must match dense OWAFRS for all result fields."""
    dense = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="owafrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        block_size=block_size,
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)
    assert blockwise.model == "owafrs"
    assert blockwise.similarity == "linear"
    assert blockwise.similarity_matrix is None


def test_blockwise_owafrs_matches_dense_with_custom_components():
    """@brief Blockwise OWAFRS must reuse T-norm, implicator, and OWA settings."""
    kwargs = dict(
        model="owafrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
        lb_owa_method_name="harmonic",
        ub_owa_method_name="exponential",
        ub_owa_method_base=1.25,
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


def test_blockwise_owafrs_can_materialize_similarity_matrix_when_requested():
    """@brief return_similarity_matrix=True remains available for debugging/contract checks."""
    expected_matrix = build_similarity_matrix(X_BLOCKWISE, similarity="linear")
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
    )

    np.testing.assert_allclose(blockwise.similarity_matrix, expected_matrix, atol=1e-12)


def test_compute_positive_region_wrapper_supports_blockwise_owafrs():
    """@brief Convenience wrappers pass the OWAFRS blockwise execution options through."""
    dense_scores = compute_positive_region(X_BLOCKWISE, Y_BLOCKWISE, model="owafrs", similarity="linear")
    blockwise_scores = compute_positive_region(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
    )

    np.testing.assert_allclose(blockwise_scores, dense_scores, atol=1e-12)
