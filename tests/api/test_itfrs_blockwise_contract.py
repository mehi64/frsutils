# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for exact blockwise ITFRS approximation."""

import numpy as np
import pytest

from frsutils import build_similarity_matrix, compute_approximations, compute_positive_region


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
def test_blockwise_itfrs_matches_dense_itfrs_for_multiple_block_sizes(block_size):
    """Exact blockwise ITFRS must match dense ITFRS for all accumulator fields."""
    dense = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=block_size,
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)
    assert blockwise.model == "itfrs"
    assert blockwise.similarity == "linear"
    assert blockwise.similarity_matrix is None


def test_blockwise_itfrs_matches_dense_with_gaussian_similarity():
    """Blockwise ITFRS must reuse the same similarity params as dense construction."""
    dense = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
    )
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
        engine="blockwise",
        block_size=2,
    )

    np.testing.assert_allclose(blockwise.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)


def test_blockwise_itfrs_can_materialize_similarity_matrix_when_requested():
    """return_similarity_matrix=True remains available for debugging/contract checks."""
    expected_matrix = build_similarity_matrix(X_BLOCKWISE, similarity="linear")
    blockwise = compute_approximations(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
    )

    np.testing.assert_allclose(blockwise.similarity_matrix, expected_matrix, atol=1e-12)


def test_compute_positive_region_wrapper_supports_blockwise_itfrs():
    """Convenience wrappers pass the blockwise execution options through."""
    dense_scores = compute_positive_region(X_BLOCKWISE, Y_BLOCKWISE, model="itfrs", similarity="linear")
    blockwise_scores = compute_positive_region(
        X_BLOCKWISE,
        Y_BLOCKWISE,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
    )

    np.testing.assert_allclose(blockwise_scores, dense_scores, atol=1e-12)


def test_blockwise_now_supports_owafrs_after_row_buffer_accumulator_was_added():
    """Exact blockwise support replaces the old OWAFRS guardrail."""
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
        block_size=2,
    )

    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)


def test_blockwise_requires_x_not_precomputed_similarity_matrix():
    """Blockwise execution generates blocks from X and rejects precomputed matrices."""
    sim = build_similarity_matrix(X_BLOCKWISE, similarity="linear")

    with pytest.raises(ValueError, match="does not accept precomputed similarity_matrix"):
        compute_approximations(
            X=None,
            y=Y_BLOCKWISE,
            model="itfrs",
            similarity_matrix=sim,
            engine="blockwise",
        )
