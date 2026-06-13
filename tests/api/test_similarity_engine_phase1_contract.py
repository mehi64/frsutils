# SPDX-License-Identifier: BSD-3-Clause
"""Phase 1 contract tests for the similarity-engine abstraction."""

import numpy as np
import pytest

from FRsutils.api import (
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    build_similarity_engine,
    build_similarity_matrix,
    normalize_flat_config_to_nested,
)
from FRsutils.core.similarity_engine import calculate_similarity_block
from FRsutils.core.similarities import Similarity
from FRsutils.core.tnorms import TNorm


X_ENGINE = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.7],
        [0.9, 1.0],
        [0.4, 0.5],
    ],
    dtype=float,
)


@pytest.mark.parametrize(
    "flat_config",
    [
        {"similarity": "linear", "similarity_tnorm": "minimum"},
        {"similarity": "gaussian", "similarity_sigma": 0.5, "similarity_tnorm": "minimum"},
    ],
)
def test_dense_similarity_engine_matches_existing_dense_builder(flat_config):
    """DenseSimilarityEngine must be a compatibility wrapper for the current dense path."""
    expected = build_similarity_matrix(X_ENGINE, **flat_config)
    engine = build_similarity_engine(X_ENGINE, engine="dense", **flat_config)

    assert isinstance(engine, DenseSimilarityEngine)
    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)

    blocks = list(engine.iter_blocks())
    assert len(blocks) == 1
    assert blocks[0].row_slice == slice(0, X_ENGINE.shape[0])
    assert blocks[0].col_slice == slice(0, X_ENGINE.shape[0])
    np.testing.assert_allclose(blocks[0].values, expected, atol=1e-12)


@pytest.mark.parametrize("block_size", [1, 2, 3, 10])
def test_blockwise_similarity_engine_materializes_existing_dense_matrix(block_size):
    """Blockwise materialization must match the legacy exact dense matrix."""
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="linear",
        similarity_tnorm="minimum",
    )
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=block_size,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    assert isinstance(engine, BlockwiseSimilarityEngine)
    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)



def test_blockwise_similarity_engine_exposes_expected_block_slices_and_shapes():
    """Block iterator exposes reusable row/column slices for future accumulators."""
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    blocks = list(engine.iter_blocks())
    assert len(blocks) == 9

    expected_shapes = [
        (2, 2),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 2),
        (2, 1),
        (1, 2),
        (1, 2),
        (1, 1),
    ]
    assert [block.values.shape for block in blocks] == expected_shapes

    # Reconstruct manually from blocks to prove the slices are directly usable.
    reconstructed = np.zeros((X_ENGINE.shape[0], X_ENGINE.shape[0]), dtype=float)
    for block in blocks:
        reconstructed[block.row_slice, block.col_slice] = block.values

    expected = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)



def test_similarity_engine_nested_config_matches_flat_config():
    """Engine construction must preserve the flat/nested config equivalence contract."""
    flat_config = {
        "type": "itfrs",
        "similarity": "linear",
        "similarity_tnorm": "minimum",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    }
    nested_config = normalize_flat_config_to_nested(flat_config)

    flat_engine = build_similarity_engine(X_ENGINE, engine="blockwise", block_size=2, **flat_config)
    nested_engine = build_similarity_engine(X_ENGINE, engine="blockwise", block_size=2, config=nested_config)

    np.testing.assert_allclose(nested_engine.to_dense(), flat_engine.to_dense(), atol=1e-12)



def test_calculate_similarity_block_matches_corresponding_dense_submatrix():
    """Low-level block helper must match the same slice from the dense matrix."""
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    block = calculate_similarity_block(
        X_ENGINE[0:2],
        X_ENGINE[2:5],
        similarity_func,
        tnorm_func,
    )
    dense = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")

    np.testing.assert_allclose(block, dense[0:2, 2:5], atol=1e-12)



def test_similarity_engine_does_not_mutate_input_matrix():
    """Engine construction/materialization must not modify the caller's feature matrix."""
    X = X_ENGINE.copy()
    before = X.copy()

    engine = build_similarity_engine(X, engine="blockwise", block_size=2, similarity="linear")
    _ = engine.to_dense()

    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("engine_name", ["unknown", "", 123])
def test_invalid_similarity_engine_name_is_rejected(engine_name):
    """Engine aliases are validated at the public construction boundary."""
    expected_error = TypeError if not isinstance(engine_name, str) or not str(engine_name).strip() else ValueError
    with pytest.raises(expected_error):
        build_similarity_engine(X_ENGINE, engine=engine_name, similarity="linear")


@pytest.mark.parametrize("block_size, expected_error", [(0, ValueError), (-1, ValueError), (1.5, TypeError)])
def test_invalid_block_size_is_rejected_for_blockwise_engine(block_size, expected_error):
    """Blockwise engine rejects invalid block sizes before computation."""
    with pytest.raises(expected_error):
        build_similarity_engine(
            X_ENGINE,
            engine="blockwise",
            block_size=block_size,
            similarity="linear",
        )


def test_similarity_engine_rejects_unknown_backend_alias():
    """Backend aliases remain validated at the public construction boundary."""
    with pytest.raises(ValueError):
        build_similarity_engine(X_ENGINE, engine="blockwise", backend="unknown", similarity="linear")
