# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2 contract tests for public execution metadata and scorer engine params."""

import numpy as np
from sklearn.base import clone

from FRsutils.api import FuzzyRoughPositiveRegionScorer, compute_approximations


X_META = np.array(
    [
        [0.00, 0.00],
        [0.20, 0.10],
        [0.70, 0.80],
        [0.90, 0.75],
    ],
    dtype=float,
)
Y_META = np.array([0, 0, 1, 1])


def test_dense_result_exposes_execution_metadata():
    """@brief Dense results explicitly report CPU/dense provenance."""
    result = compute_approximations(X_META, Y_META, model="itfrs", similarity="linear")

    assert result.engine == "dense"
    assert result.backend == "numpy"
    assert result.block_size is None
    assert result.used_blockwise is False
    assert result.used_gpu_similarity_blocks is False

    result_dict = result.as_dict()
    assert result_dict["engine"] == "dense"
    assert result_dict["backend"] == "numpy"
    assert result_dict["block_size"] is None
    assert result_dict["used_blockwise"] is False
    assert result_dict["used_gpu_similarity_blocks"] is False


def test_blockwise_result_exposes_execution_metadata():
    """@brief Blockwise results preserve engine, backend, and block-size metadata."""
    dense = compute_approximations(X_META, Y_META, model="itfrs", similarity="linear")
    blockwise = compute_approximations(
        X_META,
        Y_META,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=2,
        backend="numpy",
    )

    np.testing.assert_allclose(blockwise.positive_region, dense.positive_region, atol=1e-12)
    assert blockwise.engine == "blockwise"
    assert blockwise.backend == "numpy"
    assert blockwise.block_size == 2
    assert blockwise.used_blockwise is True
    assert blockwise.used_gpu_similarity_blocks is False


def test_positive_region_scorer_forwards_engine_backend_and_block_size():
    """@brief Public scorer can opt into the blockwise backend-aware path."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="numpy",
        block_size=2,
    )

    scores = scorer.fit_score(X_META, Y_META)
    expected = compute_approximations(
        X_META,
        Y_META,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="numpy",
        block_size=2,
    )

    np.testing.assert_allclose(scores, expected.positive_region, atol=1e-12)
    assert scorer.result_.engine == "blockwise"
    assert scorer.result_.backend == "numpy"
    assert scorer.result_.block_size == 2


def test_positive_region_scorer_engine_params_are_sklearn_compatible():
    """@brief New scorer params survive sklearn get_params/set_params/clone."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="numpy",
        block_size=3,
    )

    params = scorer.get_params(deep=False)
    assert params["engine"] == "blockwise"
    assert params["backend"] == "numpy"
    assert params["block_size"] == 3

    cloned = clone(scorer)
    cloned_params = cloned.get_params(deep=False)
    assert cloned_params["engine"] == "blockwise"
    assert cloned_params["backend"] == "numpy"
    assert cloned_params["block_size"] == 3

    scorer.set_params(engine="dense", block_size=5)
    assert scorer.engine == "dense"
    assert scorer.block_size == 5
