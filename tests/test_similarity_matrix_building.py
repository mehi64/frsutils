# SPDX-License-Identifier: BSD-3-Clause
"""Tests for similarity-matrix construction helpers."""

import numpy as np

from frsutils.core.similarities import build_similarity_matrix
from frsutils.utils.init_helpers import normalize_flat_config_to_nested


def test_build_similarity_matrix_from_flat_params():
    X = np.array(
        [
            [0.0, 0.2],
            [0.1, 0.3],
            [0.9, 0.8],
        ],
        dtype=float,
    )

    sim = build_similarity_matrix(
        X,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="minimum",
    )

    assert sim.shape == (3, 3)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.all(sim >= 0.0) and np.all(sim <= 1.0)


def test_build_similarity_matrix_from_nested_config():
    X = np.random.RandomState(0).rand(5, 3)
    flat = {
        "similarity": "gaussian",
        "similarity_sigma": 0.2,
        "similarity_tnorm": "minimum",
    }
    nested = normalize_flat_config_to_nested(flat)

    sim = build_similarity_matrix(X, config=nested)

    assert sim.shape == (5, 5)
    assert np.allclose(np.diag(sim), 1.0)
