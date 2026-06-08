"""
@file test_dense_similarity_baseline_contract.py
@brief Phase 0 baseline tests for the current dense similarity-matrix contract.

These tests intentionally protect the existing dense similarity behavior before
introducing blockwise, GPU, or approximate similarity engines. Future engines can
use this file as the reference contract for small exact computations.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# test_dense_linear_similarity...      Locks exact linear/minimum dense output
# test_dense_gaussian_similarity...    Locks exact Gaussian dense output
# test_dense_similarity_nested...      Ensures flat and nested config stay equivalent
# test_dense_similarity_does_not...    Guards against accidental input mutation

# ✅ Design Patterns & Clean Code Notes
# - Contract Testing: freezes public dense behavior before engine refactoring
# - Regression Testing: future blockwise/GPU paths can compare against these cases
# - Boundary Testing: uses FRsutils.api instead of deep internal imports
##############################################
"""

import numpy as np

from FRsutils.api import build_similarity_matrix, normalize_flat_config_to_nested


X_ONE_FEATURE = np.array([[0.0], [0.1], [0.8], [0.9]], dtype=float)
X_TWO_FEATURES = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.8, 0.7],
        [0.9, 1.0],
    ],
    dtype=float,
)


EXPECTED_LINEAR_ONE_FEATURE = np.array(
    [
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.9],
        [0.1, 0.2, 0.9, 1.0],
    ],
    dtype=float,
)


EXPECTED_LINEAR_TWO_FEATURES_MINIMUM = np.array(
    [
        [1.0, 0.8, 0.2, 0.0],
        [0.8, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.7],
        [0.0, 0.2, 0.7, 1.0],
    ],
    dtype=float,
)


EXPECTED_GAUSSIAN_ONE_FEATURE_SIGMA_05 = np.array(
    [
        [1.0, 0.9801986733067553, 0.27803730045319414, 0.19789869908361465],
        [0.9801986733067553, 1.0, 0.37531109885139957, 0.27803730045319414],
        [0.27803730045319414, 0.37531109885139957, 1.0, 0.9801986733067553],
        [0.19789869908361465, 0.27803730045319414, 0.9801986733067553, 1.0],
    ],
    dtype=float,
)


def test_dense_linear_similarity_baseline_exact_one_feature():
    """@brief Dense linear similarity with one feature returns the locked baseline matrix."""
    sim = build_similarity_matrix(
        X_ONE_FEATURE,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(sim, EXPECTED_LINEAR_ONE_FEATURE, atol=1e-12)
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_ONE_FEATURE.shape[0]))


def test_dense_linear_similarity_baseline_exact_two_features_minimum_tnorm():
    """@brief Dense linear similarity across two features uses the minimum T-norm baseline."""
    sim = build_similarity_matrix(
        X_TWO_FEATURES,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(sim, EXPECTED_LINEAR_TWO_FEATURES_MINIMUM, atol=1e-12)
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_TWO_FEATURES.shape[0]))


def test_dense_gaussian_similarity_baseline_exact_one_feature():
    """@brief Dense Gaussian similarity keeps the current sigma parameter contract."""
    sim = build_similarity_matrix(
        X_ONE_FEATURE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(sim, EXPECTED_GAUSSIAN_ONE_FEATURE_SIGMA_05, atol=1e-12)
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_ONE_FEATURE.shape[0]))


def test_dense_similarity_nested_config_matches_flat_config():
    """@brief Normalized nested config must produce the same dense matrix as flat params."""
    flat_config = {
        "type": "itfrs",
        "similarity": "linear",
        "similarity_tnorm": "minimum",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    }
    nested_config = normalize_flat_config_to_nested(flat_config)

    flat_sim = build_similarity_matrix(X_TWO_FEATURES, **flat_config)
    nested_sim = build_similarity_matrix(X_TWO_FEATURES, config=nested_config)

    np.testing.assert_allclose(nested_sim, flat_sim, atol=1e-12)


def test_dense_similarity_does_not_mutate_input_matrix():
    """@brief Dense similarity construction must not modify the caller's feature matrix."""
    X = X_TWO_FEATURES.copy()
    before = X.copy()

    _ = build_similarity_matrix(X, similarity="linear", similarity_tnorm="minimum")

    np.testing.assert_array_equal(X, before)
