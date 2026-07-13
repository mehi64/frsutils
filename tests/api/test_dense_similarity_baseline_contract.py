# SPDX-License-Identifier: BSD-3-Clause
"""Baseline tests for the dense similarity-matrix contract."""

import numpy as np

from frsutils import build_similarity_matrix, normalize_flat_config_to_nested
from tests import reference_data_store as ds


DENSE_BASELINES = {
    case["name"]: case for case in ds.get_dense_similarity_baseline_testsets()
}
LINEAR_ONE_FEATURE = DENSE_BASELINES["dense_linear_one_feature"]
LINEAR_TWO_FEATURES_MINIMUM = DENSE_BASELINES[
    "dense_linear_two_features_minimum"
]
GAUSSIAN_ONE_FEATURE_SIGMA_05 = DENSE_BASELINES[
    "dense_gaussian_one_feature_sigma_0_5"
]
X_ONE_FEATURE = LINEAR_ONE_FEATURE["X"]
X_TWO_FEATURES = LINEAR_TWO_FEATURES_MINIMUM["X"]


def _build_similarity_from_reference_case(reference_case):
    """Build a similarity matrix using one reference case configuration."""
    similarity = reference_case["similarity"]
    params = similarity["params"]
    kwargs = {
        "similarity": similarity["name"],
        "similarity_tnorm": params["tnorm"],
    }
    if "sigma" in params:
        kwargs["similarity_sigma"] = params["sigma"]
    return build_similarity_matrix(reference_case["X"], **kwargs)


def test_dense_linear_similarity_baseline_exact_one_feature():
    """Dense linear similarity with one feature returns the locked baseline matrix."""
    sim = _build_similarity_from_reference_case(LINEAR_ONE_FEATURE)

    np.testing.assert_allclose(sim, LINEAR_ONE_FEATURE["expected"], atol=1e-12)
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_ONE_FEATURE.shape[0]))


def test_dense_linear_similarity_baseline_exact_two_features_minimum_tnorm():
    """Dense linear similarity across two features uses the minimum T-norm baseline."""
    sim = _build_similarity_from_reference_case(LINEAR_TWO_FEATURES_MINIMUM)

    np.testing.assert_allclose(
        sim,
        LINEAR_TWO_FEATURES_MINIMUM["expected"],
        atol=1e-12,
    )
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_TWO_FEATURES.shape[0]))


def test_dense_gaussian_similarity_baseline_exact_one_feature():
    """Dense Gaussian similarity keeps the current sigma parameter contract."""
    sim = _build_similarity_from_reference_case(GAUSSIAN_ONE_FEATURE_SIGMA_05)

    np.testing.assert_allclose(
        sim,
        GAUSSIAN_ONE_FEATURE_SIGMA_05["expected"],
        atol=1e-12,
    )
    np.testing.assert_allclose(sim, sim.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(sim), np.ones(X_ONE_FEATURE.shape[0]))


def test_dense_similarity_nested_config_matches_flat_config():
    """Normalized nested config must produce the same dense matrix as flat params."""
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
    """Dense similarity construction must not modify the caller's feature matrix."""
    X = X_TWO_FEATURES.copy()
    before = X.copy()

    _ = build_similarity_matrix(X, similarity="linear", similarity_tnorm="minimum")

    np.testing.assert_array_equal(X, before)
