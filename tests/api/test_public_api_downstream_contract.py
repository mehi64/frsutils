"""
@file test_public_api_downstream_contract.py
@brief Contract tests for downstream packages that depend on FRsutils.api.

These tests model how a standalone sampling package such as frsampling should
consume FRsutils after FRSMOTE has been moved out of the core package.
"""

import numpy as np

from FRsutils.api import (
    build_fuzzy_rough_model,
    build_similarity_matrix,
    compute_approximations,
    compute_positive_region,
)


def test_downstream_can_use_only_public_api_for_model_building():
    X = np.array([[0.0, 0.0], [0.2, 0.2], [0.8, 0.8], [1.0, 1.0]], dtype=float)
    y = np.array([0, 0, 1, 1])

    similarity_matrix = build_similarity_matrix(X, similarity="linear")
    model = build_fuzzy_rough_model(
        "itfrs",
        similarity_matrix=similarity_matrix,
        labels=y,
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
    )

    positive_region = model.positive_region()
    assert positive_region.shape == (len(y),)


def test_downstream_can_reuse_precomputed_similarity_matrix_in_task_api():
    X = np.array([[0.0], [0.1], [0.8], [0.9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    similarity_matrix = build_similarity_matrix(X, similarity="linear")
    direct_scores = compute_positive_region(X, y, model="itfrs")
    reused_scores = compute_positive_region(
        X=None,
        y=y,
        model="itfrs",
        similarity_matrix=similarity_matrix,
    )

    np.testing.assert_allclose(direct_scores, reused_scores)


def test_downstream_gets_named_result_not_tuple():
    X = np.array([[0.0], [0.2], [0.7], [0.9]], dtype=float)
    y = np.array([0, 0, 1, 1])

    result = compute_approximations(X, y, model="itfrs")

    assert hasattr(result, "lower")
    assert hasattr(result, "upper")
    assert hasattr(result, "positive_region")
