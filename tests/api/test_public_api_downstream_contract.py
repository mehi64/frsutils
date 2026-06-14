# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for downstream packages that depend on FRsutils.api."""

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


def test_public_api_star_import_exposes_expected_facade_names():
    """The canonical facade exposes the stable user-facing names via __all__."""
    namespace = {}

    exec("from FRsutils.api import *", namespace)

    expected_names = {
        "compute_approximations",
        "compute_lower_approximation",
        "compute_upper_approximation",
        "compute_boundary_region",
        "compute_positive_region",
        "build_similarity_matrix",
        "build_similarity_engine",
        "build_fuzzy_rough_model",
        "FuzzyRoughApproximationResult",
        "FuzzyRoughPositiveRegionScorer",
        "ITFRS",
        "VQRS",
        "OWAFRS",
    }

    assert expected_names <= set(namespace)


def test_top_level_package_keeps_public_api_under_api_namespace():
    """The package root stays compact while FRsutils.api remains canonical."""
    import FRsutils

    assert hasattr(FRsutils, "api")
    assert FRsutils.__all__ == [
        "api",
        "tnorms",
        "implicators",
        "similarities",
        "itfrs",
    ]
    assert not hasattr(FRsutils, "compute_approximations")
    assert not hasattr(FRsutils, "build_fuzzy_rough_model")


def test_public_api_does_not_expose_internal_constructor_utilities():
    """Downstream users should not depend on internal construction helpers."""
    import FRsutils.api as public_api

    internal_names = {
        "LazyConstructibleMixin",
        "RegistryFactoryMixin",
        "normalize_config",
    }

    assert internal_names.isdisjoint(set(public_api.__all__))

