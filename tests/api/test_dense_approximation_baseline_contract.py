# SPDX-License-Identifier: BSD-3-Clause
"""Baseline tests for dense fuzzy-rough approximation outputs."""

import numpy as np
import pytest

from frsutils import (
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)
from tests import reference_data_store as reference_data


DENSE_BASELINE_CASE = reference_data.get_dense_approximation_baseline_testsets()[0]
X_BASELINE = DENSE_BASELINE_CASE["X"]
Y_BASELINE = DENSE_BASELINE_CASE["labels"]
SIMILARITY_SPEC = DENSE_BASELINE_CASE["similarity"]
EXPECTED_BY_MODEL = DENSE_BASELINE_CASE["expected_by_model"]
MODEL_NAMES = tuple(EXPECTED_BY_MODEL)
LOCKED_MODEL_CONFIG = {
    "vqrs": {
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.1,
        "lb_fuzzy_quantifier_beta": 0.6,
        "ub_fuzzy_quantifier_name": "linear",
        "ub_fuzzy_quantifier_alpha": 0.1,
        "ub_fuzzy_quantifier_beta": 0.6,
    }
}


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_dense_approximation_baseline_exact_values(model_name):
    """Public dense execution matches the locked configured baseline fixture."""
    result = compute_approximations(
        X_BASELINE,
        Y_BASELINE,
        model=model_name,
        similarity=SIMILARITY_SPEC["name"],
        return_similarity_matrix=True,
        **LOCKED_MODEL_CONFIG.get(model_name, {}),
    )
    expected = EXPECTED_BY_MODEL[model_name]

    np.testing.assert_allclose(result.lower, expected["lower"], atol=1e-12)
    np.testing.assert_allclose(result.upper, expected["upper"], atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected["boundary"], atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected["positive_region"], atol=1e-12)
    assert result.model == model_name
    assert result.similarity == SIMILARITY_SPEC["name"]
    assert result.similarity_matrix.shape == (len(Y_BASELINE), len(Y_BASELINE))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_precomputed_similarity_path_matches_x_path_for_dense_baseline(model_name):
    """Reusing a dense similarity matrix must match the X-driven public path."""
    similarity_matrix = build_similarity_matrix(
        X_BASELINE,
        similarity=SIMILARITY_SPEC["name"],
        **SIMILARITY_SPEC["params"],
    )

    from_x = compute_approximations(
        X_BASELINE,
        Y_BASELINE,
        model=model_name,
        similarity=SIMILARITY_SPEC["name"],
    )
    from_matrix = compute_approximations(
        X=None,
        y=Y_BASELINE,
        model=model_name,
        similarity_matrix=similarity_matrix,
    )

    np.testing.assert_allclose(from_matrix.lower, from_x.lower, atol=1e-12)
    np.testing.assert_allclose(from_matrix.upper, from_x.upper, atol=1e-12)
    np.testing.assert_allclose(from_matrix.boundary, from_x.boundary, atol=1e-12)
    np.testing.assert_allclose(from_matrix.positive_region, from_x.positive_region, atol=1e-12)


def test_convenience_wrappers_match_dense_full_result():
    """Public wrapper functions must remain aliases for the full dense result fields."""
    result = compute_approximations(X_BASELINE, Y_BASELINE, model="itfrs", similarity=SIMILARITY_SPEC["name"])

    np.testing.assert_allclose(
        compute_lower_approximation(
            X_BASELINE,
            Y_BASELINE,
            model="itfrs",
            similarity=SIMILARITY_SPEC["name"],
        ),
        result.lower,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_upper_approximation(
            X_BASELINE,
            Y_BASELINE,
            model="itfrs",
            similarity=SIMILARITY_SPEC["name"],
        ),
        result.upper,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_boundary_region(
            X_BASELINE,
            Y_BASELINE,
            model="itfrs",
            similarity=SIMILARITY_SPEC["name"],
        ),
        result.boundary,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_positive_region(
            X_BASELINE,
            Y_BASELINE,
            model="itfrs",
            similarity=SIMILARITY_SPEC["name"],
        ),
        result.positive_region,
        atol=1e-12,
    )


def test_dense_result_as_dict_excludes_similarity_matrix_by_default():
    """Result serialization keeps the optional dense matrix out unless requested."""
    result = compute_approximations(
        X_BASELINE,
        Y_BASELINE,
        model="itfrs",
        similarity=SIMILARITY_SPEC["name"],
        return_similarity_matrix=True,
    )

    default_dict = result.as_dict()
    full_dict = result.as_dict(include_similarity_matrix=True)

    assert "similarity_matrix" not in default_dict
    assert "similarity_matrix" in full_dict
    np.testing.assert_allclose(full_dict["similarity_matrix"], result.similarity_matrix, atol=1e-12)
