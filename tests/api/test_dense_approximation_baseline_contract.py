# SPDX-License-Identifier: BSD-3-Clause
"""Phase 0 baseline tests for dense fuzzy-rough approximation outputs."""

import numpy as np
import pytest

from FRsutils.api import (
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)


X_BASELINE = np.array([[0.0], [0.1], [0.8], [0.9]], dtype=float)
Y_BASELINE = np.array([0, 0, 1, 1])


EXPECTED_BY_MODEL = {
    "itfrs": {
        "lower": np.array([0.8, 0.7, 0.7, 0.8], dtype=float),
        "upper": np.array([0.9, 0.9, 0.9, 0.9], dtype=float),
        "boundary": np.array([0.1, 0.2, 0.2, 0.1], dtype=float),
        "positive_region": np.array([0.8, 0.7, 0.7, 0.8], dtype=float),
    },
    "vqrs": {
        "lower": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        "upper": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        "boundary": np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        "positive_region": np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
    },
    "owafrs": {
        "lower": np.array([0.8666666666666667, 0.7833333333333333, 0.7833333333333333, 0.8666666666666667]),
        "upper": np.array([0.45, 0.45, 0.45, 0.45], dtype=float),
        "boundary": np.array([-0.4166666666666667, -0.3333333333333333, -0.3333333333333333, -0.4166666666666667]),
        "positive_region": np.array([0.8666666666666667, 0.7833333333333333, 0.7833333333333333, 0.8666666666666667]),
    },
}


@pytest.mark.parametrize("model_name", ["itfrs", "vqrs", "owafrs"])
def test_dense_approximation_baseline_exact_values(model_name):
    """@brief Public dense approximation output stays fixed for the small Phase 0 fixture."""
    result = compute_approximations(
        X_BASELINE,
        Y_BASELINE,
        model=model_name,
        similarity="linear",
        return_similarity_matrix=True,
    )
    expected = EXPECTED_BY_MODEL[model_name]

    np.testing.assert_allclose(result.lower, expected["lower"], atol=1e-12)
    np.testing.assert_allclose(result.upper, expected["upper"], atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected["boundary"], atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected["positive_region"], atol=1e-12)
    assert result.model == model_name
    assert result.similarity == "linear"
    assert result.similarity_matrix.shape == (len(Y_BASELINE), len(Y_BASELINE))


@pytest.mark.parametrize("model_name", ["itfrs", "vqrs", "owafrs"])
def test_precomputed_similarity_path_matches_x_path_for_dense_baseline(model_name):
    """@brief Reusing a dense similarity matrix must match the X-driven public path."""
    similarity_matrix = build_similarity_matrix(X_BASELINE, similarity="linear")

    from_x = compute_approximations(X_BASELINE, Y_BASELINE, model=model_name, similarity="linear")
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
    """@brief Public wrapper functions must remain aliases for the full dense result fields."""
    result = compute_approximations(X_BASELINE, Y_BASELINE, model="itfrs", similarity="linear")

    np.testing.assert_allclose(
        compute_lower_approximation(X_BASELINE, Y_BASELINE, model="itfrs", similarity="linear"),
        result.lower,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_upper_approximation(X_BASELINE, Y_BASELINE, model="itfrs", similarity="linear"),
        result.upper,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_boundary_region(X_BASELINE, Y_BASELINE, model="itfrs", similarity="linear"),
        result.boundary,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compute_positive_region(X_BASELINE, Y_BASELINE, model="itfrs", similarity="linear"),
        result.positive_region,
        atol=1e-12,
    )


def test_dense_result_as_dict_excludes_similarity_matrix_by_default():
    """@brief Result serialization keeps the optional dense matrix out unless requested."""
    result = compute_approximations(
        X_BASELINE,
        Y_BASELINE,
        model="itfrs",
        similarity="linear",
        return_similarity_matrix=True,
    )

    default_dict = result.as_dict()
    full_dict = result.as_dict(include_similarity_matrix=True)

    assert "similarity_matrix" not in default_dict
    assert "similarity_matrix" in full_dict
    np.testing.assert_allclose(full_dict["similarity_matrix"], result.similarity_matrix, atol=1e-12)
