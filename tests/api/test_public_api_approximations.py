# SPDX-License-Identifier: BSD-3-Clause
"""Tests for frsutils public approximation API."""

import numpy as np
import pytest

from frsutils import (
    FuzzyRoughApproximationResult,
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)


X_SMALL = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.8, 0.8],
        [0.9, 0.9],
    ],
    dtype=float,
)
Y_SMALL = np.array([0, 0, 1, 1])


def test_compute_approximations_returns_named_result():
    result = compute_approximations(X_SMALL, Y_SMALL, model="itfrs")

    assert isinstance(result, FuzzyRoughApproximationResult)
    assert result.lower.shape == (len(Y_SMALL),)
    assert result.upper.shape == (len(Y_SMALL),)
    assert result.boundary.shape == (len(Y_SMALL),)
    assert result.positive_region.shape == (len(Y_SMALL),)
    assert result.model == "itfrs"


def test_convenience_wrappers_match_full_result():
    result = compute_approximations(X_SMALL, Y_SMALL, model="itfrs")

    np.testing.assert_allclose(compute_lower_approximation(X_SMALL, Y_SMALL), result.lower)
    np.testing.assert_allclose(compute_upper_approximation(X_SMALL, Y_SMALL), result.upper)
    np.testing.assert_allclose(compute_boundary_region(X_SMALL, Y_SMALL), result.boundary)
    np.testing.assert_allclose(compute_positive_region(X_SMALL, Y_SMALL), result.positive_region)


def test_compute_approximations_accepts_precomputed_similarity_matrix():
    similarity_matrix = build_similarity_matrix(X_SMALL, similarity="linear")

    result = compute_approximations(
        X=None,
        y=Y_SMALL,
        model="itfrs",
        similarity_matrix=similarity_matrix,
        return_similarity_matrix=True,
    )

    np.testing.assert_allclose(result.similarity_matrix, similarity_matrix)
    assert result.positive_region.shape == (len(Y_SMALL),)


def test_compute_positive_region_supports_alternative_models():
    scores = compute_positive_region(X_SMALL, Y_SMALL, model="vqrs")

    assert scores.shape == (len(Y_SMALL),)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
@pytest.mark.parametrize("engine", ["dense", "blockwise"])
def test_compute_approximations_supports_all_models_and_engines(model, engine):
    result = compute_approximations(
        X_SMALL,
        Y_SMALL,
        model=model,
        engine=engine,
        block_size=2,
    )

    assert result.model == model
    assert result.engine == engine
    assert result.used_blockwise is (engine == "blockwise")
    assert result.similarity_matrix is None
    for values in (result.lower, result.upper, result.boundary, result.positive_region):
        assert isinstance(values, np.ndarray)
        assert values.shape == (len(Y_SMALL),)
    np.testing.assert_allclose(result.boundary, result.upper - result.lower)
    np.testing.assert_allclose(result.positive_region, result.lower)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_compute_approximations_can_return_similarity_matrix_for_all_models(model):
    result = compute_approximations(
        X_SMALL,
        Y_SMALL,
        model=model,
        return_similarity_matrix=True,
    )

    assert isinstance(result.similarity_matrix, np.ndarray)
    assert result.similarity_matrix.shape == (len(Y_SMALL), len(Y_SMALL))


@pytest.mark.parametrize(
    ("bad_y", "message"),
    [
        (None, "provided"),
        ([[0, 0], [1, 1]], "labels must be a 1D"),
    ],
)
def test_compute_approximations_rejects_invalid_public_labels(bad_y, message):
    with pytest.raises(ValueError, match=message):
        compute_approximations(X_SMALL, bad_y, model="itfrs")


def test_compute_approximations_rejects_x_y_length_mismatch():
    with pytest.raises(ValueError, match="Length of labels must match"):
        compute_approximations(X_SMALL[:3], Y_SMALL, model="itfrs")


@pytest.mark.parametrize(
    ("similarity_matrix", "message"),
    [
        (np.ones(4), "2D square"),
        (np.ones((3, 4)), "must be square"),
        (np.eye(3), "size must match"),
    ],
)
def test_compute_approximations_validates_precomputed_similarity_matrix(similarity_matrix, message):
    with pytest.raises(ValueError, match=message):
        compute_approximations(
            X=None,
            y=Y_SMALL,
            model="itfrs",
            similarity_matrix=similarity_matrix,
        )


def test_compute_approximations_rejects_unknown_model_alias():
    with pytest.raises(ValueError, match="Unknown alias"):
        compute_approximations(X_SMALL, Y_SMALL, model="unknown-model")


def test_compute_approximations_rejects_unknown_execution_engine():
    with pytest.raises(ValueError, match="Unknown approximation engine"):
        compute_approximations(X_SMALL, Y_SMALL, model="itfrs", engine="streaming")


def test_compute_approximations_rejects_unknown_blockwise_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            engine="blockwise",
            backend="not-a-backend",
        )


def test_compute_approximations_rejects_nested_config():
    nested_config = {
        "similarity": {"name": "linear", "params": {}},
        "fr_model": {
            "type": "itfrs",
            "ub_tnorm": {"name": "minimum", "params": {}},
            "lb_implicator": {"name": "lukasiewicz", "params": {}},
        },
    }

    with pytest.raises(ValueError, match="Nested configuration is internal"):
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            config=nested_config,
            ub_tnorm_name="minimum",
        )


def test_blockwise_compute_approximations_rejects_precomputed_similarity_matrix():
    similarity_matrix = build_similarity_matrix(X_SMALL, similarity="linear")

    with pytest.raises(ValueError, match="does not accept precomputed"):
        compute_approximations(
            X=None,
            y=Y_SMALL,
            model="itfrs",
            engine="blockwise",
            similarity_matrix=similarity_matrix,
        )

