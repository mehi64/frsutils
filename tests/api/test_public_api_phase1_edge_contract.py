# SPDX-License-Identifier: BSD-3-Clause
"""Edge-case and validation contracts for the public fuzzy-rough API."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from frsutils import (
    FuzzyRoughPositiveRegionScorer,
    build_fuzzy_rough_model,
    build_similarity_engine,
    build_similarity_matrix,
    calculate_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)
from frsutils.core.similarities import Similarity
from frsutils.core.tnorms import TNorm


X_VALID = np.array(
    [
        [0.0, 0.1],
        [0.2, 0.3],
        [0.8, 0.7],
        [1.0, 0.9],
    ],
    dtype=float,
)
Y_VALID = np.array(["low", "low", "high", "high"], dtype=object)


@pytest.mark.parametrize("sample_count", [0, 1])
@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
@pytest.mark.parametrize("engine", ["dense", "blockwise"])
def test_compute_approximations_rejects_fewer_than_two_samples(
    sample_count: int,
    model: str,
    engine: str,
) -> None:
    """All public model and engine combinations share one minimum-size contract."""
    X = np.zeros((sample_count, 2), dtype=float)
    y = np.arange(sample_count)

    with pytest.raises(ValueError, match="at least two samples"):
        compute_approximations(
            X,
            y,
            model=model,
            engine=engine,
            block_size=1,
        )


@pytest.mark.parametrize(
    "wrapper",
    [
        compute_lower_approximation,
        compute_upper_approximation,
        compute_boundary_region,
        compute_positive_region,
    ],
)
def test_approximation_wrappers_share_minimum_sample_contract(
    wrapper: Callable[..., np.ndarray],
) -> None:
    """Convenience wrappers propagate the public minimum-sample validation."""
    with pytest.raises(ValueError, match="at least two samples"):
        wrapper(np.array([[0.0, 1.0]]), np.array([0]), model="itfrs")


@pytest.mark.parametrize("sample_count", [0, 1])
def test_positive_region_scorer_shares_minimum_sample_contract(
    sample_count: int,
) -> None:
    """The sklearn-style scorer rejects undersized fuzzy-rough datasets."""
    scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
    X = np.zeros((sample_count, 2), dtype=float)
    y = np.arange(sample_count)

    with pytest.raises(ValueError, match="at least two samples"):
        scorer.fit(X, y)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
@pytest.mark.parametrize("sample_count", [0, 1])
def test_public_model_builder_rejects_fewer_than_two_samples(
    model: str,
    sample_count: int,
) -> None:
    """Direct public model construction uses the same minimum-size contract."""
    with pytest.raises(ValueError, match="at least two samples"):
        build_fuzzy_rough_model(
            model,
            similarity_matrix=np.eye(sample_count),
            labels=np.arange(sample_count),
        )


def test_calculate_similarity_matrix_matches_configured_public_builder() -> None:
    """The direct-component helper matches the flat-config similarity builder."""
    X = X_VALID.copy()
    original = X.copy()
    similarity = Similarity.create("gaussian", sigma=0.4)
    tnorm = TNorm.create("product")

    actual = calculate_similarity_matrix(X, similarity, tnorm)
    expected = build_similarity_matrix(
        X,
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="product",
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(X, original)


def test_calculate_similarity_matrix_accepts_array_like_input() -> None:
    """The direct-component helper accepts finite two-dimensional array-like data."""
    similarity = Similarity.create("linear")
    tnorm = TNorm.create("minimum")

    matrix = calculate_similarity_matrix(
        [[0.0, 0.5], [1.0, 0.25]],
        similarity,
        tnorm,
    )

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(matrix), np.ones(2), atol=0.0)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("entry_point", ["matrix", "engine", "direct"])
def test_similarity_public_entry_points_reject_non_finite_features(
    invalid_value: float,
    entry_point: str,
) -> None:
    """Every public similarity entry point rejects NaN and infinite features."""
    X = X_VALID.copy()
    X[1, 0] = invalid_value

    with pytest.raises(ValueError, match="finite numeric values"):
        if entry_point == "matrix":
            build_similarity_matrix(X, similarity="linear")
        elif entry_point == "engine":
            build_similarity_engine(X, engine="dense", similarity="linear")
        else:
            calculate_similarity_matrix(
                X,
                Similarity.create("linear"),
                TNorm.create("minimum"),
            )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_public_approximation_rejects_non_finite_precomputed_similarity(
    invalid_value: float,
) -> None:
    """Precomputed matrices with non-finite values fail at the public boundary."""
    similarity_matrix = np.eye(len(Y_VALID), dtype=float)
    similarity_matrix[0, 1] = invalid_value

    with pytest.raises(ValueError, match="similarity_matrix.*finite"):
        compute_approximations(
            None,
            Y_VALID,
            model="itfrs",
            similarity_matrix=similarity_matrix,
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_public_model_builder_rejects_non_finite_similarity(
    invalid_value: float,
) -> None:
    """Direct public model construction rejects non-finite matrix values."""
    similarity_matrix = np.eye(len(Y_VALID), dtype=float)
    similarity_matrix[0, 1] = invalid_value

    with pytest.raises(ValueError, match="similarity_matrix.*finite"):
        build_fuzzy_rough_model(
            "itfrs",
            similarity_matrix=similarity_matrix,
            labels=Y_VALID,
        )


def test_compute_approximations_rejects_non_mapping_config() -> None:
    """Approximation configuration must be supplied as a mapping."""
    with pytest.raises(TypeError, match="config must be a mapping"):
        compute_approximations(X_VALID, Y_VALID, config=[])


@pytest.mark.parametrize("engine", ["dense", "blockwise"])
def test_compute_approximations_requires_x_without_precomputed_similarity(
    engine: str,
) -> None:
    """Dense and blockwise execution reject a missing feature matrix clearly."""
    with pytest.raises(ValueError, match="X must be provided"):
        compute_approximations(None, Y_VALID, model="itfrs", engine=engine)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"similarity_matrix": None, "labels": Y_VALID}, "similarity_matrix"),
        ({"similarity_matrix": np.eye(len(Y_VALID)), "labels": None}, "labels"),
    ],
)
def test_public_model_builder_requires_matrix_and_labels(kwargs: dict, match: str) -> None:
    """The public model builder rejects either missing required data input."""
    with pytest.raises(ValueError, match=match):
        build_fuzzy_rough_model("itfrs", **kwargs)


def test_public_model_builder_rejects_non_string_model_selector() -> None:
    """Model selectors must be non-empty string aliases."""
    with pytest.raises(TypeError, match="model type must be a non-empty string"):
        build_fuzzy_rough_model(
            123,
            similarity_matrix=np.eye(len(Y_VALID)),
            labels=Y_VALID,
        )


def test_public_config_rejects_non_string_component_selector() -> None:
    """Component selector values must be non-empty string aliases."""
    with pytest.raises(TypeError, match="ub_tnorm_name must be a non-empty string"):
        compute_approximations(
            X_VALID,
            Y_VALID,
            model="itfrs",
            ub_tnorm_name=123,
        )


def test_public_config_rejects_direct_component_and_flat_selector_mix() -> None:
    """A direct component cannot be mixed with its flat selector or parameters."""
    similarity_matrix = build_similarity_matrix(X_VALID)

    with pytest.raises(ValueError, match="Do not mix direct component 'ub_tnorm'"):
        build_fuzzy_rough_model(
            "itfrs",
            similarity_matrix=similarity_matrix,
            labels=Y_VALID,
            ub_tnorm=TNorm.create("minimum"),
            ub_tnorm_name="minimum",
        )


def test_canonical_similarity_parameter_precedes_legacy_alias() -> None:
    """Canonical flat names win deterministically when a legacy alias is also present."""
    actual = build_similarity_matrix(
        X_VALID,
        config={
            "similarity": "gaussian",
            "similarity_sigma": 0.25,
            "sigma": 0.9,
        },
    )
    expected = build_similarity_matrix(
        X_VALID,
        similarity="gaussian",
        similarity_sigma=0.25,
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)
