# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for public model resolution and default configuration."""

import numpy as np
import pytest

from frsutils import (
    ITFRS,
    OWAFRS,
    VQRS,
    build_fuzzy_rough_model,
    build_similarity_matrix,
    compute_approximations,
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
MODEL_CLASSES = {
    "itfrs": ITFRS,
    "owafrs": OWAFRS,
    "vqrs": VQRS,
}


def test_compute_approximations_defaults_to_itfrs():
    """Approximation calls use ITFRS only when no source selects a model."""
    result = compute_approximations(X_SMALL, Y_SMALL)

    assert result.model == "itfrs"
    assert result.config["type"] == "itfrs"


def test_compute_approximations_resolves_model_from_flat_config():
    """Flat config model type controls approximation dispatch."""
    result = compute_approximations(X_SMALL, Y_SMALL, config={"type": "vqrs"})
    explicit = compute_approximations(X_SMALL, Y_SMALL, model="vqrs")

    assert result.model == "vqrs"
    np.testing.assert_allclose(result.lower, explicit.lower)
    np.testing.assert_allclose(result.upper, explicit.upper)


def test_compute_approximations_rejects_nested_model_config():
    """Normalized nested model config is not part of the public API contract."""
    with pytest.raises(ValueError, match="Nested configuration is internal"):
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            config={"fr_model": {"type": "vqrs"}},
        )


def test_compute_approximations_rejects_conflicting_model_sources():
    """Explicit model and config model types must agree."""
    with pytest.raises(ValueError, match="Conflicting fuzzy-rough model types"):
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            config={"type": "vqrs"},
        )


@pytest.mark.parametrize("model_type", ["itfrs", "owafrs", "vqrs"])
def test_public_model_builder_supplies_defaults_for_each_model(model_type):
    """The public model builder constructs every model without manual components."""
    similarity_matrix = build_similarity_matrix(X_SMALL)

    model = build_fuzzy_rough_model(
        model_type,
        similarity_matrix=similarity_matrix,
        labels=Y_SMALL,
    )

    assert isinstance(model, MODEL_CLASSES[model_type])
    assert model.positive_region().shape == (len(Y_SMALL),)


def test_public_model_builder_resolves_model_from_flat_config():
    """The public builder resolves the model type from flat config."""
    similarity_matrix = build_similarity_matrix(X_SMALL)

    model = build_fuzzy_rough_model(
        similarity_matrix=similarity_matrix,
        labels=Y_SMALL,
        config={"type": "owafrs"},
    )

    assert isinstance(model, OWAFRS)


def test_public_model_builder_rejects_nested_model_config():
    """The public model builder rejects internal nested configuration."""
    similarity_matrix = build_similarity_matrix(X_SMALL)

    with pytest.raises(ValueError, match="Nested configuration is internal"):
        build_fuzzy_rough_model(
            similarity_matrix=similarity_matrix,
            labels=Y_SMALL,
            config={"fr_model": {"type": "vqrs"}},
        )


@pytest.mark.parametrize("model_type", ["itfrs", "owafrs", "vqrs"])
def test_public_builder_and_approximation_api_share_model_defaults(model_type):
    """Builder and task API use identical default components for each model."""
    similarity_matrix = build_similarity_matrix(X_SMALL)
    model = build_fuzzy_rough_model(
        model_type,
        similarity_matrix=similarity_matrix,
        labels=Y_SMALL,
    )
    result = compute_approximations(
        None,
        Y_SMALL,
        model=model_type,
        similarity_matrix=similarity_matrix,
    )

    np.testing.assert_allclose(model.lower_approximation(), result.lower)
    np.testing.assert_allclose(model.upper_approximation(), result.upper)


def test_public_model_builder_defaults_to_itfrs():
    """The builder uses ITFRS when neither argument nor config selects a model."""
    similarity_matrix = build_similarity_matrix(X_SMALL)

    model = build_fuzzy_rough_model(
        similarity_matrix=similarity_matrix,
        labels=Y_SMALL,
    )

    assert isinstance(model, ITFRS)
