# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2 contract tests for downstream-oriented public builders."""

import numpy as np
import pytest

from FRsutils.api import (
    FuzzyRoughModel,
    build_fuzzy_rough_model,
    build_similarity_matrix,
    list_fuzzy_rough_models,
    list_similarities,
    normalize_flat_config_to_nested,
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


def test_public_similarity_builder_accepts_flat_config():
    """Downstream packages can build matrices from flat params."""
    sim = build_similarity_matrix(
        X_SMALL,
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="minimum",
    )

    assert sim.shape == (len(Y_SMALL), len(Y_SMALL))
    np.testing.assert_allclose(np.diag(sim), np.ones(len(Y_SMALL)))
    assert np.all(sim >= 0.0)
    assert np.all(sim <= 1.0)


def test_public_similarity_builder_accepts_nested_config():
    """Downstream packages can reuse normalized nested config."""
    config = normalize_flat_config_to_nested(
        {
            "type": "itfrs",
            "similarity": "linear",
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "minimum",
            "lb_implicator_name": "lukasiewicz",
        }
    )

    flat_sim = build_similarity_matrix(X_SMALL, similarity="linear")
    nested_sim = build_similarity_matrix(X_SMALL, config=config)

    np.testing.assert_allclose(flat_sim, nested_sim)


def test_public_model_builder_accepts_type_from_flat_config():
    """Downstream packages may provide model type as flat `type`."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    model = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="itfrs",
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
    )

    assert isinstance(model, FuzzyRoughModel)
    assert model.positive_region().shape == (len(Y_SMALL),)


def test_public_model_builder_accepts_nested_config_with_extra_flat_kwargs():
    """Mixed nested config plus flat runtime kwargs supports frsampling's bridge."""
    config = normalize_flat_config_to_nested(
        {
            "type": "itfrs",
            "similarity": "linear",
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "minimum",
            "lb_implicator_name": "lukasiewicz",
        }
    )
    sim = build_similarity_matrix(X_SMALL, config=config)

    model = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        config=config,
        type="itfrs",
        k_neighbors=3,
        sampling_strategy="auto",
    )

    assert isinstance(model, FuzzyRoughModel)
    assert model.lower_approximation().shape == (len(Y_SMALL),)


def test_public_model_builder_rejects_conflicting_model_types():
    """Public boundary fails fast when explicit and config model types conflict."""
    config = normalize_flat_config_to_nested(
        {
            "type": "itfrs",
            "similarity": "linear",
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "minimum",
            "lb_implicator_name": "lukasiewicz",
        }
    )
    sim = build_similarity_matrix(X_SMALL, config=config)

    with pytest.raises(ValueError, match="Conflicting fuzzy-rough model types"):
        build_fuzzy_rough_model(
            "vqrs",
            similarity_matrix=sim,
            labels=Y_SMALL,
            config=config,
        )


def test_public_registry_listing_helpers_are_available():
    """Downstream packages can inspect public registries through the facade."""
    models = list_fuzzy_rough_models()
    similarities = list_similarities()

    assert "itfrs" in models
    assert "linear" in similarities
