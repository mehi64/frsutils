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

from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.implicators import Implicator
from FRsutils.core.owa_weights import OWAWeights
from FRsutils.core.tnorms import TNorm


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



MODEL_FLAT_CONFIGS = {
    "itfrs": {
        "type": "itfrs",
        "ub_tnorm_name": "yager",
        "ub_tnorm_p": 2.5,
        "lb_implicator_name": "lukasiewicz",
    },
    "vqrs": {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.0,
        "lb_fuzzy_quantifier_beta": 0.5,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.1,
        "ub_fuzzy_quantifier_beta": 0.8,
    },
    "owafrs": {
        "type": "owafrs",
        "ub_tnorm_name": "yager",
        "ub_tnorm_p": 2.0,
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "exponential",
        "ub_owa_method_base": 2.0,
        "lb_owa_method_name": "harmonic",
    },
}


@pytest.mark.parametrize("model_type", ["itfrs", "vqrs", "owafrs"])
def test_public_model_builder_supports_each_model_from_flat_config(model_type):
    """The public builder constructs every supported fuzzy-rough model."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")

    model = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        **MODEL_FLAT_CONFIGS[model_type],
    )

    assert isinstance(model, FuzzyRoughModel)
    assert model.positive_region().shape == (len(Y_SMALL),)


@pytest.mark.parametrize("model_type", ["itfrs", "vqrs", "owafrs"])
def test_public_model_builder_supports_each_model_from_nested_config(model_type):
    """Nested configs produced by the public normalizer work for all models."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    nested_config = normalize_flat_config_to_nested(MODEL_FLAT_CONFIGS[model_type])

    model = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        config=nested_config,
    )

    assert isinstance(model, FuzzyRoughModel)
    assert model.lower_approximation().shape == (len(Y_SMALL),)


@pytest.mark.parametrize("model_type", ["itfrs", "vqrs", "owafrs"])
def test_public_model_builder_accepts_list_and_string_labels(model_type):
    """Public construction normalizes non-NumPy labels before model creation."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    string_labels = ["left", "left", "right", "right"]

    model = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=string_labels,
        **MODEL_FLAT_CONFIGS[model_type],
    )

    assert isinstance(model.labels, np.ndarray)
    assert model.positive_region().shape == (len(string_labels),)


def test_public_model_builder_accepts_direct_component_instances():
    """Direct component instances pass through public flat config safely."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")

    itfrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="itfrs",
        ub_tnorm=TNorm.create("minimum"),
        lb_implicator=Implicator.create("lukasiewicz"),
    )
    vqrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="vqrs",
        lb_fuzzy_quantifier=FuzzyQuantifier.create("linear", alpha=0.0, beta=0.5),
        ub_fuzzy_quantifier=FuzzyQuantifier.create("quadratic", alpha=0.1, beta=0.8),
    )
    owafrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="owafrs",
        ub_tnorm=TNorm.create("minimum"),
        lb_implicator=Implicator.create("lukasiewicz"),
        ub_owa_method=OWAWeights.create("linear"),
        lb_owa_method=OWAWeights.create("harmonic"),
    )

    assert itfrs.positive_region().shape == (len(Y_SMALL),)
    assert vqrs.positive_region().shape == (len(Y_SMALL),)
    assert owafrs.positive_region().shape == (len(Y_SMALL),)


def test_public_model_builder_accepts_serialized_component_specs():
    """Serialized component specs are valid public builder inputs."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")

    itfrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="itfrs",
        ub_tnorm=TNorm.create("yager", p=2.0).to_dict(),
        lb_implicator=Implicator.create("lukasiewicz").to_dict(),
    )
    vqrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="vqrs",
        lb_fuzzy_quantifier=FuzzyQuantifier.create("linear", alpha=0.0, beta=0.5).to_dict(),
        ub_fuzzy_quantifier=FuzzyQuantifier.create("quadratic", alpha=0.1, beta=0.8).to_dict(),
    )
    owafrs = build_fuzzy_rough_model(
        similarity_matrix=sim,
        labels=Y_SMALL,
        type="owafrs",
        ub_tnorm=TNorm.create("minimum").to_dict(),
        lb_implicator=Implicator.create("lukasiewicz").to_dict(),
        ub_owa_method=OWAWeights.create("exponential", base=2.0).to_dict(),
        lb_owa_method=OWAWeights.create("linear").to_dict(),
    )

    assert itfrs.upper_approximation().shape == (len(Y_SMALL),)
    assert vqrs.upper_approximation().shape == (len(Y_SMALL),)
    assert owafrs.upper_approximation().shape == (len(Y_SMALL),)


def test_public_model_builder_does_not_mutate_config():
    """The public builder must not rewrite caller-owned config dictionaries."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    config = normalize_flat_config_to_nested(MODEL_FLAT_CONFIGS["vqrs"])
    original = {
        "fr_type": config["fr_model"]["type"],
        "lb_name": config["fr_model"]["lb_fuzzy_quantifier"]["name"],
        "ub_name": config["fr_model"]["ub_fuzzy_quantifier"]["name"],
    }

    build_fuzzy_rough_model(similarity_matrix=sim, labels=Y_SMALL, config=config)

    assert config["fr_model"]["type"] == original["fr_type"]
    assert config["fr_model"]["lb_fuzzy_quantifier"]["name"] == original["lb_name"]
    assert config["fr_model"]["ub_fuzzy_quantifier"]["name"] == original["ub_name"]


def test_public_model_builder_rejects_unknown_model_alias():
    """Unknown fuzzy-rough model aliases fail at the public boundary."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")

    with pytest.raises(ValueError, match="Unknown alias"):
        build_fuzzy_rough_model(
            similarity_matrix=sim,
            labels=Y_SMALL,
            type="not-a-model",
        )


def test_public_model_builder_rejects_non_mapping_config():
    """Non-mapping config input fails before internal normalization."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")

    with pytest.raises(TypeError, match="config must be a mapping"):
        build_fuzzy_rough_model(
            similarity_matrix=sim,
            labels=Y_SMALL,
            config=[("type", "itfrs")],
        )


@pytest.mark.parametrize(
    "similarity_matrix, labels, match",
    [
        (np.array([1.0, 0.5]), Y_SMALL, "2D"),
        (np.ones((3, 4)), Y_SMALL, "square"),
        (np.eye(4), np.array([[0, 0, 1, 1]]), "1D"),
        (np.eye(4), np.array([0, 1]), "Length of labels"),
    ],
)
def test_public_model_builder_rejects_invalid_matrix_or_labels(similarity_matrix, labels, match):
    """Matrix and label validation is enforced by the public model builder."""
    with pytest.raises(ValueError, match=match):
        build_fuzzy_rough_model(
            similarity_matrix=similarity_matrix,
            labels=labels,
            type="itfrs",
            ub_tnorm_name="minimum",
            lb_implicator_name="lukasiewicz",
        )


def test_public_model_builder_rejects_invalid_similarity_values():
    """Invalid similarity values are rejected by model validation."""
    sim = build_similarity_matrix(X_SMALL, similarity="linear")
    sim[0, 1] = 1.5

    with pytest.raises(ValueError, match="range"):
        build_fuzzy_rough_model(
            similarity_matrix=sim,
            labels=Y_SMALL,
            type="itfrs",
            ub_tnorm_name="minimum",
            lb_implicator_name="lukasiewicz",
        )
