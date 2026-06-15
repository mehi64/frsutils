# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for similarity engines in approximation APIs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from frsutils import (
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
    normalize_flat_config_to_nested,
)
from tests._fake_cupy_backend import install_fake_cupy_module


X_INTEGRATION = np.array(
    [
        [0.00, 0.00, 0.10],
        [0.08, 0.20, 0.15],
        [0.22, 0.15, 0.05],
        [0.65, 0.70, 0.55],
        [0.78, 0.75, 0.60],
        [0.90, 0.88, 0.72],
        [0.42, 0.52, 0.90],
    ],
    dtype=float,
)
Y_INTEGRATION = np.array([0, 0, 0, 1, 1, 1, 2])


MODEL_CONFIGS = {
    "itfrs": {
        "model": "itfrs",
        "similarity": "linear",
        "similarity_tnorm": "minimum",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    },
    "vqrs": {
        "model": "vqrs",
        "similarity": "gaussian",
        "similarity_sigma": 0.45,
        "similarity_tnorm": "product",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.0,
        "lb_fuzzy_quantifier_beta": 0.65,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.0,
        "ub_fuzzy_quantifier_beta": 0.9,
    },
    "owafrs": {
        "model": "owafrs",
        "similarity": "gaussian",
        "similarity_sigma": 0.55,
        "similarity_tnorm": "minimum",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "lb_owa_method_name": "linear",
        "ub_owa_method_name": "exponential",
        "ub_owa_method_base": 1.25,
    },
}


RESULT_FIELDS = ("lower", "upper", "boundary", "positive_region")
WRAPPER_FUNCTIONS = {
    "lower": compute_lower_approximation,
    "upper": compute_upper_approximation,
    "boundary": compute_boundary_region,
    "positive_region": compute_positive_region,
}


def _assert_approximation_results_match(actual, expected, *, atol=1e-12):
    """Assert that two public approximation results contain equivalent arrays."""
    for field in RESULT_FIELDS:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=atol)


def _without_model(flat_config):
    """Return model alias and flat config without the public model key."""
    config = dict(flat_config)
    model = config.pop("model")
    return model, config


def _assert_mapping_subset(expected_subset, actual_mapping):
    """Assert recursively that all expected mapping items appear in actual_mapping."""
    for key, expected_value in expected_subset.items():
        assert key in actual_mapping
        actual_value = actual_mapping[key]
        if isinstance(expected_value, dict):
            assert isinstance(actual_value, dict)
            _assert_mapping_subset(expected_value, actual_value)
        else:
            assert actual_value == expected_value


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
@pytest.mark.parametrize("block_size", [1, 2, 3, 20])
def test_blockwise_public_approximations_match_dense_for_all_models_and_block_sizes(model_name, flat_config, block_size):
    model, kwargs = _without_model(flat_config)

    dense = compute_approximations(X_INTEGRATION, Y_INTEGRATION, model=model, engine="dense", **kwargs)
    blockwise = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=block_size,
        **kwargs,
    )

    assert model_name == model
    _assert_approximation_results_match(blockwise, dense)
    assert blockwise.engine == "blockwise"
    assert blockwise.backend == "numpy"
    assert blockwise.block_size == block_size
    assert blockwise.used_blockwise is True
    assert blockwise.used_gpu_similarity_blocks is False
    assert blockwise.similarity_matrix is None


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
@pytest.mark.parametrize("engine_alias", ["blockwise", "chunkwise", "blocked", " BLOCKWISE "])
def test_blockwise_public_approximations_accept_engine_aliases_for_all_models(model_name, flat_config, engine_alias):
    model, kwargs = _without_model(flat_config)

    dense = compute_approximations(X_INTEGRATION, Y_INTEGRATION, model=model, engine="dense", **kwargs)
    aliased = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine=engine_alias,
        block_size=2,
        **kwargs,
    )

    assert model_name == model
    assert aliased.engine == "blockwise"
    _assert_approximation_results_match(aliased, dense)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_nested_config_matches_flat_config_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    nested_config = normalize_flat_config_to_nested({"type": model, **kwargs})

    flat = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        **kwargs,
    )
    nested = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        config=nested_config,
    )

    assert model_name == model
    _assert_approximation_results_match(nested, flat)
    _assert_mapping_subset(nested_config, nested.config)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_do_not_mutate_nested_config_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    nested_config = normalize_flat_config_to_nested({"type": model, **kwargs})
    before = deepcopy(nested_config)

    _ = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=3,
        config=nested_config,
    )

    assert model_name == model
    assert nested_config == before


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_return_similarity_matrix_when_requested_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    expected_matrix = build_similarity_matrix(X_INTEGRATION, **kwargs)

    result = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
        **kwargs,
    )

    assert model_name == model
    assert result.similarity_matrix is not None
    np.testing.assert_allclose(result.similarity_matrix, expected_matrix, atol=1e-12)
    np.testing.assert_allclose(np.diag(result.similarity_matrix), np.ones(X_INTEGRATION.shape[0]), atol=0.0)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
@pytest.mark.parametrize("field,wrapper", WRAPPER_FUNCTIONS.items())
def test_blockwise_public_wrapper_functions_match_compute_approximations_for_all_models(model_name, flat_config, field, wrapper):
    model, kwargs = _without_model(flat_config)
    full_result = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        **kwargs,
    )

    wrapped = wrapper(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        **kwargs,
    )

    assert model_name == model
    np.testing.assert_allclose(wrapped, getattr(full_result, field), atol=1e-12)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_dense_public_approximations_accept_precomputed_similarity_matrix_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    similarity_matrix = build_similarity_matrix(X_INTEGRATION, **kwargs)

    from_x = compute_approximations(X_INTEGRATION, Y_INTEGRATION, model=model, engine="dense", **kwargs)
    from_matrix = compute_approximations(
        None,
        Y_INTEGRATION,
        model=model,
        engine="dense",
        similarity_matrix=similarity_matrix,
        **kwargs,
    )

    assert model_name == model
    _assert_approximation_results_match(from_matrix, from_x)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_reject_precomputed_similarity_matrix_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    similarity_matrix = build_similarity_matrix(X_INTEGRATION, **kwargs)

    with pytest.raises(ValueError, match="does not accept precomputed similarity_matrix"):
        compute_approximations(
            None,
            Y_INTEGRATION,
            model=model,
            engine="blockwise",
            block_size=2,
            similarity_matrix=similarity_matrix,
            **kwargs,
        )

    assert model_name == model


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_reject_label_length_mismatch_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)

    with pytest.raises(ValueError, match="Length of labels must match"):
        compute_approximations(
            X_INTEGRATION,
            Y_INTEGRATION[:-1],
            model=model,
            engine="blockwise",
            block_size=2,
            **kwargs,
        )

    assert model_name == model


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_reject_multidimensional_labels_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)

    with pytest.raises(ValueError, match="labels must be a 1D"):
        compute_approximations(
            X_INTEGRATION,
            Y_INTEGRATION.reshape(1, -1),
            model=model,
            engine="blockwise",
            block_size=2,
            **kwargs,
        )

    assert model_name == model


@pytest.mark.parametrize("invalid_block_size", [0, -1, 1.5, "2", True])
def test_blockwise_public_approximations_propagate_block_size_validation(invalid_block_size):
    with pytest.raises((TypeError, ValueError)):
        compute_approximations(
            X_INTEGRATION,
            Y_INTEGRATION,
            model="itfrs",
            similarity="linear",
            engine="blockwise",
            block_size=invalid_block_size,
        )


@pytest.mark.parametrize("invalid_engine", ["unknown", "", None, 123])
def test_public_approximations_reject_invalid_engine_aliases(invalid_engine):
    with pytest.raises((TypeError, ValueError)):
        compute_approximations(
            X_INTEGRATION,
            Y_INTEGRATION,
            model="itfrs",
            similarity="linear",
            engine=invalid_engine,
        )


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_approximations_reject_nested_config_mixed_with_flat_kwargs_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)
    nested_config = normalize_flat_config_to_nested({"type": model, **kwargs})

    with pytest.raises(ValueError, match="Do not mix nested config"):
        compute_approximations(
            X_INTEGRATION,
            Y_INTEGRATION,
            model=model,
            engine="blockwise",
            block_size=2,
            config=nested_config,
            similarity_sigma=0.99,
        )

    assert model_name == model


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_fake_cupy_path_matches_dense_and_reports_backend_metadata_for_all_models(
    monkeypatch,
    model_name,
    flat_config,
):
    install_fake_cupy_module(monkeypatch)
    model, kwargs = _without_model(flat_config)

    dense = compute_approximations(X_INTEGRATION, Y_INTEGRATION, model=model, engine="dense", **kwargs)
    gpu_like = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        backend="cupy",
        block_size=2,
        **kwargs,
    )

    _assert_approximation_results_match(gpu_like, dense)
    assert model_name == model
    assert gpu_like.engine == "blockwise"
    assert gpu_like.backend == "cupy"
    assert gpu_like.block_size == 2
    assert gpu_like.used_blockwise is True
    assert gpu_like.used_gpu_similarity_blocks is True
    assert gpu_like.used_gpu_approximation_accumulators is (model != "owafrs")


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_fake_cupy_path_can_return_numpy_similarity_matrix_for_all_models(
    monkeypatch,
    model_name,
    flat_config,
):
    install_fake_cupy_module(monkeypatch)
    model, kwargs = _without_model(flat_config)
    expected_matrix = build_similarity_matrix(X_INTEGRATION, **kwargs)

    result = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        backend="cupy",
        block_size=2,
        return_similarity_matrix=True,
        **kwargs,
    )

    assert model_name == model
    assert isinstance(result.similarity_matrix, np.ndarray)
    np.testing.assert_allclose(result.similarity_matrix, expected_matrix, atol=1e-12)


@pytest.mark.parametrize("model_name,flat_config", MODEL_CONFIGS.items())
def test_blockwise_public_result_as_dict_preserves_similarity_engine_metadata_for_all_models(model_name, flat_config):
    model, kwargs = _without_model(flat_config)

    result = compute_approximations(
        X_INTEGRATION,
        Y_INTEGRATION,
        model=model,
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
        **kwargs,
    )
    payload = result.as_dict(include_similarity_matrix=True)

    assert model_name == model
    assert payload["engine"] == "blockwise"
    assert payload["backend"] == "numpy"
    assert payload["block_size"] == 2
    assert payload["used_blockwise"] is True
    assert payload["used_gpu_similarity_blocks"] is False
    assert payload["used_gpu_approximation_accumulators"] is False
    np.testing.assert_allclose(payload["similarity_matrix"], result.similarity_matrix, atol=0.0)
