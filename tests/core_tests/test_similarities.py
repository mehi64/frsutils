# SPDX-License-Identifier: BSD-3-Clause
"""Tests for similarity functions and matrix construction."""

import numpy as np
import pytest
from frsutils.core.similarities import (
    GaussianSimilarity,
    LinearSimilarity,
    Similarity,
    build_similarity_matrix,
    calculate_similarity_matrix,
)
from frsutils.core.tnorms import TNorm
from tests import reference_data_store as ds
from tests._cupy_test_support import require_usable_cupy
from frsutils.utils.logger.logger_util import get_logger


logger = get_logger(env="test",
                    experiment_name="test_similarities1")
similarity_testsets = ds.get_similarity_testing_testsets()
registered_similarities = Similarity.list_available()

def _cupy_to_numpy(value, cp):
    """Convert a CuPy scalar or array result to a NumPy-compatible value."""
    return cp.asnumpy(cp.asarray(value))


def test_linear_similarity_formula_matches_definition():
    diff = np.array([[0.0, 0.25, 1.0, 1.5, -0.5]])
    expected = np.maximum(0.0, 1.0 - np.abs(diff))

    result = LinearSimilarity()._compute(diff)

    np.testing.assert_allclose(result, expected)


def test_gaussian_similarity_formula_matches_definition_with_default_sigma():
    diff = np.array([[0.0, 0.1, -0.2, 0.5]])
    sigma = 0.1
    expected = np.exp(-((diff ** 2) / (2.0 * sigma ** 2)))

    result = GaussianSimilarity()._compute(diff)

    np.testing.assert_allclose(result, expected)


def test_gaussian_similarity_formula_matches_definition_with_custom_sigma():
    diff = np.array([[0.0, 0.25, -0.5, 1.0]])
    sigma = 0.67
    expected = np.exp(-((diff ** 2) / (2.0 * sigma ** 2)))

    result = GaussianSimilarity(sigma=sigma)._compute(diff)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_similarity_call_computes_pairwise_values_for_broadcastable_2d_arrays(similarity):
    x = np.array([[0.0], [0.25], [0.75]])
    y = np.array([[0.0, 0.5, 1.0]])
    expected = similarity._compute(x - y)

    result = similarity(x, y)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_compute_backend_numpy_matches_numpy_compute(similarity):
    diff = np.array([[0.0, 0.25], [-0.5, 1.0]])
    expected = similarity._compute(diff)

    result = similarity.compute_backend(diff, xp=np)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_compute_backend_numpy_preserves_broadcasted_shape(similarity):
    column = np.array([[0.0], [0.5], [1.0]])
    row = np.array([[0.25, 0.75]])
    diff = column - row

    result = similarity.compute_backend(diff, xp=np)

    assert result.shape == (3, 2)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@pytest.mark.parametrize("sigma", [0.0, -0.1, None, "0.1"])
def test_gaussian_similarity_rejects_invalid_sigma_values(sigma):
    with pytest.raises(ValueError, match="positive number"):
        GaussianSimilarity(sigma=sigma)


@pytest.mark.parametrize("sigma", [0.0, -0.1, None, "0.1"])
def test_similarity_factory_rejects_invalid_gaussian_sigma_values(sigma):
    with pytest.raises(ValueError, match="positive number"):
        Similarity.create("gaussian", sigma=sigma)


def test_similarity_factory_rejects_unknown_similarity_name():
    with pytest.raises(ValueError, match="Unknown alias"):
        Similarity.create("unknown_similarity")


def test_build_similarity_matrix_rejects_unknown_similarity_name():
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="Unknown alias"):
        build_similarity_matrix(X, similarity="unknown_similarity")


def test_build_similarity_matrix_rejects_unknown_similarity_tnorm_name():
    X = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="Unknown alias"):
        build_similarity_matrix(X, similarity="linear", similarity_tnorm="unknown_tnorm")


@pytest.mark.parametrize(
    "X",
    [
        None,
        np.array([1.0, 2.0]),
        np.array([[[0.0], [1.0]]]),
        [[0.0], [1.0]],
    ],
)
def test_build_similarity_matrix_rejects_non_2d_numpy_inputs(X):
    with pytest.raises(ValueError, match="X must be a 2D NumPy array"):
        build_similarity_matrix(X, similarity="linear", similarity_tnorm="minimum")


@pytest.mark.parametrize(
    "X",
    [
        None,
        np.array([1.0, 2.0]),
        np.array([[[0.0], [1.0]]]),
        [[0.0], [1.0]],
    ],
)
def test_calculate_similarity_matrix_rejects_non_2d_numpy_inputs(X):
    with pytest.raises(ValueError, match="X must be a 2D NumPy array"):
        calculate_similarity_matrix(X, LinearSimilarity(), TNorm.create("minimum"))


def test_similarity_compute_rejects_non_numpy_diff_input():
    with pytest.raises(TypeError, match="NumPy array"):
        LinearSimilarity()._compute([[0.0, 0.5]])


@pytest.mark.parametrize(
    "diff",
    [
        np.array(0.0),
        np.array([0.0, 0.5]),
        np.zeros((1, 1, 1)),
    ],
)
def test_compute_backend_rejects_non_2d_diff(diff):
    with pytest.raises(ValueError, match="Expected a 2D pairwise difference matrix"):
        LinearSimilarity().compute_backend(diff, xp=np)


def test_build_similarity_matrix_accepts_gauss_alias():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])

    result = build_similarity_matrix(
        X,
        similarity="gauss",
        sigma=0.67,
        similarity_tnorm="minimum",
    )
    expected = build_similarity_matrix(
        X,
        similarity="gaussian",
        sigma=0.67,
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_accepts_nested_linear_config():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])
    config = {
        "similarity": {"name": "linear", "params": {}},
        "similarity_tnorm": {"name": "minimum", "params": {}},
    }

    result = build_similarity_matrix(X, config=config)
    expected = calculate_similarity_matrix(
        X,
        LinearSimilarity(),
        TNorm.create("minimum"),
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_accepts_nested_gaussian_config():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])
    config = {
        "similarity": {"name": "gaussian", "params": {"sigma": 0.67}},
        "similarity_tnorm": {"name": "product", "params": {}},
    }

    result = build_similarity_matrix(X, config=config)
    expected = calculate_similarity_matrix(
        X,
        GaussianSimilarity(sigma=0.67),
        TNorm.create("product"),
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_accepts_new_flat_similarity_names():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])

    result = build_similarity_matrix(
        X,
        similarity_name="gaussian",
        similarity_sigma=0.67,
        similarity_tnorm_name="minimum",
    )
    expected = calculate_similarity_matrix(
        X,
        GaussianSimilarity(sigma=0.67),
        TNorm.create("minimum"),
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_accepts_legacy_gaussian_similarity_sigma_alias():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])

    result = build_similarity_matrix(
        X,
        similarity="gaussian",
        gaussian_similarity_sigma=0.67,
        similarity_tnorm="minimum",
    )
    expected = build_similarity_matrix(
        X,
        similarity="gaussian",
        sigma=0.67,
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_routes_flat_similarity_tnorm_parameters():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])

    result = build_similarity_matrix(
        X,
        similarity="linear",
        similarity_tnorm_name="yager",
        similarity_tnorm_p=2.5,
    )
    expected = calculate_similarity_matrix(
        X,
        LinearSimilarity(),
        TNorm.create("yager", p=2.5),
    )

    np.testing.assert_allclose(result, expected)


def test_build_similarity_matrix_prefers_nested_config_over_conflicting_kwargs():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])
    config = {
        "similarity": {"name": "linear", "params": {}},
        "similarity_tnorm": {"name": "minimum", "params": {}},
    }

    result = build_similarity_matrix(
        X,
        config=config,
        similarity="gaussian",
        sigma=0.1,
        similarity_tnorm="product",
    )
    expected = calculate_similarity_matrix(
        X,
        LinearSimilarity(),
        TNorm.create("minimum"),
    )

    np.testing.assert_allclose(result, expected)


def test_nested_similarity_params_override_legacy_sigma_kwargs():
    X = np.array([[0.0, 0.2], [0.3, 0.8], [1.0, 0.4]])
    config = {
        "similarity": {"name": "gaussian", "params": {"sigma": 0.67}},
        "similarity_tnorm": {"name": "minimum", "params": {}},
    }

    result = build_similarity_matrix(X, config=config, sigma=99.0)
    expected = calculate_similarity_matrix(
        X,
        GaussianSimilarity(sigma=0.67),
        TNorm.create("minimum"),
    )
    conflicting = calculate_similarity_matrix(
        X,
        GaussianSimilarity(sigma=99.0),
        TNorm.create("minimum"),
    )

    np.testing.assert_allclose(result, expected)
    assert not np.allclose(result, conflicting)


def _assert_similarity_matrix_invariants(matrix, n_samples):
    assert matrix.shape == (n_samples, n_samples)
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), np.ones(n_samples))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"similarity": "linear", "similarity_tnorm": "minimum"},
        {"similarity": "linear", "similarity_tnorm": "product"},
        {"similarity": "gaussian", "sigma": 0.67, "similarity_tnorm": "minimum"},
        {"similarity": "gaussian", "sigma": 0.67, "similarity_tnorm": "product"},
        {
            "similarity": "gaussian",
            "sigma": 0.67,
            "similarity_tnorm_name": "yager",
            "similarity_tnorm_p": 2.5,
        },
    ],
)
def test_build_similarity_matrix_satisfies_core_matrix_invariants(kwargs):
    X = np.array(
        [
            [0.0, 0.2, 0.7],
            [0.3, 0.8, 0.4],
            [1.0, 0.4, 0.1],
            [0.6, 0.6, 0.6],
        ]
    )

    result = build_similarity_matrix(X, **kwargs)

    _assert_similarity_matrix_invariants(result, n_samples=X.shape[0])


def test_calculate_similarity_matrix_satisfies_core_matrix_invariants():
    X = np.array(
        [
            [0.0, 0.2],
            [0.3, 0.8],
            [1.0, 0.4],
        ]
    )

    result = calculate_similarity_matrix(
        X,
        GaussianSimilarity(sigma=0.67),
        TNorm.create("product"),
    )

    _assert_similarity_matrix_invariants(result, n_samples=X.shape[0])


def test_similarity_matrix_for_single_sample_is_one_by_one_identity():
    X = np.array([[0.25, 0.75, 1.0]])

    result = build_similarity_matrix(
        X,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(result, np.array([[1.0]]))


def test_similarity_matrix_for_empty_sample_axis_is_empty_square_matrix():
    X = np.empty((0, 3))

    result = build_similarity_matrix(
        X,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    assert result.shape == (0, 0)
    assert result.size == 0


def test_zero_feature_matrix_preserves_sample_axis_as_empty_conjunction_identity():
    X = np.empty((3, 0))

    result = calculate_similarity_matrix(
        X,
        LinearSimilarity(),
        TNorm.create("minimum"),
    )

    np.testing.assert_allclose(result, np.ones((3, 3)))


@pytest.mark.parametrize(
    "similarity,tnorm",
    [
        (LinearSimilarity(), TNorm.create("minimum")),
        (LinearSimilarity(), TNorm.create("product")),
        (GaussianSimilarity(sigma=0.67), TNorm.create("minimum")),
        (GaussianSimilarity(sigma=0.67), TNorm.create("product")),
        (GaussianSimilarity(sigma=0.67), TNorm.create("yager", p=2.5)),
    ],
)
def test_calculate_similarity_matrix_supports_multiple_similarity_tnorm_pairs(
    similarity,
    tnorm,
):
    X = np.array(
        [
            [0.0, 0.2, 0.7],
            [0.3, 0.8, 0.4],
            [1.0, 0.4, 0.1],
        ]
    )

    result = calculate_similarity_matrix(X, similarity, tnorm)

    _assert_similarity_matrix_invariants(result, n_samples=X.shape[0])


@pytest.mark.parametrize("testset", similarity_testsets)
def test_calculate_similarity_matrix(testset):
    """
    check the results with data provided
    """
    for key, expected in testset["expected"].items():
        parts = key.split("_")
        sim_type = parts[3]
        tnorm_type = parts[5].replace("tnorm", "")

        sim_obj = Similarity.create(sim_type, **({"sigma": testset.get("sigma_for_gaussian_similarity", 0.67)} if sim_type == "gaussian" else {}))
        tnorm_obj = TNorm.create(tnorm_type)

        result = calculate_similarity_matrix(testset["X"], sim_obj, tnorm_obj)
        np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("testset", similarity_testsets)
def test_build_similarity_matrix_matches_expected(testset):
    for key, expected in testset["expected"].items():
        parts = key.split("_")
        sim_type = parts[3]
        tnorm_type = parts[5].replace("tnorm", "")
        kwargs = {
            "similarity": sim_type,
            "similarity_tnorm": tnorm_type
        }
        if sim_type == "gaussian":
            kwargs["sigma"] = testset.get("sigma_for_gaussian_similarity", 0.67)
        result = build_similarity_matrix(testset["X"], **kwargs)
        np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_compute_backend_cupy_matches_numpy_for_matrices(similarity):
    cp = require_usable_cupy()
    diff_np = np.array([[0.0, 0.25, -0.5], [0.75, -1.0, 0.33]])

    result_cp = similarity.compute_backend(cp.asarray(diff_np), xp=cp)
    expected = similarity.compute_backend(diff_np, xp=np)

    assert isinstance(result_cp, cp.ndarray)
    assert result_cp.shape == expected.shape
    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-12)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_compute_backend_cupy_preserves_column_row_broadcasting(similarity):
    cp = require_usable_cupy()
    column_np = np.array([[0.0], [0.25], [0.5], [0.75]])
    row_np = np.array([[0.1, 0.4, 0.7]])
    diff_np = column_np - row_np

    result_cp = similarity.compute_backend(
        cp.asarray(column_np) - cp.asarray(row_np),
        xp=cp,
    )
    expected = similarity.compute_backend(diff_np, xp=np)

    assert result_cp.shape == expected.shape
    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-12)


@pytest.mark.parametrize(
    "similarity",
    [LinearSimilarity(), GaussianSimilarity(sigma=0.67)],
)
def test_compute_backend_cupy_values_stay_in_unit_interval(similarity):
    cp = require_usable_cupy()
    diff_np = np.linspace(-2.0, 2.0, num=25).reshape(5, 5)

    result_np = _cupy_to_numpy(
        similarity.compute_backend(cp.asarray(diff_np), xp=cp),
        cp,
    )

    assert np.all(result_np >= 0.0)
    assert np.all(result_np <= 1.0)


def test_linear_similarity_round_trips_through_dict_without_parameters():
    obj = LinearSimilarity()

    serialized = obj.to_dict()
    restored = Similarity.from_dict(serialized)

    assert serialized == {"type": "LinearSimilarity", "name": "linear", "params": {}}
    assert isinstance(restored, LinearSimilarity)
    assert restored.name == obj.name


def test_gaussian_similarity_round_trips_through_dict_with_sigma_parameter():
    obj = GaussianSimilarity(sigma=0.67)

    serialized = obj.to_dict()
    restored = Similarity.from_dict(serialized)

    assert serialized["type"] == "GaussianSimilarity"
    assert serialized["name"] == "gaussian"
    assert serialized["params"] == {"sigma": 0.67}
    assert isinstance(restored, GaussianSimilarity)
    assert restored.sigma == pytest.approx(0.67)


def test_similarity_from_dict_preserves_gaussian_behavior_after_round_trip():
    diff = np.array([[0.0, 0.25, -0.5]])
    original = GaussianSimilarity(sigma=0.67)

    restored = Similarity.from_dict(original.to_dict())

    np.testing.assert_allclose(restored._compute(diff), original._compute(diff))


@pytest.mark.parametrize(
    "spec,expected_type,expected_params",
    [
        ({"name": "linear", "params": {}}, LinearSimilarity, {}),
        ({"name": "gaussian", "params": {"sigma": 0.67}}, GaussianSimilarity, {"sigma": 0.67}),
        ({"name": "gauss", "params": {"sigma": 0.67}}, GaussianSimilarity, {"sigma": 0.67}),
        ({"type": "linear"}, LinearSimilarity, {}),
        ({"type": "gaussian", "sigma": 0.67}, GaussianSimilarity, {"sigma": 0.67}),
    ],
)
def test_similarity_create_from_spec_accepts_supported_serialized_forms(
    spec,
    expected_type,
    expected_params,
):
    obj = Similarity.create_from_spec(spec)

    assert isinstance(obj, expected_type)
    for name, expected in expected_params.items():
        assert getattr(obj, name) == pytest.approx(expected)


def test_similarity_create_from_spec_returns_existing_similarity_instance():
    obj = GaussianSimilarity(sigma=0.67)

    result = Similarity.create_from_spec(obj)

    assert result is obj


@pytest.mark.parametrize("sim_name", list(registered_similarities.keys()))
def test_to_dict_and_from_dict(sim_name):
    params = {"sigma": 0.67} if sim_name == "gaussian" else {}
    obj = Similarity.create(sim_name, **params)
    d = obj.to_dict()
    new_obj = Similarity.from_dict(d)
    assert isinstance(new_obj, Similarity)
    assert new_obj.name == obj.name


@pytest.mark.parametrize("sim_name", list(registered_similarities.keys()))
def test_describe_params_and_registered_name(sim_name):
    params = {"sigma": 0.67} if sim_name == "gaussian" else {}
    cls = Similarity.get_class(sim_name)
    instance = cls(**params)

    name = Similarity.get_registered_name(instance)
    assert isinstance(name, str)
    described = instance.describe_params_detailed()
    for k in instance._get_params():
        assert k in described
    logger.info(sim_name + ', registered_name:' + str(name))


@pytest.mark.parametrize("sim_name", list(registered_similarities.keys()))
def test_full_range_combinations(sim_name):
    obj = Similarity.create(sim_name, **({"sigma": 0.67} if sim_name == "gaussian" else {}))
    values = np.round(np.arange(0.0, 1.01, 0.01), 2)
    a, b = np.meshgrid(values, values)
    diff = a - b
    diff = diff.reshape(-1, 1)
    try:
        sim_vals = obj._compute(diff)
        assert np.all(sim_vals >= 0.0) and np.all(sim_vals <= 1.0)
    except Exception as e:
        pytest.fail(f"Exception in {sim_name}: {str(e)}")


def test_invalid_similarity_diff_dimension():
    sim = Similarity.create("linear")
    with pytest.raises(ValueError, match="Expected a 2D pairwise difference matrix"):
        sim._compute(np.array([0.1, 0.2, 0.3]))


def test_invalid_build_similarity_matrix_input():
    with pytest.raises(ValueError, match="X must be a 2D NumPy array"):
        build_similarity_matrix(np.array([1.0, 2.0]), similarity="linear", similarity_tnorm="minimum")
