import numpy as np
import pytest
from FRsutils.core.similarities import Similarity, calculate_similarity_matrix, build_similarity_matrix
from FRsutils.core.tnorms import TNorm
from tests import synthetic_data_store as ds
from FRsutils.utils.logger.logger_util import get_logger


logger = get_logger(env="test",
                    experiment_name="test_similarities1")
similarity_testsets = ds.get_similarity_testing_testsets()
registered_similarities = Similarity.list_available()


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
