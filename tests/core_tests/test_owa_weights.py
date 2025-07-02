"""
@file test_owa_weights.py
@brief Pytest unit tests for OWAWeights and its subclasses.

This test suite verifies:
- Output correctness of each weighting strategy
- Param validation for parameterized strategies
- Consistency across scalar, vector, and matrix inputs
- Registry-based instantiation
- to_dict / from_dict serialization
- Reflection metadata and docstring availability

##############################################
# ✅ Summary of Test Coverage
# - Scalar output shape, sum to 1.0
# - Correct ascending and descending order
# - create() and from_dict() correctness
# - Parameter validation triggers
# - Distinct instances from different construction paths
# - describe_params_detailed and help()
##############################################

@example
>>> strategy = OWAWeights.create("linear")
>>> weights = strategy.weights(5, order='asc')
>>> assert np.isclose(weights.sum(), 1.0)
"""

import pytest
import numpy as np
from FRsutils.core.owa_weights import OWAWeights
from tests import synthetic_data_store as ds
import time

data_entry = ds.owa_weights_testing_testsets()[0]  # single dictionary wrapper

@pytest.mark.parametrize("name", list(OWAWeights.list_available().keys()))
def test_weights_sum_to_1(name):
    """Test that all OWA strategies produce weights summing to 1 and within [0,1]."""
    obj = OWAWeights.create(name)
    for order in ["asc", "desc"]:
        w = obj.weights(10, order=order)
        assert np.isclose(np.sum(w), 1.0), f"Sum not 1 for {name} in {order} order"
        assert len(w) == 10
        assert np.all((w >= 0) & (w <= 1))


@pytest.mark.parametrize("name", list(OWAWeights.list_available().keys()))
def test_ascending_vs_descending_order(name):
    """Ensure ascending order is the reverse of descending for all OWA strategies."""
    obj = OWAWeights.create(name)
    asc = obj.weights(10, order="asc")
    desc = obj.weights(10, order="desc")
    np.testing.assert_allclose(asc[::-1], desc, atol=1e-7)

    sorted_asc = np.sort(asc)
    np.testing.assert_allclose(asc, sorted_asc, atol=1e-7)

    sorted_desc = np.sort(desc)[::-1]
    np.testing.assert_allclose(desc, sorted_desc, atol=1e-7)



@pytest.mark.parametrize("name", ["linear", "harmonic", "logarithmic"])
def test_paramless_strategies_serialization(name):
    """Test that serialization roundtrip returns consistent outputs for non-param strategies."""
    obj = OWAWeights.create(name)
    d = obj.to_dict()
    reconstructed = OWAWeights.from_dict(d)
    np.testing.assert_array_almost_equal(obj.weights(10), reconstructed.weights(10))


@pytest.mark.parametrize("base", [1.5, 2.0, 5.0])
def test_exponential_strategy_param_roundtrip(base):
    """Test exponential strategy retains 'base' across serialization."""
    obj = OWAWeights.create("exponential", base=base)
    d = obj.to_dict()
    assert d["params"]["base"] == base
    new_obj = OWAWeights.from_dict(d)
    np.testing.assert_array_almost_equal(obj.weights(10), new_obj.weights(10))


def test_exponential_strategy_invalid_base():
    """Test that exponential strategy fails with base <= 1."""
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", base=1.0)
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", base=-5)
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", base=0)
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", base=None)
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", base='fodor')


def test_invalid_order_string_and_None():
    """Test ValueError is raised for invalid weight order value."""
    strategy = OWAWeights.create("linear")
    with pytest.raises(ValueError):
        strategy.weights(5, order="invalid")
    with pytest.raises(ValueError):
        strategy.weights(5, order=None)


def test_invalid_n():
    """Test ValueError is raised for invalid 'n' values in weights()."""
    strategy = OWAWeights.create("linear")
    with pytest.raises(ValueError):
        strategy.weights(0)
    with pytest.raises(ValueError):
        strategy.weights(-3)


def test_help_and_describe_params():
    """Test each strategy has valid docstring and parameter description."""
    for name in OWAWeights.list_available().keys():
        obj = OWAWeights.create(name)
        assert isinstance(obj.help(), str)
        details = obj.describe_params_detailed()
        assert isinstance(details, dict)
        for param in obj._get_params().keys():
            assert param in details


def test_registry_instances_are_distinct():
    """Ensure different construction paths produce distinct objects in memory."""
    obj1 = OWAWeights.create("linear")
    obj2 = OWAWeights.from_dict(obj1.to_dict())
    obj3 = OWAWeights.get_class("linear")()
    assert id(obj1) != id(obj2)
    assert id(obj2) != id(obj3)
    assert id(obj1) != id(obj3)


def test_registry_get_class_and_name():
    """Ensure registry provides consistent class and name lookup."""
    for name in OWAWeights.list_available().keys():
        cls = OWAWeights.get_class(name)
        instance = cls()
        reg_name = OWAWeights.get_registered_name(instance)
        assert isinstance(reg_name, str)
        assert reg_name == name or reg_name in OWAWeights.list_available()[name]



def test_edge_case_single_element():
    """
    Ensure all strategies return [1.0] when n=1
    """
    for name in OWAWeights.list_available().keys():
        obj = OWAWeights.create(name) if name != "exponential" else OWAWeights.create(name, base=2.0)
        out = obj.weights(1, order='asc')
        np.testing.assert_allclose(out, [1.0], atol=1e-10)
        out_desc = obj.weights(1, order='desc')
        np.testing.assert_allclose(out_desc, [1.0], atol=1e-10)


@pytest.mark.parametrize("name", ["linear", "harmonic", "log", "exponential"])
def test_owa_weight_generation_large_n(name):
    """
    Benchmark large-n OWA weight generation possibility.
    """
    strategy = OWAWeights.create(name, base=2.0) if name == "exponential" else OWAWeights.create(name)
    n = 100_000

    if(name == "exponential"):
        with pytest.raises(ValueError):
            weights = strategy.weights(n, order='asc')

    else:
        weights = strategy.weights(n, order='asc')

        sum_w = weights.sum()

        assert np.isclose(sum_w, 1.0, atol=1e-6)
        assert len(weights) == n



#region <test with data>
###############################################
###                                         ###
###             test with data              ###
###                                         ###
###############################################

@pytest.mark.parametrize("name", ["linear", "harmonic", "log"])
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_owa_weights_against_expected_vectors(name, order):
    """
    Test OWAWeights subclasses (linear, harmonic, log) against known expected vectors.
    Uses both ascending and descending order.
    """
    strategy = OWAWeights.create(name)
    expected_sets = data_entry[name][f"{order}_OWA"]

    for key, expected in expected_sets.items():
        n = int(key.split("_")[-1])
        actual = strategy.weights(n, order=order)
        np.testing.assert_allclose(actual, expected, atol=1e-8, err_msg=f"Mismatch for {name}, {order}, {key}")


@pytest.mark.parametrize("dataset_id", ["dataset_1", "dataset_2"])
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_exponential_weights_against_expected(dataset_id, order):
    """
    Test Exponential OWAWeights with base=2.0 and base=3.0 against stored outputs.
    """
    exp_data = data_entry["exp"][dataset_id]
    base = exp_data["base"]
    strategy = OWAWeights.create("exponential", base=base)

    for key, expected in exp_data[f"{order}_OWA"].items():
        n = int(key.split("_")[-1])
        actual = strategy.weights(n, order=order)
        np.testing.assert_allclose(actual, expected, atol=1e-8, err_msg=f"Mismatch for exp-{base}, {order}, {key}")
#endregion