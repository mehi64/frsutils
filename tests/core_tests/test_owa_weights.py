# SPDX-License-Identifier: BSD-3-Clause
"""Regression and contract tests for OWA weight strategies."""

import numpy as np
import pytest
from frsutils.core.owa_weights import (
    ExponentialOWAWeights,
    HarmonicOWAWeights,
    LinearOWAWeights,
    LogarithmicOWAWeights,
    OWAWeights,
)
from tests import synthetic_data_store as ds

data_entry = ds.owa_weights_testing_testsets()[0]  # single dictionary wrapper


def test_registry_exposes_expected_primary_names_and_aliases():
    """Ensure the public OWA registry exposes the expected stable strategies."""
    available = OWAWeights.list_available()

    assert available["linear"] == ["linear"]
    assert available["exponential"] == ["exponential", "exp"]
    assert available["harmonic"] == ["harmonic", "harm"]
    assert available["logarithmic"] == ["logarithmic", "log"]


@pytest.mark.parametrize(
    ("canonical_name", "alias"),
    [
        ("exponential", "exp"),
        ("harmonic", "harm"),
        ("logarithmic", "log"),
    ],
)
def test_owa_aliases_generate_same_weights_as_canonical_names(canonical_name, alias):
    """Ensure registered aliases behave identically to their canonical names."""
    canonical = OWAWeights.create(canonical_name)
    alias_instance = OWAWeights.create(alias)

    for order in ["asc", "desc"]:
        np.testing.assert_allclose(
            canonical.weights(10, order=order),
            alias_instance.weights(10, order=order),
            atol=1e-10,
        )


@pytest.mark.parametrize(
    ("mixed_case_name", "expected_class"),
    [
        ("LINEAR", LinearOWAWeights),
        ("Exp", ExponentialOWAWeights),
        ("LOG", LogarithmicOWAWeights),
    ],
)
def test_owa_factory_names_are_case_insensitive(mixed_case_name, expected_class):
    """Ensure factory lookup accepts mixed-case registered names."""
    assert isinstance(OWAWeights.create(mixed_case_name), expected_class)


@pytest.mark.parametrize(
    "factory_call",
    [
        lambda: OWAWeights.create("unknown"),
        lambda: OWAWeights.get_class("unknown"),
    ],
)
def test_owa_factory_rejects_unknown_aliases(factory_call):
    """Ensure factory lookup fails clearly for unregistered OWA aliases."""
    with pytest.raises(ValueError):
        factory_call()


@pytest.mark.parametrize(
    ("spec", "expected_class", "expected_params"),
    [
        ({"name": "linear", "params": {}}, LinearOWAWeights, {}),
        ({"name": "exponential", "params": {"base": 3.0}}, ExponentialOWAWeights, {"base": 3.0}),
        ({"type": "exponential", "base": 2.5}, ExponentialOWAWeights, {"base": 2.5}),
        ({"type": "harmonic"}, HarmonicOWAWeights, {}),
    ],
)
def test_owa_create_from_spec_builds_expected_strategy(spec, expected_class, expected_params):
    """Ensure nested and legacy compact specs instantiate OWA strategies."""
    strategy = OWAWeights.create_from_spec(spec)

    assert isinstance(strategy, expected_class)
    for param_name, expected_value in expected_params.items():
        assert getattr(strategy, param_name) == expected_value


def test_owa_create_from_spec_returns_existing_instance_unchanged():
    """Ensure direct OWA instances pass through create_from_spec unchanged."""
    strategy = ExponentialOWAWeights(base=4.0)

    assert OWAWeights.create_from_spec(strategy) is strategy


def test_owa_create_from_spec_returns_internal_instance_marker_unchanged():
    """Ensure internal instance markers pass through create_from_spec unchanged."""
    strategy = HarmonicOWAWeights()

    assert OWAWeights.create_from_spec({"__instance__": strategy}) is strategy


def test_owa_create_from_spec_accepts_none_as_missing_optional_component():
    """Ensure None keeps the shared factory convention for optional configs."""
    assert OWAWeights.create_from_spec(None) is None


@pytest.mark.parametrize(
    "spec",
    [
        {"name": "exponential", "params": 2.0},
        {"foo": "bar"},
        123,
    ],
)
def test_owa_create_from_spec_rejects_invalid_specs(spec):
    """Ensure malformed OWA specs fail at the factory boundary."""
    with pytest.raises(TypeError):
        OWAWeights.create_from_spec(spec)


def test_owa_create_from_spec_rejects_unknown_name_spec():
    """Ensure specs with unknown names fail with the registry error type."""
    with pytest.raises(ValueError):
        OWAWeights.create_from_spec({"name": "unknown", "params": {}})


def test_owa_create_uses_namespaced_parameters_for_constructor_args():
    """Ensure namespaced flat parameters are routed to the target constructor."""
    strategy = OWAWeights.create("exponential", namespace="lb", lb_base=3.0, ub_base=7.0)

    assert isinstance(strategy, ExponentialOWAWeights)
    assert strategy.base == 3.0


def test_owa_create_strict_mode_rejects_unused_parameters():
    """Ensure strict factory mode rejects unused constructor parameters."""
    with pytest.raises(ValueError):
        OWAWeights.create("exponential", strict=True, base=2.0, unused=123)


def test_owa_create_strict_mode_accepts_fully_consumed_parameters():
    """Ensure strict factory mode accepts valid constructor parameters."""
    strategy = OWAWeights.create("exponential", strict=True, base=2.0)

    assert isinstance(strategy, ExponentialOWAWeights)
    assert strategy.base == 2.0


@pytest.mark.parametrize(
    "strategy",
    [
        LinearOWAWeights(),
        ExponentialOWAWeights(base=2.0),
        HarmonicOWAWeights(),
        LogarithmicOWAWeights(),
    ],
)
def test_direct_strategy_constructors_generate_valid_weight_vectors(strategy):
    """Ensure public strategy constructors generate normalized NumPy vectors."""
    weights = strategy.weights(8, order="asc")

    assert isinstance(weights, np.ndarray)
    assert weights.shape == (8,)
    assert np.issubdtype(weights.dtype, np.floating)
    assert np.all(np.isfinite(weights))
    assert np.all((weights >= 0) & (weights <= 1))
    assert np.isclose(weights.sum(), 1.0)


def _normalized_expected_weights(raw_weights):
    """Return normalized expected OWA weights for formula-based tests."""
    raw_weights = np.asarray(raw_weights, dtype=np.longdouble)
    return raw_weights / raw_weights.sum()


def _order_expected_weights(weights, order):
    """Sort expected OWA weights according to the public order contract."""
    sorted_weights = np.sort(weights)
    if order == "desc":
        return sorted_weights[::-1]
    return sorted_weights


@pytest.mark.parametrize(
    ("name", "kwargs", "raw_weights_factory"),
    [
        (
            "linear",
            {},
            lambda n: np.arange(1, n + 1, dtype=np.longdouble),
        ),
        (
            "exponential",
            {"base": 3.0},
            lambda n: 3.0 ** np.arange(1, n + 1, dtype=np.longdouble),
        ),
        (
            "harmonic",
            {},
            lambda n: 1.0 / np.arange(1, n + 1, dtype=np.longdouble),
        ),
        (
            "logarithmic",
            {},
            lambda n: np.log(np.arange(1, n + 1, dtype=np.longdouble) + 1.0),
        ),
    ],
)
@pytest.mark.parametrize("order", ["asc", "desc"])
@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_owa_strategies_match_their_scientific_weight_formulas(
    name,
    kwargs,
    raw_weights_factory,
    order,
    n,
):
    """Ensure each OWA strategy follows its documented mathematical formula."""
    strategy = OWAWeights.create(name, **kwargs)
    expected = _normalized_expected_weights(raw_weights_factory(n))
    expected = _order_expected_weights(expected, order)

    actual = strategy.weights(n, order=order)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("linear", {}),
        ("exponential", {"base": 2.0}),
        ("harmonic", {}),
        ("logarithmic", {}),
    ],
)
@pytest.mark.parametrize("order", ["asc", "desc"])
def test_formula_based_weights_are_nonnegative_normalized_and_ordered(name, kwargs, order):
    """Ensure formula-based weights preserve OWA normalization and ordering contracts."""
    weights = OWAWeights.create(name, **kwargs).weights(20, order=order)

    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)

    if order == "asc":
        np.testing.assert_allclose(weights, np.sort(weights), rtol=1e-12, atol=1e-12)
    else:
        np.testing.assert_allclose(weights, np.sort(weights)[::-1], rtol=1e-12, atol=1e-12)


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


@pytest.mark.parametrize(
    ("strategy", "expected_schema"),
    [
        (
            LinearOWAWeights(),
            {"type": "LinearOWAWeights", "name": "linear", "params": {}},
        ),
        (
            ExponentialOWAWeights(base=3.0),
            {"type": "ExponentialOWAWeights", "name": "exponential", "params": {"base": 3.0}},
        ),
        (
            HarmonicOWAWeights(),
            {"type": "HarmonicOWAWeights", "name": "harmonic", "params": {}},
        ),
        (
            LogarithmicOWAWeights(),
            {"type": "LogarithmicOWAWeights", "name": "logarithmic", "params": {}},
        ),
    ],
)
def test_owa_strategy_to_dict_uses_exact_public_schema(strategy, expected_schema):
    """Ensure serialized OWA strategies expose the expected config schema."""
    assert strategy.to_dict() == expected_schema


@pytest.mark.parametrize(
    ("alias", "canonical_name"),
    [
        ("exp", "exponential"),
        ("harm", "harmonic"),
        ("log", "logarithmic"),
    ],
)
def test_owa_alias_serialization_uses_canonical_names(alias, canonical_name):
    """Ensure alias-created strategies serialize with canonical registry names."""
    strategy = OWAWeights.create(alias)

    assert strategy.to_dict()["name"] == canonical_name


@pytest.mark.parametrize(
    ("strategy", "n", "orders"),
    [
        (LinearOWAWeights(), 7, ["asc", "desc"]),
        (ExponentialOWAWeights(base=2.5), 7, ["asc", "desc"]),
        (HarmonicOWAWeights(), 7, ["asc", "desc"]),
        (LogarithmicOWAWeights(), 7, ["asc", "desc"]),
    ],
)
def test_owa_to_dict_create_from_spec_roundtrip_preserves_weight_behavior(strategy, n, orders):
    """Ensure to_dict output can rebuild an equivalent OWA strategy from a spec."""
    reconstructed = OWAWeights.create_from_spec(strategy.to_dict())

    assert reconstructed is not strategy
    assert isinstance(reconstructed, strategy.__class__)
    assert reconstructed.to_dict() == strategy.to_dict()

    for order in orders:
        np.testing.assert_allclose(
            reconstructed.weights(n, order=order),
            strategy.weights(n, order=order),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "LinearOWAWeights", "name": "linear", "params": {}},
        {"type": "ExponentialOWAWeights", "name": "exponential", "params": {"base": 4.0}},
        {"type": "HarmonicOWAWeights", "name": "harmonic", "params": {}},
        {"type": "LogarithmicOWAWeights", "name": "logarithmic", "params": {}},
    ],
)
def test_owa_from_dict_accepts_to_dict_compatible_schema(spec):
    """Ensure from_dict accepts the exact schema emitted by OWA strategies."""
    strategy = OWAWeights.from_dict(spec)

    assert strategy.to_dict() == spec


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


@pytest.mark.parametrize("base", [np.nan, np.inf, -np.inf, True])
def test_exponential_strategy_rejects_non_finite_and_boolean_base(base):
    """Ensure exponential OWA rejects unsafe base values at construction time."""
    with pytest.raises(ValueError):
        ExponentialOWAWeights(base=base)


def test_exponential_strategy_accepts_boundary_safe_n_and_rejects_unsafe_n():
    """Ensure the exponential strategy enforces its documented upper n limit."""
    strategy = ExponentialOWAWeights(base=2.0)

    weights = strategy.weights(20)
    assert weights.shape == (20,)
    assert np.isclose(weights.sum(), 1.0)

    with pytest.raises(ValueError):
        strategy.weights(21)


def test_invalid_order_string_and_None():
    """Test ValueError is raised for invalid weight order value."""
    strategy = OWAWeights.create("linear")
    with pytest.raises(ValueError):
        strategy.weights(5, order="invalid")
    with pytest.raises(ValueError):
        strategy.weights(5, order=None)


@pytest.mark.parametrize("order", ["ASC", "DESC", "Asc"])
def test_order_argument_is_case_insensitive(order):
    """Ensure order parsing accepts case-insensitive asc and desc values."""
    strategy = OWAWeights.create("linear")
    expected_order = order.lower()

    np.testing.assert_allclose(
        strategy.weights(5, order=order),
        strategy.weights(5, order=expected_order),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("order", [1, True, []])
def test_non_string_order_values_raise_value_error(order):
    """Ensure non-string order values fail with the public validation error type."""
    strategy = OWAWeights.create("linear")

    with pytest.raises(ValueError):
        strategy.weights(5, order=order)


def test_invalid_n():
    """Test ValueError is raised for invalid 'n' values in weights()."""
    strategy = OWAWeights.create("linear")
    with pytest.raises(ValueError):
        strategy.weights(0)
    with pytest.raises(ValueError):
        strategy.weights(-3)


@pytest.mark.parametrize("n", [3.5, "5", None, True])
def test_invalid_n_type_values_raise_value_error(n):
    """Ensure non-integral and boolean n values are rejected consistently."""
    strategy = OWAWeights.create("linear")

    with pytest.raises(ValueError):
        strategy.weights(n)


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