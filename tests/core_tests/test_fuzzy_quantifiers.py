# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fuzzy quantifier functions."""

import pytest
import numpy as np
from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier, validate_range_0_1
from frsutils.utils.logger.logger_util import get_logger
from tests import reference_data_store as ds
from tests._cupy_test_support import require_usable_cupy


logger = get_logger(env="test",
                    experiment_name="test_fuzzy_quantifiers")

registered_fqs = FuzzyQuantifier.list_available()
REFERENCE_CASES = {
    case["name"]: case for case in ds.get_fuzzy_quantifier_testsets()
}
LINEAR_1D_REFERENCE = REFERENCE_CASES["fuzzy_quantifier_linear_piecewise_1d"]
QUADRATIC_1D_REFERENCE = REFERENCE_CASES[
    "fuzzy_quantifier_quadratic_piecewise_1d"
]
LINEAR_2D_REFERENCE = REFERENCE_CASES["fuzzy_quantifier_linear_piecewise_2d"]
QUADRATIC_2D_REFERENCE = REFERENCE_CASES[
    "fuzzy_quantifier_quadratic_piecewise_2d"
]


def _reference_subset(
    reference_case: dict,
    indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected input and expected values from a 1D reference case."""
    index_array = np.asarray(indices, dtype=int)
    return reference_case["x"][index_array], reference_case["expected"][index_array]

BOUNDARY_REFERENCE_X, BOUNDARY_REFERENCE_EXPECTED = _reference_subset(
    LINEAR_1D_REFERENCE,
    (1, 3, 5),
)
QUADRATIC_INTERNAL_X, QUADRATIC_INTERNAL_EXPECTED = _reference_subset(
    QUADRATIC_1D_REFERENCE,
    (1, 2, 3, 4, 5),
)

# ----------------------------
# Functional Behavior Testing
# ----------------------------
#region <Functional Behavior Testing>
@pytest.mark.parametrize("quant_type, alpha, beta", [
    ("linear", 0.25, 0.75),
    ("quadratic", 0.25, 0.75),
    ("linear", 0.1, 0.9),
    ("quadratic", 0.1, 0.9)
])
def test_quantifier_output_shape_and_type(quant_type, alpha, beta):
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    x = np.linspace(0, 1, 500)
    result = fq(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert (0.0 <= result).all() and (result <= 1.0).all()


@pytest.mark.parametrize("quant_type, alpha, beta", [
    ("linear", -0.25, 0.75),
    ("quadratic", 0.25, -0.75),
    ("linear", 0.9, 0.1),
    ("quadratic", 0.5, 0.5),
    ("linear", 1.9, 2.1),
    ("quadratic", 3.5, 1.5)
])
def test_quantifier_output_exceptions(quant_type, alpha, beta):
    with pytest.raises(ValueError) as exc_info:
        _ = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    val =str(exc_info.value)
    logger.info(quant_type + ', alpha:' + str(alpha)+ ', beta:' + str(beta) + "Caught error message: " + val)



@pytest.mark.parametrize(
    "reference_case",
    [LINEAR_1D_REFERENCE, QUADRATIC_1D_REFERENCE],
    ids=lambda case: case["quantifier"]["name"],
)
def test_quantifier_known_outputs(reference_case):
    quantifier = reference_case["quantifier"]
    x, expected = _reference_subset(reference_case, (0, 1, 3, 5, 6))
    fq = FuzzyQuantifier.create(quantifier["name"], **quantifier["params"])

    result = fq(x)

    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_linear_quantifier_matches_piecewise_formula_at_internal_points():
    quantifier = LINEAR_1D_REFERENCE["quantifier"]
    fq = FuzzyQuantifier.create(quantifier["name"], **quantifier["params"])

    result = fq(LINEAR_1D_REFERENCE["x"])

    np.testing.assert_allclose(
        result,
        LINEAR_1D_REFERENCE["expected"],
        atol=1e-12,
    )


def test_quadratic_quantifier_matches_piecewise_formula_at_internal_points():
    quantifier = QUADRATIC_1D_REFERENCE["quantifier"]
    fq = FuzzyQuantifier.create(quantifier["name"], **quantifier["params"])

    result = fq(QUADRATIC_1D_REFERENCE["x"])

    np.testing.assert_allclose(
        result,
        QUADRATIC_1D_REFERENCE["expected"],
        atol=1e-12,
    )


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_quantifier_preserves_two_dimensional_input_shape(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x = np.array([[0.0, 0.2, 0.35], [0.5, 0.8, 1.0]], dtype=float)

    result = fq(x)

    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
@pytest.mark.parametrize("x", [0.5, np.float64(0.5), np.array(0.5, dtype=float)])
def test_quantifier_accepts_scalar_like_float_inputs(quant_type, x):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    result = fq(x)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    np.testing.assert_allclose(result, np.array(0.5), atol=1e-12)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_quantifier_outputs_are_bounded_for_dense_valid_inputs(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x = np.linspace(0.0, 1.0, 1001, dtype=float)

    result = fq(x)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_quantifier_is_non_decreasing_for_dense_valid_inputs(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x = np.linspace(0.0, 1.0, 1001, dtype=float)

    result = fq(x)

    assert np.all(np.diff(result) >= -1e-12)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_quantifier_returns_zero_below_alpha_and_one_above_beta(quant_type):
    alpha = 0.2
    beta = 0.8
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    below_alpha = np.array([0.0, 0.05, 0.1, alpha], dtype=float)
    above_beta = np.array([beta, 0.9, 0.95, 1.0], dtype=float)

    np.testing.assert_allclose(fq(below_alpha), np.zeros_like(below_alpha), atol=1e-12)
    np.testing.assert_allclose(fq(above_beta), np.ones_like(above_beta), atol=1e-12)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_quantifier_boundary_and_midpoint_values_are_continuous(quant_type):
    alpha = 0.2
    beta = 0.8
    mid = (alpha + beta) / 2.0
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    x = np.array([alpha, mid, beta], dtype=float)
    expected = np.array([0.0, 0.5, 1.0], dtype=float)

    result = fq(x)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_linear_quantifier_matches_formula_on_two_dimensional_input():
    quantifier = LINEAR_2D_REFERENCE["quantifier"]
    fq = FuzzyQuantifier.create(quantifier["name"], **quantifier["params"])

    result = fq(LINEAR_2D_REFERENCE["x"])

    np.testing.assert_allclose(
        result,
        LINEAR_2D_REFERENCE["expected"],
        atol=1e-12,
    )


def test_quadratic_quantifier_matches_formula_on_two_dimensional_input():
    quantifier = QUADRATIC_2D_REFERENCE["quantifier"]
    fq = FuzzyQuantifier.create(quantifier["name"], **quantifier["params"])

    result = fq(QUADRATIC_2D_REFERENCE["x"])

    np.testing.assert_allclose(
        result,
        QUADRATIC_2D_REFERENCE["expected"],
        atol=1e-12,
    )
#endregion


# ----------------------------
# Factory and Serialization
# ----------------------------
#region<Factory and Serialization>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_to_dict_from_dict(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    d = fq.to_dict()
    fq2 = FuzzyQuantifier.from_dict(d)

    assert isinstance(fq2, FuzzyQuantifier)
    assert fq2.name == fq.name

    np.testing.assert_allclose(fq2._get_params()["alpha"], fq._get_params()["alpha"])
    np.testing.assert_allclose(fq2._get_params()["beta"], fq._get_params()["beta"])


def test_quadratic_alias_quad_creates_equivalent_quantifier():
    x = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0], dtype=float)
    quadratic = FuzzyQuantifier.create("quadratic", alpha=0.2, beta=0.8)
    quad = FuzzyQuantifier.create("quad", alpha=0.2, beta=0.8)

    assert quad.name == quadratic.name
    np.testing.assert_allclose(quad(x), quadratic(x), atol=1e-12)


def test_list_available_includes_quadratic_alias_quad():
    assert "quadratic" in registered_fqs
    assert "quad" in registered_fqs["quadratic"]


@pytest.mark.parametrize("quant_type", ["LINEAR", "Quadratic", "QUAD"])
def test_factory_alias_lookup_is_case_insensitive(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    result = fq(BOUNDARY_REFERENCE_X)

    assert isinstance(fq, FuzzyQuantifier)
    np.testing.assert_allclose(
        result,
        BOUNDARY_REFERENCE_EXPECTED,
        atol=1e-12,
    )


def test_factory_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Unknown alias"):
        FuzzyQuantifier.create("unknown", alpha=0.2, beta=0.8)


@pytest.mark.parametrize("data", [
    {"name": "linear", "params": {"alpha": 0.2, "beta": 0.8}},
    {"type": "linear", "params": {"alpha": 0.2, "beta": 0.8}},
    {"type": "linear", "alpha": 0.2, "beta": 0.8},
])
def test_from_dict_accepts_supported_linear_formats(data):
    fq = FuzzyQuantifier.from_dict(data)
    assert fq.name == "linear"
    np.testing.assert_allclose(
        fq(BOUNDARY_REFERENCE_X),
        BOUNDARY_REFERENCE_EXPECTED,
        atol=1e-12,
    )


@pytest.mark.parametrize("data", [
    {"name": "quadratic", "params": {"alpha": 0.2, "beta": 0.8}},
    {"type": "quadratic", "params": {"alpha": 0.2, "beta": 0.8}},
    {"type": "quadratic", "alpha": 0.2, "beta": 0.8},
])
def test_from_dict_accepts_supported_quadratic_formats(data):
    fq = FuzzyQuantifier.from_dict(data)
    assert fq.name == "quadratic"
    np.testing.assert_allclose(
        fq(QUADRATIC_INTERNAL_X),
        QUADRATIC_INTERNAL_EXPECTED,
        atol=1e-12,
    )


@pytest.mark.parametrize("data", [
    ["linear", {"alpha": 0.2, "beta": 0.8}],
    "linear",
    None,
])
def test_from_dict_rejects_non_dict_input(data):
    with pytest.raises(TypeError):
        FuzzyQuantifier.from_dict(data)


@pytest.mark.parametrize("data", [
    {"name": "linear", "params": [0.2, 0.8]},
    {"type": "linear", "params": [0.2, 0.8]},
])
def test_from_dict_rejects_non_dict_params(data):
    with pytest.raises(TypeError):
        FuzzyQuantifier.from_dict(data)


@pytest.mark.parametrize("data", [
    {},
    {"name": "linear"},
    {"params": {"alpha": 0.2, "beta": 0.8}},
    {"type": "linear", "alpha": 0.2},
])
def test_from_dict_rejects_unsupported_dict_formats(data):
    with pytest.raises(ValueError):
        FuzzyQuantifier.from_dict(data)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_to_dict_from_dict_preserves_validate_inputs(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)

    restored = FuzzyQuantifier.from_dict(fq.to_dict())

    assert restored._get_params()["validate_inputs"] is False


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_direct_constructor_factory_and_from_dict_are_equivalent(quant_type):
    cls = FuzzyQuantifier.get_class(quant_type)
    direct = cls(alpha=0.2, beta=0.8, validate_inputs=False)
    factory = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)
    restored = FuzzyQuantifier.from_dict(direct.to_dict())
    x = np.linspace(0.0, 1.0, 101, dtype=float)

    assert id(direct) != id(factory)
    assert id(factory) != id(restored)
    assert id(direct) != id(restored)
    assert direct._get_params() == factory._get_params() == restored._get_params()
    np.testing.assert_allclose(direct(x), factory(x), atol=1e-12)
    np.testing.assert_allclose(factory(x), restored(x), atol=1e-12)
#endregion

# ----------------------------
# Registry and Nested-Spec Integration
# ----------------------------
#region<Registry and Nested-Spec Integration>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_from_spec_accepts_preferred_nested_spec(quant_type):
    spec = {"name": quant_type, "params": {"alpha": 0.2, "beta": 0.8}}
    fq = FuzzyQuantifier.create_from_spec(spec)
    assert isinstance(fq, FuzzyQuantifier)
    assert fq.name == FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8).name
    np.testing.assert_allclose(
        fq(BOUNDARY_REFERENCE_X),
        BOUNDARY_REFERENCE_EXPECTED,
        atol=1e-12,
    )


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_from_spec_accepts_existing_instance_as_pass_through(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    restored = FuzzyQuantifier.create_from_spec(fq)

    assert restored is fq


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_from_spec_accepts_internal_instance_marker_as_pass_through(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    restored = FuzzyQuantifier.create_from_spec({"__instance__": fq})

    assert restored is fq


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_from_spec_accepts_legacy_compact_type_spec(quant_type):
    spec = {"type": quant_type, "alpha": 0.2, "beta": 0.8, "validate_inputs": False}
    fq = FuzzyQuantifier.create_from_spec(spec)
    x = np.array([-0.1, 0.5, 1.1], dtype=float)

    assert isinstance(fq, FuzzyQuantifier)
    assert fq._get_params()["validate_inputs"] is False
    result = fq(x)
    assert result.shape == x.shape


def test_create_from_spec_returns_none_for_none_spec():
    assert FuzzyQuantifier.create_from_spec(None) is None


@pytest.mark.parametrize("spec", [
    {"name": "linear", "params": [0.2, 0.8]},
    {"name": "linear", "params": None},
    {"params": {"alpha": 0.2, "beta": 0.8}},
    ["linear", {"alpha": 0.2, "beta": 0.8}],
])
def test_create_from_spec_rejects_invalid_specs(spec):
    with pytest.raises((TypeError, ValueError)):
        FuzzyQuantifier.create_from_spec(spec)


def test_create_extracts_namespaced_parameters():
    fq = FuzzyQuantifier.create(
        "linear",
        namespace="lb",
        lb_alpha=0.2,
        lb_beta=0.8,
        ub_alpha=0.1,
        ub_beta=0.9,
    )
    assert fq._get_params()["alpha"] == 0.2
    assert fq._get_params()["beta"] == 0.8
    np.testing.assert_allclose(
        fq(BOUNDARY_REFERENCE_X),
        BOUNDARY_REFERENCE_EXPECTED,
        atol=1e-12,
    )


def test_create_extracts_namespaced_parameters_case_insensitively_for_alias():
    fq = FuzzyQuantifier.create(
        "QUAD",
        namespace="ub",
        ub_alpha=0.2,
        ub_beta=0.8,
        lb_alpha=0.1,
        lb_beta=0.9,
    )
    assert fq.name == "quadratic"
    assert fq._get_params()["alpha"] == 0.2
    assert fq._get_params()["beta"] == 0.8
    np.testing.assert_allclose(
        fq(QUADRATIC_INTERNAL_X),
        QUADRATIC_INTERNAL_EXPECTED,
        atol=1e-12,
    )


def test_create_with_strict_true_rejects_unused_parameters():
    with pytest.raises(ValueError, match="Unused parameters"):
        FuzzyQuantifier.create(
            "linear",
            alpha=0.2,
            beta=0.8,
            extra_param=123,
            strict=True,
        )


def test_create_with_strict_false_ignores_unused_parameters():
    fq = FuzzyQuantifier.create(
        "linear",
        alpha=0.2,
        beta=0.8,
        extra_param=123,
        strict=False,
    )
    assert isinstance(fq, FuzzyQuantifier)
    np.testing.assert_allclose(
        fq(BOUNDARY_REFERENCE_X),
        BOUNDARY_REFERENCE_EXPECTED,
        atol=1e-12,
    )


def test_create_with_strict_true_checks_filtered_namespaced_parameters():
    with pytest.raises(ValueError, match="Unused parameters"):
        FuzzyQuantifier.create(
            "linear",
            namespace="lb",
            lb_alpha=0.2,
            lb_beta=0.8,
            lb_extra_param=123,
            ub_extra_param=456,
            strict=True,
        )


def test_create_with_namespace_ignores_unrelated_namespace_even_in_strict_mode():
    fq = FuzzyQuantifier.create(
        "linear",
        namespace="lb",
        lb_alpha=0.2,
        lb_beta=0.8,
        ub_extra_param=456,
        strict=True,
    )

    assert fq._get_params()["alpha"] == 0.2
    assert fq._get_params()["beta"] == 0.8


def test_create_from_spec_passes_strict_flag_to_factory():
    spec = {"name": "linear", "params": {"alpha": 0.2, "beta": 0.8, "extra_param": 123}}

    with pytest.raises(ValueError, match="Unused parameters"):
        FuzzyQuantifier.create_from_spec(spec, strict=True)


def test_create_from_spec_ignores_unused_params_when_not_strict():
    spec = {"name": "linear", "params": {"alpha": 0.2, "beta": 0.8, "extra_param": 123}}
    fq = FuzzyQuantifier.create_from_spec(spec, strict=False)

    assert fq._get_params()["alpha"] == 0.2
    assert fq._get_params()["beta"] == 0.8

#endregion

# ----------------------------
# Validation and Fail-Fast
# ----------------------------
#region<Validation and Fail-Fast>
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_validate_range_0_1_accepts_valid_float_scalars(value):
    assert validate_range_0_1(value, name="value") == value


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("-inf")])
def test_validate_range_0_1_rejects_invalid_float_scalars(value):
    with pytest.raises(ValueError):
        validate_range_0_1(value, name="value")


def test_validate_range_0_1_accepts_valid_float_array():
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    result = validate_range_0_1(values, name="values")

    assert result is values


@pytest.mark.parametrize("values", [
    np.array([-0.1, 0.5], dtype=float),
    np.array([0.5, 1.1], dtype=float),
    np.array([0.5, np.inf], dtype=float),
    np.array([-np.inf, 0.5], dtype=float),
])
def test_validate_range_0_1_rejects_out_of_range_float_arrays(values):
    with pytest.raises(ValueError):
        validate_range_0_1(values, name="values")


@pytest.mark.parametrize("values", [
    np.array([0, 1], dtype=int),
    np.array([True, False], dtype=bool),
])
def test_validate_range_0_1_rejects_non_float_arrays(values):
    with pytest.raises(TypeError):
        validate_range_0_1(values, name="values")


@pytest.mark.parametrize("value", [
    [0.0, 1.0],
    (0.0, 1.0),
    "0.5",
    None,
])
def test_validate_range_0_1_rejects_unsupported_input_types(value):
    with pytest.raises(TypeError):
        validate_range_0_1(value, name="value")


def test_validate_range_0_1_rejects_scalar_nan():
    with pytest.raises(ValueError):
        validate_range_0_1(float("nan"), name="value")


def test_validate_range_0_1_rejects_array_nan():
    with pytest.raises(ValueError):
        validate_range_0_1(np.array([0.5, np.nan], dtype=float), name="values")


@pytest.mark.parametrize("params", [
    {"typ":"linear", "alpha": None, "beta": 0.6},
    {"typ":"linear", "alpha": 0.2, "beta": None},
    {"typ":"linear", "alpha": "a", "beta": 0.6},
    {"typ":"linear", "alpha": 0.2, "beta": "b"},
    {"typ":"linear", "alpha": 0.7, "beta": 0.6},
    {"typ":"linear", "alpha": -0.1, "beta": 1.2},
    {"typ":"quadratic", "alpha": None, "beta": 0.6},
    {"typ":"quadratic", "alpha": 0.2, "beta": None},
    {"typ":"quadratic", "alpha": "a", "beta": 0.6},
    {"typ":"quadratic", "alpha": 0.2, "beta": "b"},
    {"typ":"quadratic", "alpha": 0.7, "beta": 0.6},
    {"typ":"quadratic", "alpha": -0.1, "beta": 1.2}
])
def test_invalid_alpha_beta(params):
    with pytest.raises(ValueError) as exc_info:
        FuzzyQuantifier.create(params["typ"], **params)

    val =str(exc_info.value)
    logger.info(params["typ"] + ', alpha:' + str(params["alpha"])+ ', beta:' + str(params["beta"]) + "Caught error message: " + val)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
@pytest.mark.parametrize("x", [
    np.array([-0.1, 0.5], dtype=float),
    np.array([0.5, 1.1], dtype=float),
    -0.1,
    1.1,
])
def test_quantifier_call_rejects_out_of_range_inputs(quant_type, x):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    with pytest.raises(ValueError):
        fq(x)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
@pytest.mark.parametrize("x", [
    np.array([0, 1], dtype=int),
    0,
    1,
])
def test_quantifier_call_rejects_non_float_inputs(quant_type, x):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    with pytest.raises(TypeError):
        fq(x)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
@pytest.mark.parametrize("x", [float("nan"), np.array([0.5, np.nan], dtype=float)])
def test_quantifier_call_rejects_nan_inputs(quant_type, x):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)

    with pytest.raises(ValueError):
        fq(x)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_validate_inputs_false_bypasses_input_range_validation(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)
    x = np.array([-0.1, 0.5, 1.1], dtype=float)

    result = fq(x)

    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


#endregion

# ----------------------------
# Optional CuPy Backend Behavior
# ----------------------------
#region<Optional CuPy Backend Behavior>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_compute_backend_cupy_returns_cupy_array_and_matches_numpy(quant_type):
    cp = require_usable_cupy()
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x_np = np.array([[0.0, 0.2, 0.35], [0.5, 0.65, 0.8]], dtype=float)
    x_cp = cp.asarray(x_np)

    result_cp = fq.compute_backend(x_cp, xp=cp)
    expected_np = fq(x_np)

    assert isinstance(result_cp, cp.ndarray)
    assert result_cp.shape == x_cp.shape
    np.testing.assert_allclose(cp.asnumpy(result_cp), expected_np, atol=1e-12)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_compute_backend_cupy_preserves_scalar_like_shape(quant_type):
    cp = require_usable_cupy()
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x_cp = cp.asarray(0.5, dtype=float)

    result_cp = fq.compute_backend(x_cp, xp=cp)

    assert isinstance(result_cp, cp.ndarray)
    assert result_cp.shape == ()
    np.testing.assert_allclose(cp.asnumpy(result_cp), np.array(0.5), atol=1e-12)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
@pytest.mark.parametrize("x_np", [
    np.array([-0.1, 0.5], dtype=float),
    np.array([0.5, 1.1], dtype=float),
])
def test_compute_backend_cupy_rejects_out_of_range_inputs(quant_type, x_np):
    cp = require_usable_cupy()
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x_cp = cp.asarray(x_np)

    with pytest.raises(ValueError):
        fq.compute_backend(x_cp, xp=cp)


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_compute_backend_cupy_validate_inputs_false_bypasses_range_validation(quant_type):
    cp = require_usable_cupy()
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)
    x_np = np.array([-0.1, 0.5, 1.1], dtype=float)
    x_cp = cp.asarray(x_np)

    result_cp = fq.compute_backend(x_cp, xp=cp, validate_inputs=fq.validate_inputs)

    assert isinstance(result_cp, cp.ndarray)
    assert result_cp.shape == x_cp.shape


@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_compute_backend_cupy_rejects_nan_inputs(quant_type):
    cp = require_usable_cupy()
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    x_cp = cp.asarray([0.5, cp.nan], dtype=float)

    with pytest.raises(ValueError):
        fq.compute_backend(x_cp, xp=cp)

#endregion

# ----------------------------
# Metadata & Reflection
# ----------------------------
#region<Metadata & Reflection>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_describe_and_params_match(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)
    described = fq.describe_params_detailed()
    params = fq._get_params()
    for k in params:
        assert k in described
        assert described[k]["value"] == params[k]

#endregion

@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_help(quant_type):
    """
    tests values of params and deteriled params. check them in log
    """
    # TODO: when a class does not have docstring, it returns the base calss docstring. This is wrong
    obj = FuzzyQuantifier.create(quant_type, alpha=0.25, beta=0.75)
    details = obj.help()
    assert isinstance(details, str)
    
    logger.info(quant_type + ', class docstring:' + details)
