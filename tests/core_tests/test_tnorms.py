# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fuzzy T-norm operators."""

import numpy as np
import pytest

from frsutils.core.tnorms import TNorm
from tests import reference_data_store as ds
from tests._cupy_test_support import require_usable_cupy


CALL_TESTSETS = ds.get_tnorm_call_testsets()
MATRIX_MASK_TESTSETS = ds.get_tnorm_reduce_testsets()
TNORM_REGRESSION_TESTSETS = ds.get_tnorm_regression_testsets()
REGISTERED_TNORMS = TNorm.list_available()
REGISTERED_TNORM_NAMES = list(REGISTERED_TNORMS.keys())
REGISTERED_TNORM_ALIASES = [
    (primary_name, alias)
    for primary_name, aliases in REGISTERED_TNORMS.items()
    for alias in aliases
]
EXPECTED_TNORM_ALIASES = {
    "minimum": {"minimum", "min", "goedel", "standardintersection"},
    "product": {"product", "prod", "algebraic"},
    "lukasiewicz": {"lukasiewicz", "luk", "bounded", "boundeddifference"},
    "drastic": {"drastic", "drasticproduct"},
    "einstein": {"einstein", "einsteinproduct"},
    "hamacher": {"hamacher", "hamacherproduct"},
    "nilpotent": {"nilpotent", "nilpotentminimum"},
    "yager": {"yager", "yg"},
}
REDUCE_TEST_ARRAY = TNORM_REGRESSION_TESTSETS["axis_zero_reduce"]["input"]
REDUCE_EXPECTED = TNORM_REGRESSION_TESTSETS["axis_zero_reduce"]["expected"]
YAGER_DEFAULT_EXPECTED = TNORM_REGRESSION_TESTSETS["yager_default"]["expected"]


def _default_params_for_tnorm(tnorm_name):
    """Return constructor parameters needed by parametrized T-norm tests."""
    return {"p": 2.0} if tnorm_name == "yager" else {}


def _build_tnorm(tnorm_name):
    """Create a registered T-norm using stable default test parameters."""
    return TNorm.create(tnorm_name, **_default_params_for_tnorm(tnorm_name))


def _cupy_to_numpy(value, cp):
    """Convert a CuPy scalar or array result to a NumPy-compatible value."""
    return cp.asnumpy(cp.asarray(value))


#region <test output correctness>
###############################################
###                                         ###
###         test output correctness         ###
###                                         ###
###############################################

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_outputs_stay_within_unit_interval_on_dense_grid(tnorm_name):
    obj = _build_tnorm(tnorm_name)
    values = np.linspace(0, 1, 101)  # 0.0 to 1.0 step 0.01

    for a in values:
        for b in values:
            try:
                result = obj(np.array(a), np.array(b))
                assert 0.0 <= result <= 1.0
            except Exception as e:
                raise AssertionError(f"{tnorm_name} failed for a={a}, b={b}: {e}")


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_accepts_scalar_inputs_and_returns_scalar_result(tnorm_name):
    """Check that scalar inputs are handled correctly by T-norms."""
    obj = TNorm.create(tnorm_name, **({"p": 0.835} if tnorm_name == "yager" else {}))
    a, b = 0.73, 0.18
    result = obj(a, b)
    assert np.isscalar(result) or np.shape(result) == ()


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_vectorized_call_matches_reference_values(tnorm_name, testset):
    """Test that vectorized T-norm calls match hand-calculated outputs."""
    obj = _build_tnorm(tnorm_name)
    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]
    expected = (
        YAGER_DEFAULT_EXPECTED
        if tnorm_name == "yager"
        else testset["expected"][tnorm_name]
    )

    result = obj(a, b)

    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    ("tnorm_name", "expected_key"),
    [
        pytest.param("minimum", "minimum_outputs", id="minimum"),
        pytest.param("product", "product_outputs", id="product"),
        pytest.param("lukasiewicz", "luk_outputs", id="lukasiewicz"),
    ],
)
@pytest.mark.parametrize(
    "testset",
    MATRIX_MASK_TESTSETS,
    ids=lambda testset: testset["name"],
)
def test_tnorm_matrix_call_matches_binary_mask_reference_values(
    tnorm_name,
    expected_key,
    testset,
):
    """Validate matrix T-norm calls against binary-label-mask references."""
    tnorm = _build_tnorm(tnorm_name)

    result = tnorm(testset["similarity_matrix"], testset["label_mask"])

    np.testing.assert_allclose(
        result,
        testset["expected"][expected_key],
        atol=1e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_scalar_calls_match_reference_values(tnorm_name, testset):
    """Validate that scalar T-norm calls match expected values."""
    obj = _build_tnorm(tnorm_name)
    expected = (
        YAGER_DEFAULT_EXPECTED
        if tnorm_name == "yager"
        else testset["expected"][tnorm_name]
    )

    for i, (a_val, b_val) in enumerate(testset["a_b"]):
        result = obj(a_val, b_val)
        exp_val = expected[i]
        assert np.isclose(result, exp_val, atol=1e-6), (
            f"{tnorm_name} scalar call mismatch at index {i}: "
            f"got {result}, expected {exp_val}"
        )


@pytest.mark.parametrize("tnorm_name", ["yager"])
@pytest.mark.parametrize("p", [0.835, 5.0])
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_yager_tnorm_matches_reference_values_for_custom_p(tnorm_name, p, testset):
    """Test the Yager T-norm with predefined parameterized datasets."""
    obj = TNorm.create(tnorm_name, p=p)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    key = f"yager_p={p}" if p == 0.835 else "yager_p=5.0"

    result = obj(a, b)
    exp = testset["expected"][key]

    np.testing.assert_allclose(result, exp, atol=1e-5)


@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_yager_tnorm_default_p_matches_reference_values(testset):
    """Validate the default Yager T-norm parameter against expected values."""
    obj = TNorm.create("yager")
    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]

    result = obj(a, b)

    np.testing.assert_allclose(result, YAGER_DEFAULT_EXPECTED, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_numpy_backend_matches_reference_values(tnorm_name, testset):
    """Validate direct NumPy backend computations against expected values."""
    obj = _build_tnorm(tnorm_name)
    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]

    result = obj.compute_backend(a, b, xp=np)
    expected = (
        YAGER_DEFAULT_EXPECTED
        if tnorm_name == "yager"
        else testset["expected"][tnorm_name]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_cupy_backend_matches_reference_values(tnorm_name, testset):
    """Validate CuPy backend computations against NumPy expected values."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    a_cp = cp.asarray(testset["a_b"][:, 0])
    b_cp = cp.asarray(testset["a_b"][:, 1])

    result_cp = obj.compute_backend(a_cp, b_cp, xp=cp)
    expected = (
        YAGER_DEFAULT_EXPECTED
        if tnorm_name == "yager"
        else testset["expected"][tnorm_name]
    )

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-6)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_backend_matches_numpy_for_zero_dimensional_inputs(tnorm_name):
    """Check CuPy backend behavior for scalar-like zero-dimensional arrays."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    a_np = np.asarray(0.73)
    b_np = np.asarray(0.18)
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)

    result_cp = obj.compute_backend(a_cp, b_cp, xp=cp)
    expected = obj.compute_backend(a_np, b_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_backend_matches_numpy_for_matrix_inputs(tnorm_name):
    """Compare CuPy and NumPy backend computations on matrix inputs."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    a_np = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.33]])
    b_np = np.array([[1.0, 0.75, 0.5], [0.25, 0.0, 0.67]])

    result_cp = obj.compute_backend(cp.asarray(a_np), cp.asarray(b_np), xp=cp)
    expected = obj.compute_backend(a_np, b_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_backend_broadcasts_column_and_row_inputs(tnorm_name):
    """Check CuPy backend broadcasting for column-row array inputs."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    column_np = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    row_np = np.array([[0.1, 0.4, 0.7, 1.0]])

    result_cp = obj.compute_backend(cp.asarray(column_np), cp.asarray(row_np), xp=cp)
    expected = obj.compute_backend(column_np, row_np, xp=np)

    assert result_cp.shape == expected.shape
    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_call_broadcasts_scalar_and_vector_inputs(tnorm_name):
    """Check broadcasting for scalar-array and array-scalar inputs."""
    obj = _build_tnorm(tnorm_name)
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scalar = 0.6

    array_scalar = obj(values, scalar)
    scalar_array = obj(scalar, values)
    expected_array_scalar = np.array([obj(value, scalar) for value in values])
    expected_scalar_array = np.array([obj(scalar, value) for value in values])

    np.testing.assert_allclose(array_scalar, expected_array_scalar, atol=1e-8)
    np.testing.assert_allclose(scalar_array, expected_scalar_array, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_call_broadcasts_column_and_row_inputs(tnorm_name):
    """Check two-dimensional NumPy broadcasting for T-norm calls."""
    obj = _build_tnorm(tnorm_name)
    column = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    row = np.array([[0.1, 0.4, 0.7, 1.0]])

    result = obj(column, row)
    expected = np.empty((column.shape[0], row.shape[1]))
    for i, a_value in enumerate(column[:, 0]):
        for j, b_value in enumerate(row[0]):
            expected[i, j] = obj(a_value, b_value)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_reduce_matches_repeated_binary_application(tnorm_name):
    """Check that reduce is consistent with repeated binary calls."""
    obj = _build_tnorm(tnorm_name)
    rng = np.random.default_rng(seed=20240613)
    data_ = rng.random((200, 200))

    reduced = obj.reduce(data_)
    data_ = data_.T

    results = []
    for row in data_:
        res = row[0]
        for i in range(1, len(row)):
            res = obj(np.array(res), np.array(row[i]))
        results.append(float(res))

    np.testing.assert_allclose(reduced, results, atol=1e-7)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_reduce_matches_reference_values_along_axis_zero(tnorm_name):
    """Validate NumPy reduce outputs against independent expected values."""
    obj = _build_tnorm(tnorm_name)

    result = obj.reduce(REDUCE_TEST_ARRAY)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_numpy_reduce_backend_matches_reference_values_along_axis_zero(tnorm_name):
    """Validate direct NumPy backend reduction against expected values."""
    obj = _build_tnorm(tnorm_name)

    result = obj.reduce_backend(REDUCE_TEST_ARRAY, xp=np)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_reduce_backend_matches_reference_values_along_axis_zero(tnorm_name):
    """Validate CuPy backend reduction against independent expected values."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)

    result_cp = obj.reduce_backend(cp.asarray(REDUCE_TEST_ARRAY), xp=cp)

    np.testing.assert_allclose(
        _cupy_to_numpy(result_cp, cp),
        REDUCE_EXPECTED[tnorm_name],
        atol=1e-8,
    )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_reduce_backend_matches_numpy_for_vector_input(tnorm_name):
    """Compare CuPy and NumPy reduction for one-dimensional input."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    vector_np = REDUCE_TEST_ARRAY[:, 0]

    result_cp = obj.reduce_backend(cp.asarray(vector_np), xp=cp)
    expected = obj.reduce_backend(vector_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_cupy_reduce_backend_returns_single_row_values(tnorm_name):
    """Check CuPy reduction behavior for a single-row matrix."""
    cp = require_usable_cupy()
    obj = _build_tnorm(tnorm_name)
    single_row_np = REDUCE_TEST_ARRAY[:1, :]

    result_cp = obj.reduce_backend(cp.asarray(single_row_np), xp=cp)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), single_row_np[0], atol=1e-8)

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_reduce_vector_matches_reference_scalar(tnorm_name):
    """Check reduce behavior for a one-dimensional membership vector."""
    obj = _build_tnorm(tnorm_name)
    vector = REDUCE_TEST_ARRAY[:, 0]

    result = obj.reduce(vector)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name][0], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_reduce_returns_single_row_values(tnorm_name):
    """Check that reducing a single row returns that row unchanged."""
    obj = _build_tnorm(tnorm_name)
    single_row = REDUCE_TEST_ARRAY[:1, :]

    result = obj.reduce(single_row)

    np.testing.assert_allclose(result, single_row[0], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_construction_paths_produce_equal_outputs_on_random_data(tnorm_name):
    """Test that constructor, create(), and from_dict() produce identical outputs."""
    rng = np.random.default_rng(seed=123)
    a = rng.uniform(0, 1, size=1000)
    b = rng.uniform(0, 1, size=1000)

    params = _default_params_for_tnorm(tnorm_name)
    cls = TNorm.get_class(tnorm_name)
    tnorm1 = cls(**params)
    tnorm2 = TNorm.create(tnorm_name, **params)
    tnorm3 = TNorm.from_dict(tnorm1.to_dict())

    out1 = tnorm1(a, b)
    out2 = tnorm2(a, b)
    out3 = tnorm3(a, b)

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    np.testing.assert_allclose(out2, out3, atol=1e-7)
    np.testing.assert_allclose(out1, out3, atol=1e-7)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_construction_paths_produce_equal_outputs_on_reference_data(tnorm_name, testset):
    """Verify constructor, factory, and from_dict equivalence on test data."""
    cls = TNorm.get_class(tnorm_name)
    params = _default_params_for_tnorm(tnorm_name)
    tnorm1 = cls(**params)
    tnorm2 = TNorm.create(tnorm_name, **params)
    tnorm3 = TNorm.from_dict(tnorm1.to_dict())

    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]

    out1 = tnorm1(a, b)
    out2 = tnorm2(a, b)
    out3 = tnorm3(a, b)

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    np.testing.assert_allclose(out2, out3, atol=1e-7)
    np.testing.assert_allclose(out1, out3, atol=1e-7)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_scalar_calls_match_vectorized_call_outputs(tnorm_name, testset):
    """Ensure scalar T-norm calls match vectorized results for each test pair."""
    obj = _build_tnorm(tnorm_name)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    vectorized = obj(a, b)

    for idx, (a_val, b_val) in enumerate(a_b):
        result = obj(a_val, b_val)
        expected_val = vectorized[idx]
        assert np.isclose(result, expected_val, atol=1e-6), (
            f"Mismatch at index {idx} for {tnorm_name}: "
            f"got {result}, expected {expected_val}"
        )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_matrix_call_matches_elementwise_scalar_calls(tnorm_name):
    """Ensure matrix-wise T-norm application matches scalar application."""
    rng = np.random.default_rng(seed=42)
    n = 64
    A = rng.uniform(0, 1, size=(n, n))
    B = rng.uniform(0, 1, size=(n, n))

    tnorm = _build_tnorm(tnorm_name)

    # Direct matrix-wise application.
    matrix_result = tnorm(A, B)

    # Scalar-wise application.
    scalar_result = np.empty_like(A)
    for i in range(n):
        for j in range(n):
            scalar_result[i, j] = tnorm(A[i, j], B[i, j])

    np.testing.assert_allclose(
        matrix_result,
        scalar_result,
        atol=1e-7,
        err_msg=f"{tnorm_name}: matrix vs scalar mismatch",
    )

#endregion

#region <test non-calculational aspects>
###############################################
###                                         ###
###     test non-calculational aspects      ###
###                                         ###
###############################################

def test_tnorm_registry_contains_expected_public_aliases():
    """Ensure the registry exposes the expected public T-norm aliases."""
    for primary_name, expected_aliases in EXPECTED_TNORM_ALIASES.items():
        assert primary_name in REGISTERED_TNORMS
        assert expected_aliases.issubset(set(REGISTERED_TNORMS[primary_name]))


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_tnorm_create_resolves_each_registered_alias_to_primary_class(primary_name, alias):
    """Check that every registered alias creates the expected T-norm class."""
    primary = _build_tnorm(primary_name)
    alias_obj = TNorm.create(alias, **_default_params_for_tnorm(primary_name))

    assert isinstance(alias_obj, primary.__class__)


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_tnorm_get_class_resolves_each_registered_alias_to_primary_class(primary_name, alias):
    """Check that get_class resolves every alias to its primary class."""
    assert TNorm.get_class(alias) is TNorm.get_class(primary_name)


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_tnorm_create_resolves_registered_aliases_case_insensitively(primary_name, alias):
    """Check that aliases can be used with mixed-case names."""
    mixed_case_alias = alias.upper()
    obj = TNorm.create(mixed_case_alias, **_default_params_for_tnorm(primary_name))

    assert isinstance(obj, TNorm.get_class(primary_name))


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_serialization_roundtrip_preserves_constructor_parameters(tnorm_name):
    """Check that serialized T-norm parameters survive from_dict round-tripping."""
    params = {"p": 5.0} if tnorm_name == "yager" else {}
    obj = TNorm.create(tnorm_name, **params)

    data = obj.to_dict()
    restored = TNorm.from_dict(data)

    assert data["params"] == obj._get_params()
    assert restored._get_params() == obj._get_params()
    if tnorm_name == "yager":
        assert data["params"]["p"] == 5.0
        assert restored.p == 5.0


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_create_from_spec_builds_instance_from_nested_spec(tnorm_name):
    """Create T-norms from the preferred internal nested spec format."""
    params = _default_params_for_tnorm(tnorm_name)
    spec = {"name": tnorm_name, "params": params}

    obj = TNorm.create_from_spec(spec)

    assert isinstance(obj, TNorm.get_class(tnorm_name))
    assert obj._get_params() == params


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_create_from_spec_builds_instance_from_legacy_compact_spec(tnorm_name):
    """Create T-norms from the backward-compatible compact spec format."""
    params = _default_params_for_tnorm(tnorm_name)
    spec = {"type": tnorm_name, **params}

    obj = TNorm.create_from_spec(spec)

    assert isinstance(obj, TNorm.get_class(tnorm_name))
    assert obj._get_params() == params


def test_tnorm_create_from_spec_returns_existing_instance_by_identity():
    """Check that existing T-norm instances pass through create_from_spec."""
    obj = TNorm.create("minimum")

    result = TNorm.create_from_spec(obj)

    assert result is obj


def test_tnorm_create_from_spec_returns_internal_marker_instance_by_identity():
    """Check the internal __instance__ marker pass-through contract."""
    obj = TNorm.create("yager", p=3.0)

    result = TNorm.create_from_spec({"__instance__": obj})

    assert result is obj


def test_tnorm_create_from_spec_returns_none_for_none_spec():
    """Document the optional-spec behavior for create_from_spec(None)."""
    assert TNorm.create_from_spec(None) is None


def test_tnorm_create_raises_value_error_for_unknown_name():
    """Check factory error handling for unknown T-norm names."""
    with pytest.raises(ValueError):
        TNorm.create("unknown_tnorm")


def test_tnorm_get_class_raises_value_error_for_unknown_name():
    """Check registry lookup error handling for unknown T-norm names."""
    with pytest.raises(ValueError):
        TNorm.get_class("unknown_tnorm")


def test_tnorm_from_dict_raises_key_error_when_name_is_missing():
    """Check from_dict error handling for malformed serialized data."""
    with pytest.raises(KeyError):
        TNorm.from_dict({"params": {}})


def test_tnorm_create_from_spec_raises_type_error_for_non_mapping_params():
    """Check nested spec validation when params is not a dictionary."""
    with pytest.raises(TypeError):
        TNorm.create_from_spec({"name": "minimum", "params": [("unused", 1)]})


def test_tnorm_create_from_spec_raises_type_error_for_unsupported_spec():
    """Check create_from_spec error handling for unsupported objects."""
    with pytest.raises(TypeError):
        TNorm.create_from_spec(object())


def test_tnorm_create_strict_mode_raises_value_error_for_unused_parameters():
    """Check that strict factory construction rejects unused parameters."""
    with pytest.raises(ValueError):
        TNorm.create("minimum", strict=True, unused_parameter=1.0)


@pytest.mark.parametrize("bad_p", [0, -1.0, None, "2.0"])
def test_yager_tnorm_raises_value_error_for_invalid_p(bad_p):
    """Validate constructor-time rejection of invalid Yager parameters."""
    with pytest.raises(ValueError):
        TNorm.create("yager", p=bad_p)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_construction_paths_return_distinct_instances(tnorm_name):
    """Ensure direct, create(), and from_dict instances are separate objects."""
    params = _default_params_for_tnorm(tnorm_name)
    cls = TNorm.get_class(tnorm_name)
    tnorm1 = cls(**params)
    tnorm2 = TNorm.create(tnorm_name, **params)
    tnorm3 = TNorm.from_dict(tnorm1.to_dict())

    id1, id2, id3 = id(tnorm1), id(tnorm2), id(tnorm3)

    assert id1 != id2, f"{tnorm_name}: tnorm1 and tnorm2 share the same object ID!"
    assert id2 != id3, f"{tnorm_name}: tnorm2 and tnorm3 share the same object ID!"
    assert id1 != id3, f"{tnorm_name}: tnorm1 and tnorm3 share the same object ID!"


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_serialized_schema_and_roundtrip_preserve_registered_name(tnorm_name):
    """Test to_dict(), from_dict(), and create() round-tripping."""
    obj = _build_tnorm(tnorm_name)
    assert isinstance(obj, TNorm)

    data = obj.to_dict()
    assert "name" in data
    assert "type" in data
    assert "params" in data

    obj2 = TNorm.from_dict(data)
    assert isinstance(obj2, TNorm)
    assert obj2.name == obj.name


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_describe_params_detailed_returns_all_parameter_entries(tnorm_name):
    """Check detailed parameter introspection for all registered T-norms."""
    obj = _build_tnorm(tnorm_name)
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for k in obj._get_params():
        assert k in details


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_get_registered_name_returns_string_for_registered_instance(tnorm_name):
    """Check class lookup and reverse registered-name lookup."""
    cls = TNorm.get_class(tnorm_name)
    instance = cls(**_default_params_for_tnorm(tnorm_name))
    name = TNorm.get_registered_name(instance)
    assert isinstance(name, str)

#endregion

#region <axiom testing>

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_dense_grid_satisfies_range_commutativity_and_identity_axioms(tnorm_name):
    """Check output range, commutativity, and identity over a fixed grid."""
    obj = _build_tnorm(tnorm_name)
    values = np.linspace(0, 1, 11)

    for a in values:
        for b in values:
            a_np = np.array(a)
            b_np = np.array(b)

            result = obj(a_np, b_np)
            assert 0.0 <= result <= 1.0, (
                f"{tnorm_name} gave result {result} for a={a}, b={b}"
            )

            # Commutativity.
            result_rev = obj(b_np, a_np)
            assert np.isclose(result, result_rev, atol=1e-8), (
                f"{tnorm_name} is not commutative: "
                f"T({a},{b})={result} vs T({b},{a})={result_rev}"
            )

            # Boundary condition.
            result_boundary = obj(a_np, np.array(1.0))
            assert np.isclose(result_boundary, a, atol=1e-8), (
                f"{tnorm_name} failed boundary T({a},1.0)={result_boundary}, "
                f"expected {a}"
            )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_satisfies_zero_annihilator_and_one_identity_axioms(tnorm_name):
    """Verify the standard zero and one boundary axioms for T-norms."""
    obj = _build_tnorm(tnorm_name)
    values = np.linspace(0.0, 1.0, 21)
    zeros = np.zeros_like(values)
    ones = np.ones_like(values)

    np.testing.assert_allclose(obj(values, zeros), zeros, atol=1e-8)
    np.testing.assert_allclose(obj(zeros, values), zeros, atol=1e-8)
    np.testing.assert_allclose(obj(values, ones), values, atol=1e-8)
    np.testing.assert_allclose(obj(ones, values), values, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_is_monotone_in_both_arguments(tnorm_name):
    """Verify that each T-norm is monotone in both arguments."""
    obj = _build_tnorm(tnorm_name)
    rng = np.random.default_rng(seed=20240614)
    a_low = rng.uniform(0.0, 1.0, size=1000)
    b_low = rng.uniform(0.0, 1.0, size=1000)
    a_high = a_low + rng.uniform(0.0, 1.0 - a_low)
    b_high = b_low + rng.uniform(0.0, 1.0 - b_low)

    low_result = obj(a_low, b_low)
    high_result = obj(a_high, b_high)

    assert np.all(low_result <= high_result + 1e-12)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_is_associative_on_fixed_grid(tnorm_name):
    """Verify associativity over a fixed grid of membership values."""
    obj = _build_tnorm(tnorm_name)
    values = np.linspace(0, 1, 11)

    for a in values:
        for b in values:
            for c in values:
                a_np = np.array(a)
                b_np = np.array(b)
                c_np = np.array(c)

                try:
                    left = obj(a_np, obj(b_np, c_np))
                    right = obj(obj(a_np, b_np), c_np)

                    assert np.isclose(left, right, atol=1e-6), (
                        f"{tnorm_name} failed associativity: "
                        f"T({a},T({b},{c}))={left} vs "
                        f"T(T({a},{b}),{c})={right}"
                    )
                except Exception as e:
                    raise AssertionError(
                        f"{tnorm_name} error on associativity for "
                        f"a={a}, b={b}, c={c}: {e}"
                    )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_help_returns_nonempty_string(tnorm_name):
    """Check that each T-norm provides a valid help string."""
    obj = _build_tnorm(tnorm_name)
    doc = obj.help()
    assert isinstance(doc, str)
    assert len(doc.strip()) > 0

#endregion
