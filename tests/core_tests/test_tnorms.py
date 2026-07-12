# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fuzzy T-norm operators."""

import numpy as np
import pytest

from frsutils.core.tnorms import TNorm
from tests import synthetic_data_store as ds


CALL_TESTSETS = ds.get_tnorm_call_testsets()
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
REDUCE_TEST_ARRAY = np.array([
    [0.9, 1.0, 0.7, 0.2, 1.0],
    [0.8, 0.6, 0.9, 0.5, 1.0],
    [0.7, 0.8, 0.6, 1.0, 0.75],
])
REDUCE_EXPECTED = {
    "minimum": np.array([0.7, 0.6, 0.6, 0.2, 0.75]),
    "product": np.array([0.504, 0.48, 0.378, 0.1, 0.75]),
    "lukasiewicz": np.array([0.4, 0.4, 0.2, 0.0, 0.75]),
    "drastic": np.array([0.0, 0.0, 0.0, 0.0, 0.75]),
    "einstein": np.array([0.4540540541, 0.4444444444, 0.3176470588, 0.0714285714, 0.75]),
    "hamacher": np.array([0.5587583149, 0.5217391304, 0.4532374101, 0.1666666667, 0.75]),
    "nilpotent": np.array([0.7, 0.6, 0.6, 0.0, 0.75]),
    "yager": np.array([0.6258342613, 0.5527864045, 0.4900980486, 0.0566018868, 0.75]),
}
YAGER_DEFAULT_EXPECTED = np.array([
    0.13669241,
    0.13669241,
    0.83029437,
    0.47226901,
    1.0,
    0.0,
    0.65,
    0.37,
])


def _default_params_for_tnorm(tnorm_name):
    """Return constructor parameters needed by parametrized T-norm tests."""
    return {"p": 2.0} if tnorm_name == "yager" else {}


def _build_tnorm(tnorm_name):
    """Create a registered T-norm using stable default test parameters."""
    return TNorm.create(tnorm_name, **_default_params_for_tnorm(tnorm_name))


def _cupy_module():
    """Return CuPy or skip when the installed CuPy runtime is unusable."""
    cp = pytest.importorskip("cupy")

    # Importing CuPy is not enough: environments with missing NVRTC/CUDA runtime
    # pieces can import CuPy but fail when the first ufunc kernel is compiled.
    # Run a tiny ufunc smoke check so such environments skip CuPy backend tests
    # instead of reporting false failures in T-norm logic.
    try:
        x = cp.asarray([0.0, 1.0])
        y = cp.asarray([1.0, 0.0])
        cp.asnumpy(cp.minimum(x, y))
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime.
        pytest.skip(f"CuPy is importable but unusable in this environment: {exc}")

    return cp


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
def test_tnorm_all_pairs_from_0_to_1(tnorm_name):
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
def test_scalar_inputs(tnorm_name):
    """Check that scalar inputs are handled correctly by T-norms."""
    obj = TNorm.create(tnorm_name, **({"p": 0.835} if tnorm_name == "yager" else {}))
    a, b = 0.73, 0.18
    result = obj(a, b)
    assert np.isscalar(result) or np.shape(result) == ()


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_tnorm_call_output_matches_expected(tnorm_name, testset):
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


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_scalar_call_output_matches_expected_values(tnorm_name, testset):
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
def test_yager_parametrized_behavior(tnorm_name, p, testset):
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
def test_yager_default_p_matches_expected_values(testset):
    """Validate the default Yager T-norm parameter against expected values."""
    obj = TNorm.create("yager")
    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]

    result = obj(a, b)

    np.testing.assert_allclose(result, YAGER_DEFAULT_EXPECTED, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
@pytest.mark.parametrize("testset", CALL_TESTSETS)
def test_compute_backend_numpy_matches_expected_values(tnorm_name, testset):
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
def test_compute_backend_cupy_matches_numpy_expected_values(tnorm_name, testset):
    """Validate CuPy backend computations against NumPy expected values."""
    cp = _cupy_module()
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
def test_compute_backend_cupy_handles_scalar_like_arrays(tnorm_name):
    """Check CuPy backend behavior for scalar-like zero-dimensional arrays."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)
    a_np = np.asarray(0.73)
    b_np = np.asarray(0.18)
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)

    result_cp = obj.compute_backend(a_cp, b_cp, xp=cp)
    expected = obj.compute_backend(a_np, b_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_compute_backend_cupy_matches_numpy_for_matrices(tnorm_name):
    """Compare CuPy and NumPy backend computations on matrix inputs."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)
    a_np = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.33]])
    b_np = np.array([[1.0, 0.75, 0.5], [0.25, 0.0, 0.67]])

    result_cp = obj.compute_backend(cp.asarray(a_np), cp.asarray(b_np), xp=cp)
    expected = obj.compute_backend(a_np, b_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_compute_backend_cupy_broadcasts_column_and_row_arrays(tnorm_name):
    """Check CuPy backend broadcasting for column-row array inputs."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)
    column_np = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    row_np = np.array([[0.1, 0.4, 0.7, 1.0]])

    result_cp = obj.compute_backend(cp.asarray(column_np), cp.asarray(row_np), xp=cp)
    expected = obj.compute_backend(column_np, row_np, xp=np)

    assert result_cp.shape == expected.shape
    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_broadcasts_scalar_and_array_inputs(tnorm_name):
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
def test_tnorm_broadcasts_column_and_row_arrays(tnorm_name):
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
def test_reduce_consistency_with_call(tnorm_name):
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
def test_reduce_matches_expected_axis0_values(tnorm_name):
    """Validate NumPy reduce outputs against independent expected values."""
    obj = _build_tnorm(tnorm_name)

    result = obj.reduce(REDUCE_TEST_ARRAY)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_backend_numpy_matches_expected_axis0_values(tnorm_name):
    """Validate direct NumPy backend reduction against expected values."""
    obj = _build_tnorm(tnorm_name)

    result = obj.reduce_backend(REDUCE_TEST_ARRAY, xp=np)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_backend_cupy_matches_expected_axis0_values(tnorm_name):
    """Validate CuPy backend reduction against independent expected values."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)

    result_cp = obj.reduce_backend(cp.asarray(REDUCE_TEST_ARRAY), xp=cp)

    np.testing.assert_allclose(
        _cupy_to_numpy(result_cp, cp),
        REDUCE_EXPECTED[tnorm_name],
        atol=1e-8,
    )


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_backend_cupy_matches_numpy_for_one_dimensional_input(tnorm_name):
    """Compare CuPy and NumPy reduction for one-dimensional input."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)
    vector_np = REDUCE_TEST_ARRAY[:, 0]

    result_cp = obj.reduce_backend(cp.asarray(vector_np), xp=cp)
    expected = obj.reduce_backend(vector_np, xp=np)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), expected, atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_backend_cupy_single_row_returns_same_values(tnorm_name):
    """Check CuPy reduction behavior for a single-row matrix."""
    cp = _cupy_module()
    obj = _build_tnorm(tnorm_name)
    single_row_np = REDUCE_TEST_ARRAY[:1, :]

    result_cp = obj.reduce_backend(cp.asarray(single_row_np), xp=cp)

    np.testing.assert_allclose(_cupy_to_numpy(result_cp, cp), single_row_np[0], atol=1e-8)

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_matches_expected_for_one_dimensional_input(tnorm_name):
    """Check reduce behavior for a one-dimensional membership vector."""
    obj = _build_tnorm(tnorm_name)
    vector = REDUCE_TEST_ARRAY[:, 0]

    result = obj.reduce(vector)

    np.testing.assert_allclose(result, REDUCE_EXPECTED[tnorm_name][0], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_reduce_single_row_returns_same_row_values(tnorm_name):
    """Check that reducing a single row returns that row unchanged."""
    obj = _build_tnorm(tnorm_name)
    single_row = REDUCE_TEST_ARRAY[:1, :]

    result = obj.reduce(single_row)

    np.testing.assert_allclose(result, single_row[0], atol=1e-8)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_equivalence_of_constructor_create_fromdict_with_random_data(tnorm_name):
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
def test_equivalence_of_constructor_create_fromdict(tnorm_name, testset):
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
def test_scalar_call_matches_vectorized_outputs(tnorm_name, testset):
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
def test_tnorm_matrix_consistency_with_scalar_application(tnorm_name):
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

def test_registered_tnorm_aliases_include_expected_names():
    """Ensure the registry exposes the expected public T-norm aliases."""
    for primary_name, expected_aliases in EXPECTED_TNORM_ALIASES.items():
        assert primary_name in REGISTERED_TNORMS
        assert expected_aliases.issubset(set(REGISTERED_TNORMS[primary_name]))


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_all_registered_aliases_create_equivalent_tnorms(primary_name, alias):
    """Check that every registered alias creates the expected T-norm class."""
    primary = _build_tnorm(primary_name)
    alias_obj = TNorm.create(alias, **_default_params_for_tnorm(primary_name))

    assert isinstance(alias_obj, primary.__class__)


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_all_registered_aliases_get_same_class(primary_name, alias):
    """Check that get_class resolves every alias to its primary class."""
    assert TNorm.get_class(alias) is TNorm.get_class(primary_name)


@pytest.mark.parametrize("primary_name, alias", REGISTERED_TNORM_ALIASES)
def test_registered_aliases_are_case_insensitive(primary_name, alias):
    """Check that aliases can be used with mixed-case names."""
    mixed_case_alias = alias.upper()
    obj = TNorm.create(mixed_case_alias, **_default_params_for_tnorm(primary_name))

    assert isinstance(obj, TNorm.get_class(primary_name))


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_yager_serialization_preserves_parameters_and_roundtrips(tnorm_name):
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
def test_create_from_spec_with_preferred_nested_format(tnorm_name):
    """Create T-norms from the preferred internal nested spec format."""
    params = _default_params_for_tnorm(tnorm_name)
    spec = {"name": tnorm_name, "params": params}

    obj = TNorm.create_from_spec(spec)

    assert isinstance(obj, TNorm.get_class(tnorm_name))
    assert obj._get_params() == params


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_create_from_spec_with_legacy_compact_format(tnorm_name):
    """Create T-norms from the backward-compatible compact spec format."""
    params = _default_params_for_tnorm(tnorm_name)
    spec = {"type": tnorm_name, **params}

    obj = TNorm.create_from_spec(spec)

    assert isinstance(obj, TNorm.get_class(tnorm_name))
    assert obj._get_params() == params


def test_create_from_spec_returns_existing_instance_unchanged():
    """Check that existing T-norm instances pass through create_from_spec."""
    obj = TNorm.create("minimum")

    result = TNorm.create_from_spec(obj)

    assert result is obj


def test_create_from_spec_returns_internal_marker_instance_unchanged():
    """Check the internal __instance__ marker pass-through contract."""
    obj = TNorm.create("yager", p=3.0)

    result = TNorm.create_from_spec({"__instance__": obj})

    assert result is obj


def test_create_from_spec_allows_none_as_empty_optional_spec():
    """Document the optional-spec behavior for create_from_spec(None)."""
    assert TNorm.create_from_spec(None) is None


def test_create_raises_for_unknown_alias():
    """Check factory error handling for unknown T-norm names."""
    with pytest.raises(ValueError):
        TNorm.create("unknown_tnorm")


def test_get_class_raises_for_unknown_alias():
    """Check registry lookup error handling for unknown T-norm names."""
    with pytest.raises(ValueError):
        TNorm.get_class("unknown_tnorm")


def test_from_dict_raises_for_missing_name():
    """Check from_dict error handling for malformed serialized data."""
    with pytest.raises(KeyError):
        TNorm.from_dict({"params": {}})


def test_create_from_spec_raises_for_non_dict_params():
    """Check nested spec validation when params is not a dictionary."""
    with pytest.raises(TypeError):
        TNorm.create_from_spec({"name": "minimum", "params": [("unused", 1)]})


def test_create_from_spec_raises_for_invalid_object():
    """Check create_from_spec error handling for unsupported objects."""
    with pytest.raises(TypeError):
        TNorm.create_from_spec(object())


def test_create_strict_mode_rejects_unused_parameters():
    """Check that strict factory construction rejects unused parameters."""
    with pytest.raises(ValueError):
        TNorm.create("minimum", strict=True, unused_parameter=1.0)


@pytest.mark.parametrize("bad_p", [0, -1.0, None, "2.0"])
def test_yager_rejects_invalid_p_values(bad_p):
    """Validate constructor-time rejection of invalid Yager parameters."""
    with pytest.raises(ValueError):
        TNorm.create("yager", p=bad_p)


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_instances_are_distinct(tnorm_name):
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
def test_create_and_to_dict_from_dict_roundtrip(tnorm_name):
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
def test_describe_params_detailed(tnorm_name):
    """Check detailed parameter introspection for all registered T-norms."""
    obj = _build_tnorm(tnorm_name)
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for k in obj._get_params():
        assert k in details


@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_registry_get_class_and_name(tnorm_name):
    """Check class lookup and reverse registered-name lookup."""
    cls = TNorm.get_class(tnorm_name)
    instance = cls(**_default_params_for_tnorm(tnorm_name))
    name = TNorm.get_registered_name(instance)
    assert isinstance(name, str)

#endregion

#region <axiom testing>

@pytest.mark.parametrize("tnorm_name", REGISTERED_TNORM_NAMES)
def test_tnorm_exhaustive_validity_and_properties(tnorm_name):
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
def test_tnorm_zero_and_one_boundary_axioms(tnorm_name):
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
def test_tnorm_monotonicity_axiom(tnorm_name):
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
def test_tnorm_associativity(tnorm_name):
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
def test_help(tnorm_name):
    """Check that each T-norm provides a valid help string."""
    obj = _build_tnorm(tnorm_name)
    doc = obj.help()
    assert isinstance(doc, str)
    assert len(doc.strip()) > 0

#endregion
