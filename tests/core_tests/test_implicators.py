# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fuzzy implicator operators."""

import numpy as np
import pytest
from frsutils.core.implicators import Implicator
from tests import reference_data_store as ds
from frsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test", experiment_name="test_implicators")
call_testsets = ds.get_implicator_scalar_testsets()
BOUNDARY_CASES = [
    (case["implicator"], case["a"], case["b"], case["expected"])
    for case in ds.get_implicator_boundary_cases()
]
BRANCH_EDGE_CASES = [
    (case["implicator"], case["a"], case["b"], case["expected"])
    for case in ds.get_implicator_branch_edge_cases()
]
registered_implicators = Implicator.list_available()

#region <Output correctness>
###############################################
###           Output correctness            ###
###############################################

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_scalar_call_valid(implicator_name):
    """
    checks if implicator can be called with scalars and return scalar
    """
    obj = Implicator.create(implicator_name)
    a, b = 0.73, 0.18
    result = obj(a, b)
    logger.info(f"{implicator_name} result for (0.73, 0.18): {result}")
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_scalar_call_matches_vectorized_outputs(implicator_name, testset):
    """Validates that scalar __call__(a, b) matches vectorized results for each test pair.
    
    This ensures implicator logic is consistent when called per-element.
    """

    if implicator_name not in testset["expected"]:
        pytest.skip(f"Expected output not available for {implicator_name}")

    obj = Implicator.create(implicator_name)
    a_b = testset["a_b"]
    expected = testset["expected"][implicator_name]

    for idx, (a_scalar, b_scalar) in enumerate(a_b):
        result = obj(a_scalar, b_scalar)
        expected_val = expected[idx]
        logger.info(f"{implicator_name} scalar call {idx}: ({a_scalar}, {b_scalar}) => {result:.6f}, expected {expected_val:.6f}")
        assert np.isclose(result, expected_val, atol=1e-6), f"Mismatch at index {idx} for {implicator_name}"


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_implicator_call_vector_output(implicator_name, testset):
    obj = Implicator.create(implicator_name)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]

    if implicator_name not in testset["expected"]:
        pytest.skip(f"Expected output not available for {implicator_name}")

    expected = testset["expected"][implicator_name]
    calculated = obj(a, b)
    np.testing.assert_allclose(calculated, expected, atol=1e-6)

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_small_grid_outputs_stay_in_unit_interval(implicator_name):
    """Check that each implicator maps a small [0, 1] grid into [0, 1]."""
    obj = Implicator.create(implicator_name)

    values = np.linspace(0.0, 1.0, 101)
    a_grid, b_grid = np.meshgrid(values, values)
    a_flat = a_grid.ravel()
    b_flat = b_grid.ravel()

    result = obj(a_flat, b_flat)

    assert result.shape == a_flat.shape
    assert np.all((0.0 <= result) & (result <= 1.0)), f"Out-of-range result from {implicator_name}"


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_matches_public_numpy_call(implicator_name):
    """Check that NumPy backend execution matches the public NumPy call."""
    obj = Implicator.create(implicator_name)
    a = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    b = np.array([0.0, 0.4, 0.5, 0.3, 1.0])

    public_result = obj(a, b)
    backend_result = obj.compute_backend(a, b, xp=np)

    np.testing.assert_allclose(backend_result, public_result, atol=1e-12)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_rejects_mismatched_shapes(implicator_name):
    """Check that public and backend calls reject incompatible input shapes."""
    obj = Implicator.create(implicator_name)
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([[0.1, 0.2, 0.3]])

    with pytest.raises(ValueError, match="Incompatible shapes"):
        obj(a, b)

    with pytest.raises(ValueError, match="Incompatible shapes"):
        obj.compute_backend(a, b, xp=np)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize(
    "a, b",
    [
        (-0.1, 0.5),
        (0.5, -0.1),
        (1.1, 0.5),
        (0.5, 1.1),
        (np.array([0.2, 1.2]), np.array([0.3, 0.4])),
        (np.array([0.2, 0.4]), np.array([0.3, 1.2])),
    ],
)
def test_implicator_rejects_out_of_range_values(implicator_name, a, b):
    """Check that validated calls reject membership values outside [0, 1]."""
    obj = Implicator.create(implicator_name)

    with pytest.raises(ValueError, match="range"):
        obj(a, b)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_rejects_out_of_range_values(implicator_name):
    """Check that backend validation rejects values outside [0, 1]."""
    obj = Implicator.create(implicator_name)
    a = np.array([0.2, 1.2])
    b = np.array([0.3, 0.4])

    with pytest.raises(ValueError, match="range"):
        obj.compute_backend(a, b, xp=np)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_validate_inputs_false_skips_validation(implicator_name):
    """Check that backend validation can be disabled by approximation engines."""
    obj = Implicator.create(implicator_name)
    a = np.array([1.2, 0.2])
    b = np.array([0.4, 1.1])

    result = obj.compute_backend(a, b, xp=np, validate_inputs=False)

    assert result.shape == a.shape


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_monotonicity_on_grid(implicator_name):
    """Check standard implicator monotonicity on a compact unit grid."""
    obj = Implicator.create(implicator_name)
    grid = np.linspace(0.0, 1.0, 11)

    for fixed_b in grid:
        values = obj(grid, np.full_like(grid, fixed_b))
        assert np.all(np.diff(values) <= 1e-12), (
            f"{implicator_name} must be non-increasing in the first argument."
        )

    for fixed_a in grid:
        values = obj(np.full_like(grid, fixed_a), grid)
        assert np.all(np.diff(values) >= -1e-12), (
            f"{implicator_name} must be non-decreasing in the second argument."
        )




@pytest.mark.parametrize("implicator_name, a, b, expected", BOUNDARY_CASES)
def test_implicator_boundary_cases_match_contract(implicator_name, a, b, expected):
    """Check representative boundary values against each implicator contract."""
    obj = Implicator.create(implicator_name)

    result = obj(a, b)

    assert np.isclose(result, expected, atol=1e-12)




@pytest.mark.parametrize("implicator_name, a, b, expected", BRANCH_EDGE_CASES)
def test_implicator_branch_edge_cases_match_expected_values(implicator_name, a, b, expected):
    """Check branch-sensitive formulas on vector inputs."""
    obj = Implicator.create(implicator_name)

    result = obj(a, b)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def _import_cupy_or_skip():
    """Return CuPy when a working CUDA-backed installation is available."""
    cp = pytest.importorskip("cupy")
    try:
        cp.asnumpy(cp.asarray([0.0]))
    except Exception as exc:  # pragma: no cover - depends on local CUDA setup.
        pytest.skip(f"CuPy is installed but unavailable in this environment: {exc}")
    return cp


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_cupy_matches_numpy(implicator_name):
    """Check that CuPy backend execution matches NumPy backend execution."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a_np = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    b_np = np.array([0.0, 0.4, 0.5, 0.3, 1.0])
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)

    numpy_result = obj.compute_backend(a_np, b_np, xp=np)
    cupy_result = obj.compute_backend(a_cp, b_cp, xp=cp)

    np.testing.assert_allclose(cp.asnumpy(cupy_result), numpy_result, atol=1e-12)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_cupy_returns_cupy_array(implicator_name):
    """Check that CuPy backend execution keeps the result on the CuPy backend."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a = cp.asarray([0.0, 0.2, 0.5, 0.8, 1.0])
    b = cp.asarray([0.0, 0.4, 0.5, 0.3, 1.0])

    result = obj.compute_backend(a, b, xp=cp)

    assert isinstance(result, cp.ndarray)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_implicator_compute_backend_cupy_handles_scalar_like_arrays(implicator_name, shape):
    """Check that CuPy backend execution handles scalar-like equal-shape arrays."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a_np = np.full(shape, 0.73)
    b_np = np.full(shape, 0.18)
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)

    numpy_result = obj.compute_backend(a_np, b_np, xp=np)
    cupy_result = obj.compute_backend(a_cp, b_cp, xp=cp)

    assert isinstance(cupy_result, cp.ndarray)
    assert cupy_result.shape == a_np.shape
    np.testing.assert_allclose(cp.asnumpy(cupy_result), numpy_result, atol=1e-12)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_cupy_rejects_mismatched_shapes(implicator_name):
    """Check that CuPy backend validation rejects incompatible input shapes."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a = cp.asarray([0.1, 0.2, 0.3])
    b = cp.asarray([[0.1, 0.2, 0.3]])

    with pytest.raises(ValueError, match="Incompatible shapes"):
        obj.compute_backend(a, b, xp=cp)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_compute_backend_cupy_rejects_out_of_range_values(implicator_name):
    """Check that CuPy backend validation rejects values outside [0, 1]."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a = cp.asarray([0.2, 1.2])
    b = cp.asarray([0.3, 0.4])

    with pytest.raises(ValueError, match="range"):
        obj.compute_backend(a, b, xp=cp)


@pytest.mark.parametrize("implicator_name, a, b, expected", BRANCH_EDGE_CASES)
def test_implicator_compute_backend_cupy_branch_edge_cases(implicator_name, a, b, expected):
    """Check branch-sensitive formulas with CuPy backend arrays."""
    cp = _import_cupy_or_skip()
    obj = Implicator.create(implicator_name)
    a_cp = cp.asarray(a)
    b_cp = cp.asarray(b)

    result = obj.compute_backend(a_cp, b_cp, xp=cp)

    assert isinstance(result, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(result), expected, atol=1e-12)


def test_implicator_create_unknown_alias_raises_value_error():
    """Check that the factory rejects unknown implicator aliases."""
    with pytest.raises(ValueError, match="Unknown alias"):
        Implicator.create("not_an_implicator")



ALIAS_EQUIVALENCE_CASES = [
    ("luk", "lukasiewicz"),
    ("kleene", "kleenedienes"),
    ("kd", "kleenedienes"),
    ("product", "goguen"),
]


@pytest.mark.parametrize("alias_name, canonical_name", ALIAS_EQUIVALENCE_CASES)
def test_implicator_aliases_create_same_class_and_same_outputs(alias_name, canonical_name):
    """Check that public aliases resolve to the canonical implicator behavior."""
    alias_obj = Implicator.create(alias_name)
    canonical_obj = Implicator.create(canonical_name)
    a = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    b = np.array([0.0, 0.4, 0.5, 0.3, 1.0])

    assert type(alias_obj) is type(canonical_obj)
    assert Implicator.get_class(alias_name) is Implicator.get_class(canonical_name)
    np.testing.assert_allclose(alias_obj(a, b), canonical_obj(a, b), atol=1e-12)


@pytest.mark.parametrize("mixed_case_name", ["LUK", "GoEdEl", "KLEENE", "Product"])
def test_implicator_create_is_case_insensitive(mixed_case_name):
    """Check that factory aliases are case-insensitive."""
    obj = Implicator.create(mixed_case_name)

    assert isinstance(obj, Implicator)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_create_from_spec_accepts_name_params_spec(implicator_name):
    """Check preferred nested specs with name and params keys."""
    obj = Implicator.create_from_spec({"name": implicator_name, "params": {}})

    assert isinstance(obj, Implicator)
    assert type(obj) is Implicator.get_class(implicator_name)


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_create_from_spec_accepts_legacy_type_spec(implicator_name):
    """Check legacy compact specs with a type key."""
    obj = Implicator.create_from_spec({"type": implicator_name})

    assert isinstance(obj, Implicator)
    assert type(obj) is Implicator.get_class(implicator_name)


def test_implicator_create_from_spec_returns_existing_instance():
    """Check that already constructed implicator instances pass through unchanged."""
    existing = Implicator.create("lukasiewicz")

    result = Implicator.create_from_spec(existing)

    assert result is existing


def test_implicator_create_from_spec_accepts_internal_instance_marker():
    """Check the internal normalizer marker used for pass-through instances."""
    existing = Implicator.create("goedel")

    result = Implicator.create_from_spec({"__instance__": existing})

    assert result is existing


def test_implicator_create_from_spec_rejects_invalid_params_type():
    """Check that nested spec params must be a dictionary."""
    with pytest.raises(TypeError, match="params.*dict"):
        Implicator.create_from_spec({"name": "lukasiewicz", "params": ["bad"]})


@pytest.mark.parametrize(
    "unsupported_spec",
    [
        [],
        {"params": {}},
        {"unknown": "lukasiewicz"},
        object(),
    ],
)
def test_implicator_create_from_spec_rejects_unsupported_spec(unsupported_spec):
    """Check that unsupported spec formats fail with a clear type error."""
    with pytest.raises(TypeError, match="Unsupported component spec"):
        Implicator.create_from_spec(unsupported_spec)

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_equivalence_of_constructor_create_fromdict_with_synthetic_data(implicator_name, testset):
    """Verifies that explicitly constructed, factory-created, and deserialized implicators yield the same outputs.
    
    For each implicator:
    - Instantiate explicitly using the class
    - Instantiate via mixin's .create()
    - Instantiate via to_dict/from_dict
    Then compare all outputs from get_implicator_scalar_testsets().
    """

    # Skip if implicator not in expected testset
    if implicator_name not in testset["expected"]:
        pytest.skip(f"Expected output not available for {implicator_name}")

    cls = Implicator.get_class(implicator_name)
    implicator1 = cls()  # Explicit constructor
    implicator2 = Implicator.create(implicator_name)  # Registry-based constructor
    implicator3 = Implicator.from_dict(implicator1.to_dict())  # Serialization roundtrip

    id1, id2, id3 = id(implicator1), id(implicator2), id(implicator3)

    assert id1 != id2, f"{implicator_name}: implicator1 and implicator2 share the same object ID!"
    assert id2 != id3, f"{implicator_name}: implicator2 and implicator3 share the same object ID!"
    assert id1 != id3, f"{implicator_name}: implicator1 and implicator3 share the same object ID!"

    a = testset["a_b"][:, 0]
    b = testset["a_b"][:, 1]

    expected = testset["expected"][implicator_name]
    

    out1 = implicator1(a, b)
    out2 = implicator2(a, b)
    out3 = implicator3(a, b)

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    np.testing.assert_allclose(out2, out3, atol=1e-7)
    np.testing.assert_allclose(out1, out3, atol=1e-7)
    
    np.testing.assert_allclose(out1, expected, atol=1e-6)


    logger.info(f"{implicator_name} equivalence check passed.")

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_equivalence_of_constructor_create_fromdict_with_random_data(implicator_name):
    """Check constructor, factory, and from_dict equivalence on random vectors."""

    rng = np.random.default_rng(seed=42)
    a = rng.uniform(0, 1, size=10_000)
    b = rng.uniform(0, 1, size=10_000)

    cls = Implicator.get_class(implicator_name)
    implicator1 = cls()
    implicator2 = Implicator.create(implicator_name)
    implicator3 = Implicator.from_dict(implicator1.to_dict())

    id1, id2, id3 = id(implicator1), id(implicator2), id(implicator3)

    assert id1 != id2, f"{implicator_name}: implicator1 and implicator2 share the same object ID!"
    assert id2 != id3, f"{implicator_name}: implicator2 and implicator3 share the same object ID!"
    assert id1 != id3, f"{implicator_name}: implicator1 and implicator3 share the same object ID!"

    out1 = implicator1(a, b)
    out2 = implicator2(a, b)
    out3 = implicator3(a, b)

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    np.testing.assert_allclose(out2, out3, atol=1e-7)
    np.testing.assert_allclose(out1, out3, atol=1e-7)

    logger.info(f"{implicator_name}: random input equivalence test passed.")


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_matrix_consistency_with_scalar_application(implicator_name):
    """Check matrix evaluation against scalar element-wise evaluation."""
    rng = np.random.default_rng(seed=42)
    n = 30
    A = rng.uniform(0, 1, size=(n, n))
    B = rng.uniform(0, 1, size=(n, n))

    implicator = Implicator.create(implicator_name)

    # Apply implicator directly on matrices
    direct_result = implicator(A, B)

    # Scalar application for each (i, j)
    scalar_result = np.empty_like(A)
    for i in range(n):
        for j in range(n):
            scalar_result[i, j] = implicator(A[i, j], B[i, j])

    np.testing.assert_allclose(
        scalar_result, direct_result, atol=1e-7,
        err_msg=f"{implicator_name}: matrix and scalar application mismatch"
    )

    logger.info(f"{implicator_name}: scalar vs matrix application match confirmed.")

#endregion

#region <Non-calculational behaviors>
###############################################
###       Non-calculational behaviors       ###
###############################################

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_implicator_instances_are_distinct(implicator_name):
    """Ensures that implicator1 (direct), implicator2 (create), and implicator3 (from_dict)
    
    are separate instances in memory.
    """
    cls = Implicator.get_class(implicator_name)
    implicator1 = cls()
    implicator2 = Implicator.create(implicator_name)
    implicator3 = Implicator.from_dict(implicator1.to_dict())

    id1, id2, id3 = id(implicator1), id(implicator2), id(implicator3)

    assert id1 != id2, f"{implicator_name}: implicator1 and implicator2 share the same object ID!"
    assert id2 != id3, f"{implicator_name}: implicator2 and implicator3 share the same object ID!"
    assert id1 != id3, f"{implicator_name}: implicator1 and implicator3 share the same object ID!"

    logger.info(f"{implicator_name}: All three implicators are distinct in memory.")


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_create_and_to_dict_from_dict_roundtrip(implicator_name):
    """
    tests to_dict(), from_dict() and create() 
    """
    obj = Implicator.create(implicator_name)
    serialized = obj.to_dict()

    assert "type" in serialized
    assert "name" in serialized
    assert "params" in serialized

    reconstructed = Implicator.from_dict(serialized)
    assert isinstance(reconstructed, Implicator)
    assert reconstructed.name == obj.name


@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_describe_params_detailed_keys(implicator_name):
    """
    tests values of params and deteriled params. check them in log
    """
    # TODO: This test needs consideration when implicators with parameters introduce
    obj = Implicator.create(implicator_name)
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)

    for param in obj._get_params().keys():
        assert param in details

    logger.info(implicator_name + ', params:' + str(obj._get_params())+ ', detailed params:' + str(details))

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_registry_get_class_and_name(implicator_name):
    # TODO: just checks class name is of str type. not very helpful
    cls = Implicator.get_class(implicator_name)
    instance = cls()
    name = Implicator.get_registered_name(instance)
    assert isinstance(name, str)
    logger.info(f"{implicator_name}, registered_name: {name}")

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
def test_help(implicator_name):
    """
    tests values of params and deteriled params. check them in log
    """
    # TODO: when a class does not have docstring, it returns the base calss docstring. This is wrong
    obj = Implicator.create(implicator_name)
    details = obj.help()
    assert isinstance(details, str)
    
    logger.info(implicator_name + ', class docstring:' + details)

#endregion
