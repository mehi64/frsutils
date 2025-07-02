import numpy as np
import pytest
from FRsutils.core.implicators import Implicator
from tests import synthetic_data_store as ds
from FRsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test", experiment_name="test_implicators")
call_testsets = ds.get_implicator_scalar_testsets()
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
    """
    @brief Validates that scalar __call__(a, b) matches vectorized results for each test pair.

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
def test_implicator_exhaustive_grid_no_exception(implicator_name):
    """
    @brief Tests whether implicator handles full [0,1] grid without exceptions.

    Uses meshgrid of 0.0 to 1.0 with 101 values for each axis (step=0.01).
    Applies implicator elementwise on all combinations.
    """
    obj = Implicator.create(implicator_name)
    
    values = np.linspace(0.0, 1.0, 1001)
    a_grid, b_grid = np.meshgrid(values, values)
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()

    try:
        result = obj(a_flat, b_flat)
        assert result.shape == a_flat.shape
        assert np.all((0.0 <= result) & (result <= 1.0)), f"Out-of-range result from {implicator_name}"
    except Exception as e:
        pytest.fail(f"{implicator_name} raised an exception during grid evaluation: {e}")

@pytest.mark.parametrize("implicator_name", list(registered_implicators.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_equivalence_of_constructor_create_fromdict_with_synthetic_data(implicator_name, testset):
    """
    @brief Verifies that explicitly constructed, factory-created, and deserialized implicators yield the same outputs.

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
    """
    @brief Ensures that direct instantiation, mixin-based creation, and from_dict deserialization
           produce consistent results for random input data in [0, 1].

    For each implicator:
    - Instantiate directly using the class
    - Instantiate using the create() factory
    - Instantiate using to_dict() + from_dict()
    Then compare their outputs on randomly generated input vectors.
    """

    rng = np.random.default_rng(seed=42)  # deterministic random input
    a = rng.uniform(0, 1, size=1000000)
    b = rng.uniform(0, 1, size=1000000)

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
    """
    @brief Ensures that applying implicator to A and B matrices is the same as
           applying scalar implicator to A[i,j], B[i,j] for each element.

    This verifies consistency between vectorized and scalar logic.
    """
    rng = np.random.default_rng(seed=42)
    n = 400  # small enough for fast testing, large enough for generality
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
    """
    @brief Ensures that implicator1 (direct), implicator2 (create), and implicator3 (from_dict)
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