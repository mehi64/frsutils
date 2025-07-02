import numpy as np
import pytest
from FRsutils.core.tnorms import TNorm
from tests import synthetic_data_store as ds
from FRsutils.utils.logger.logger_util import get_logger

logger = get_logger(env="test",
                    experiment_name="test_tnorms1")

call_testsets = ds.get_tnorm_call_testsets()
registered_tnorms = TNorm.list_available()

#region <test output correctness>
###############################################
###                                         ###
###         test output correctness         ###
###                                         ###
###############################################

@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
def test_tnorm_all_pairs_from_0_to_1(tnorm_name):
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    values = np.linspace(0, 1, 101)  # 0.0 to 1.0 step 0.01

    for a in values:
        for b in values:
            try:
                result = obj(np.array(a), np.array(b))
                assert 0.0 <= result <= 1.0
            except Exception as e:
                raise AssertionError(f"{tnorm_name} failed for a={a}, b={b}: {e}")

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_scalar_inputs(tnorm_name):
    """
    cheks if the scalar inputs can be handeled correctly by tnorms
    """
    obj = TNorm.create(tnorm_name, **({"p": 0.835} if tnorm_name == "yager" else {}))
    a, b = 0.73, 0.18
    result = obj(a, b)
    logger.info(tnorm_name + ', ' + str(result))
    assert np.isscalar(result) or np.shape(result) == ()

    
@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_tnorm_call_output_matches_expected(tnorm_name, testset):
    """
    tests if generated outputs of __call__ are the same as calculated by hand
    this test uses get_tnorm_call_testsets() in synthetic_data_store
    Yager tnorm is not tested here
    """
    if "yager" in tnorm_name and "p=" not in tnorm_name:
        return
    obj = TNorm.create(tnorm_name)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    expected_key = tnorm_name
    if "yager" in tnorm_name:
        if "p=" in tnorm_name:
            expected_key = tnorm_name
        else:
            return
        
    calc = obj(a, b)
    exp = testset["expected"][expected_key]
    np.testing.assert_allclose(calc, exp, atol=1e-6)


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_scalar_call_output_matches_expected_values(tnorm_name, testset):
    """
    @brief Validates that scalar TNorm calls produce the correct result as per the expected values.

    This ensures that the scalar invocation path of __call__ yields the same output
    as vectorized usage for each (a, b) test pair in get_tnorm_call_testsets().
    """
    if "yager" in tnorm_name and "p=" not in tnorm_name:
        return  # skip unparameterized yager fallback

    # Detect Yager testset key
    params = {"p": 0.835} if "p=0.835" in testset["expected"] else \
             {"p": 5.0} if "p=5.0" in testset["expected"] else {}

    obj = TNorm.create(tnorm_name, **params)
    a_b = testset["a_b"]

    if "yager" in tnorm_name:
        expected_key = [k for k in testset["expected"].keys() if "yager" in k and str(params["p"]) in k][0]
    else:
        expected_key = tnorm_name

    expected = testset["expected"][expected_key]

    for i, (a_val, b_val) in enumerate(a_b):
        result = obj(a_val, b_val)
        exp_val = expected[i]
        logger.info(f"{tnorm_name} scalar test {i}: ({a_val}, {b_val}) => {result:.6f} (expected {exp_val:.6f})")
        assert np.isclose(result, exp_val, atol=1e-6), \
            f"{tnorm_name} scalar call mismatch at index {i}: got {result}, expected {exp_val}"


@pytest.mark.parametrize("tnorm_name", ["yager"])
@pytest.mark.parametrize("p", [0.835, 5.0])
@pytest.mark.parametrize("testset", call_testsets)
def test_yager_parametrized_behavior(tnorm_name, p, testset):
    """
    tests yager __call__ function with get_tnorm_call_testsets()
    from shnthetic_data_store.py
    """
    obj = TNorm.create("yager", p=p)
    a_b = testset["a_b"]
    a = a_b[:, 0]
    b = a_b[:, 1]
    key = f"yager_p={p}" if p == 0.835 else "yager_p=5.0"
    
    result = obj(a, b)
    exp = testset["expected"][key]

    np.testing.assert_allclose(result, exp, atol=1e-5)


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_reduce_consistency_with_call(tnorm_name):
    """
    tests if the output the __call__ on a random 1D array
    is the same as output for reduce"""
    
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    data_ = np.random.rand(200,200)

    reduced = obj.reduce(data_)
    data_ = data_.T

    results = []
    for row in data_:
        res = row[0]
        for i in range(1, len(row)):
            res = obj(np.array(res), np.array(row[i]))
        results.append(float(res))

    np.testing.assert_allclose(reduced, results, atol=1e-7)

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_equivalence_of_constructor_create_fromdict_with_random_data(tnorm_name):
    """
    @brief Tests whether constructor, create(), and from_dict() produce identical outputs
           for randomly generated vectors.
    """
    rng = np.random.default_rng(seed=123)
    a = rng.uniform(0, 1, size=1000)
    b = rng.uniform(0, 1, size=1000)

    cls = TNorm.get_class(tnorm_name)
    tnorm1 = cls(**({"p": 2.0} if tnorm_name == "yager" else {}))
    tnorm2 = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    tnorm3 = TNorm.from_dict(tnorm1.to_dict())

    out1 = tnorm1(a, b)
    out2 = tnorm2(a, b)
    out3 = tnorm3(a, b)

    np.testing.assert_allclose(out1, out2, atol=1e-7)
    np.testing.assert_allclose(out2, out3, atol=1e-7)
    np.testing.assert_allclose(out1, out3, atol=1e-7)

    logger.info(f"{tnorm_name}: random input equivalence test passed.")

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_equivalence_of_constructor_create_fromdict(tnorm_name, testset):
    """
    @brief Verifies that constructor, factory, and from_dict instances all behave identically
           on predefined test data from get_tnorm_call_testsets().
    """
    # Skip parametric Yager if p is not defined
    if "yager" in tnorm_name and "p=" not in tnorm_name:
        return

    cls = TNorm.get_class(tnorm_name)
    params = {"p": 0.835} if "yager" in tnorm_name and "p=0.835" in testset["expected"] else \
             {"p": 5.0} if "yager" in tnorm_name and "p=5.0" in testset["expected"] else {}

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

    logger.info(f"{tnorm_name}: synthetic testset equivalence passed.")


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
@pytest.mark.parametrize("testset", call_testsets)
def test_scalar_call_matches_vectorized_outputs(tnorm_name, testset):
    """
    @brief Ensures TNorm scalar calls match vectorized results for each test pair.
    """

    if "yager" in tnorm_name and "p=" not in tnorm_name:
        return

    params = {"p": 0.835} if "p=0.835" in testset["expected"] else \
             {"p": 5.0} if "p=5.0" in testset["expected"] else {}

    obj = TNorm.create(tnorm_name, **params)
    a_b = testset["a_b"]

    # Get the matching expected key
    if "yager" in tnorm_name:
        expected_key = [k for k in testset["expected"].keys() if "yager" in k and str(params["p"]) in k][0]
    else:
        expected_key = tnorm_name

    expected = testset["expected"][expected_key]

    for idx, (a_val, b_val) in enumerate(a_b):
        result = obj(a_val, b_val)
        expected_val = expected[idx]
        assert np.isclose(result, expected_val, atol=1e-6), \
            f"Mismatch at index {idx} for {tnorm_name}: got {result}, expected {expected_val}"

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_tnorm_matrix_consistency_with_scalar_application(tnorm_name):
    """
    @brief Ensures that applying T-norm to A and B matrices is the same as
           applying scalar T-norm to A[i,j], B[i,j] for each element.

    This verifies consistency between vectorized and scalar logic.
    """
    rng = np.random.default_rng(seed=42)
    n = 400
    A = rng.uniform(0, 1, size=(n, n))
    B = rng.uniform(0, 1, size=(n, n))

    tnorm = TNorm.create(tnorm_name)

    # Direct matrix-wise application
    matrix_result = tnorm(A, B)

    # Scalar-wise application
    scalar_result = np.empty_like(A)
    for i in range(n):
        for j in range(n):
            scalar_result[i, j] = tnorm(A[i, j], B[i, j])

    np.testing.assert_allclose(
        matrix_result, scalar_result, atol=1e-7,
        err_msg=f"{tnorm_name}: matrix vs scalar mismatch"
    )

    logger.info(f"{tnorm_name}: scalar vs matrix application match confirmed.")

#endregion

#region <test non-calculational aspects>
###############################################
###                                         ###
###     test non-calculational aspects      ###
###                                         ###
###############################################

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_tnorm_instances_are_distinct(tnorm_name):
    """
    @brief Ensures that tnorm1 (direct), tnorm2 (create), and tnorm3 (from_dict)
           are separate instances in memory.
    """
    cls = TNorm.get_class(tnorm_name)
    tnorm1 = cls(**({"p": 2.0} if tnorm_name == "yager" else {}))
    tnorm2 = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    tnorm3 = TNorm.from_dict(tnorm1.to_dict())

    id1, id2, id3 = id(tnorm1), id(tnorm2), id(tnorm3)

    assert id1 != id2, f"{tnorm_name}: tnorm1 and tnorm2 share the same object ID!"
    assert id2 != id3, f"{tnorm_name}: tnorm2 and tnorm3 share the same object ID!"
    assert id1 != id3, f"{tnorm_name}: tnorm1 and tnorm3 share the same object ID!"

    logger.info(f"{tnorm_name}: All three tnorms are distinct in memory.")


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_create_and_to_dict_from_dict_roundtrip(tnorm_name):
    """
    tests to_dict(), from_dict() and create()
    """
    obj = TNorm.create(tnorm_name)
    assert isinstance(obj, TNorm)

    data = obj.to_dict()
    assert "name" in data
    assert "type" in data
    assert "params" in data

    obj2 = TNorm.from_dict(data)
    assert isinstance(obj2, TNorm)
    assert obj2.name == obj.name

    logger.info(tnorm_name + ', 1st obj to_dict:' + str(data)+ ', 2nd obj from_dict:' + str(obj2.to_dict()))


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_describe_params_detailed(tnorm_name):
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    details = obj.describe_params_detailed()
    assert isinstance(details, dict)
    for k in obj._get_params():
        assert k in details
    
    logger.info(tnorm_name + ', params:' + str(obj._get_params())+ ', detailed params:' + str(details))

@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_registry_get_class_and_name(tnorm_name):
    # TODO: not really helpfull. just checks if help returns string
    cls = TNorm.get_class(tnorm_name)
    instance = cls(**({"p": 2.0} if tnorm_name == "yager" else {}))
    name = TNorm.get_registered_name(instance)
    assert isinstance(name, str)
    logger.info(tnorm_name + ', registered_name:' + str(name))

#endregion

#region <axiom testing>

@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
def test_tnorm_exhaustive_validity_and_properties(tnorm_name):
    # this test might not be valid for all tnorms
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
    values = np.linspace(0, 1, 11)

    for a in values:
        for b in values:
            a_np = np.array(a)
            b_np = np.array(b)

            result = obj(a_np, b_np)
            assert 0.0 <= result <= 1.0, f"{tnorm_name} gave result {result} for a={a}, b={b}"

            # Commutativity
            result_rev = obj(b_np, a_np)
            assert np.isclose(result, result_rev, atol=1e-8), f"{tnorm_name} is not commutative: T({a},{b})={result} vs T({b},{a})={result_rev}"

            # Boundary condition
            result_boundary = obj(a_np, np.array(1.0))
            assert np.isclose(result_boundary, a, atol=1e-8), f"{tnorm_name} failed boundary T({a},1.0)={result_boundary}, expected {a}"

@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
def test_tnorm_associativity(tnorm_name):
    # this test might not be valid for all tnorms
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if tnorm_name == "yager" else {}))
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

                    assert np.isclose(left, right, atol=1e-6), \
                        f"{tnorm_name} failed associativity: T({a},T({b},{c}))={left} vs T(T({a},{b}),{c})={right}"
                except Exception as e:
                    raise AssertionError(f"{tnorm_name} error on associativity for a={a}, b={b}, c={c}: {e}")


@pytest.mark.parametrize("tnorm_name", list(registered_tnorms.keys()))
def test_help(tnorm_name):
    """
    @brief Checks that each TNorm provides a valid help string (docstring).
    """
    # not very helpfull test
    obj = TNorm.create(tnorm_name, **({"p": 2.0} if "yager" in tnorm_name else {}))
    doc = obj.help()
    assert isinstance(doc, str)
    assert len(doc.strip()) > 0
    logger.info(f"{tnorm_name} help text: {doc[:60]}...")



#endregion