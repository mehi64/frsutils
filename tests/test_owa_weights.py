# Test Type	Covered ✅
# ==============================
# Valid input (n=4)	✅
# n=1 case	✅
# Zero or negative n	✅
# Non-integer inputs	✅
# Very large n	✅
# Sum ≈ 1 validation	✅


import numpy as np
import pytest
from numpy.testing import assert_allclose
from tests import syntetic_data_for_tests as sdt

from FRsutils.core.owa_weights import _owa_suprimum_weights_linear, _owa_infimum_weights_linear



def test_suprimum_weights_normal_case():
    # Test correct computation of suprimum weights for n = 4
    n = 4
    expected = np.array([4, 3, 2, 1]) / (4 * (4 + 1) / 2)
    result = _owa_suprimum_weights_linear(n)

    # Assert that weights match expected values
    assert_allclose(result, expected)

    # Assert that the sum of weights is 1
    assert np.isclose(np.sum(result), 1.0)

def test_infimum_weights_normal_case():
    # Test correct computation of infimum weights for n = 4
    n = 4
    expected = np.array([1, 2, 3, 4]) / (4 * (4 + 1) / 2)
    result = _owa_infimum_weights_linear(n)

    # Assert that weights match expected values
    assert_allclose(result, expected)

    # Assert that the sum of weights is 1
    assert np.isclose(np.sum(result), 1.0)

def test_single_element_case():
    # Test edge case where n = 1, should return [1.0] for both functions
    assert np.array_equal(_owa_suprimum_weights_linear(1), np.array([1.0]))
    assert np.array_equal(_owa_infimum_weights_linear(1), np.array([1.0]))

@pytest.mark.parametrize("func", [
    _owa_suprimum_weights_linear,
    _owa_infimum_weights_linear
])
@pytest.mark.parametrize("invalid_n", [0, -1, -100])
def test_invalid_input_raises_value_error(func, invalid_n):
    # Test that n <= 0 raises a ValueError
    with pytest.raises(ValueError, match="n must be an integer number >= 1"):
        func(invalid_n)

@pytest.mark.parametrize("func", [
    _owa_suprimum_weights_linear,
    _owa_infimum_weights_linear
])
@pytest.mark.parametrize("non_int", [2.5, "5", None, [1], {2: 3}])
def test_non_integer_inputs_raise_type_error(func, non_int):
    # Test that non-integer inputs raise TypeError or ValueError
    with pytest.raises((TypeError, ValueError)):
        func(non_int)

@pytest.mark.parametrize("n", range(2, 21))
def test_sum_to_one_property_small_n(n):
    # Test that the weights sum to 1 for a range of small n values
    sup = _owa_suprimum_weights_linear(n)
    inf = _owa_infimum_weights_linear(n)

    # Ensure the weights are normalized
    assert np.isclose(np.sum(sup), 1.0, atol=1e-9)
    assert np.isclose(np.sum(inf), 1.0, atol=1e-9)

@pytest.mark.parametrize("n", [10_000, 100_000])
def test_large_n_behaves_correctly(n):
    # Test performance and correctness for large n
    sup = _owa_suprimum_weights_linear(n)
    inf = _owa_infimum_weights_linear(n)

    # Ensure output length is correct
    assert len(sup) == n
    assert len(inf) == n

    # Ensure weights are non-negative and sum to 1
    assert np.isclose(np.sum(sup), 1.0, atol=1e-6)
    assert np.isclose(np.sum(inf), 1.0, atol=1e-6)
    assert np.all(sup >= 0)
    assert np.all(inf >= 0)

# Helper to unpack test data
@pytest.fixture(scope="module")
def owa_test_data():
    return sdt.syntetic_dataset_factory().owa_weights_linear_testing_data()  # call as a method if it's inside a class
# Test cases as tuples: (function_to_test, input_n, expected_key)
test_cases = [
    (_owa_infimum_weights_linear, 5, "owa_infimum_weights_linear_len_5"),
    (_owa_infimum_weights_linear, 10, "owa_infimum_weights_linear_len_10"),
    (_owa_suprimum_weights_linear, 8, "owa_suprimum_weights_linear_len_8"),
    (_owa_suprimum_weights_linear, 13, "owa_suprimum_weights_linear_len_13"),
]


@pytest.mark.parametrize("func,n,key", test_cases)
def test_owa_weights_against_known_data(func, n, key, owa_test_data):
    expected = owa_test_data[key]
    result = func(n)
    np.testing.assert_allclose(result, expected, rtol=1e-6)