# import numpy as np
# import tests.syntetic_data_for_tests as sds
# import pytest
# from numpy.testing import assert_almost_equal

# from FRsutils.core.fuzzy_quantifiers import fuzzy_quantifier1, fuzzy_quantifier_quad  # Replace 'your_module' with the actual module name


# def test_fuzzy_quantifier1_increasing_scalar():
#     assert fuzzy_quantifier1(0.0, 0.3, 0.7) == 0
#     assert fuzzy_quantifier1(0.7, 0.3, 0.7) == 1
#     assert_almost_equal(fuzzy_quantifier1(0.5, 0.3, 0.7), 0.5)


# def test_fuzzy_quantifier1_decreasing_scalar():
#     assert fuzzy_quantifier1(0.0, 0.3, 0.7, increasing=False) == 1
#     assert fuzzy_quantifier1(0.7, 0.3, 0.7, increasing=False) == 0
#     assert_almost_equal(fuzzy_quantifier1(0.5, 0.3, 0.7, increasing=False), 0.5)


# def test_fuzzy_quantifier1_array():
#     p = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
#     expected_incr = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
#     expected_decr = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
#     assert_almost_equal(fuzzy_quantifier1(p, 0.3, 0.7), expected_incr)
#     assert_almost_equal(fuzzy_quantifier1(p, 0.3, 0.7, increasing=False), expected_decr)


# def test_fuzzy_quantifier1_alpha_equals_beta():
#     with pytest.raises(ZeroDivisionError):
#         fuzzy_quantifier1(0.5, 0.5, 0.5)


# def test_fuzzy_quantifier_quad_scalar():
#     assert fuzzy_quantifier_quad(0.0, 0.3, 0.7) == 0.0
#     assert fuzzy_quantifier_quad(1.0, 0.3, 0.7) == 1.0
#     assert_almost_equal(fuzzy_quantifier_quad(0.5, 0.3, 0.7), 0.5)


# def test_fuzzy_quantifier_quad_array():
#     x = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
#     expected = np.array([
#         0.0,        # x <= alpha
#         0.0,        # x == alpha
#         0.125,      # transition zone
#         0.5,
#         0.875,
#         1.0,        # x == beta
#         1.0         # x > beta
#     ])
#     output = fuzzy_quantifier_quad(x, 0.3, 0.7)
#     assert_almost_equal(output, expected)


# def test_fuzzy_quantifier_quad_invalid_alpha_beta():
#     result = fuzzy_quantifier_quad(np.array([0.5]), 0.7, 0.3)
#     assert np.all(np.isfinite(result))


