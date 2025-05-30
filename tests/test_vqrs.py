import numpy as np
import tests.syntetic_data_for_tests as sds

from FRsutils.core.models.vqrs import VQRS
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp


def test_vqrs_lower_upper_approximations_quadratic_fuzzyquantifier():
    data_dict = sds.syntetic_dataset_factory().VQRS_testing_dataset()
    expected_lowerBound = data_dict["lower_bound"]
    expected_upperBound = data_dict["upper_bound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    alpha_lower = data_dict["alpha_lower"]
    beta_lower  = data_dict["beta_lower"]
    alpha_upper = data_dict["alpha_upper"]
    beta_upper  = data_dict["beta_upper"]

    model = VQRS(sim_matrix,
                     y, 
                     alpha_lower= alpha_lower,
                     beta_lower= beta_lower,
                     alpha_upper= alpha_upper,
                     beta_upper= beta_upper
                     )
    
    upper = model.upper_approximation()
    lower = model.lower_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "LB outputs are not the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "UB outputs are not the expected values"