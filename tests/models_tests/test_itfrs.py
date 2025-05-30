import numpy as np

import tests.syntetic_data_for_tests as sds
from FRsutils.core.models.itfrs import ITFRS
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp

def test_itfrs_approximations_reichenbach_imp_product_tnorm():
    data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
    expected_lowerBound = data_dict["Reichenbach_lowerBound"]
    expected_upperBound = data_dict["prod_tn_upperBound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    tnrm = tn.ProductTNorm()

    model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_reichenbach)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "outputs are not similatr to the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "outputs are not similar to the expected values"

def test_itfrs_approximations_KD_imp_product_tnorm():
    data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
    expected_lowerBound = data_dict["KD_lowerBound"]
    expected_upperBound = data_dict["prod_tn_upperBound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    tnrm = tn.ProductTNorm()
    
    model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_kleene_dienes)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "outputs are not the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "outputs are not similar to the expected values"


def test_itfrs_approximations_Luk_imp_product_tnorm():
    data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
    expected_lowerBound = data_dict["Luk_lowerBound"]
    expected_upperBound = data_dict["prod_tn_upperBound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    tnrm = tn.ProductTNorm()

    model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_lukasiewicz)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "outputs are not the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "outputs are not similar to the expected values"


def test_itfrs_approximations_Goedel_imp_product_tnorm():
    data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
    expected_lowerBound = data_dict["Goedel_lowerBound"]
    expected_upperBound = data_dict["prod_tn_upperBound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    tnrm = tn.ProductTNorm()

    model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_goedel)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "outputs are not the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "outputs are not similar to the expected values"


def test_itfrs_approximations_Gaines_imp_product_tnorm():
    data_dict = sds.syntetic_dataset_factory().ITFRS_testing_dataset()
    expected_lowerBound = data_dict["Gaines_lowerBound"]
    expected_upperBound = data_dict["prod_tn_upperBound"]
    sim_matrix = data_dict["sim_matrix"]
    y = data_dict["y"]

    tnrm = tn.ProductTNorm()

    model = ITFRS(sim_matrix, y, tnorm=tnrm, implicator=imp.imp_gaines)
    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == (5,)
    assert upper.shape == (5,)
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    closeness_LB = np.isclose(lower, expected_lowerBound)
    assert np.all(closeness_LB), "outputs are not the expected values"

    closeness_UB = np.isclose(upper, expected_upperBound)
    assert np.all(closeness_UB), "outputs are not similar to the expected values"
