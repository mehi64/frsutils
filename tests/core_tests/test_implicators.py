import pytest
import numpy as np

import FRsutils.core.implicators as imp

import tests.syntetic_data_for_tests as sds


@pytest.mark.parametrize("func", [
    imp.imp_goedel,
    imp.imp_lukasiewicz,
    imp.imp_gaines,
    imp.imp_kleene_dienes,
    imp.imp_reichenbach
])
def test_implicators_valid_range(func):
    a = 0.3
    b = 0.7
    result = func(a, b)
    assert 0.0 <= result <= 1.0, f"{func.__name__} produced value out of range"

@pytest.mark.parametrize("func", [
    imp.imp_goedel,
    imp.imp_lukasiewicz,
    imp.imp_gaines,
    imp.imp_kleene_dienes,
    imp.imp_reichenbach
])
@pytest.mark.parametrize("a,b", [
    (-0.1, 0.5),
    (1.1, 0.5),
    (0.5, -0.2),
    (0.5, 1.2)
])
def test_implicators_invalid_input(func, a, b):
    with pytest.raises(ValueError):
        func(a, b)

def test_imp_goedel_behavior():
    assert imp.imp_goedel(0.3, 0.5) == 1.0
    assert imp.imp_goedel(0.8, 0.8) == 1.0
    assert imp.imp_goedel(0.8, 0.5) == 0.5
    assert imp.imp_goedel(0.8, 0.1) == 0.1

def test_imp_lukasiewicz_behavior():
    assert imp.imp_lukasiewicz(0.3, 0.5) == min(1.0, 1.0 - 0.3 + 0.5)



def test_imp_kleene_dienes_behavior():
    assert imp.imp_kleene_dienes(0.6, 0.3) == max(1.0 - 0.6, 0.3)

def test_imp_reichenbach_behavior():
    assert imp.imp_reichenbach(0.4, 0.7) == 1.0 - 0.4 + 0.4 * 0.7


    # imp.imp_lukasiewicz,
    # imp.imp_gaines,
    # imp.imp_kleene_dienes,
    # imp.imp_reichenbach

def test_goedel_implicator_outputs():
   
    data_dict = sds.syntetic_dataset_factory().implicator_testing_data()
    a_b = data_dict["a_b"].T
    expected = data_dict["goedel_outputs"]
    temp_implicator = np.vectorize(imp.imp_goedel)

    
    result = temp_implicator(a_b[0], a_b[1])
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_gaines_implicator_outputs():
   
    data_dict = sds.syntetic_dataset_factory().implicator_testing_data()
    a_b = data_dict["a_b"].T
    expected = data_dict["gaines_outputs"]
    temp_implicator = np.vectorize(imp.imp_gaines)

    
    result = temp_implicator(a_b[0], a_b[1])
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_luk_implicator_outputs():
   
    data_dict = sds.syntetic_dataset_factory().implicator_testing_data()
    a_b = data_dict["a_b"].T
    expected = data_dict["luk_outputs"]
    temp_implicator = np.vectorize(imp.imp_lukasiewicz)

    
    result = temp_implicator(a_b[0], a_b[1])
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_kd_implicator_outputs():
   
    data_dict = sds.syntetic_dataset_factory().implicator_testing_data()
    a_b = data_dict["a_b"].T
    expected = data_dict["kleene_dienes_outputs"]
    temp_implicator = np.vectorize(imp.imp_kleene_dienes)

    
    result = temp_implicator(a_b[0], a_b[1])
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_reichenbach_implicator_outputs():
   
    data_dict = sds.syntetic_dataset_factory().implicator_testing_data()
    a_b = data_dict["a_b"].T
    expected = data_dict["reichenbach_outputs"]
    temp_implicator = np.vectorize(imp.imp_reichenbach)

    
    result = temp_implicator(a_b[0], a_b[1])
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"


