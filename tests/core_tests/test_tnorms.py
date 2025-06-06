
import FRsutils.core.tnorms as tn
import tests.syntetic_data_for_tests as sds
import numpy as np
import sys

def test_tn_minimum_scalar_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
    a_b = data_dict["a_b"]
    expected = data_dict["minimum_outputs"]
    temp_tnorm = tn.MinTNorm()

    result = []

    l = len(a_b)
    for i in range(l):
        result.append(temp_tnorm.reduce(a_b[i]))
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

    # if "pytest" in sys.modules:
    #     aa = sys.modules["pytest"]
    if hasattr(sys, 'gettrace') and sys.gettrace():
        assert False

def test_tn_product_scalar_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
    a_b = data_dict["a_b"]
    expected = data_dict["product_outputs"]
    temp_tnorm = tn.ProductTNorm()

    result = []

    l = len(a_b)
    for i in range(l):
        result.append(temp_tnorm.reduce(a_b[i]))
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_tn_luk_scalar_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_scalar_testing_data()
    a_b = data_dict["a_b"]
    expected = data_dict["luk_outputs"]
    temp_tnorm = tn.LukasiewiczTNorm()

    result = []

    l = len(a_b)
    for i in range(l):
        result.append(temp_tnorm.reduce(a_b[i]))
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_tn_minimum_nxnx2_map_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()
    similarity_matrix = data_dict["similarity_matrix"]
    label_mask  = data_dict["label_mask"]
    expected = data_dict["minimum_outputs"]
    temp_tnorm = tn.MinTNorm()

    result = temp_tnorm(similarity_matrix, label_mask)
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_tn_product_nxnx2_map_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()
    similarity_matrix = data_dict["similarity_matrix"]
    label_mask  = data_dict["label_mask"]
    expected = data_dict["product_outputs"]
    temp_tnorm = tn.ProductTNorm()

    result = temp_tnorm(similarity_matrix, label_mask)
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"

def test_tn_luk_nxnx2_map_values():
    data_dict = sds.syntetic_dataset_factory().tnorm_nxnx2_testing_dataset()
    similarity_matrix = data_dict["similarity_matrix"]
    label_mask  = data_dict["label_mask"]
    expected = data_dict["luk_outputs"]
    temp_tnorm = tn.LukasiewiczTNorm()

    result = temp_tnorm(similarity_matrix, label_mask)
    
    closeness = np.isclose(result, expected)
    assert np.all(closeness), "outputs are not the expected values"
