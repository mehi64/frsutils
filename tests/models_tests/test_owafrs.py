import pytest
import numpy as np
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.tnorms import MinTNorm
from FRsutils.core.implicators import LukasiewiczImplicator
from FRsutils.utils.logger.logger_util import get_logger
from tests import synthetic_data_store as ds
from FRsutils.core.tnorms import TNorm
from FRsutils.core.implicators import Implicator
import FRsutils.core.owa_weights as oww


@pytest.fixture
def synthetic_data_():
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])
    return sim_matrix, labels

@pytest.fixture
def model_instance(synthetic_data_):
    sim, lbl = synthetic_data_
    tnorm = MinTNorm()
    implicator = LukasiewiczImplicator()
    ub_owa_weights = oww.LinearOWAWeights()
    lb_owa_weights = oww.LinearOWAWeights()
    logger = get_logger("test")
    return OWAFRS(sim, lbl, tnorm, implicator, ub_owa_weights, lb_owa_weights, logger=logger)

@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_lower_upper_approximation_shape_ndarray_range_all_combinations(tnorm_name, 
                                                                  implicator_name,
                                                                  ub_owa_method_name,
                                                                  lb_owa_method_name):
    """
    @brief Test lower_approximation shape, type, and value range for all combinations of TNorm and Implicator
    but correctness of values are not checked
    """
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])

    tnorm = TNorm.create(tnorm_name, p=0.83)
    implicator = Implicator.create(implicator_name)
    
    ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
    lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


    model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)

    lower = model.lower_approximation()

    assert isinstance(lower, np.ndarray)
    assert lower.shape == (3,)
    assert np.all((0.0 <= lower) & (lower <= 1.0)), f"Out of range values in lower approx for {tnorm_name} + {implicator_name}"

    upper = model.upper_approximation()

    assert isinstance(upper, np.ndarray)
    assert upper.shape == (3,)
    assert np.all((0.0 <= upper) & (upper <= 1.0)), f"Out of range values in upp approx for {tnorm_name} + {implicator_name}"




# @pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
# @pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
# @pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
# @pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
# def test_boundary_region_all_combinations_shape_range(tnorm_name, 
#                                                     implicator_name,
#                                                     ub_owa_method_name,
#                                                     lb_owa_method_name):
#     """
#     @brief Validates boundary_region = upper - lower across all ITFRS combinations just for shape and range
#     correctness of values are not checked
#     """
#     sim_matrix = np.array([
#         [1.0, 0.8, 0.0],
#         [0.8, 1.0, 0.3],
#         [0.0, 0.3, 1.0]
#     ])
#     labels = np.array([1, 1, 0])

#     tnorm = TNorm.create(tnorm_name, p=0.83)
#     implicator = Implicator.create(implicator_name)
    
#     ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
#     lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


#     model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)


#     boundary = model.boundary_region()
#     expected = model.upper_approximation() - model.lower_approximation()
#     np.testing.assert_allclose(boundary, expected, err_msg=f"Failed for {tnorm_name} + {implicator_name}")

#     assert isinstance(boundary, np.ndarray)
#     assert boundary.shape == (3,)
#     assert np.all((0.0 <= boundary) & (boundary <= 1.0)), f"Out of range values in bouundry region for {tnorm_name} + {implicator_name}"


@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_positive_region_all_combinations_shape_range(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):
    """
    @brief Validates boundary_region = upper - lower across all ITFRS combinations just for shape and range
    correctness of values are not checked
    """
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])

    tnorm = TNorm.create(tnorm_name, p=0.83)
    implicator = Implicator.create(implicator_name)
    
    ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
    lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


    model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)

    pos = model.positive_region()
    expected = model.lower_approximation()
    np.testing.assert_allclose(pos, expected, err_msg=f"Failed for {tnorm_name} + {implicator_name}")

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert np.all((0.0 <= pos) & (pos <= 1.0)), f"Out of range values in positive region for {tnorm_name} + {implicator_name}"


@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_to_dict_include_data_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):

    """
    @brief Validates presence of all fields in `to_dict(include_data=True)`
    """
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])

    tnorm = TNorm.create(tnorm_name, p=0.83)
    implicator = Implicator.create(implicator_name)
    
    ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
    lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


    model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)

    
    data = model.to_dict(include_data=True)
    assert "type" in data
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "similarity_matrix" in data
    assert "labels" in data
    assert "ub_owa_method" in data
    assert "lb_owa_method" in data
    
    
        
@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_to_dict_exclude_data_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):

    """
    @brief Validates presence/absence of fields in `to_dict(include_data=False)`
    """
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])

    tnorm = TNorm.create(tnorm_name, p=0.83)
    implicator = Implicator.create(implicator_name)
    
    ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
    lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


    model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)

    
    data = model.to_dict(include_data=False)
    assert "type" in data
    assert "ub_tnorm" in data
    assert "lb_implicator" in data
    assert "ub_owa_method" in data
    assert "lb_owa_method" in data



@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_from_dict_roundtrip_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):

    """
    @brief Tests whether ITFRS.from_dict correctly restores all fields for all combinations of TNorm and Implicator.
    """
    sim_matrix = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    labels = np.array([1, 1, 0])

    tnorm = TNorm.create(tnorm_name, p=0.83)
    implicator = Implicator.create(implicator_name)
    
    ub_owa_method = oww.OWAWeights.create(ub_owa_method_name)
    lb_owa_method = oww.OWAWeights.create(lb_owa_method_name)


    model = OWAFRS(sim_matrix, labels, tnorm, implicator,ub_owa_method, lb_owa_method)

    serialized = model.to_dict(include_data=True)
    restored = OWAFRS.from_dict(serialized)

    np.testing.assert_array_equal(restored.similarity_matrix, sim_matrix)
    np.testing.assert_array_equal(restored.labels, labels)

    np.testing.assert_allclose(restored.lower_approximation(), model.lower_approximation(), err_msg=f"Lower mismatch for {tnorm_name} + {implicator_name}")
    np.testing.assert_allclose(restored.upper_approximation(), model.upper_approximation(), err_msg=f"Upper mismatch for {tnorm_name} + {implicator_name}")


@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_from_config_equivalence_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):
    """
    @brief Test OWAFRS.from_config builds a valid model for all combinations.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])

    config = {
        "ub_tnorm_name": tnorm_name
        ,"lb_implicator_name": implicator_name
        ,"p": 0.83  # included for parametric T-norms like Yager
        ,"ub_owa_method_name": ub_owa_method_name
        ,"lb_owa_method_name": lb_owa_method_name
        ,"base":3.0
    }

    model = OWAFRS.from_config(similarity_matrix=sim, labels=lbl, **config)
    assert isinstance(model, OWAFRS)
    assert model.similarity_matrix.shape == (3, 3)
    assert model.labels.shape == (3,)
    
    model2 = OWAFRS.from_dict(model.to_dict(include_data=True))
    
    assert model2.similarity_matrix.shape == (3, 3)
    assert model2.labels.shape == (3,)
    
    mdl_lwr = model.lower_approximation()
    mdl2_lwr = model2.lower_approximation()
    
    np.testing.assert_allclose(mdl_lwr, mdl2_lwr, atol=1e-6)
    
    mdl_upr = model.upper_approximation()
    mdl2_upr = model2.upper_approximation()
    
    np.testing.assert_allclose(mdl_upr, mdl2_upr, atol=1e-6)
    
    

@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_describe_params_detailed_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):

    """
    @brief Ensure that ITFRS.describe_params_detailed() contains keys.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])

    config = {
        "ub_tnorm_name": tnorm_name
        ,"lb_implicator_name": implicator_name
        ,"p": 0.83  # included for parametric T-norms like Yager
        ,"ub_owa_method_name": ub_owa_method_name
        ,"lb_owa_method_name": lb_owa_method_name
        ,"base":3.0
    }

    model = OWAFRS.from_config(similarity_matrix=sim, labels=lbl, **config)
    desc = model.describe_params_detailed()
    assert isinstance(desc, dict)
    assert "ub_tnorm" in desc
    assert "lb_implicator" in desc
    assert "ub_owa_method" in desc
    assert "lb_owa_method" in desc



@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_get_params_internal_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):

    """
    @brief Ensure _get_params returns full param dict.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])

    config = {
        "ub_tnorm_name": tnorm_name
        ,"lb_implicator_name": implicator_name
        ,"p": 0.83  # included for parametric T-norms like Yager
        ,"ub_owa_method_name": ub_owa_method_name
        ,"lb_owa_method_name": lb_owa_method_name
        ,"base":3.0
    }

    model = OWAFRS.from_config(similarity_matrix=sim, labels=lbl, **config)

    params = model._get_params()
    assert "ub_tnorm" in params
    assert "lb_implicator" in params
    assert "similarity_matrix" in params
    assert "labels" in params
    assert "ub_owa_method" in params
    assert "lb_owa_method" in params


def test_validate_params_invalid_tnorm():
    """
    @brief Validation must fail when T-norm is missing or invalid.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        OWAFRS.validate_params(lb_implicator=Implicator.create("lukasiewicz")
                               ,ub_owa_method=oww.LinearOWAWeights()
                               ,lb_owa_method=oww.LinearOWAWeights()
                               ,ub_tnorm=None)


def test_validate_params_invalid_implicator():
    """
    @brief Validation must fail when implicator is missing or invalid.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        OWAFRS.validate_params(lb_implicator=None
                               ,ub_owa_method=oww.LinearOWAWeights()
                               ,lb_owa_method=oww.LinearOWAWeights()
                               ,ub_tnorm=TNorm.create("min")
                               )
        
        
def test_validate_params_invalid_ub_owa_method():
    """
    @brief Validation must fail when lb_owa_method is missing or invalid.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        OWAFRS.validate_params(lb_implicator=Implicator.create("lukasiewicz")
                               ,ub_owa_method=None
                               ,lb_owa_method=oww.LinearOWAWeights()
                               ,ub_tnorm=TNorm.create("min")
                               )
        
def test_validate_params_invalid_lb_owa_method():
    """
    @brief Validation must fail when lb_owa_method is missing or invalid.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])
    with pytest.raises(ValueError):
        OWAFRS.validate_params(lb_implicator=Implicator.create("lukasiewicz")
                               ,ub_owa_method=oww.LinearOWAWeights()
                               ,lb_owa_method=None
                               ,ub_tnorm=TNorm.create("min")
                               )
        
                        
@pytest.mark.parametrize("ub_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("lb_owa_method_name", list(oww.OWAWeights.list_available().keys()))
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
def test_logger_works_all_combinations(tnorm_name, 
                                                    implicator_name,
                                                    ub_owa_method_name,
                                                    lb_owa_method_name):
    """
    @brief Sanity check for logger presence in all ITFRS combinations.
    """
    sim = np.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.3],
        [0.0, 0.3, 1.0]
    ])
    lbl = np.array([1, 1, 0])
    config = {
        "ub_tnorm_name": tnorm_name
        ,"lb_implicator_name": implicator_name
        ,"ub_owa_method_name": ub_owa_method_name
        ,"lb_owa_method_name": lb_owa_method_name
        ,"p": 0.83  # included for parametric T-norms like Yager
    }

    model = OWAFRS.from_config(similarity_matrix=sim, labels=lbl, **config)
    model.logger.info(f"Logger active for {tnorm_name} + {implicator_name}")

#################################
###                           ###
###      Tests with data      ###
###                           ###
#################################

@pytest.mark.parametrize("test_case", ds.get_OWAFRS_testing_testsets())
@pytest.mark.parametrize("implicator_name, expected_lower_key", [
    ("reichenbach", "reichenbach_lowerBound"),
    ("kleenedienes", "kd_lowerBound"),
    ("lukasiewicz", "luk_lowerBound"),
    ("goedel", "goedel_lowerBound"),
    # ("gaines", "Gaines_lowerBound"),
    ("goguen", "goguen_lowerBound"),
    ("rescher", "rescher_lowerBound"),
    ("weber", "weber_lowerBound"),
    ("fodor", "fodor_lowerBound"),
    ("yager", "yager_lowerBound")
])
@pytest.mark.parametrize("tnorm_name, expected_upper_key", [
    ("product", "prod_tn_upperBound"),
    ("minimum", "min_tn_upperBound"),
    ("lukasiewicz", "luk_tn_upperBound"),
    ("einstein", "einstein_tn_upperBound"),
    ("drastic", "drastic_tn_upperBound"),
    ("nilpotent", "nilpotent_tn_upperBound"),
    ("hamacher", "hamacher_tn_upperBound"),
    ("yager", "yager_tn_upperBound_p_0_83")
])
def test_owafrs_model_with_all_settings(test_case, implicator_name, expected_lower_key, tnorm_name, expected_upper_key):
    """
    @brief Test for `owafrs_model_with_all_settings` method of OWAFRS model.
    """
    sim = test_case["sim_matrix"]
    y = test_case["y"]
    expected = test_case["expected"]["owa_linear"]

    config={
        "ub_tnorm_name": tnorm_name,
        "lb_implicator_name": implicator_name,
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
        "p": 0.83
    }


    model = OWAFRS.from_config(similarity_matrix=sim, labels=y, **config)

    actual_lower = model.lower_approximation()
    actual_upper = model.upper_approximation()

    np.testing.assert_allclose(actual_lower, expected[expected_lower_key], atol=1e-5, err_msg=f"Failed for {implicator_name}")
    np.testing.assert_allclose(actual_upper, expected[expected_upper_key], atol=1e-5, err_msg=f"Failed for {tnorm_name}")



@pytest.fixture
def synthetic_data():
    return ds.get_OWAFRS_testing_testsets()[0]

@pytest.mark.parametrize("implicator_name", ['reichenbach', 'kleenedienes', 'lukasiewicz', 'goedel', 'goguen', 'yager', 'rescher', 'weber', 'fodor'])
@pytest.mark.parametrize("tnorm_name", ['product', 'minimum', 'yager', 'luk', 'drastic', 'hamacher', 'einstein', 'nilpotent'])
@pytest.mark.parametrize("similarity_name", ['gaussian', 'linear'])
def test_owafrs_all_combinations(implicator_name, tnorm_name, similarity_name, synthetic_data):
    sim_matrix_raw = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    expected = synthetic_data["expected"]["owa_linear"]

    # lb_owa_method = oww.LinearOWAWeights()
    # ub_owa_method = oww.LinearOWAWeights()
    
    sim_matrix = sim_matrix_raw

    model = OWAFRS.from_config(
        similarity_matrix=sim_matrix,
        labels=labels,
        lb_implicator_name=implicator_name,
        ub_tnorm_name=tnorm_name,
        lb_owa_method_name="linear",
        ub_owa_method_name="linear",
        p=0.83
    )

    lower = model.lower_approximation()
    upper = model.upper_approximation()

    assert lower.shape == labels.shape
    assert upper.shape == labels.shape
    assert np.all((0.0 <= lower) & (lower <= 1.0))
    assert np.all((0.0 <= upper) & (upper <= 1.0))

    expected_lower_keys = {
        'reichenbach': "Reichenbach_lowerBound",
        'kleene-dienes': "KD_lowerBound",
        'lukasiewicz': "Luk_lowerBound",
        'goedel': "Goedel_lowerBound",
        # 'gaines': "Gaines_lowerBound",
        "goguen": "Goguen_lowerBound",
        "rescher": "Rescher_lowerBound",
        "weber": "Weber_lowerBound",
        "fodor": "Fodor_lowerBound",
        "yager": "Yager_lowerBound"
    }
    expected_upper_keys = {
        'product': "prod_tn_upperBound",
        'minimum': "min_tn_upperBound",
        "lukasiewicz": "luk_tn_upperBound",
        "einstein": "einstein_tn_upperBound",
        "drastic": "drastic_tn_upperBound",
        "nilpotent": "nilpotent_tn_upperBound",
        "hamacher": "hamacher_tn_upperBound",
        "yager": "yager_tn_upperBound_p_0_83"
    }

    if implicator_name in expected_lower_keys:
        expected_key = expected_lower_keys[implicator_name]
        if expected_key in expected:
            np.testing.assert_allclose(lower, expected[expected_key], atol=1e-5)

    if tnorm_name in expected_upper_keys:
        expected_key = expected_upper_keys[tnorm_name]
        if expected_key in expected:
            np.testing.assert_allclose(upper, expected[expected_key], atol=1e-5)


def test_logger_and_params_describe(synthetic_data):
    sim_matrix = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    model = OWAFRS.from_config(similarity_matrix=sim_matrix,
                                labels=labels,
                              lb_implicator_name="lukasiewicz", 
                              ub_tnorm_name="minimum",
                              ub_owa_method_name="linear",
                              lb_owa_method_name="linear",
                              logger_name="test")
    
    params = model.describe_params_detailed()
    assert "ub_tnorm" in params
    assert "lb_implicator" in params
    assert "lb_owa_method" in params
    assert "ub_owa_method" in params
    model.logger.info("Logger test message")

def test_to_dict_and_from_dict_roundtrip(synthetic_data):
    sim_matrix = synthetic_data["sim_matrix"]
    labels = synthetic_data["y"]
    model = OWAFRS.from_config(similarity_matrix=sim_matrix,
                                labels=labels,
                              lb_implicator_name="lukasiewicz", 
                              ub_tnorm_name="minimum",
                              ub_owa_method_name="linear",
                              lb_owa_method_name="linear",
                              logger_name="test")
    
    model_dict = model.to_dict(include_data=True)
    reconstructed = OWAFRS.from_dict(model_dict)
    np.testing.assert_allclose(reconstructed.lower_approximation(), model.lower_approximation())
    np.testing.assert_allclose(reconstructed.upper_approximation(), model.upper_approximation())    
    
def normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")
