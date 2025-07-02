import pytest
import numpy as np
from FRsutils.core.models.vqrs import VQRS
from FRsutils.utils.logger.logger_util import get_logger
from tests import synthetic_data_store as ds
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier, LinearFuzzyQuantifier, QuadraticFuzzyQuantifier

logger = get_logger("test")

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
    ub_fq = FuzzyQuantifier.create("linear", alpha=0.2, beta=0.8)
    lb_fq = FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)

    logger = get_logger("test")
    return VQRS(sim, lbl, lb_fq, ub_fq, logger=logger)

@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_lower_approximation_shape_ndarray_range_all_combinations(quantifier_name, synthetic_data_, model_instance):
    """
    @brief Test lower_approximation shape, type, and value range for all possibilities of fuzzy quantifiers.
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)

    lower = model.lower_approximation()

    assert isinstance(lower, np.ndarray)
    assert lower.shape == (3,)
    assert np.all((0.0 <= lower) & (lower <= 1.0)), f"Out of range values for {quantifier_name}"

@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_upper_approximation_shape_ndarray_range_all_combinations(quantifier_name, synthetic_data_, model_instance):
    """
    @brief Test upper_approximation shape, type, and value range for all possibilities of fuzzy quantifiers.
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)

    upper = model.upper_approximation()

    assert isinstance(upper, np.ndarray)
    assert upper.shape == (3,)
    assert np.all((0.0 <= upper) & (upper <= 1.0)), f"Out of range values for {quantifier_name}"


# @pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
# @pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
# def test_boundary_region_all_combinations_shape_range(tnorm_name, implicator_name):
#     """
#     @brief Validates boundary_region = upper - lower across all ITFRS combinations just for shape and range
#     correctness of values are not checked
#     """
#     sim = np.array([
#         [1.0, 0.8, 0.0],
#         [0.8, 1.0, 0.3],
#         [0.0, 0.3, 1.0]
#     ])
#     lbl = np.array([1, 1, 0])

#     tnorm = TNorm.create(tnorm_name, p=0.83)
#     implicator = Implicator.create(implicator_name)
#     model = ITFRS(sim, lbl, tnorm, implicator)

#     boundary = model.boundary_region()
#     expected = model.upper_approximation() - model.lower_approximation()
#     np.testing.assert_allclose(boundary, expected, err_msg=f"Failed for {tnorm_name} + {implicator_name}")

#     assert isinstance(boundary, np.ndarray)
#     assert boundary.shape == (3,)
#     assert np.all((0.0 <= boundary) & (boundary <= 1.0)), f"Out of range values in bouundry region for {tnorm_name} + {implicator_name}"


@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_positive_region_all_combinations_shape_range(quantifier_name, synthetic_data_):
    """
    @brief Validates positive_region == lower_approximation across all ITFRS combinations
    correctness of values are not checked
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)

    pos = model.positive_region()
    expected = model.lower_approximation()
    np.testing.assert_allclose(pos, expected, err_msg=f"Failed for {quantifier_name}")

    assert isinstance(pos, np.ndarray)
    assert pos.shape == (3,)
    assert np.all((0.0 <= pos) & (pos <= 1.0)), f"Out of range values in positive region for {quantifier_name}"

@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_to_dict_include_data_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Validates presence of all fields in `to_dict(include_data=True)`
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)
    
    data = model.to_dict(include_data=True)
    assert "type" in data
    assert "ub_fuzzy_quantifier" in data
    assert "ub_fuzzy_quantifier" in data
    assert "similarity_matrix" in data
    assert "labels" in data
    

@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_to_dict_excluse_data_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Validates presence of all fields in `to_dict(include_data=False)`
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)
    
    data = model.to_dict(include_data=False)
    assert "type" in data
    assert "ub_fuzzy_quantifier" in data
    assert "ub_fuzzy_quantifier" in data
    assert "similarity_matrix" not in data
    assert "labels" not in data


@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_from_dict_roundtrip_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Tests whether VQRS.from_dict correctly restores all fields for all combinations of TNorm and Implicator.
    """
    sim_matrix, labels = synthetic_data_

    fq_cls = FuzzyQuantifier.get_class(quantifier_name)
    ub_fq = fq_cls(alpha=0.2, beta=0.8)
    lb_fq = fq_cls(alpha=0.1, beta=0.6)

    model = VQRS(sim_matrix, labels, lb_fq, ub_fq)
    
    serialized = model.to_dict(include_data=True)
    restored = VQRS.from_dict(serialized)

    np.testing.assert_array_equal(restored.similarity_matrix, sim_matrix)
    np.testing.assert_array_equal(restored.labels, labels)

    np.testing.assert_allclose(restored.lower_approximation(), model.lower_approximation(), err_msg=f"Lower mismatch for {quantifier_name}")
    np.testing.assert_allclose(restored.upper_approximation(), model.upper_approximation(), err_msg=f"Upper mismatch for {quantifier_name}")


@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_from_config_equivalence_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Test VQRS.from_config builds a valid model for all combinations.
    """
    sim_matrix, labels = synthetic_data_

    config = {
        "ub_fuzzy_quantifier_name": quantifier_name,
        "ub_alpha" : 0.2,
        "ub_beta": 0.8,
        "lb_fuzzy_quantifier_name": quantifier_name,
        "lb_alpha" : 0.1,
        "lb_beta": 0.6,
        "logger":logger
    }

    model = VQRS.from_config(similarity_matrix=sim_matrix,
                              labels=labels, **config)
    assert isinstance(model, VQRS)
    assert model.similarity_matrix.shape == (3, 3)
    assert model.labels.shape == (3,)
    
    model2 = VQRS.from_dict(model.to_dict(include_data=True))
    
    assert model2.similarity_matrix.shape == (3, 3)
    assert model2.labels.shape == (3,)
    
    mdl_lwr = model.lower_approximation()
    mdl2_lwr = model2.lower_approximation()
    
    np.testing.assert_allclose(mdl_lwr, mdl2_lwr, atol=1e-6)
    
    mdl_upr = model.upper_approximation()
    mdl2_upr = model2.upper_approximation()
    
    np.testing.assert_allclose(mdl_upr, mdl2_upr, atol=1e-6)
    

@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_describe_params_detailed_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Ensure that VQRS.describe_params_detailed() contains keys.
    """
    sim_matrix, labels = synthetic_data_

    config = {
        "ub_fuzzy_quantifier_name": quantifier_name,
        "ub_alpha" : 0.2,
        "ub_beta": 0.8,
        "lb_fuzzy_quantifier_name": quantifier_name,
        "lb_alpha" : 0.1,
        "lb_beta": 0.6,
        "logger":logger
    }

    model = VQRS.from_config(similarity_matrix=sim_matrix,
                              labels=labels, **config)
    
    desc = model.describe_params_detailed()
    assert isinstance(desc, dict)
    assert "ub_fuzzy_quantifier" in desc
    # assert "ub_alpha" in desc
    # assert "ub_beta" in desc
    assert "lb_fuzzy_quantifier" in desc
    # assert "lb_alpha" in desc
    # assert "lb_beta" in desc
    # assert "logger" in desc


@pytest.mark.parametrize("quantifier_name", list(FuzzyQuantifier.list_available().keys()))
def test_get_params_internal_all_combinations(quantifier_name, synthetic_data_):
    """
    @brief Ensure _get_params returns full param dict.
    """
    sim_matrix, labels = synthetic_data_

    config = {
        "ub_fuzzy_quantifier_name": quantifier_name,
        "ub_alpha" : 0.2,
        "ub_beta": 0.8,
        "lb_fuzzy_quantifier_name": quantifier_name,
        "lb_alpha" : 0.1,
        "lb_beta": 0.6,
        "logger":logger
    }

    model = VQRS.from_config(similarity_matrix=sim_matrix,
                              labels=labels, **config)
    
    params = model._get_params()
    assert "ub_fuzzy_quantifier" in params
    assert "lb_fuzzy_quantifier" in params
    assert "similarity_matrix" in params
    assert "labels" in params
    assert "tnorm" in params


# def test_validate_params_invalid_lb_fuzzy_quantifier():
#     """
#     @brief Validation must fail when lb_fuzzy_quantifier is missing or invalid.
#     """
#     sim = np.array([
#         [1.0, 0.8, 0.0],
#         [0.8, 1.0, 0.3],
#         [0.0, 0.3, 1.0]
#     ])
#     lbl = np.array([1, 1, 0])
#     with pytest.raises(ValueError):
#         VQRS.validate_params(ub_fuzzy_quantifier=FuzzyQuantifier.create("linear", alpha=0.3, beta=0.8), lb_fuzzy_quantifier=None)


# def test_validate_params_invalid_implicator():
#     """
#     @brief Validation must fail when implicator is missing or invalid.
#     """
#     sim = np.array([
#         [1.0, 0.8, 0.0],
#         [0.8, 1.0, 0.3],
#         [0.0, 0.3, 1.0]
#     ])
#     lbl = np.array([1, 1, 0])
#     with pytest.raises(ValueError):
#         ITFRS.validate_params(lb_implicator=None, ub_tnorm=TNorm.create("minimum"))


# @pytest.mark.parametrize("tnorm_name", list(TNorm.list_available().keys()))
# @pytest.mark.parametrize("implicator_name", list(Implicator.list_available().keys()))
# def test_logger_works_all_combinations(tnorm_name, implicator_name):
#     """
#     @brief Sanity check for logger presence in all ITFRS combinations.
#     """
#     sim = np.array([
#         [1.0, 0.8, 0.0],
#         [0.8, 1.0, 0.3],
#         [0.0, 0.3, 1.0]
#     ])
#     lbl = np.array([1, 1, 0])
#     config = {
#         "ub_tnorm_name": tnorm_name,
#         "lb_implicator_name": implicator_name
#         ,"p": 0.83  # included for parametric T-norms like Yager
#     }

#     model = ITFRS.from_config(similarity_matrix=sim, labels=lbl, **config)
#     model.logger.info(f"Logger active for {tnorm_name} + {implicator_name}")

# #################################
# ###                           ###
# ###      Tests with data      ###
# ###                           ###
# #################################

# @pytest.mark.parametrize("test_case", ds.get_ITFRS_testing_testsets())
# @pytest.mark.parametrize("implicator_name, expected_lower_key", [
#     ("reichenbach", "Reichenbach_lowerBound"),
#     ("kleenedienes", "KD_lowerBound"),
#     ("lukasiewicz", "Luk_lowerBound"),
#     ("goedel", "Goedel_lowerBound"),
#     # ("gaines", "Gaines_lowerBound"),
#     ("goguen", "Goguen_lowerBound"),
#     ("rescher", "Rescher_lowerBound"),
#     ("weber", "Weber_lowerBound"),
#     ("fodor", "Fodor_lowerBound"),
#     ("yager", "Yager_lowerBound")
# ])
# @pytest.mark.parametrize("tnorm_name, expected_upper_key", [
#     ("product", "prod_tn_upperBound"),
#     ("minimum", "min_tn_upperBound"),
#     ("lukasiewicz", "luk_tn_upperBound"),
#     ("einstein", "einstein_tn_upperBound"),
#     ("drastic", "drastic_tn_upperBound"),
#     ("nilpotent", "nilpotent_tn_upperBound"),
#     ("hamacher", "hamacher_tn_upperBound"),
#     ("yager", "yager_tn_upperBound_p_0_83")
# ])
# def test_itfrs_model_with_all_settings(test_case, implicator_name, expected_lower_key, tnorm_name, expected_upper_key):
#     """
#     @brief Test for `itfrs_model_with_all_settings` method of ITFRS model.
#     """
#     sim = test_case["sim_matrix"]
#     y = test_case["y"]
#     expected = test_case["expected"]

#     config={
#         "ub_tnorm_name": tnorm_name,
#         "lb_implicator_name": implicator_name,
#         "p": 0.83
#     }

#     model = ITFRS.from_config(similarity_matrix=sim, labels=y, **config)

#     actual_lower = model.lower_approximation()
#     actual_upper = model.upper_approximation()

#     np.testing.assert_allclose(actual_lower, expected[expected_lower_key], atol=1e-5, err_msg=f"Failed for {implicator_name}")
#     np.testing.assert_allclose(actual_upper, expected[expected_upper_key], atol=1e-5, err_msg=f"Failed for {tnorm_name}")

# @pytest.fixture
# def synthetic_data():
#     return ds.get_ITFRS_testing_testsets()[0]

# @pytest.mark.parametrize("implicator_name", ['reichenbach', 'kleenedienes', 'lukasiewicz', 'goedel', 'goguen', 'yager', 'rescher', 'weber', 'fodor'])
# @pytest.mark.parametrize("tnorm_name", ['product', 'minimum', 'yager', 'luk', 'drastic', 'hamacher', 'einstein', 'nilpotent'])
# @pytest.mark.parametrize("similarity_name", ['gaussian', 'linear'])
# def test_itfrs_all_combinations(implicator_name, tnorm_name, similarity_name, synthetic_data):
#     sim_matrix_raw = synthetic_data["sim_matrix"]
#     labels = synthetic_data["y"]
#     expected = synthetic_data["expected"]

    
#     sim_matrix = sim_matrix_raw

#     model = ITFRS.from_config(
#         similarity_matrix=sim_matrix,
#         labels=labels,
#         lb_implicator_name=implicator_name,
#         ub_tnorm_name=tnorm_name
#         ,p=0.83
#     )

#     lower = model.lower_approximation()
#     upper = model.upper_approximation()

#     assert lower.shape == labels.shape
#     assert upper.shape == labels.shape
#     assert np.all((0.0 <= lower) & (lower <= 1.0))
#     assert np.all((0.0 <= upper) & (upper <= 1.0))

#     expected_lower_keys = {
#         'reichenbach': "Reichenbach_lowerBound",
#         'kleene-dienes': "KD_lowerBound",
#         'lukasiewicz': "Luk_lowerBound",
#         'goedel': "Goedel_lowerBound",
#         # 'gaines': "Gaines_lowerBound",
#         "goguen": "Goguen_lowerBound",
#         "rescher": "Rescher_lowerBound",
#         "weber": "Weber_lowerBound",
#         "fodor": "Fodor_lowerBound",
#         "yager": "Yager_lowerBound"
#     }
#     expected_upper_keys = {
#         'product': "prod_tn_upperBound",
#         'minimum': "min_tn_upperBound",
#         "lukasiewicz": "luk_tn_upperBound",
#         "einstein": "einstein_tn_upperBound",
#         "drastic": "drastic_tn_upperBound",
#         "nilpotent": "nilpotent_tn_upperBound",
#         "hamacher": "hamacher_tn_upperBound",
#         "yager": "yager_tn_upperBound_p_0_83"
#     }

#     if implicator_name in expected_lower_keys:
#         expected_key = expected_lower_keys[implicator_name]
#         if expected_key in expected:
#             np.testing.assert_allclose(lower, expected[expected_key], atol=1e-5)

#     if tnorm_name in expected_upper_keys:
#         expected_key = expected_upper_keys[tnorm_name]
#         if expected_key in expected:
#             np.testing.assert_allclose(upper, expected[expected_key], atol=1e-5)

# def test_logger_and_params_describe(synthetic_data):
#     sim_matrix = synthetic_data["sim_matrix"]
#     labels = synthetic_data["y"]
#     model = ITFRS.from_config(similarity_matrix=sim_matrix, labels=labels,
#                               lb_implicator_name="lukasiewicz", ub_tnorm_name="minimum")
#     params = model.describe_params_detailed()
#     assert "ub_tnorm" in params
#     assert "lb_implicator" in params
#     model.logger.info("Logger test message")

# def test_to_dict_and_from_dict_roundtrip(synthetic_data):
#     sim_matrix = synthetic_data["sim_matrix"]
#     labels = synthetic_data["y"]
#     model = ITFRS.from_config(similarity_matrix=sim_matrix, labels=labels,
#                               lb_implicator_name="lukasiewicz", ub_tnorm_name="minimum")
#     model_dict = model.to_dict(include_data=True)
#     reconstructed = ITFRS.from_dict(model_dict)
#     np.testing.assert_allclose(reconstructed.lower_approximation(), model.lower_approximation())
#     np.testing.assert_allclose(reconstructed.upper_approximation(), model.upper_approximation())    
    
# def normalize(name: str) -> str:
#     return name.lower().replace("-", "").replace("_", "")