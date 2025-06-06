# TODO: check this file thoroughly

import pytest

from FRsutils.utils.validation_utils import (
    _validate_string_param_choice,
    validate_strategy_compatibility,
    validate_fr_model_params,
    validate_tnorm_params,
    validate_similarity_choice,
    validate_implicator_choice,
    validate_quantifier_choice,
    validate_owa_strategy_choice,
    get_param_schema
)

# ---------------------------------------------
# _validate_string_param_choice
# ---------------------------------------------
def test_validate_string_param_choice_valid():
    assert _validate_string_param_choice("similarity_name", "linear", {"linear", "gaussian"}) == "linear"

def test_validate_string_param_choice_invalid():
    with pytest.raises(ValueError):
        _validate_string_param_choice("similarity_name", "bad", {"linear", "gaussian"})

# ---------------------------------------------
# validate_strategy_compatibility
# ---------------------------------------------
def test_validate_strategy_compatibility_valid():
    validate_strategy_compatibility("MyClass", "pos", {"pos", "upper"})

def test_validate_strategy_compatibility_invalid():
    with pytest.raises(ValueError):
        validate_strategy_compatibility("MyClass", "bad", {"pos", "upper"})

# ---------------------------------------------
# validate_fr_model_params
# ---------------------------------------------
def test_validate_fr_model_params_valid():
    validate_fr_model_params("ITFRS", {"lb_tnorm": "minimum", "ub_implicator": "goedel"})

def test_validate_fr_model_params_missing_key():
    with pytest.raises(ValueError):
        validate_fr_model_params("ITFRS", {"lb_tnorm": "minimum"})

def test_validate_fr_model_params_invalid_type():
    with pytest.raises(TypeError):
        validate_fr_model_params("VQRS", {
            "alpha_Q_lower": "wrong",
            "beta_Q_lower": 0.5,
            "alpha_Q_upper": 0.6,
            "beta_Q_upper": 0.7
        })

def test_validate_fr_model_params_out_of_range():
    with pytest.raises(ValueError):
        validate_fr_model_params("VQRS", {
            "alpha_Q_lower": -0.1,
            "beta_Q_lower": 0.5,
            "alpha_Q_upper": 0.6,
            "beta_Q_upper": 0.7
        })

# ---------------------------------------------
# validate_tnorm_params
# ---------------------------------------------
def test_validate_tnorm_params_valid():
    validate_tnorm_params("yager", {"p": 2.0})

def test_validate_tnorm_params_missing():
    with pytest.raises(ValueError):
        validate_tnorm_params("yager", {})

def test_validate_tnorm_params_type_error():
    with pytest.raises(TypeError):
        validate_tnorm_params("yager", {"p": "wrong"})

def test_validate_tnorm_params_range_error():
    with pytest.raises(ValueError):
        validate_tnorm_params("yager", {"p": 0.5})

# ---------------------------------------------
# validate_X_choice functions
# ---------------------------------------------
def test_validate_similarity_choice():
    assert validate_similarity_choice("linear") == "linear"

def test_validate_similarity_choice_invalid():
    with pytest.raises(ValueError):
        validate_similarity_choice("bad")

def test_validate_implicator_choice():
    assert validate_implicator_choice("goedel") == "goedel"

def test_validate_implicator_choice_invalid():
    with pytest.raises(ValueError):
        validate_implicator_choice("bad")

def test_validate_quantifier_choice():
    assert validate_quantifier_choice("quadratic") == "quadratic"

def test_validate_quantifier_choice_invalid():
    with pytest.raises(ValueError):
        validate_quantifier_choice("bad")

def test_validate_owa_strategy_choice():
    assert validate_owa_strategy_choice("linear_sup") == "linear_sup"

def test_validate_owa_strategy_choice_invalid():
    with pytest.raises(ValueError):
        validate_owa_strategy_choice("bad")

# ---------------------------------------------
# get_param_schema
# ---------------------------------------------
def test_get_param_schema_fr_model():
    schema = get_param_schema("fr_model", "ITFRS")
    assert "lb_tnorm" in schema

def test_get_param_schema_tnorm():
    schema = get_param_schema("tnorm", "minimum")
    assert schema == {}

def test_get_param_schema_invalid_type():
    with pytest.raises(ValueError):
        get_param_schema("unknown_type", "ITFRS")

def test_get_param_schema_invalid_name():
    with pytest.raises(ValueError):
        get_param_schema("tnorm", "not-a-tnorm")
