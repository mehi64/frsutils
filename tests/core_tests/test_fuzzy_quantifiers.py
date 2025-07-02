import pytest
import numpy as np
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.utils.logger.logger_util import get_logger


logger = get_logger(env="test",
                    experiment_name="test_fuzzy_quantifiers")

registered_fqs = FuzzyQuantifier.list_available()
# ----------------------------
# Functional Behavior Testing
# ----------------------------
#region <Functional Behavior Testing>
@pytest.mark.parametrize("quant_type, alpha, beta", [
    ("linear", 0.25, 0.75),
    ("quadratic", 0.25, 0.75),
    ("linear", 0.1, 0.9),
    ("quadratic", 0.1, 0.9)
])
def test_quantifier_output_shape_and_type(quant_type, alpha, beta):
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    x = np.linspace(0, 1, 500)
    result = fq(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert (0.0 <= result).all() and (result <= 1.0).all()


@pytest.mark.parametrize("quant_type, alpha, beta", [
    ("linear", -0.25, 0.75),
    ("quadratic", 0.25, -0.75),
    ("linear", 0.9, 0.1),
    ("quadratic", 0.5, 0.5),
    ("linear", 1.9, 2.1),
    ("quadratic", 3.5, 1.5)
])
def test_quantifier_output_exceptions(quant_type, alpha, beta):
    with pytest.raises(ValueError) as exc_info:
        _ = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    val =str(exc_info.value)
    logger.info(quant_type + ', alpha:' + str(alpha)+ ', beta:' + str(beta) + "Caught error message: " + val)



@pytest.mark.parametrize("quant_type, alpha, beta, x, expected", [
    ("linear", 0.2, 0.8, np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.array([0.0, 0.0, 0.5, 1.0, 1.0])),
    ("quadratic", 0.2, 0.8, np.array([0.0, 0.2, 0.5, 0.8, 1.0]), np.array([0.0, 0.0, 0.5, 1.0, 1.0]))
])
def test_quantifier_known_outputs(quant_type, alpha, beta, x, expected):
    fq = FuzzyQuantifier.create(quant_type, alpha=alpha, beta=beta)
    result = fq(x)
    np.testing.assert_allclose(result, expected, atol=1e-5)
#endregion


# ----------------------------
# Factory and Serialization
# ----------------------------
#region<Factory and Serialization>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_create_to_dict_from_dict(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8)
    d = fq.to_dict()
    fq2 = FuzzyQuantifier.from_dict(d)

    assert isinstance(fq2, FuzzyQuantifier)
    assert fq2.name == fq.name

    np.testing.assert_allclose(fq2._get_params()["alpha"], fq._get_params()["alpha"])
    np.testing.assert_allclose(fq2._get_params()["beta"], fq._get_params()["beta"])
#endregion

# ----------------------------
# Validation and Fail-Fast
# ----------------------------
#region<Validation and Fail-Fast>
@pytest.mark.parametrize("params", [
    {"typ":"linear", "alpha": None, "beta": 0.6},
    {"typ":"linear", "alpha": 0.2, "beta": None},
    {"typ":"linear", "alpha": "a", "beta": 0.6},
    {"typ":"linear", "alpha": 0.2, "beta": "b"},
    {"typ":"linear", "alpha": 0.7, "beta": 0.6},
    {"typ":"linear", "alpha": -0.1, "beta": 1.2},
    {"typ":"quadratic", "alpha": None, "beta": 0.6},
    {"typ":"quadratic", "alpha": 0.2, "beta": None},
    {"typ":"quadratic", "alpha": "a", "beta": 0.6},
    {"typ":"quadratic", "alpha": 0.2, "beta": "b"},
    {"typ":"quadratic", "alpha": 0.7, "beta": 0.6},
    {"typ":"quadratic", "alpha": -0.1, "beta": 1.2}
])
def test_invalid_alpha_beta(params):
    with pytest.raises(ValueError) as exc_info:
        FuzzyQuantifier.create(params["typ"], **params)

    val =str(exc_info.value)
    logger.info(params["typ"] + ', alpha:' + str(params["alpha"])+ ', beta:' + str(params["beta"]) + "Caught error message: " + val)

#endregion

# ----------------------------
# Metadata & Reflection
# ----------------------------
#region<Metadata & Reflection>
@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_describe_and_params_match(quant_type):
    fq = FuzzyQuantifier.create(quant_type, alpha=0.2, beta=0.8, validate_inputs=False)
    described = fq.describe_params_detailed()
    params = fq._get_params()
    for k in params:
        assert k in described
        assert described[k]["value"] == params[k]

#endregion

@pytest.mark.parametrize("quant_type", list(registered_fqs.keys()))
def test_help(quant_type):
    """
    tests values of params and deteriled params. check them in log
    """
    # TODO: when a class does not have docstring, it returns the base calss docstring. This is wrong
    obj = FuzzyQuantifier.create(quant_type, alpha=0.25, beta=0.75)
    details = obj.help()
    assert isinstance(details, str)
    
    logger.info(quant_type + ', class docstring:' + details)