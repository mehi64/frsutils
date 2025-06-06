import pytest
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.core.tnorms import MinTNorm, ProductTNorm, LukasiewiczTNorm


@pytest.mark.parametrize("name, expected_type", [
    ("minimum", MinTNorm),
    ("product", ProductTNorm),
    ("lukasiewicz", LukasiewiczTNorm)
])
def test_valid_tnorms(name, expected_type):
    """
    @brief Test that valid T-norm names build correct instances.
    """
    tnorm = build_tnorm(name)
    assert isinstance(tnorm, expected_type), f"{name} did not return {expected_type.__name__}"


@pytest.mark.parametrize("invalid_name", [
    "maximum", "unknown", "", "lula", None, 123
])
def test_invalid_tnorms(invalid_name):
    """
    @brief Test that invalid T-norm names raise ValueError.
    """
    with pytest.raises(ValueError):
        build_tnorm(invalid_name)
