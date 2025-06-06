import pytest
from FRsutils.utils.constructor_utils.similarity_builder import build_similarity
from FRsutils.core.similarities import LinearSimilarity, GaussianSimilarity

def test_build_linear_similarity():
    """
    @brief Test that 'linear' returns a LinearSimilarity instance.
    """
    sim = build_similarity('linear')
    assert isinstance(sim, LinearSimilarity)

def test_build_gaussian_similarity_with_default_sigma():
    """
    @brief Test that 'gaussian' returns a GaussianSimilarity with default sigma.
    """
    sim = build_similarity('gaussian', sigma=0.1)
    assert isinstance(sim, GaussianSimilarity)
    assert abs(sim.sigma - 0.1) < 1e-6

def test_build_gaussian_similarity_with_custom_sigma():
    """
    @brief Test that 'gaussian' returns a GaussianSimilarity with a specified sigma.
    """
    sim = build_similarity('gaussian', sigma=0.3)
    assert isinstance(sim, GaussianSimilarity)
    assert abs(sim.sigma - 0.3) < 1e-6

def test_build_similarity_invalid_name():
    """
    @brief Test that an unknown similarity name raises ValueError.
    """
    with pytest.raises(ValueError) as excinfo:
        build_similarity('unknown')
    assert "Invalid value 'unknown' for parameter 'similarity_name'" in str(excinfo.value)