"""
@file test_backend_aware_components_phase1_contract.py
@brief Phase 1 contract tests for backend-aware component formulas.

These tests ensure that Similarity, TNorm, Implicator, and FuzzyQuantifier
components own their NumPy/CuPy-ready formulas through compute_backend-style
hooks. The tests use NumPy as the available backend namespace so they remain
stable in CPU-only CI while still protecting the backend boundary contract.
"""

import numpy as np

from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.implicators import Implicator
from FRsutils.core.similarities import Similarity
from FRsutils.core.similarity_engine import calculate_similarity_block
from FRsutils.core.tnorms import TNorm


def test_similarity_components_expose_backend_formula_equivalent_to_numpy_call():
    """@brief Linear/Gaussian backend formulas match existing NumPy formulas."""
    diff = np.array([[0.0, 0.25], [-0.5, 0.75]])

    for similarity in [Similarity.create("linear"), Similarity.create("gaussian", sigma=0.67)]:
        expected = similarity._compute(diff)
        actual = similarity.compute_backend(diff, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_tnorm_components_expose_backend_formula_equivalent_to_numpy_call():
    """@brief Supported T-norm backend formulas match existing NumPy calls."""
    a = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.2]])
    b = np.array([[1.0, 0.5, 0.2], [0.25, 0.9, 0.8]])

    for name in ["minimum", "product", "lukasiewicz", "drastic", "einstein", "hamacher", "nilpotent", "yager"]:
        tnorm = TNorm.create(name)
        expected = tnorm(a, b)
        actual = tnorm.compute_backend(a, b, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_implicator_components_expose_vectorized_backend_formulas():
    """@brief Implicator backend formulas match public NumPy calls without np.vectorize mirroring."""
    a = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.2]])
    b = np.array([[1.0, 0.5, 0.2], [0.25, 0.9, 0.8]])

    for name in ["lukasiewicz", "goedel", "kleenedienes", "reichenbach", "goguen", "rescher", "yager", "weber", "fodor"]:
        implicator = Implicator.create(name)
        expected = implicator(a, b)
        actual = implicator.compute_backend(a, b, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_fuzzy_quantifier_components_expose_backend_formulas():
    """@brief Fuzzy quantifier backend formulas match public NumPy calls."""
    x = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])

    for name in ["linear", "quadratic"]:
        quantifier = FuzzyQuantifier.create(name, alpha=0.2, beta=0.8)
        expected = quantifier(x)
        actual = quantifier.compute_backend(x, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_similarity_block_uses_component_backend_contract_for_numpy_path():
    """@brief Similarity engine block output remains equivalent after delegating formulas to components."""
    X = np.array([[0.0, 0.1], [0.4, 0.3], [1.0, 0.8]])
    rows = X[:2]
    cols = X[1:]
    similarity = Similarity.create("gaussian", sigma=0.67)
    tnorm = TNorm.create("minimum")

    block = calculate_similarity_block(rows, cols, similarity, tnorm)

    manual = np.ones((rows.shape[0], cols.shape[0]))
    for feature_idx in range(X.shape[1]):
        diff = rows[:, feature_idx].reshape(-1, 1) - cols[:, feature_idx].reshape(1, -1)
        manual = np.minimum(manual, similarity.compute_backend(diff, xp=np))

    np.testing.assert_allclose(block, manual)
