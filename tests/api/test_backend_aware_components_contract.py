# SPDX-License-Identifier: BSD-3-Clause
"""Phase 1 contract tests for backend-aware component formulas."""

import numpy as np

from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.implicators import Implicator
from FRsutils.core.similarities import Similarity
from FRsutils.core.similarity_engine import calculate_similarity_block
from FRsutils.core.tnorms import TNorm


def test_similarity_components_expose_backend_formula_equivalent_to_numpy_call():
    """Linear/Gaussian backend formulas match existing NumPy formulas."""
    diff = np.array([[0.0, 0.25], [-0.5, 0.75]])

    for similarity in [Similarity.create("linear"), Similarity.create("gaussian", sigma=0.67)]:
        expected = similarity._compute(diff)
        actual = similarity.compute_backend(diff, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_tnorm_components_expose_backend_formula_equivalent_to_numpy_call():
    """Supported T-norm backend formulas match existing NumPy calls."""
    a = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.2]])
    b = np.array([[1.0, 0.5, 0.2], [0.25, 0.9, 0.8]])

    for name in ["minimum", "product", "lukasiewicz", "drastic", "einstein", "hamacher", "nilpotent", "yager"]:
        tnorm = TNorm.create(name)
        expected = tnorm(a, b)
        actual = tnorm.compute_backend(a, b, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_implicator_components_expose_vectorized_backend_formulas():
    """Implicator backend formulas match public NumPy calls without np.vectorize mirroring."""
    a = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.2]])
    b = np.array([[1.0, 0.5, 0.2], [0.25, 0.9, 0.8]])

    for name in ["lukasiewicz", "goedel", "kleenedienes", "reichenbach", "goguen", "rescher", "yager", "weber", "fodor"]:
        implicator = Implicator.create(name)
        expected = implicator(a, b)
        actual = implicator.compute_backend(a, b, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_fuzzy_quantifier_components_expose_backend_formulas():
    """Fuzzy quantifier backend formulas match public NumPy calls."""
    x = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])

    for name in ["linear", "quadratic"]:
        quantifier = FuzzyQuantifier.create(name, alpha=0.2, beta=0.8)
        expected = quantifier(x)
        actual = quantifier.compute_backend(x, xp=np)
        np.testing.assert_allclose(actual, expected)


def test_similarity_block_uses_component_backend_contract_for_numpy_path():
    """Similarity engine block output remains equivalent after delegating formulas to components.
    """
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
