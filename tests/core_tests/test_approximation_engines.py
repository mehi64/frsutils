# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for fuzzy-rough approximation engine helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from FRsutils.core.approximation_engines import (
    ITFRSBlockwiseApproximation,
    OWAFRSBlockwiseApproximation,
    VQRSBlockwiseApproximation,
    _as_labels,
    _as_nested_config,
    _backend_index_array,
    _diagonal_positions_for_block,
    _set_block_diagonal_values,
    build_itfrs_components_from_config,
    compute_itfrs_blockwise,
    compute_vqrs_blockwise,
    build_owafrs_components_from_config,
    build_vqrs_components_from_config,
)
from FRsutils.core.fuzzy_quantifiers import LinearFuzzyQuantifier, QuadraticFuzzyQuantifier
from FRsutils.core.implicators import GoedelImplicator, KleeneDienesImplicator, LukasiewiczImplicator
from FRsutils.core.owa_weights import ExponentialOWAWeights, HarmonicOWAWeights, LinearOWAWeights
from FRsutils.core.similarity_engine import BaseSimilarityEngine, SimilarityBlock
from FRsutils.core.tnorms import MinTNorm, ProductTNorm


class FakeArrayNamespace:
    """Minimal array namespace used to verify backend index conversion."""

    def __init__(self):
        self.asarray_calls = []

    def asarray(self, values, dtype=None):
        """Record conversion arguments and return a NumPy-backed index array."""
        self.asarray_calls.append((values, dtype))
        return np.asarray(values, dtype=dtype)


class FixedBlockSimilarityEngine(BaseSimilarityEngine):
    """Similarity engine fixture yielding a fixed matrix in row/column blocks."""

    def __init__(self, similarity_matrix, *, block_size):
        self._similarity_matrix = np.asarray(similarity_matrix, dtype=float)
        self.block_size = block_size
        self.backend = None
        self.X = np.zeros((self._similarity_matrix.shape[0], 1), dtype=float)

    @property
    def n_samples(self):
        """Return the number of rows in the fixed similarity matrix."""
        return self._similarity_matrix.shape[0]

    def iter_blocks(self):
        """Yield fixed similarity submatrices in row-major block order."""
        n_samples = self.n_samples
        for row_start in range(0, n_samples, self.block_size):
            row_stop = min(row_start + self.block_size, n_samples)
            row_slice = slice(row_start, row_stop)
            for col_start in range(0, n_samples, self.block_size):
                col_stop = min(col_start + self.block_size, n_samples)
                col_slice = slice(col_start, col_stop)
                yield SimilarityBlock(
                    row_slice=row_slice,
                    col_slice=col_slice,
                    values=self._similarity_matrix[row_slice, col_slice],
                )


def _manual_itfrs_expected(similarity_matrix, labels, *, ub_tnorm=None, lb_implicator=None):
    """Compute ITFRS lower and upper values from a dense similarity matrix."""
    labels_array = np.asarray(labels)
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    if ub_tnorm is None:
        ub_tnorm = MinTNorm()
    if lb_implicator is None:
        lb_implicator = LukasiewiczImplicator()

    label_mask = (labels_array[:, None] == labels_array[None, :]).astype(float)
    implication_values = lb_implicator.compute_backend(
        similarity_matrix,
        label_mask,
        xp=np,
        validate_inputs=False,
    )
    tnorm_values = ub_tnorm.compute_backend(similarity_matrix, label_mask, xp=np)
    if similarity_matrix.shape[0]:
        np.fill_diagonal(implication_values, 1.0)
        np.fill_diagonal(tnorm_values, 0.0)

    lower = np.min(implication_values, axis=1) if similarity_matrix.shape[0] else np.array([], dtype=float)
    upper = np.max(tnorm_values, axis=1) if similarity_matrix.shape[0] else np.array([], dtype=float)
    return lower, upper, upper - lower, lower.copy()


def _manual_vqrs_expected(
    similarity_matrix,
    labels,
    *,
    lb_fuzzy_quantifier=None,
    ub_fuzzy_quantifier=None,
):
    """Compute VQRS values from a dense similarity matrix."""
    labels_array = np.asarray(labels)
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    if lb_fuzzy_quantifier is None:
        lb_fuzzy_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
    if ub_fuzzy_quantifier is None:
        ub_fuzzy_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)

    label_mask = (labels_array[:, None] == labels_array[None, :]).astype(float)
    tnorm_values = np.minimum(similarity_matrix, label_mask)
    if similarity_matrix.shape[0]:
        np.fill_diagonal(tnorm_values, 0.0)

    numerator = np.sum(tnorm_values, axis=1)
    denominator = np.sum(similarity_matrix, axis=1) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        interim = numerator / denominator

    lower = lb_fuzzy_quantifier.compute_backend(
        interim,
        xp=np,
        validate_inputs=lb_fuzzy_quantifier.validate_inputs,
    )
    upper = ub_fuzzy_quantifier.compute_backend(
        interim,
        xp=np,
        validate_inputs=ub_fuzzy_quantifier.validate_inputs,
    )
    return lower, upper, upper - lower, lower.copy(), interim


def test_itfrs_blockwise_approximation_dataclass_keeps_arrays_and_defaults():
    lower = np.array([0.1, 0.2], dtype=float)
    upper = np.array([0.7, 0.9], dtype=float)
    boundary = upper - lower
    positive_region = lower.copy()

    result = ITFRSBlockwiseApproximation(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
    )

    assert result.lower is lower
    assert result.upper is upper
    assert result.boundary is boundary
    assert result.positive_region is positive_region
    assert result.execution_backend == "numpy"
    assert result.used_gpu_approximation_accumulators is False


def test_vqrs_blockwise_approximation_dataclass_keeps_interim_and_defaults():
    lower = np.array([0.0, 0.5], dtype=float)
    upper = np.array([0.5, 1.0], dtype=float)
    boundary = upper - lower
    positive_region = lower.copy()
    interim = np.array([0.25, 0.75], dtype=float)

    result = VQRSBlockwiseApproximation(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
        interim=interim,
    )

    assert result.lower is lower
    assert result.upper is upper
    assert result.boundary is boundary
    assert result.positive_region is positive_region
    assert result.interim is interim
    assert result.execution_backend == "numpy"
    assert result.used_gpu_approximation_accumulators is False


def test_owafrs_blockwise_approximation_dataclass_keeps_arrays():
    lower = np.array([0.3, 0.4], dtype=float)
    upper = np.array([0.8, 0.9], dtype=float)
    boundary = upper - lower
    positive_region = lower.copy()

    result = OWAFRSBlockwiseApproximation(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
    )

    assert result.lower is lower
    assert result.upper is upper
    assert result.boundary is boundary
    assert result.positive_region is positive_region


@pytest.mark.parametrize(
    "approximation",
    [
        ITFRSBlockwiseApproximation(
            lower=np.array([0.1]),
            upper=np.array([0.2]),
            boundary=np.array([0.1]),
            positive_region=np.array([0.1]),
        ),
        VQRSBlockwiseApproximation(
            lower=np.array([0.1]),
            upper=np.array([0.2]),
            boundary=np.array([0.1]),
            positive_region=np.array([0.1]),
            interim=np.array([0.3]),
        ),
        OWAFRSBlockwiseApproximation(
            lower=np.array([0.1]),
            upper=np.array([0.2]),
            boundary=np.array([0.1]),
            positive_region=np.array([0.1]),
        ),
    ],
)
def test_blockwise_approximation_dataclasses_are_frozen(approximation):
    with pytest.raises(FrozenInstanceError):
        approximation.lower = np.array([1.0])


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["minority", "majority", "minority"], np.array(["minority", "majority", "minority"])),
        (np.array([10, 10, 5], dtype=int), np.array([10, 10, 5], dtype=int)),
        (np.array(["a", 1, None], dtype=object), np.array(["a", 1, None], dtype=object)),
    ],
)
def test_as_labels_accepts_one_dimensional_label_vectors(labels, expected):
    result = _as_labels(labels, expected_length=3)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "labels, expected_length",
    [
        ([[0, 1], [1, 0]], 2),
        (np.array(1), 1),
        ([0, 1], 3),
    ],
)
def test_as_labels_rejects_non_1d_or_length_mismatched_inputs(labels, expected_length):
    with pytest.raises(ValueError):
        _as_labels(labels, expected_length=expected_length)


@pytest.mark.parametrize(
    "default_model_type, expected_model_type, expected_keys",
    [
        ("itfrs", "itfrs", {"ub_tnorm", "lb_implicator"}),
        ("vqrs", "vqrs", {"lb_fuzzy_quantifier", "ub_fuzzy_quantifier"}),
        ("owafrs", "owafrs", {"ub_tnorm", "lb_implicator", "ub_owa_method", "lb_owa_method"}),
    ],
)
def test_as_nested_config_builds_model_specific_defaults(
    default_model_type,
    expected_model_type,
    expected_keys,
):
    result = _as_nested_config(None, default_model_type=default_model_type)

    assert result["fr_model"]["type"] == expected_model_type
    assert expected_keys.issubset(result["fr_model"].keys())


def test_as_nested_config_normalizes_flat_config_without_mutating_input():
    config = {
        "type": "itfrs",
        "similarity": "gaussian",
        "similarity_sigma": 0.5,
        "ub_tnorm_name": "product",
        "lb_implicator_name": "kleene_dienes",
    }
    before = dict(config)

    result = _as_nested_config(config)

    assert config == before
    assert result["similarity"]["name"] == "gaussian"
    assert result["similarity"]["params"]["sigma"] == 0.5
    assert result["fr_model"]["ub_tnorm"]["name"] == "product"
    assert result["fr_model"]["lb_implicator"]["name"] == "kleene_dienes"


def test_as_nested_config_returns_existing_nested_config_unchanged():
    nested_config = {
        "fr_model": {
            "type": "itfrs",
            "ub_tnorm": {"name": "minimum", "params": {}},
            "lb_implicator": {"name": "lukasiewicz", "params": {}},
        }
    }

    result = _as_nested_config(nested_config)

    assert result is nested_config


@pytest.mark.parametrize("config", ["itfrs", [("type", "itfrs")], 1])
def test_as_nested_config_rejects_non_mapping_config(config):
    with pytest.raises(TypeError):
        _as_nested_config(config)


@pytest.mark.parametrize(
    "row_slice, col_slice, expected_rows, expected_cols",
    [
        (slice(0, 3), slice(0, 3), np.array([0, 1, 2]), np.array([0, 1, 2])),
        (slice(1, 4), slice(2, 5), np.array([1, 2]), np.array([0, 1])),
        (slice(5, 8), slice(0, 4), np.array([], dtype=int), np.array([], dtype=int)),
        (slice(None, 2), slice(0, 2), np.array([0, 1]), np.array([0, 1])),
    ],
)
def test_diagonal_positions_for_block_returns_local_overlap_positions(
    row_slice,
    col_slice,
    expected_rows,
    expected_cols,
):
    row_positions, col_positions = _diagonal_positions_for_block(row_slice, col_slice)

    np.testing.assert_array_equal(row_positions, expected_rows)
    np.testing.assert_array_equal(col_positions, expected_cols)
    assert row_positions.dtype == int
    assert col_positions.dtype == int


def test_backend_index_array_returns_numpy_indices_unchanged_for_numpy_namespace():
    indices = np.array([0, 2, 4], dtype=np.int64)

    result = _backend_index_array(indices, xp=np)

    assert result is indices


def test_backend_index_array_converts_indices_for_non_numpy_namespace():
    indices = np.array([0, 2, 4], dtype=np.int64)
    xp = FakeArrayNamespace()

    result = _backend_index_array(indices, xp=xp)

    np.testing.assert_array_equal(result, indices)
    assert xp.asarray_calls == [(indices, int)]


def test_set_block_diagonal_values_mutates_requested_numpy_positions():
    values = np.zeros((3, 4), dtype=float)
    row_indices = np.array([0, 2], dtype=int)
    col_indices = np.array([1, 3], dtype=int)

    _set_block_diagonal_values(
        values,
        row_indices,
        col_indices,
        xp=np,
        lower_value=7.0,
    )

    expected = np.zeros((3, 4), dtype=float)
    expected[0, 1] = 7.0
    expected[2, 3] = 7.0
    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_set_block_diagonal_values_returns_without_mutation_for_empty_indices():
    values = np.ones((2, 2), dtype=float)
    before = values.copy()

    _set_block_diagonal_values(
        values,
        np.array([], dtype=int),
        np.array([], dtype=int),
        xp=np,
        lower_value=0.0,
    )

    np.testing.assert_allclose(values, before, atol=1e-12)


def test_set_block_diagonal_values_uses_backend_index_conversion_for_non_numpy_namespace():
    values = np.zeros((2, 2), dtype=float)
    xp = FakeArrayNamespace()

    _set_block_diagonal_values(
        values,
        np.array([0, 1], dtype=int),
        np.array([1, 0], dtype=int),
        xp=xp,
        lower_value=3.0,
        upper_value=99.0,
    )

    expected = np.array([[0.0, 3.0], [3.0, 0.0]], dtype=float)
    np.testing.assert_allclose(values, expected, atol=1e-12)
    assert len(xp.asarray_calls) == 2
    assert xp.asarray_calls[0][1] is int
    assert xp.asarray_calls[1][1] is int


# -----------------------------------------------------------------------------
# Phase 2: component builders
# -----------------------------------------------------------------------------


def test_build_itfrs_components_from_none_returns_default_components():
    ub_tnorm, lb_implicator = build_itfrs_components_from_config(None)

    assert isinstance(ub_tnorm, MinTNorm)
    assert isinstance(lb_implicator, LukasiewiczImplicator)


def test_build_itfrs_components_from_flat_config_resolves_component_aliases():
    config = {
        "type": "itfrs",
        "ub_tnorm_name": "product",
        "lb_implicator_name": "kleene",
    }

    ub_tnorm, lb_implicator = build_itfrs_components_from_config(config)

    assert isinstance(ub_tnorm, ProductTNorm)
    assert isinstance(lb_implicator, KleeneDienesImplicator)


def test_build_itfrs_components_from_nested_config_resolves_specs():
    config = {
        "fr_model": {
            "type": "itfrs",
            "ub_tnorm": {"name": "product", "params": {}},
            "lb_implicator": {"name": "goedel", "params": {}},
        }
    }

    ub_tnorm, lb_implicator = build_itfrs_components_from_config(config)

    assert isinstance(ub_tnorm, ProductTNorm)
    assert isinstance(lb_implicator, GoedelImplicator)


def test_build_itfrs_components_uses_defaults_for_missing_nested_specs():
    config = {"fr_model": {"type": "itfrs", "ub_tnorm": {"name": "product", "params": {}}}}

    ub_tnorm, lb_implicator = build_itfrs_components_from_config(config)

    assert isinstance(ub_tnorm, ProductTNorm)
    assert isinstance(lb_implicator, LukasiewiczImplicator)


def test_build_itfrs_components_accepts_existing_component_instances():
    ub_tnorm = ProductTNorm()
    lb_implicator = KleeneDienesImplicator()
    config = {"fr_model": {"type": "itfrs", "ub_tnorm": ub_tnorm, "lb_implicator": lb_implicator}}

    resolved_ub_tnorm, resolved_lb_implicator = build_itfrs_components_from_config(config)

    assert resolved_ub_tnorm is ub_tnorm
    assert resolved_lb_implicator is lb_implicator


def test_build_vqrs_components_from_none_returns_default_quantifiers_and_tnorm():
    lb_quantifier, ub_quantifier, tnorm = build_vqrs_components_from_config(None)

    assert isinstance(lb_quantifier, LinearFuzzyQuantifier)
    assert isinstance(ub_quantifier, LinearFuzzyQuantifier)
    assert lb_quantifier.alpha == pytest.approx(0.1)
    assert lb_quantifier.beta == pytest.approx(0.6)
    assert ub_quantifier.alpha == pytest.approx(0.1)
    assert ub_quantifier.beta == pytest.approx(0.6)
    assert isinstance(tnorm, MinTNorm)


def test_build_vqrs_components_from_flat_config_resolves_quantifier_params():
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.2,
        "lb_fuzzy_quantifier_beta": 0.7,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.3,
        "ub_fuzzy_quantifier_beta": 0.8,
    }

    lb_quantifier, ub_quantifier, tnorm = build_vqrs_components_from_config(config)

    assert isinstance(lb_quantifier, LinearFuzzyQuantifier)
    assert lb_quantifier.alpha == pytest.approx(0.2)
    assert lb_quantifier.beta == pytest.approx(0.7)
    assert isinstance(ub_quantifier, QuadraticFuzzyQuantifier)
    assert ub_quantifier.alpha == pytest.approx(0.3)
    assert ub_quantifier.beta == pytest.approx(0.8)
    assert isinstance(tnorm, MinTNorm)


def test_build_vqrs_components_from_nested_config_resolves_specs():
    config = {
        "fr_model": {
            "type": "vqrs",
            "lb_fuzzy_quantifier": {"name": "quadratic", "params": {"alpha": 0.0, "beta": 0.5}},
            "ub_fuzzy_quantifier": {"name": "linear", "params": {"alpha": 0.4, "beta": 1.0}},
        }
    }

    lb_quantifier, ub_quantifier, tnorm = build_vqrs_components_from_config(config)

    assert isinstance(lb_quantifier, QuadraticFuzzyQuantifier)
    assert lb_quantifier.alpha == pytest.approx(0.0)
    assert lb_quantifier.beta == pytest.approx(0.5)
    assert isinstance(ub_quantifier, LinearFuzzyQuantifier)
    assert ub_quantifier.alpha == pytest.approx(0.4)
    assert ub_quantifier.beta == pytest.approx(1.0)
    assert isinstance(tnorm, MinTNorm)


def test_build_vqrs_components_uses_defaults_for_missing_nested_specs():
    config = {
        "fr_model": {
            "type": "vqrs",
            "lb_fuzzy_quantifier": {"name": "quadratic", "params": {"alpha": 0.2, "beta": 0.9}},
        }
    }

    lb_quantifier, ub_quantifier, tnorm = build_vqrs_components_from_config(config)

    assert isinstance(lb_quantifier, QuadraticFuzzyQuantifier)
    assert lb_quantifier.alpha == pytest.approx(0.2)
    assert lb_quantifier.beta == pytest.approx(0.9)
    assert isinstance(ub_quantifier, LinearFuzzyQuantifier)
    assert ub_quantifier.alpha == pytest.approx(0.1)
    assert ub_quantifier.beta == pytest.approx(0.6)
    assert isinstance(tnorm, MinTNorm)


def test_build_vqrs_components_accepts_existing_quantifier_instances():
    lb_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.4)
    ub_quantifier = QuadraticFuzzyQuantifier(alpha=0.4, beta=0.9)
    config = {
        "fr_model": {
            "type": "vqrs",
            "lb_fuzzy_quantifier": lb_quantifier,
            "ub_fuzzy_quantifier": ub_quantifier,
        }
    }

    resolved_lb_quantifier, resolved_ub_quantifier, tnorm = build_vqrs_components_from_config(config)

    assert resolved_lb_quantifier is lb_quantifier
    assert resolved_ub_quantifier is ub_quantifier
    assert isinstance(tnorm, MinTNorm)


def test_build_owafrs_components_from_none_returns_default_components():
    ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(None)

    assert isinstance(ub_tnorm, MinTNorm)
    assert isinstance(lb_implicator, LukasiewiczImplicator)
    assert isinstance(ub_owa_method, LinearOWAWeights)
    assert isinstance(lb_owa_method, LinearOWAWeights)


def test_build_owafrs_components_from_flat_config_resolves_all_component_params():
    config = {
        "type": "owafrs",
        "ub_tnorm_name": "product",
        "lb_implicator_name": "goedel",
        "ub_owa_method_name": "exponential",
        "ub_owa_method_base": 3.0,
        "lb_owa_method_name": "harmonic",
    }

    ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(config)

    assert isinstance(ub_tnorm, ProductTNorm)
    assert isinstance(lb_implicator, GoedelImplicator)
    assert isinstance(ub_owa_method, ExponentialOWAWeights)
    assert ub_owa_method.base == pytest.approx(3.0)
    assert isinstance(lb_owa_method, HarmonicOWAWeights)


def test_build_owafrs_components_from_nested_config_resolves_specs():
    config = {
        "fr_model": {
            "type": "owafrs",
            "ub_tnorm": {"name": "product", "params": {}},
            "lb_implicator": {"name": "kleene", "params": {}},
            "ub_owa_method": {"name": "exponential", "params": {"base": 2.5}},
            "lb_owa_method": {"name": "harmonic", "params": {}},
        }
    }

    ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(config)

    assert isinstance(ub_tnorm, ProductTNorm)
    assert isinstance(lb_implicator, KleeneDienesImplicator)
    assert isinstance(ub_owa_method, ExponentialOWAWeights)
    assert ub_owa_method.base == pytest.approx(2.5)
    assert isinstance(lb_owa_method, HarmonicOWAWeights)


def test_build_owafrs_components_uses_defaults_for_missing_nested_specs():
    config = {
        "fr_model": {
            "type": "owafrs",
            "ub_owa_method": {"name": "exponential", "params": {"base": 4.0}},
        }
    }

    ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(config)

    assert isinstance(ub_tnorm, MinTNorm)
    assert isinstance(lb_implicator, LukasiewiczImplicator)
    assert isinstance(ub_owa_method, ExponentialOWAWeights)
    assert ub_owa_method.base == pytest.approx(4.0)
    assert isinstance(lb_owa_method, LinearOWAWeights)


def test_build_owafrs_components_accepts_existing_component_instances():
    ub_tnorm = ProductTNorm()
    lb_implicator = GoedelImplicator()
    ub_owa_method = ExponentialOWAWeights(base=2.2)
    lb_owa_method = HarmonicOWAWeights()
    config = {
        "fr_model": {
            "type": "owafrs",
            "ub_tnorm": ub_tnorm,
            "lb_implicator": lb_implicator,
            "ub_owa_method": ub_owa_method,
            "lb_owa_method": lb_owa_method,
        }
    }

    resolved_ub_tnorm, resolved_lb_implicator, resolved_ub_owa, resolved_lb_owa = build_owafrs_components_from_config(config)

    assert resolved_ub_tnorm is ub_tnorm
    assert resolved_lb_implicator is lb_implicator
    assert resolved_ub_owa is ub_owa_method
    assert resolved_lb_owa is lb_owa_method


@pytest.mark.parametrize(
    "builder",
    [
        build_itfrs_components_from_config,
        build_vqrs_components_from_config,
        build_owafrs_components_from_config,
    ],
)
def test_component_builders_reject_non_mapping_config(builder):
    with pytest.raises(TypeError):
        builder("not-a-config")


@pytest.mark.parametrize(
    "builder",
    [
        build_itfrs_components_from_config,
        build_vqrs_components_from_config,
        build_owafrs_components_from_config,
    ],
)
def test_component_builders_reject_non_mapping_nested_fr_model_section(builder):
    config = {"similarity": {"name": "l1", "params": {}}, "fr_model": "not-a-mapping"}

    with pytest.raises(TypeError, match="fr_model"):
        builder(config)


@pytest.mark.parametrize(
    "builder, config",
    [
        (
            build_itfrs_components_from_config,
            {"fr_model": {"type": "itfrs", "ub_tnorm": {"name": "unknown_tnorm", "params": {}}}},
        ),
        (
            build_vqrs_components_from_config,
            {
                "fr_model": {
                    "type": "vqrs",
                    "lb_fuzzy_quantifier": {"name": "unknown_quantifier", "params": {"alpha": 0.1, "beta": 0.6}},
                }
            },
        ),
        (
            build_owafrs_components_from_config,
            {"fr_model": {"type": "owafrs", "ub_owa_method": {"name": "unknown_owa", "params": {}}}},
        ),
    ],
)
def test_component_builders_reject_unknown_component_aliases(builder, config):
    with pytest.raises(ValueError):
        builder(config)


@pytest.mark.parametrize(
    "builder, config",
    [
        (
            build_itfrs_components_from_config,
            {"fr_model": {"type": "itfrs", "ub_tnorm": {"name": "product", "params": ["not", "a", "mapping"]}}},
        ),
        (
            build_vqrs_components_from_config,
            {
                "fr_model": {
                    "type": "vqrs",
                    "lb_fuzzy_quantifier": {"name": "linear", "params": ["not", "a", "mapping"]},
                }
            },
        ),
        (
            build_owafrs_components_from_config,
            {"fr_model": {"type": "owafrs", "ub_owa_method": {"name": "linear", "params": ["not", "a", "mapping"]}}},
        ),
    ],
)
def test_component_builders_reject_component_specs_with_non_mapping_params(builder, config):
    with pytest.raises(TypeError, match="params"):
        builder(config)


# -----------------------------------------------------------------------------
# Phase 3: direct ITFRS blockwise engine tests
# -----------------------------------------------------------------------------


ITFRS_SIMILARITY_MATRIX = np.array(
    [
        [1.00, 0.80, 0.30, 0.10],
        [0.80, 1.00, 0.45, 0.20],
        [0.30, 0.45, 1.00, 0.70],
        [0.10, 0.20, 0.70, 1.00],
    ],
    dtype=float,
)
ITFRS_LABELS = np.array(["minority", "minority", "majority", "majority"], dtype=object)


@pytest.mark.parametrize("block_size", [1, 2, 10])
def test_compute_itfrs_blockwise_matches_manual_default_components_for_block_sizes(block_size):
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=block_size)
    expected_lower, expected_upper, expected_boundary, expected_positive_region = _manual_itfrs_expected(
        ITFRS_SIMILARITY_MATRIX,
        ITFRS_LABELS,
    )

    result = compute_itfrs_blockwise(engine, ITFRS_LABELS)

    assert isinstance(result, ITFRSBlockwiseApproximation)
    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)
    np.testing.assert_allclose(result.boundary, result.upper - result.lower, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)
    assert result.execution_backend == "numpy"
    assert result.used_gpu_approximation_accumulators is False


def test_compute_itfrs_blockwise_uses_flat_config_components():
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=2)
    config = {
        "type": "itfrs",
        "ub_tnorm_name": "product",
        "lb_implicator_name": "kleene",
    }
    expected_lower, expected_upper, expected_boundary, expected_positive_region = _manual_itfrs_expected(
        ITFRS_SIMILARITY_MATRIX,
        ITFRS_LABELS,
        ub_tnorm=ProductTNorm(),
        lb_implicator=KleeneDienesImplicator(),
    )

    result = compute_itfrs_blockwise(engine, ITFRS_LABELS, config=config)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)


def test_compute_itfrs_blockwise_uses_nested_config_components():
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=3)
    config = {
        "fr_model": {
            "type": "itfrs",
            "ub_tnorm": {"name": "product", "params": {}},
            "lb_implicator": {"name": "goedel", "params": {}},
        }
    }
    expected_lower, expected_upper, expected_boundary, expected_positive_region = _manual_itfrs_expected(
        ITFRS_SIMILARITY_MATRIX,
        ITFRS_LABELS,
        ub_tnorm=ProductTNorm(),
        lb_implicator=GoedelImplicator(),
    )

    result = compute_itfrs_blockwise(engine, ITFRS_LABELS, config=config)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([10, 10, 5, 5], dtype=int),
        np.array(["a", "a", "b", "b"], dtype=str),
        np.array(["a", 1, "a", 1], dtype=object),
    ],
)
def test_compute_itfrs_blockwise_supports_non_canonical_label_values(labels):
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=1)
    expected_lower, expected_upper, expected_boundary, expected_positive_region = _manual_itfrs_expected(
        ITFRS_SIMILARITY_MATRIX,
        labels,
    )

    result = compute_itfrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)


def test_compute_itfrs_blockwise_handles_all_same_class_labels():
    labels = np.array(["same", "same", "same", "same"], dtype=object)
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=2)

    result = compute_itfrs_blockwise(engine, labels)

    expected_upper = np.array([0.80, 0.80, 0.70, 0.70], dtype=float)
    np.testing.assert_allclose(result.lower, np.ones(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_upper - 1.0, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_compute_itfrs_blockwise_handles_all_singleton_classes():
    labels = np.array(["a", "b", "c", "d"], dtype=object)
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=2)
    expected_lower = 1.0 - np.max(ITFRS_SIMILARITY_MATRIX - np.eye(4), axis=1)

    result = compute_itfrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.boundary, -expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_compute_itfrs_blockwise_documents_single_sample_contract():
    engine = FixedBlockSimilarityEngine(np.array([[1.0]], dtype=float), block_size=1)

    result = compute_itfrs_blockwise(engine, np.array(["only"], dtype=object))

    np.testing.assert_allclose(result.lower, np.array([1.0]), atol=1e-12)
    np.testing.assert_allclose(result.upper, np.array([0.0]), atol=1e-12)
    np.testing.assert_allclose(result.boundary, np.array([-1.0]), atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_compute_itfrs_blockwise_documents_empty_dataset_contract():
    engine = FixedBlockSimilarityEngine(np.empty((0, 0), dtype=float), block_size=2)

    result = compute_itfrs_blockwise(engine, np.array([], dtype=int))

    assert result.lower.shape == (0,)
    assert result.upper.shape == (0,)
    assert result.boundary.shape == (0,)
    assert result.positive_region.shape == (0,)


def test_compute_itfrs_blockwise_rejects_invalid_similarity_engine():
    with pytest.raises(TypeError, match="BaseSimilarityEngine"):
        compute_itfrs_blockwise(object(), ITFRS_LABELS)


def test_compute_itfrs_blockwise_rejects_length_mismatched_labels():
    engine = FixedBlockSimilarityEngine(ITFRS_SIMILARITY_MATRIX, block_size=2)

    with pytest.raises(ValueError, match="Length of labels"):
        compute_itfrs_blockwise(engine, ITFRS_LABELS[:-1])


# -----------------------------------------------------------------------------
# Phase 4: direct VQRS blockwise engine tests
# -----------------------------------------------------------------------------


VQRS_SIMILARITY_MATRIX = np.array(
    [
        [1.00, 0.40, 0.60, 0.50],
        [0.40, 1.00, 0.30, 0.50],
        [0.60, 0.30, 1.00, 0.20],
        [0.50, 0.50, 0.20, 1.00],
    ],
    dtype=float,
)
VQRS_LABELS = np.array(["minority", "minority", "majority", "majority"], dtype=object)


@pytest.mark.parametrize("block_size", [1, 2, 10])
def test_compute_vqrs_blockwise_matches_manual_default_quantifiers_for_block_sizes(block_size):
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=block_size)
    expected_lower, expected_upper, expected_boundary, expected_positive_region, expected_interim = _manual_vqrs_expected(
        VQRS_SIMILARITY_MATRIX,
        VQRS_LABELS,
    )

    result = compute_vqrs_blockwise(engine, VQRS_LABELS)

    assert isinstance(result, VQRSBlockwiseApproximation)
    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)
    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)
    np.testing.assert_allclose(result.boundary, result.upper - result.lower, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)
    assert result.execution_backend == "numpy"
    assert result.used_gpu_approximation_accumulators is False


def test_compute_vqrs_blockwise_uses_flat_config_quantifiers():
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=2)
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.0,
        "lb_fuzzy_quantifier_beta": 0.5,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.2,
        "ub_fuzzy_quantifier_beta": 0.8,
    }
    expected_lower, expected_upper, expected_boundary, expected_positive_region, expected_interim = _manual_vqrs_expected(
        VQRS_SIMILARITY_MATRIX,
        VQRS_LABELS,
        lb_fuzzy_quantifier=LinearFuzzyQuantifier(alpha=0.0, beta=0.5),
        ub_fuzzy_quantifier=QuadraticFuzzyQuantifier(alpha=0.2, beta=0.8),
    )

    result = compute_vqrs_blockwise(engine, VQRS_LABELS, config=config)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)
    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)


def test_compute_vqrs_blockwise_uses_nested_config_quantifiers():
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=3)
    config = {
        "fr_model": {
            "type": "vqrs",
            "lb_fuzzy_quantifier": {"name": "quadratic", "params": {"alpha": 0.0, "beta": 0.5}},
            "ub_fuzzy_quantifier": {"name": "linear", "params": {"alpha": 0.2, "beta": 0.9}},
        }
    }
    expected_lower, expected_upper, expected_boundary, expected_positive_region, expected_interim = _manual_vqrs_expected(
        VQRS_SIMILARITY_MATRIX,
        VQRS_LABELS,
        lb_fuzzy_quantifier=QuadraticFuzzyQuantifier(alpha=0.0, beta=0.5),
        ub_fuzzy_quantifier=LinearFuzzyQuantifier(alpha=0.2, beta=0.9),
    )

    result = compute_vqrs_blockwise(engine, VQRS_LABELS, config=config)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)
    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)


def test_compute_vqrs_blockwise_interim_uses_denominator_without_self_similarity():
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=2)
    expected_denominator = np.sum(VQRS_SIMILARITY_MATRIX, axis=1) - 1.0
    expected_numerator = np.array([0.40, 0.40, 0.20, 0.20], dtype=float)
    expected_interim = expected_numerator / expected_denominator

    result = compute_vqrs_blockwise(engine, VQRS_LABELS)

    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)


def test_compute_vqrs_blockwise_excludes_self_comparison_from_numerator():
    matrix = np.array(
        [
            [1.0, 0.25, 0.50],
            [0.25, 1.0, 0.75],
            [0.50, 0.75, 1.0],
        ],
        dtype=float,
    )
    labels = np.array(["same", "same", "other"], dtype=object)
    engine = FixedBlockSimilarityEngine(matrix, block_size=1)
    expected_interim = np.array(
        [
            0.25 / (0.25 + 0.50),
            0.25 / (0.25 + 0.75),
            0.0 / (0.50 + 0.75),
        ],
        dtype=float,
    )

    result = compute_vqrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([10, 10, 5, 5], dtype=int),
        np.array(["a", "a", "b", "b"], dtype=str),
        np.array(["a", 1, "a", 1], dtype=object),
    ],
)
def test_compute_vqrs_blockwise_supports_non_canonical_label_values(labels):
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=1)
    expected_lower, expected_upper, expected_boundary, expected_positive_region, expected_interim = _manual_vqrs_expected(
        VQRS_SIMILARITY_MATRIX,
        labels,
    )

    result = compute_vqrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.lower, expected_lower, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, expected_positive_region, atol=1e-12)
    np.testing.assert_allclose(result.interim, expected_interim, atol=1e-12)


def test_compute_vqrs_blockwise_handles_all_same_class_labels():
    matrix = np.array(
        [
            [1.0, 0.5, 0.25, 0.25],
            [0.5, 1.0, 0.25, 0.25],
            [0.25, 0.25, 1.0, 0.5],
            [0.25, 0.25, 0.5, 1.0],
        ],
        dtype=float,
    )
    labels = np.array(["same", "same", "same", "same"], dtype=object)
    engine = FixedBlockSimilarityEngine(matrix, block_size=2)

    result = compute_vqrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.interim, np.ones(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.lower, np.ones(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.upper, np.ones(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.boundary, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_compute_vqrs_blockwise_handles_all_singleton_classes():
    labels = np.array(["a", "b", "c", "d"], dtype=object)
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=2)

    result = compute_vqrs_blockwise(engine, labels)

    np.testing.assert_allclose(result.interim, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.lower, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.upper, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.boundary, np.zeros(4, dtype=float), atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_compute_vqrs_blockwise_documents_single_sample_default_quantifier_contract():
    engine = FixedBlockSimilarityEngine(np.array([[1.0]], dtype=float), block_size=1)

    with pytest.raises(ValueError, match="finite values"):
        compute_vqrs_blockwise(engine, np.array(["only"], dtype=object))


def test_compute_vqrs_blockwise_documents_empty_dataset_contract():
    engine = FixedBlockSimilarityEngine(np.empty((0, 0), dtype=float), block_size=2)

    result = compute_vqrs_blockwise(engine, np.array([], dtype=int))

    assert result.lower.shape == (0,)
    assert result.upper.shape == (0,)
    assert result.boundary.shape == (0,)
    assert result.positive_region.shape == (0,)
    assert result.interim.shape == (0,)


def test_compute_vqrs_blockwise_rejects_invalid_similarity_engine():
    with pytest.raises(TypeError, match="BaseSimilarityEngine"):
        compute_vqrs_blockwise(object(), VQRS_LABELS)


def test_compute_vqrs_blockwise_rejects_length_mismatched_labels():
    engine = FixedBlockSimilarityEngine(VQRS_SIMILARITY_MATRIX, block_size=2)

    with pytest.raises(ValueError, match="Length of labels"):
        compute_vqrs_blockwise(engine, VQRS_LABELS[:-1])
