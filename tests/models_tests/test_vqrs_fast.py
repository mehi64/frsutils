# SPDX-License-Identifier: BSD-3-Clause
"""Fast regression and contract tests for the dense VQRS model."""

import numpy as np
import pytest

from FRsutils.core.fuzzy_quantifiers import (
    FuzzyQuantifier,
    LinearFuzzyQuantifier,
    QuadraticFuzzyQuantifier,
)
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from FRsutils.core.models.vqrs import VQRS


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


def _manual_vqrs_values(
    similarity_matrix,
    labels,
    *,
    lb_fuzzy_quantifier=None,
    ub_fuzzy_quantifier=None,
):
    """Compute expected dense VQRS values without using the VQRS class."""
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    labels = np.asarray(labels)
    if lb_fuzzy_quantifier is None:
        lb_fuzzy_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
    if ub_fuzzy_quantifier is None:
        ub_fuzzy_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)

    label_mask = (labels[:, None] == labels[None, :]).astype(float)
    tnorm_values = np.minimum(similarity_matrix, label_mask)
    if similarity_matrix.shape[0]:
        np.fill_diagonal(tnorm_values, 0.0)

    numerator = np.sum(tnorm_values, axis=1)
    denominator = np.sum(similarity_matrix, axis=1) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        interim = numerator / denominator

    lower = lb_fuzzy_quantifier(interim)
    upper = ub_fuzzy_quantifier(interim)
    boundary = upper - lower
    positive_region = lower.copy()
    return lower, upper, boundary, positive_region, interim


def _build_reference_model(labels=VQRS_LABELS):
    """Return a dense VQRS model with asymmetric quantifier settings."""
    return VQRS(
        VQRS_SIMILARITY_MATRIX.copy(),
        labels,
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        QuadraticFuzzyQuantifier(alpha=0.0, beta=0.8),
    )


def test_vqrs_is_registered_under_model_alias():
    """The direct VQRS model must stay available through the model registry."""
    assert FuzzyRoughModel.get_class("vqrs") is VQRS


def test_dense_vqrs_matches_manual_interim_and_approximations():
    """Dense VQRS must match a hand-computed reference formula."""
    lb_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
    ub_quantifier = QuadraticFuzzyQuantifier(alpha=0.0, beta=0.8)
    model = VQRS(VQRS_SIMILARITY_MATRIX.copy(), VQRS_LABELS, lb_quantifier, ub_quantifier)

    expected_lower, expected_upper, expected_boundary, expected_positive, expected_interim = _manual_vqrs_values(
        VQRS_SIMILARITY_MATRIX,
        VQRS_LABELS,
        lb_fuzzy_quantifier=lb_quantifier,
        ub_fuzzy_quantifier=ub_quantifier,
    )

    np.testing.assert_allclose(model._interim_calculations(), expected_interim, atol=1e-12)
    np.testing.assert_allclose(model.lower_approximation(), expected_lower, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), expected_upper, atol=1e-12)
    np.testing.assert_allclose(model.boundary_region(), expected_boundary, atol=1e-12)
    np.testing.assert_allclose(model.positive_region(), expected_positive, atol=1e-12)


def test_dense_vqrs_accepts_list_labels_and_stores_numpy_labels():
    """Array-like labels should work in the direct dense reference model."""
    model = _build_reference_model(labels=["minority", "minority", "majority", "majority"])

    assert isinstance(model.labels, np.ndarray)
    np.testing.assert_allclose(model.lower_approximation(), _build_reference_model().lower_approximation(), atol=1e-12)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([10, 10, 5, 5], dtype=int),
        np.array(["a", "a", "b", "b"], dtype=str),
        np.array(["a", 1, "a", 1], dtype=object),
    ],
)
def test_dense_vqrs_supports_non_canonical_label_values(labels):
    """Dense VQRS should support numeric, string, and object labels."""
    model = _build_reference_model(labels=labels)
    expected_lower, expected_upper, expected_boundary, expected_positive, _ = _manual_vqrs_values(
        VQRS_SIMILARITY_MATRIX,
        labels,
        lb_fuzzy_quantifier=model.lb_fuzzy_quantifier,
        ub_fuzzy_quantifier=model.ub_fuzzy_quantifier,
    )

    np.testing.assert_allclose(model.lower_approximation(), expected_lower, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), expected_upper, atol=1e-12)
    np.testing.assert_allclose(model.boundary_region(), expected_boundary, atol=1e-12)
    np.testing.assert_allclose(model.positive_region(), expected_positive, atol=1e-12)


def test_dense_vqrs_does_not_mutate_similarity_matrix():
    """Repeated dense calculations must not mutate the stored similarity matrix."""
    matrix = VQRS_SIMILARITY_MATRIX.copy()
    model = _build_reference_model()

    model.lower_approximation()
    model.upper_approximation()
    model.boundary_region()
    model.positive_region()

    np.testing.assert_allclose(model.similarity_matrix, matrix, atol=0.0)
    np.testing.assert_allclose(model.lower_approximation(), model.lower_approximation(), atol=1e-12)


def test_dense_vqrs_handles_all_same_class_labels():
    """When all samples share a class, the VQRS interim ratio is one."""
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
    model = VQRS(
        matrix,
        labels,
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )

    np.testing.assert_allclose(model._interim_calculations(), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(model.lower_approximation(), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(model.boundary_region(), np.zeros(4), atol=1e-12)


def test_dense_vqrs_handles_all_singleton_classes():
    """When every sample is a singleton class, the interim ratio is zero."""
    model = VQRS(
        VQRS_SIMILARITY_MATRIX.copy(),
        np.array(["a", "b", "c", "d"], dtype=object),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )

    np.testing.assert_allclose(model._interim_calculations(), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(model.lower_approximation(), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(model.boundary_region(), np.zeros(4), atol=1e-12)


def test_dense_vqrs_empty_dataset_returns_empty_outputs():
    """Dense VQRS keeps the empty-dataset contract aligned with blockwise VQRS."""
    model = VQRS(
        np.empty((0, 0), dtype=float),
        np.array([], dtype=int),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )

    assert model.lower_approximation().shape == (0,)
    assert model.upper_approximation().shape == (0,)
    assert model.boundary_region().shape == (0,)
    assert model.positive_region().shape == (0,)


def test_dense_vqrs_single_sample_contract_rejects_nan_interim():
    """A single sample produces 0/0 interim and is rejected by quantifier validation."""
    model = VQRS(
        np.array([[1.0]], dtype=float),
        np.array(["only"], dtype=object),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )

    with pytest.raises(ValueError, match="finite values"):
        model.lower_approximation()


def test_dense_vqrs_zero_denominator_contract_rejects_nan_interim():
    """Rows with no non-self similarity produce invalid VQRS ratios by contract."""
    model = VQRS(
        np.eye(2, dtype=float),
        np.array(["a", "b"], dtype=object),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
    )

    with pytest.raises(ValueError, match="finite values"):
        model.lower_approximation()


def test_dense_vqrs_rejects_non_1d_labels():
    """Direct VQRS labels must be a one-dimensional vector."""
    with pytest.raises(ValueError, match="1D"):
        _build_reference_model(labels=np.array([["a"], ["a"], ["b"], ["b"]], dtype=object))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lb_fuzzy_quantifier": None}, "lower"),
        ({"ub_fuzzy_quantifier": None}, "upper"),
        ({"lb_fuzzy_quantifier": "linear"}, "lower"),
        ({"ub_fuzzy_quantifier": "linear"}, "upper"),
    ],
)
def test_dense_vqrs_rejects_invalid_quantifiers(kwargs, message):
    """Direct VQRS should reject missing or non-quantifier components."""
    lb = kwargs.get("lb_fuzzy_quantifier", LinearFuzzyQuantifier(alpha=0.1, beta=0.6))
    ub = kwargs.get("ub_fuzzy_quantifier", LinearFuzzyQuantifier(alpha=0.1, beta=0.6))

    with pytest.raises(ValueError, match=message):
        VQRS(VQRS_SIMILARITY_MATRIX.copy(), VQRS_LABELS, lb, ub)


def test_dense_vqrs_rejects_invalid_similarity_matrix_values():
    """Similarity validation should reject values outside [0, 1]."""
    matrix = VQRS_SIMILARITY_MATRIX.copy()
    matrix[0, 1] = 1.1

    with pytest.raises(ValueError, match="range"):
        VQRS(
            matrix,
            VQRS_LABELS,
            LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
            LinearFuzzyQuantifier(alpha=0.1, beta=0.6),
        )


def test_dense_vqrs_from_dict_without_embedded_data_uses_explicit_arrays():
    """Serialized VQRS components can be reconstructed with external data arrays."""
    model = _build_reference_model()
    restored = VQRS.from_dict(
        model.to_dict(include_data=False),
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS.tolist(),
    )

    np.testing.assert_allclose(restored.lower_approximation(), model.lower_approximation(), atol=1e-12)
    np.testing.assert_allclose(restored.upper_approximation(), model.upper_approximation(), atol=1e-12)
    assert isinstance(restored.labels, np.ndarray)


def test_dense_vqrs_from_dict_requires_data_or_explicit_arrays():
    """from_dict must fail clearly when model data are unavailable."""
    model = _build_reference_model()

    with pytest.raises(ValueError, match="similarity_matrix and labels"):
        VQRS.from_dict(model.to_dict(include_data=False))


def test_dense_vqrs_from_dict_rejects_invalid_quantifier_spec():
    """Invalid serialized fuzzy quantifier specs should fail at construction."""
    data = _build_reference_model().to_dict(include_data=False)
    data["lb_fuzzy_quantifier"] = {"name": "does_not_exist", "params": {}}

    with pytest.raises((KeyError, ValueError, TypeError)):
        VQRS.from_dict(data, similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(), labels=VQRS_LABELS)

from FRsutils.core.approximation_engines import compute_vqrs_blockwise
from FRsutils.core.models.vqrs_components import build_vqrs_components_from_config
from FRsutils.core.similarity_engine import BaseSimilarityEngine, SimilarityBlock


class _PrecomputedSimilarityEngine(BaseSimilarityEngine):
    """Minimal engine that yields a precomputed similarity matrix for tests."""

    def __init__(self, similarity_matrix):
        self.similarity_matrix = np.asarray(similarity_matrix, dtype=float)
        super().__init__(np.zeros((self.similarity_matrix.shape[0], 1), dtype=float))

    def iter_blocks(self):
        """Yield the stored dense similarity matrix as one block."""
        yield SimilarityBlock(
            row_slice=slice(0, self.n_samples),
            col_slice=slice(0, self.n_samples),
            values=self.similarity_matrix.copy(),
        )


def _assert_same_vqrs_components(left, right):
    """Assert two VQRS component tuples have equivalent types and parameters."""
    left_lb, left_ub, left_tnorm = left
    right_lb, right_ub, right_tnorm = right

    assert type(left_lb) is type(right_lb)
    assert type(left_ub) is type(right_ub)
    assert type(left_tnorm) is type(right_tnorm)
    assert left_lb.to_dict() == right_lb.to_dict()
    assert left_ub.to_dict() == right_ub.to_dict()


def test_vqrs_shared_component_builder_supports_flat_config():
    """Dense and blockwise VQRS should resolve flat quantifier config identically."""
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.1,
        "lb_fuzzy_quantifier_beta": 0.6,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.0,
        "ub_fuzzy_quantifier_beta": 0.8,
    }

    dense_model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **config,
    )
    shared_components = build_vqrs_components_from_config(config, require_explicit_components=True)

    _assert_same_vqrs_components(
        (dense_model.lb_fuzzy_quantifier, dense_model.ub_fuzzy_quantifier, dense_model.tnorm),
        shared_components,
    )


def test_vqrs_shared_component_builder_supports_legacy_alpha_beta_aliases():
    """Legacy ub_alpha/lb_alpha aliases should remain compatible for dense VQRS."""
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_alpha": 0.1,
        "lb_beta": 0.6,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_alpha": 0.0,
        "ub_beta": 0.8,
    }

    model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **config,
    )

    assert isinstance(model.lb_fuzzy_quantifier, LinearFuzzyQuantifier)
    assert isinstance(model.ub_fuzzy_quantifier, QuadraticFuzzyQuantifier)
    assert model.lb_fuzzy_quantifier.alpha == pytest.approx(0.1)
    assert model.ub_fuzzy_quantifier.beta == pytest.approx(0.8)


def test_vqrs_shared_component_builder_supports_nested_config():
    """Nested FRsutils config should build the same VQRS quantifiers as dense config."""
    nested_config = {
        "fr_model": {
            "type": "vqrs",
            "lb_fuzzy_quantifier": {"name": "linear", "params": {"alpha": 0.2, "beta": 0.7}},
            "ub_fuzzy_quantifier": {"name": "quadratic", "params": {"alpha": 0.0, "beta": 0.9}},
        }
    }

    model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **nested_config,
    )
    shared_components = build_vqrs_components_from_config(nested_config, require_explicit_components=True)

    _assert_same_vqrs_components(
        (model.lb_fuzzy_quantifier, model.ub_fuzzy_quantifier, model.tnorm),
        shared_components,
    )


def test_vqrs_shared_component_builder_supports_private_nested_config():
    """Private _nested_config should be accepted by dense VQRS construction."""
    private_nested = {
        "_nested_config": {
            "fr_model": {
                "type": "vqrs",
                "lb_fuzzy_quantifier": {"name": "linear", "params": {"alpha": 0.1, "beta": 0.5}},
                "ub_fuzzy_quantifier": {"name": "linear", "params": {"alpha": 0.2, "beta": 0.8}},
            }
        }
    }

    model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **private_nested,
    )
    shared_components = build_vqrs_components_from_config(private_nested, require_explicit_components=True)

    _assert_same_vqrs_components(
        (model.lb_fuzzy_quantifier, model.ub_fuzzy_quantifier, model.tnorm),
        shared_components,
    )


def test_vqrs_shared_component_builder_supports_serialized_quantifier_specs():
    """Serialized quantifier dictionaries should round-trip through from_config."""
    lb_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
    ub_quantifier = QuadraticFuzzyQuantifier(alpha=0.0, beta=0.8)
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier": lb_quantifier.to_dict(),
        "ub_fuzzy_quantifier": ub_quantifier.to_dict(),
    }

    model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **config,
    )

    assert model.lb_fuzzy_quantifier.to_dict() == lb_quantifier.to_dict()
    assert model.ub_fuzzy_quantifier.to_dict() == ub_quantifier.to_dict()


def test_vqrs_shared_component_builder_supports_direct_quantifier_instances():
    """Direct fuzzy-quantifier instances should pass through flat config safely."""
    lb_quantifier = LinearFuzzyQuantifier(alpha=0.1, beta=0.6)
    ub_quantifier = QuadraticFuzzyQuantifier(alpha=0.0, beta=0.8)

    model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        type="vqrs",
        lb_fuzzy_quantifier=lb_quantifier,
        ub_fuzzy_quantifier=ub_quantifier,
    )

    assert model.lb_fuzzy_quantifier is lb_quantifier
    assert model.ub_fuzzy_quantifier is ub_quantifier


def test_vqrs_from_config_requires_explicit_quantifiers_for_dense_model():
    """Dense VQRS should fail clearly when required quantifier specs are missing."""
    with pytest.raises(ValueError, match="lb_fuzzy_quantifier_name"):
        VQRS.from_config(
            similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
            labels=VQRS_LABELS,
            type="vqrs",
            ub_fuzzy_quantifier_name="linear",
            ub_fuzzy_quantifier_alpha=0.1,
            ub_fuzzy_quantifier_beta=0.6,
        )

    with pytest.raises(ValueError, match="ub_fuzzy_quantifier_name"):
        VQRS.from_config(
            similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
            labels=VQRS_LABELS,
            type="vqrs",
            lb_fuzzy_quantifier_name="linear",
            lb_fuzzy_quantifier_alpha=0.1,
            lb_fuzzy_quantifier_beta=0.6,
        )


def test_vqrs_from_config_and_blockwise_share_component_construction():
    """Dense from_config and blockwise engine should produce equivalent VQRS values."""
    config = {
        "type": "vqrs",
        "lb_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.1,
        "lb_fuzzy_quantifier_beta": 0.6,
        "ub_fuzzy_quantifier_name": "quadratic",
        "ub_fuzzy_quantifier_alpha": 0.0,
        "ub_fuzzy_quantifier_beta": 0.8,
    }
    dense_model = VQRS.from_config(
        similarity_matrix=VQRS_SIMILARITY_MATRIX.copy(),
        labels=VQRS_LABELS,
        **config,
    )
    engine = _PrecomputedSimilarityEngine(VQRS_SIMILARITY_MATRIX.copy())
    blockwise = compute_vqrs_blockwise(engine, VQRS_LABELS, config=config)

    np.testing.assert_allclose(blockwise.lower, dense_model.lower_approximation(), atol=1e-12)
    np.testing.assert_allclose(blockwise.upper, dense_model.upper_approximation(), atol=1e-12)
    np.testing.assert_allclose(blockwise.boundary, dense_model.boundary_region(), atol=1e-12)
    np.testing.assert_allclose(blockwise.positive_region, dense_model.positive_region(), atol=1e-12)
