# SPDX-License-Identifier: BSD-3-Clause
"""Fast contract and regression tests for the dense ITFRS model."""

import numpy as np
import pytest

from frsutils.core.implicators import Implicator, LukasiewiczImplicator
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.models.itfrs_components import build_itfrs_components_from_config
from frsutils.core.tnorms import MinTNorm, TNorm, YagerTNorm
from tests import reference_data_store as reference_data


ITFRS_REFERENCE_CASE = reference_data.get_itfrs_dense_baseline_testsets()[0]
SIMILARITY_MATRIX = ITFRS_REFERENCE_CASE["similarity_matrix"]
LABELS = ITFRS_REFERENCE_CASE["labels"]
EXPECTED_LOWER = ITFRS_REFERENCE_CASE["expected"]["lower"]
EXPECTED_UPPER = ITFRS_REFERENCE_CASE["expected"]["upper"]
EXPECTED_BOUNDARY = ITFRS_REFERENCE_CASE["expected"]["boundary"]
EXPECTED_POSITIVE_REGION = ITFRS_REFERENCE_CASE["expected"]["positive_region"]
ITFRS_COMPONENTS = ITFRS_REFERENCE_CASE["components"]


def _build_reference_model(labels=LABELS):
    """Build the small dense ITFRS reference fixture."""
    upper_spec = ITFRS_COMPONENTS["upper_tnorm"]
    lower_spec = ITFRS_COMPONENTS["lower_implicator"]
    return ITFRS(
        SIMILARITY_MATRIX.copy(),
        labels,
        ub_tnorm=TNorm.create(upper_spec["name"], **upper_spec["params"]),
        lb_implicator=Implicator.create(lower_spec["name"], **lower_spec["params"]),
    )


def test_itfrs_dense_reference_matches_hand_computed_values():
    """Check exact lower, upper, boundary, and positive-region values."""
    model = _build_reference_model()

    np.testing.assert_allclose(model.lower_approximation(), EXPECTED_LOWER, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), EXPECTED_UPPER, atol=1e-12)
    np.testing.assert_allclose(model.boundary_region(), EXPECTED_BOUNDARY, atol=1e-12)
    np.testing.assert_allclose(model.positive_region(), EXPECTED_POSITIVE_REGION, atol=1e-12)


def test_itfrs_dense_approximations_do_not_mutate_similarity_matrix():
    """Lower and upper computations must not mutate the stored similarity matrix."""
    model = _build_reference_model()
    original = model.similarity_matrix.copy()

    model.lower_approximation()
    model.upper_approximation()

    np.testing.assert_array_equal(model.similarity_matrix, original)


def test_itfrs_direct_constructor_stores_list_labels_as_numpy_array():
    """Direct dense ITFRS accepts array-like labels and stores a NumPy vector."""
    model = _build_reference_model(labels=[0, 0, 1, 1])

    assert isinstance(model.labels, np.ndarray)
    np.testing.assert_allclose(model.lower_approximation(), EXPECTED_LOWER, atol=1e-12)


def test_itfrs_from_config_supports_flat_names_without_leaking_name_params():
    """Flat component names must not be forwarded as constructor parameters."""
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
    )

    np.testing.assert_allclose(model.lower_approximation(), EXPECTED_LOWER, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), EXPECTED_UPPER, atol=1e-12)


def test_itfrs_from_config_supports_prefixed_tnorm_parameters():
    """Flat config should route prefixed T-norm parameters to the T-norm only."""
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        ub_tnorm_name="yager",
        ub_tnorm_p=0.83,
        lb_implicator_name="lukasiewicz",
    )

    assert isinstance(model.ub_tnorm, YagerTNorm)
    assert model.ub_tnorm.p == pytest.approx(0.83)


def test_itfrs_from_config_supports_legacy_top_level_tnorm_p():
    """Legacy config with top-level `p` remains supported for Yager T-norm."""
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        ub_tnorm_name="yager",
        lb_implicator_name="lukasiewicz",
        p=0.83,
    )

    assert isinstance(model.ub_tnorm, YagerTNorm)
    assert model.ub_tnorm.p == pytest.approx(0.83)


def test_itfrs_from_config_supports_serialized_operator_specs():
    """Serialized registry components should round-trip through from_config."""
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        ub_tnorm=MinTNorm().to_dict(),
        lb_implicator=LukasiewiczImplicator().to_dict(),
    )

    np.testing.assert_allclose(model.lower_approximation(), EXPECTED_LOWER, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), EXPECTED_UPPER, atol=1e-12)


def test_itfrs_from_config_supports_nested_component_specs():
    """Internal nested config should build the same dense ITFRS components."""
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        _nested_config={
            "fr_model": {
                "ub_tnorm": {"name": "minimum", "params": {}},
                "lb_implicator": {"name": "lukasiewicz", "params": {}},
            }
        },
    )

    np.testing.assert_allclose(model.lower_approximation(), EXPECTED_LOWER, atol=1e-12)
    np.testing.assert_allclose(model.upper_approximation(), EXPECTED_UPPER, atol=1e-12)


def test_itfrs_from_config_uses_shared_private_nested_component_builder():
    """Direct ITFRS should resolve private nested config through the shared helper."""
    config = {
        "_nested_config": {
            "fr_model": {
                "type": "itfrs",
                "ub_tnorm": {"name": "yager", "params": {"p": 0.83}},
                "lb_implicator": {"name": "lukasiewicz", "params": {}},
            }
        }
    }

    expected_ub_tnorm, expected_lb_implicator = build_itfrs_components_from_config(
        config,
        require_explicit_components=True,
    )
    model = ITFRS.from_config(
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
        **config,
    )

    assert type(model.ub_tnorm) is type(expected_ub_tnorm)
    assert model.ub_tnorm.p == pytest.approx(expected_ub_tnorm.p)
    assert type(model.lb_implicator) is type(expected_lb_implicator)


def test_itfrs_from_config_rejects_missing_component_names():
    """Flat config must declare both ITFRS operators when no nested spec exists."""
    with pytest.raises(ValueError, match="ub_tnorm_name"):
        ITFRS.from_config(
            similarity_matrix=SIMILARITY_MATRIX,
            labels=LABELS,
            lb_implicator_name="lukasiewicz",
        )

    with pytest.raises(ValueError, match="lb_implicator_name"):
        ITFRS.from_config(
            similarity_matrix=SIMILARITY_MATRIX,
            labels=LABELS,
            ub_tnorm_name="minimum",
        )


def test_itfrs_from_config_rejects_unknown_component_alias():
    """Unknown registry aliases should fail at the direct model boundary."""
    with pytest.raises(ValueError, match="Unknown alias"):
        ITFRS.from_config(
            similarity_matrix=SIMILARITY_MATRIX,
            labels=LABELS,
            ub_tnorm_name="unknown_tnorm",
            lb_implicator_name="lukasiewicz",
        )


def test_itfrs_from_dict_without_embedded_data_requires_external_data():
    """Serialized operators without data need matrix and labels arguments."""
    model = _build_reference_model()
    serialized = model.to_dict(include_data=False)

    with pytest.raises(ValueError, match="similarity_matrix and labels"):
        ITFRS.from_dict(serialized)

    restored = ITFRS.from_dict(
        serialized,
        similarity_matrix=SIMILARITY_MATRIX,
        labels=LABELS,
    )
    np.testing.assert_allclose(restored.lower_approximation(), EXPECTED_LOWER, atol=1e-12)


def test_itfrs_registry_contains_direct_model_alias():
    """The ITFRS model must stay registered under the public model alias."""
    assert TNorm.create("minimum").name == "min"
    assert FuzzyRoughModel._registry["itfrs"] is ITFRS
