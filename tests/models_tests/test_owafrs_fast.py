# SPDX-License-Identifier: BSD-3-Clause
"""Fast regression and contract tests for dense OWAFRS."""

import numpy as np
import pytest

from frsutils.core.implicators import (
    GoedelImplicator,
    Implicator,
    KleeneDienesImplicator,
    LukasiewiczImplicator,
)
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.owafrs import OWAFRS
from frsutils.core.owa_weights import (
    ExponentialOWAWeights,
    HarmonicOWAWeights,
    LinearOWAWeights,
    OWAWeights,
)
from frsutils.core.tnorms import MinTNorm, ProductTNorm, TNorm, YagerTNorm
from tests import reference_data_store as reference_data


OWAFRS_REFERENCE_CASE = reference_data.get_owafrs_dense_baseline_testsets()[0]
OWAFRS_SIMILARITY_MATRIX = OWAFRS_REFERENCE_CASE["similarity_matrix"]
OWAFRS_LABELS = np.asarray(OWAFRS_REFERENCE_CASE["labels"]["values"], dtype=object)
OWAFRS_LABELS.setflags(write=False)
OWAFRS_COMPONENTS = OWAFRS_REFERENCE_CASE["components"]
OWAFRS_EXPECTED = OWAFRS_REFERENCE_CASE["expected"]


def _small_similarity_matrix():
    """Return a writable copy of the JSON-backed OWAFRS matrix."""
    return OWAFRS_SIMILARITY_MATRIX.copy()


def _build_model(labels=None):
    """Return the JSON-backed dense OWAFRS reference model."""
    labels = OWAFRS_LABELS if labels is None else labels
    upper_tnorm_spec = OWAFRS_COMPONENTS["upper_tnorm"]
    lower_implicator_spec = OWAFRS_COMPONENTS["lower_implicator"]
    upper_owa_spec = OWAFRS_COMPONENTS["upper_owa"]
    lower_owa_spec = OWAFRS_COMPONENTS["lower_owa"]
    return OWAFRS(
        _small_similarity_matrix(),
        labels,
        TNorm.create(upper_tnorm_spec["name"], **upper_tnorm_spec["params"]),
        Implicator.create(
            lower_implicator_spec["name"],
            **lower_implicator_spec["params"],
        ),
        OWAWeights.create(upper_owa_spec["name"], **upper_owa_spec["params"]),
        OWAWeights.create(lower_owa_spec["name"], **lower_owa_spec["params"]),
    )


def _manual_owafrs_values(similarity_matrix, labels):
    """Compute comparison values for alternate label partitions."""
    labels = np.asarray(labels)
    label_mask = (labels[:, None] == labels[None, :]).astype(float)

    lower_evidence = np.minimum(1.0, 1.0 - similarity_matrix + label_mask)
    upper_evidence = np.minimum(similarity_matrix, label_mask)

    np.fill_diagonal(lower_evidence, 0.0)
    np.fill_diagonal(upper_evidence, 0.0)

    sorted_lower = np.sort(lower_evidence, axis=1)[:, ::-1][:, :-1]
    sorted_upper = np.sort(upper_evidence, axis=1)[:, ::-1][:, :-1]

    n_compared = similarity_matrix.shape[0] - 1
    lower_weights = LinearOWAWeights().weights(n_compared, order="asc")
    upper_weights = LinearOWAWeights().weights(n_compared, order="desc")

    lower = np.asarray(sorted_lower @ lower_weights, dtype=float)
    upper = np.asarray(sorted_upper @ upper_weights, dtype=float)
    return lower, upper, upper - lower, lower.copy()


def test_owafrs_matches_reference_lower_upper_boundary_and_positive_region():
    """Dense OWAFRS matches independently verified JSON reference values."""
    model = _build_model()

    np.testing.assert_allclose(model.lower_approximation(), OWAFRS_EXPECTED["lower"])
    np.testing.assert_allclose(model.upper_approximation(), OWAFRS_EXPECTED["upper"])
    np.testing.assert_allclose(model.boundary_region(), OWAFRS_EXPECTED["boundary"])
    np.testing.assert_allclose(
        model.positive_region(),
        OWAFRS_EXPECTED["positive_region"],
    )


def test_owafrs_accepts_list_labels_and_stores_numpy_array():
    """List labels are normalized so dense OWAFRS supports array-like labels."""
    model = _build_model(labels=["a", "a", "b", "b"])

    assert isinstance(model.labels, np.ndarray)
    np.testing.assert_allclose(
        model.lower_approximation(),
        _manual_owafrs_values(_small_similarity_matrix(), model.labels)[0],
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 1, 1]),
        np.array(["left", "left", "right", "right"], dtype=object),
        np.array(["left", 1, "left", 1], dtype=object),
    ],
)
def test_owafrs_supports_numeric_string_and_object_labels(labels):
    """Dense OWAFRS label matching works for common non-float label types."""
    model = _build_model(labels=labels)
    lower, upper, boundary, positive = _manual_owafrs_values(_small_similarity_matrix(), labels)

    np.testing.assert_allclose(model.lower_approximation(), lower)
    np.testing.assert_allclose(model.upper_approximation(), upper)
    np.testing.assert_allclose(model.boundary_region(), boundary)
    np.testing.assert_allclose(model.positive_region(), positive)


def test_owafrs_does_not_mutate_similarity_matrix():
    """Lower and upper approximation calls must not mutate stored similarities."""
    model = _build_model()
    before = model.similarity_matrix.copy()

    model.lower_approximation()
    model.upper_approximation()
    model.boundary_region()
    model.positive_region()

    np.testing.assert_array_equal(model.similarity_matrix, before)


def test_owafrs_all_same_class_and_all_singleton_classes_return_valid_vectors():
    """Dense OWAFRS handles degenerate label partitions with at least two samples."""
    for labels in (np.array([1, 1, 1, 1]), np.array([0, 1, 2, 3])):
        model = _build_model(labels=labels)
        lower = model.lower_approximation()
        upper = model.upper_approximation()

        assert lower.shape == (4,)
        assert upper.shape == (4,)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all((0.0 <= lower) & (lower <= 1.0))
        assert np.all((0.0 <= upper) & (upper <= 1.0))


@pytest.mark.parametrize(
    "similarity_matrix, labels",
    [
        (np.empty((0, 0)), []),
        (np.array([[1.0]]), ["only"]),
    ],
)
def test_owafrs_requires_at_least_two_samples(similarity_matrix, labels):
    """OWA aggregation has no non-self comparison for empty or single-sample data."""
    with pytest.raises(ValueError, match="at least two samples"):
        OWAFRS(
            similarity_matrix,
            labels,
            MinTNorm(),
            LukasiewiczImplicator(),
            LinearOWAWeights(),
            LinearOWAWeights(),
        )


def test_owafrs_rejects_non_1d_labels():
    """Dense OWAFRS rejects labels with an ambiguous two-dimensional shape."""
    with pytest.raises(ValueError, match="1D"):
        _build_model(labels=np.array([["a"], ["a"], ["b"], ["b"]], dtype=object))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ub_tnorm": None},
        {"lb_implicator": None},
        {"ub_owa_method": None},
        {"lb_owa_method": None},
    ],
)
def test_owafrs_rejects_missing_components(kwargs):
    """Dense OWAFRS rejects missing scientific component objects."""
    params = {
        "ub_tnorm": MinTNorm(),
        "lb_implicator": LukasiewiczImplicator(),
        "ub_owa_method": LinearOWAWeights(),
        "lb_owa_method": LinearOWAWeights(),
    }
    params.update(kwargs)

    with pytest.raises(ValueError):
        OWAFRS(_small_similarity_matrix(), ["a", "a", "b", "b"], **params)


def test_owafrs_from_dict_without_embedded_data_accepts_external_arrays():
    """Serialized component configuration can be reused with external data arrays."""
    model = _build_model()
    serialized = model.to_dict(include_data=False)
    restored = OWAFRS.from_dict(serialized, similarity_matrix=model.similarity_matrix, labels=model.labels)

    np.testing.assert_allclose(restored.lower_approximation(), model.lower_approximation())
    np.testing.assert_allclose(restored.upper_approximation(), model.upper_approximation())


def test_owafrs_from_dict_requires_data_when_not_embedded():
    """A serialized OWAFRS without embedded data must receive matrix and labels."""
    serialized = _build_model().to_dict(include_data=False)

    with pytest.raises(ValueError, match="similarity_matrix and labels"):
        OWAFRS.from_dict(serialized)


def test_owafrs_registry_alias_builds_dense_model():
    """The registered 'owafrs' alias constructs the dense reference model."""
    model = FuzzyRoughModel.create(
        "owafrs",
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        ub_tnorm=MinTNorm(),
        lb_implicator=LukasiewiczImplicator(),
        ub_owa_method=LinearOWAWeights(),
        lb_owa_method=LinearOWAWeights(),
    )

    assert isinstance(model, OWAFRS)
    assert model.lower_approximation().shape == (4,)



def test_owafrs_from_config_accepts_flat_component_config():
    """Flat OWAFRS config resolves every scientific component and parameter."""
    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        ub_tnorm_name="yager",
        ub_tnorm_p=2.5,
        lb_implicator_name="goedel",
        ub_owa_method_name="exponential",
        ub_owa_method_base=3.0,
        lb_owa_method_name="harmonic",
    )

    assert isinstance(model.ub_tnorm, YagerTNorm)
    assert model.ub_tnorm.p == pytest.approx(2.5)
    assert isinstance(model.lb_implicator, GoedelImplicator)
    assert isinstance(model.ub_owa_method, ExponentialOWAWeights)
    assert model.ub_owa_method.base == pytest.approx(3.0)
    assert isinstance(model.lb_owa_method, HarmonicOWAWeights)
    assert model.lower_approximation().shape == (4,)


def test_owafrs_from_config_accepts_nested_config():
    """Nested OWAFRS config follows the same component contract as blockwise code."""
    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=np.array(["a", "a", "b", "b"], dtype=object),
        fr_model={
            "type": "owafrs",
            "ub_tnorm": {"name": "product", "params": {}},
            "lb_implicator": {"name": "kleene", "params": {}},
            "ub_owa_method": {"name": "exponential", "params": {"base": 2.5}},
            "lb_owa_method": {"name": "linear", "params": {}},
        },
    )

    assert isinstance(model.ub_tnorm, ProductTNorm)
    assert isinstance(model.lb_implicator, KleeneDienesImplicator)
    assert isinstance(model.ub_owa_method, ExponentialOWAWeights)
    assert model.ub_owa_method.base == pytest.approx(2.5)
    assert isinstance(model.lb_owa_method, LinearOWAWeights)


def test_owafrs_from_config_accepts_private_nested_config():
    """Private nested config is accepted for compatibility with public builders."""
    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        _nested_config={
            "fr_model": {
                "type": "owafrs",
                "ub_tnorm": {"name": "product", "params": {}},
                "lb_implicator": {"name": "goedel", "params": {}},
                "ub_owa_method": {"name": "linear", "params": {}},
                "lb_owa_method": {"name": "harmonic", "params": {}},
            }
        },
    )

    assert isinstance(model.ub_tnorm, ProductTNorm)
    assert isinstance(model.lb_implicator, GoedelImplicator)
    assert isinstance(model.ub_owa_method, LinearOWAWeights)
    assert isinstance(model.lb_owa_method, HarmonicOWAWeights)


def test_owafrs_from_config_accepts_serialized_component_specs():
    """Serialized component specs can be passed directly to from_config."""
    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        ub_tnorm=YagerTNorm(p=3.0).to_dict(),
        lb_implicator=KleeneDienesImplicator().to_dict(),
        ub_owa_method=ExponentialOWAWeights(base=2.2).to_dict(),
        lb_owa_method=HarmonicOWAWeights().to_dict(),
    )

    assert isinstance(model.ub_tnorm, YagerTNorm)
    assert model.ub_tnorm.p == pytest.approx(3.0)
    assert isinstance(model.lb_implicator, KleeneDienesImplicator)
    assert isinstance(model.ub_owa_method, ExponentialOWAWeights)
    assert model.ub_owa_method.base == pytest.approx(2.2)
    assert isinstance(model.lb_owa_method, HarmonicOWAWeights)


def test_owafrs_from_config_accepts_direct_component_instances():
    """Direct component objects are passed through without reconstruction."""
    ub_tnorm = ProductTNorm()
    lb_implicator = GoedelImplicator()
    ub_owa_method = ExponentialOWAWeights(base=2.4)
    lb_owa_method = HarmonicOWAWeights()

    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        ub_tnorm=ub_tnorm,
        lb_implicator=lb_implicator,
        ub_owa_method=ub_owa_method,
        lb_owa_method=lb_owa_method,
    )

    assert model.ub_tnorm is ub_tnorm
    assert model.lb_implicator is lb_implicator
    assert model.ub_owa_method is ub_owa_method
    assert model.lb_owa_method is lb_owa_method


def test_owafrs_from_config_supports_legacy_p_and_base_aliases():
    """Legacy top-level p/base aliases remain usable for old OWAFRS configs."""
    model = OWAFRS.from_config(
        similarity_matrix=_small_similarity_matrix(),
        labels=["a", "a", "b", "b"],
        ub_tnorm_name="yager",
        p=2.7,
        lb_implicator_name="lukasiewicz",
        ub_owa_method_name="exponential",
        lb_owa_method_name="exponential",
        base=3.5,
    )

    assert isinstance(model.ub_tnorm, YagerTNorm)
    assert model.ub_tnorm.p == pytest.approx(2.7)
    assert isinstance(model.ub_owa_method, ExponentialOWAWeights)
    assert isinstance(model.lb_owa_method, ExponentialOWAWeights)
    assert model.ub_owa_method.base == pytest.approx(3.5)
    assert model.lb_owa_method.base == pytest.approx(3.5)


@pytest.mark.parametrize(
    "missing_key, expected_message",
    [
        ("ub_tnorm_name", "ub_tnorm_name"),
        ("lb_implicator_name", "lb_implicator_name"),
        ("ub_owa_method_name", "ub_owa_method_name"),
        ("lb_owa_method_name", "lb_owa_method_name"),
    ],
)
def test_owafrs_from_config_requires_explicit_components(missing_key, expected_message):
    """Dense from_config reports missing OWAFRS component selectors clearly."""
    config = {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
    }
    config.pop(missing_key)

    with pytest.raises(ValueError, match=expected_message):
        OWAFRS.from_config(
            similarity_matrix=_small_similarity_matrix(),
            labels=["a", "a", "b", "b"],
            **config,
        )


@pytest.mark.parametrize(
    "bad_config",
    [
        {"ub_tnorm_name": "unknown_tnorm"},
        {"lb_implicator_name": "unknown_implicator"},
        {"ub_owa_method_name": "unknown_owa"},
        {"lb_owa_method_name": "unknown_owa"},
    ],
)
def test_owafrs_from_config_rejects_unknown_component_aliases(bad_config):
    """Unknown component aliases fail at the OWAFRS config boundary."""
    config = {
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
    }
    config.update(bad_config)

    with pytest.raises(ValueError):
        OWAFRS.from_config(
            similarity_matrix=_small_similarity_matrix(),
            labels=["a", "a", "b", "b"],
            **config,
        )


def test_owafrs_from_dict_embedded_data_roundtrip_supports_object_labels():
    """Embedded object labels survive OWAFRS serialization roundtrip."""
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    model = _build_model(labels=labels)

    restored = OWAFRS.from_dict(model.to_dict(include_data=True))

    assert restored.labels.dtype == object or restored.labels.dtype.kind in {"O", "U", "S"}
    np.testing.assert_array_equal(restored.labels, labels)
    np.testing.assert_allclose(restored.lower_approximation(), model.lower_approximation())
    np.testing.assert_allclose(restored.upper_approximation(), model.upper_approximation())


def test_owafrs_from_dict_external_arrays_override_embedded_data():
    """External arrays override embedded data while reusing serialized components."""
    serialized = _build_model().to_dict(include_data=True)
    external_sim = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float)
    external_labels = np.array(["x", "y"], dtype=object)

    restored = OWAFRS.from_dict(
        serialized,
        similarity_matrix=external_sim,
        labels=external_labels,
    )

    np.testing.assert_array_equal(restored.similarity_matrix, external_sim)
    np.testing.assert_array_equal(restored.labels, external_labels)
    assert restored.lower_approximation().shape == (2,)
    assert restored.upper_approximation().shape == (2,)


@pytest.mark.parametrize(
    "component_key",
    ["ub_tnorm", "lb_implicator", "ub_owa_method", "lb_owa_method"],
)
def test_owafrs_from_dict_rejects_invalid_serialized_component_specs(component_key):
    """Malformed serialized component specs fail at the from_dict boundary."""
    serialized = _build_model().to_dict(include_data=True)
    serialized[component_key] = {"name": "not_registered", "params": {}}

    with pytest.raises((TypeError, ValueError, KeyError)):
        OWAFRS.from_dict(serialized)


@pytest.mark.parametrize(
    "similarity_matrix, labels, expected_message",
    [
        ([[1.0, 0.5], [0.5, 1.0]], [0, 1], "2D NumPy array"),
        (np.array([1.0, 0.5]), [0, 1], "2D NumPy array"),
        (np.ones((2, 3)), [0, 1], "square"),
        (np.array([[1.0, 1.2], [1.2, 1.0]]), [0, 1], "range"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), [0, 1], "finite"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), [0, 1], "finite"),
        (_small_similarity_matrix(), [0, 1, 1], "Length of labels"),
    ],
)
def test_owafrs_rejects_invalid_similarity_matrix_and_label_contracts(
    similarity_matrix,
    labels,
    expected_message,
):
    """Dense OWAFRS rejects invalid matrix and label contracts clearly."""
    with pytest.raises(ValueError, match=expected_message):
        OWAFRS(
            similarity_matrix,
            labels,
            MinTNorm(),
            LukasiewiczImplicator(),
            LinearOWAWeights(),
            LinearOWAWeights(),
        )


@pytest.mark.parametrize(
    "component_overrides",
    [
        {"ub_tnorm": "minimum"},
        {"lb_implicator": "lukasiewicz"},
        {"ub_owa_method": "linear"},
        {"lb_owa_method": "linear"},
    ],
)
def test_owafrs_rejects_component_objects_with_wrong_types(component_overrides):
    """Dense OWAFRS requires concrete component instances, not aliases."""
    params = {
        "ub_tnorm": MinTNorm(),
        "lb_implicator": LukasiewiczImplicator(),
        "ub_owa_method": LinearOWAWeights(),
        "lb_owa_method": LinearOWAWeights(),
    }
    params.update(component_overrides)

    with pytest.raises(ValueError):
        OWAFRS(_small_similarity_matrix(), ["a", "a", "b", "b"], **params)
