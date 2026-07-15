# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for flat public configuration routing and validation."""

import numpy as np
import pytest
from sklearn.base import clone

from frsutils import (
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_approximations,
)


X_SMALL = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.8, 0.8],
        [0.9, 0.9],
    ],
    dtype=float,
)
Y_SMALL = np.array([0, 0, 1, 1])


@pytest.mark.parametrize(
    "bad_param",
    [
        {"similarity_sigam": 0.5},
        {"ub_tnorm_name": "yager", "ub_tnorm_pp": 2.0},
        {"ub_owa_method_name": "exponential", "ub_owa_method_bsae": 2.0},
    ],
)
def test_compute_approximations_rejects_unknown_prefixed_parameter_names(bad_param):
    """Public approximation calls fail fast on misspelled component parameters."""
    with pytest.raises(ValueError, match="Flat parameter"):
        compute_approximations(X_SMALL, Y_SMALL, model="owafrs", **bad_param)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"similarity": "linear", "similarity_sigma": 0.5}, "similarity='linear'"),
        ({"model": "itfrs", "ub_tnorm_name": "minimum", "ub_tnorm_p": 2.0}, "minimum"),
        (
            {
                "model": "owafrs",
                "ub_owa_method_name": "harmonic",
                "ub_owa_method_base": 2.0,
            },
            "harmonic",
        ),
    ],
)
def test_public_configuration_rejects_parameters_unsupported_by_selected_alias(kwargs, match):
    """Component parameters are validated against the selected registered alias."""
    with pytest.raises(ValueError, match=match):
        compute_approximations(X_SMALL, Y_SMALL, **kwargs)


def test_public_configuration_rejects_components_not_used_by_selected_model():
    """Model-specific component parameters cannot be silently ignored."""
    with pytest.raises(ValueError, match="not used by model='itfrs'"):
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            ub_owa_method_name="exponential",
        )


def test_public_configuration_routes_all_current_parameterized_components():
    """Canonical flat prefixes route current component parameters end to end."""
    itfrs = compute_approximations(
        X_SMALL,
        Y_SMALL,
        model="itfrs",
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="yager",
        similarity_tnorm_p=2.5,
        ub_tnorm_name="yager",
        ub_tnorm_p=1.7,
        lb_implicator_name="goguen",
    )
    owafrs = compute_approximations(
        X_SMALL,
        Y_SMALL,
        model="owafrs",
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
        ub_owa_method_name="exponential",
        ub_owa_method_base=2.5,
        lb_owa_method_name="harmonic",
    )
    vqrs = compute_approximations(
        X_SMALL,
        Y_SMALL,
        model="vqrs",
        lb_fuzzy_quantifier_name="linear",
        lb_fuzzy_quantifier_alpha=0.0,
        lb_fuzzy_quantifier_beta=0.5,
        lb_fuzzy_quantifier_validate_inputs=False,
        ub_fuzzy_quantifier_name="quadratic",
        ub_fuzzy_quantifier_alpha=0.1,
        ub_fuzzy_quantifier_beta=0.8,
        ub_fuzzy_quantifier_validate_inputs=False,
    )

    for result in (itfrs, owafrs, vqrs):
        assert result.positive_region.shape == (len(Y_SMALL),)


def test_public_similarity_default_matches_approximation_default():
    """Public similarity builders and approximation helpers default to linear similarity."""
    default_matrix = build_similarity_matrix(X_SMALL)
    linear_matrix = build_similarity_matrix(X_SMALL, similarity="linear")
    result = compute_approximations(
        None,
        Y_SMALL,
        model="itfrs",
        similarity_matrix=default_matrix,
    )
    expected = compute_approximations(X_SMALL, Y_SMALL, model="itfrs")

    np.testing.assert_allclose(default_matrix, linear_matrix)
    np.testing.assert_allclose(result.positive_region, expected.positive_region)


def test_scorer_exposes_parameterized_component_contract_to_sklearn():
    """The scorer exposes current routed component parameters to clone and get_params."""
    scorer = FuzzyRoughPositiveRegionScorer(
        model="itfrs",
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="yager",
        similarity_tnorm_p=2.5,
        ub_tnorm_name="yager",
        ub_tnorm_p=1.7,
        lb_implicator_name="goguen",
    )

    cloned = clone(scorer)
    params = cloned.get_params(deep=True)

    assert params["similarity_tnorm_p"] == 2.5
    assert params["ub_tnorm_p"] == 1.7
    assert cloned.fit_score(X_SMALL, Y_SMALL).shape == (len(Y_SMALL),)
