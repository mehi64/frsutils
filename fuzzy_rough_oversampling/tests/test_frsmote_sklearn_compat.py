"""
@file test_frsmote_sklearn_compat.py
@brief scikit-learn estimator compatibility tests for standalone FRSMOTE.

The package must remain compatible with scikit-learn cloning, get_params,
set_params, imbalanced-learn Pipeline, and GridSearchCV because FRSMOTE is meant
for experimental ML workflows and model-selection pipelines.
"""

from __future__ import annotations

from sklearn.base import clone

from fuzzy_rough_oversampling import FRSMOTE


def test_frsmote_clone_preserves_flat_constructor_params() -> None:
    """@brief Verify sklearn.clone works with the flat FRSMOTE parameter API."""
    sampler = FRSMOTE(
        type="itfrs",
        similarity="gaussian",
        similarity_sigma=0.25,
        similarity_tnorm="minimum",
        ub_tnorm_name="minimum",
        lb_implicator_name="lukasiewicz",
        k_neighbors=3,
        random_state=11,
    )

    cloned = clone(sampler)

    assert isinstance(cloned, FRSMOTE)
    assert cloned is not sampler
    assert cloned.get_params(deep=False)["similarity"] == "gaussian"
    assert cloned.get_params(deep=False)["similarity_sigma"] == 0.25
    assert cloned.get_params(deep=False)["k_neighbors"] == 3
    assert cloned.get_params(deep=False)["random_state"] == 11


def test_frsmote_set_params_updates_flat_params() -> None:
    """@brief Verify set_params keeps GridSearchCV-style flat updates working."""
    sampler = FRSMOTE()

    returned = sampler.set_params(
        type="itfrs",
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="minimum",
        k_neighbors=4,
        random_state=99,
    )

    params = sampler.get_params(deep=False)
    assert returned is sampler
    assert params["similarity"] == "gaussian"
    assert params["similarity_sigma"] == 0.5
    assert params["k_neighbors"] == 4
    assert params["random_state"] == 99
