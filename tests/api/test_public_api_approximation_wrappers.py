# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for public approximation convenience wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from frsutils import (
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)


X_WRAPPER = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.7, 0.8],
        [0.9, 0.85],
    ],
    dtype=float,
)
Y_WRAPPER = np.array(["a", "a", "b", "b"], dtype=object)

WRAPPERS = {
    "lower": compute_lower_approximation,
    "upper": compute_upper_approximation,
    "boundary": compute_boundary_region,
    "positive_region": compute_positive_region,
}


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
@pytest.mark.parametrize("engine", ["dense", "blockwise"])
@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_approximation_wrappers_match_full_result_for_all_models_and_engines(model, engine, field, wrapper):
    """Wrappers return the same named array as ``compute_approximations``."""
    result = compute_approximations(
        X_WRAPPER,
        Y_WRAPPER,
        model=model,
        similarity="linear",
        engine=engine,
        block_size=2,
    )

    values = wrapper(
        X_WRAPPER,
        Y_WRAPPER,
        model=model,
        similarity="linear",
        engine=engine,
        block_size=2,
    )

    assert isinstance(values, np.ndarray)
    assert values.shape == (len(Y_WRAPPER),)
    np.testing.assert_allclose(values, getattr(result, field))


@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_approximation_wrappers_accept_precomputed_similarity_matrix(field, wrapper):
    """Dense wrappers can reuse a public precomputed similarity matrix."""
    similarity_matrix = build_similarity_matrix(X_WRAPPER, similarity="linear")
    result = compute_approximations(
        None,
        Y_WRAPPER,
        model="itfrs",
        similarity_matrix=similarity_matrix,
    )

    values = wrapper(None, Y_WRAPPER, model="itfrs", similarity_matrix=similarity_matrix)

    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, getattr(result, field))


@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_approximation_wrappers_reject_nested_config(field, wrapper):
    """Convenience wrappers preserve the flat-only public config contract."""
    nested_config = {
        "similarity": {"name": "linear", "params": {}},
        "similarity_tnorm": {"name": "minimum", "params": {}},
        "fr_model": {"type": "vqrs"},
    }

    with pytest.raises(ValueError, match="Nested configuration is internal"):
        wrapper(X_WRAPPER, Y_WRAPPER, model="vqrs", config=nested_config)


@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_approximation_wrappers_do_not_return_result_metadata(field, wrapper):
    """Convenience wrappers return arrays, not result containers or metadata."""
    values = wrapper(X_WRAPPER, Y_WRAPPER, model="owafrs", similarity="linear")

    assert isinstance(values, np.ndarray)
    assert not hasattr(values, "metadata")
    assert not hasattr(values, "as_dict")


@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_approximation_wrappers_propagate_public_validation_errors(field, wrapper):
    """Wrapper validation follows the main compute_approximations boundary."""
    with pytest.raises(ValueError, match="labels must be a 1D"):
        wrapper(X_WRAPPER, [["a", "a"], ["b", "b"]], model="itfrs")


@pytest.mark.parametrize("field,wrapper", WRAPPERS.items())
def test_blockwise_wrappers_reject_precomputed_similarity_matrix(field, wrapper):
    """Blockwise wrappers keep the public precomputed-matrix restriction."""
    similarity_matrix = build_similarity_matrix(X_WRAPPER, similarity="linear")

    with pytest.raises(ValueError, match="engine='blockwise' requires X"):
        wrapper(
            X_WRAPPER,
            Y_WRAPPER,
            model="itfrs",
            engine="blockwise",
            similarity_matrix=similarity_matrix,
        )
