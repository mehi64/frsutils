# SPDX-License-Identifier: BSD-3-Clause
"""Public result-object and execution-metadata contract tests."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from FRsutils.api import FuzzyRoughApproximationResult, compute_approximations
from tests._fake_cupy_backend import install_fake_cupy_module


X_RESULT = np.array(
    [
        [0.00, 0.00],
        [0.15, 0.05],
        [0.35, 0.25],
        [0.75, 0.80],
        [0.90, 0.85],
    ],
    dtype=float,
)
Y_RESULT = np.array(["a", "a", "a", "b", "b"], dtype=object)

_REQUIRED_DICT_KEYS = {
    "lower",
    "upper",
    "boundary",
    "positive_region",
    "model",
    "similarity",
    "config",
    "engine",
    "backend",
    "block_size",
    "used_blockwise",
    "used_gpu_similarity_blocks",
    "used_gpu_approximation_accumulators",
}


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_public_result_arrays_are_numpy_vectors_for_dense_and_blockwise(model):
    """Public approximation arrays should always be NumPy vectors."""
    dense = compute_approximations(X_RESULT, Y_RESULT, model=model, similarity="linear")
    blockwise = compute_approximations(
        X_RESULT,
        Y_RESULT,
        model=model,
        similarity="linear",
        engine="blockwise",
        block_size=2,
    )

    for result in (dense, blockwise):
        for name in ("lower", "upper", "boundary", "positive_region"):
            value = getattr(result, name)
            assert isinstance(value, np.ndarray)
            assert value.shape == (len(Y_RESULT),)
        np.testing.assert_allclose(result.boundary, result.upper - result.lower, atol=1e-12)
        np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_result_as_dict_has_stable_keys_and_omits_similarity_matrix_by_default(model):
    """as_dict should expose metadata but avoid large matrices unless requested."""
    result = compute_approximations(
        X_RESULT,
        Y_RESULT,
        model=model,
        similarity="linear",
        return_similarity_matrix=True,
    )

    payload = result.as_dict()
    full_payload = result.as_dict(include_similarity_matrix=True)

    assert set(payload) == _REQUIRED_DICT_KEYS
    assert "similarity_matrix" not in payload
    assert set(full_payload) == _REQUIRED_DICT_KEYS | {"similarity_matrix"}
    np.testing.assert_allclose(full_payload["similarity_matrix"], result.similarity_matrix, atol=1e-12)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_result_as_dict_does_not_mutate_public_result(model):
    """Serializing a public result should not change stored arrays or metadata."""
    result = compute_approximations(
        X_RESULT,
        Y_RESULT,
        model=model,
        similarity="linear",
        engine="blockwise",
        block_size=2,
        return_similarity_matrix=True,
    )
    before_lower = result.lower.copy()
    before_matrix = result.similarity_matrix.copy()

    _ = result.as_dict()
    _ = result.as_dict(include_similarity_matrix=True)

    np.testing.assert_allclose(result.lower, before_lower, atol=1e-12)
    np.testing.assert_allclose(result.similarity_matrix, before_matrix, atol=1e-12)
    assert result.engine == "blockwise"
    assert result.block_size == 2


def test_result_dataclass_is_frozen_at_attribute_level():
    """The public result container should reject attribute replacement."""
    result = FuzzyRoughApproximationResult(
        lower=np.array([0.1, 0.2]),
        upper=np.array([0.4, 0.5]),
        boundary=np.array([0.3, 0.3]),
        positive_region=np.array([0.1, 0.2]),
        model="itfrs",
    )

    with pytest.raises(FrozenInstanceError):
        result.model = "vqrs"


@pytest.mark.parametrize(
    ("model", "expected_gpu_accumulators"),
    [
        ("itfrs", True),
        ("vqrs", True),
        ("owafrs", False),
    ],
)
def test_fake_cupy_metadata_distinguishes_similarity_blocks_from_accumulators(
    monkeypatch,
    model,
    expected_gpu_accumulators,
):
    """Model-specific CuPy metadata should match the documented public claim."""
    install_fake_cupy_module(monkeypatch)

    result = compute_approximations(
        X_RESULT,
        Y_RESULT,
        model=model,
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )
    payload = result.as_dict()

    assert result.engine == "blockwise"
    assert result.backend == "cupy"
    assert result.used_blockwise is True
    assert result.used_gpu_similarity_blocks is True
    assert result.used_gpu_approximation_accumulators is expected_gpu_accumulators
    assert payload["used_gpu_similarity_blocks"] is True
    assert payload["used_gpu_approximation_accumulators"] is expected_gpu_accumulators
    for name in ("lower", "upper", "boundary", "positive_region"):
        assert isinstance(getattr(result, name), np.ndarray)
        assert isinstance(payload[name], np.ndarray)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_dense_and_blockwise_metadata_are_explicit_for_all_public_models(model):
    """All models should report the same dense/blockwise provenance fields."""
    dense = compute_approximations(X_RESULT, Y_RESULT, model=model, similarity="linear")
    blockwise = compute_approximations(
        X_RESULT,
        Y_RESULT,
        model=model,
        similarity="linear",
        engine="blockwise",
        backend="numpy",
        block_size=3,
    )

    assert dense.engine == "dense"
    assert dense.backend == "numpy"
    assert dense.block_size is None
    assert dense.used_blockwise is False
    assert dense.used_gpu_similarity_blocks is False
    assert dense.used_gpu_approximation_accumulators is False

    assert blockwise.engine == "blockwise"
    assert blockwise.backend == "numpy"
    assert blockwise.block_size == 3
    assert blockwise.used_blockwise is True
    assert blockwise.used_gpu_similarity_blocks is False
    assert blockwise.used_gpu_approximation_accumulators is False
