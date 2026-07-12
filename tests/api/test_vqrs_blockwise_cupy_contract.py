# SPDX-License-Identifier: BSD-3-Clause
"""Backend-contract tests for VQRS blockwise CuPy execution."""

import numpy as np
import pytest

from frsutils import compute_approximations
from tests._fake_cupy_backend import FakeCupyArray, install_fake_cupy_module


X_VQRS_CUPY = np.array(
    [
        [0.00, 0.00],
        [0.10, 0.20],
        [0.20, 0.10],
        [0.74, 0.82],
        [0.86, 0.76],
        [0.96, 0.90],
    ],
    dtype=float,
)
Y_VQRS_CUPY = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)


def _vqrs_kwargs():
    """Return VQRS settings that exercise custom fuzzy quantifiers."""
    return dict(
        model="vqrs",
        similarity="gaussian",
        similarity_sigma=0.35,
        similarity_tnorm="minimum",
        lb_fuzzy_quantifier_name="linear",
        lb_fuzzy_quantifier_alpha=0.0,
        lb_fuzzy_quantifier_beta=0.7,
        ub_fuzzy_quantifier_name="quadratic",
        ub_fuzzy_quantifier_alpha=0.0,
        ub_fuzzy_quantifier_beta=0.9,
    )


def _assert_same_vqrs_outputs(actual, expected):
    """Assert equality of public VQRS approximation fields."""
    np.testing.assert_allclose(actual.lower, expected.lower, atol=1e-12)
    np.testing.assert_allclose(actual.upper, expected.upper, atol=1e-12)
    np.testing.assert_allclose(actual.boundary, expected.boundary, atol=1e-12)
    np.testing.assert_allclose(actual.positive_region, expected.positive_region, atol=1e-12)


@pytest.mark.parametrize("block_size", [1, 2, 3, 20])
def test_vqrs_blockwise_matches_dense_for_multiple_block_sizes(block_size):
    """Blockwise VQRS must stay exactly equivalent to the dense reference."""
    dense = compute_approximations(X_VQRS_CUPY, Y_VQRS_CUPY, **_vqrs_kwargs(), engine="dense")
    blockwise = compute_approximations(
        X_VQRS_CUPY,
        Y_VQRS_CUPY,
        **_vqrs_kwargs(),
        engine="blockwise",
        block_size=block_size,
    )

    assert blockwise.used_blockwise is True
    assert blockwise.backend == "numpy"
    assert blockwise.similarity_matrix is None
    _assert_same_vqrs_outputs(blockwise, dense)


def test_vqrs_fake_cupy_blockwise_matches_dense_and_returns_plain_numpy(monkeypatch):
    """Fake-CuPy VQRS keeps backend work internal and returns plain NumPy arrays."""
    install_fake_cupy_module(monkeypatch)

    dense = compute_approximations(X_VQRS_CUPY, Y_VQRS_CUPY, **_vqrs_kwargs(), engine="dense")
    gpu_like = compute_approximations(
        X_VQRS_CUPY,
        Y_VQRS_CUPY,
        **_vqrs_kwargs(),
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert gpu_like.backend == "cupy"
    assert gpu_like.used_blockwise is True
    assert gpu_like.used_gpu_similarity_blocks is True
    assert gpu_like.used_gpu_approximation_accumulators is True
    assert isinstance(gpu_like.lower, np.ndarray)
    assert isinstance(gpu_like.upper, np.ndarray)
    assert isinstance(gpu_like.boundary, np.ndarray)
    assert isinstance(gpu_like.positive_region, np.ndarray)
    assert not isinstance(gpu_like.lower, FakeCupyArray)
    assert not isinstance(gpu_like.upper, FakeCupyArray)
    assert not isinstance(gpu_like.boundary, FakeCupyArray)
    assert not isinstance(gpu_like.positive_region, FakeCupyArray)
    _assert_same_vqrs_outputs(gpu_like, dense)


def test_vqrs_fake_cupy_final_conversion_uses_only_required_backend_arrays(monkeypatch):
    """Final VQRS conversion should copy only lower, upper, and interim arrays."""
    fake_cupy = install_fake_cupy_module(monkeypatch)
    original_asnumpy = fake_cupy.asnumpy
    converted_values = []

    def tracking_asnumpy(value):
        assert isinstance(value, FakeCupyArray)
        converted_values.append(value)
        return original_asnumpy(value)

    fake_cupy.asnumpy = tracking_asnumpy

    kwargs = _vqrs_kwargs()
    kwargs.update(
        lb_fuzzy_quantifier_validate_inputs=False,
        ub_fuzzy_quantifier_validate_inputs=False,
    )

    result = compute_approximations(
        X_VQRS_CUPY,
        Y_VQRS_CUPY,
        **kwargs,
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert len(converted_values) == 3
    assert [value.shape for value in converted_values] == [(len(Y_VQRS_CUPY),)] * 3
    assert result.used_gpu_approximation_accumulators is True
    assert result.as_dict()["used_gpu_approximation_accumulators"] is True


def test_vqrs_fake_cupy_supports_object_labels(monkeypatch):
    """Object labels must remain supported when VQRS uses fake-CuPy blocks."""
    install_fake_cupy_module(monkeypatch)
    labels = np.array(["c", "c", 1, 1, 1, "c"], dtype=object)

    dense = compute_approximations(X_VQRS_CUPY, labels, model="vqrs", similarity="linear", engine="dense")
    gpu_like = compute_approximations(
        X_VQRS_CUPY,
        labels,
        model="vqrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=3,
    )

    assert gpu_like.backend == "cupy"
    _assert_same_vqrs_outputs(gpu_like, dense)


def _require_cupy_device():
    """Return CuPy or skip when CuPy/CUDA is unavailable."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available.")
    except Exception as exc:  # pragma: no cover - environment-specific CUDA path
        pytest.skip(f"CuPy CUDA device is not available: {exc}")
    return cp


def test_vqrs_real_cupy_blockwise_matches_dense_when_cuda_is_available():
    """Real CuPy VQRS blockwise execution must match dense NumPy when available."""
    _require_cupy_device()

    dense = compute_approximations(X_VQRS_CUPY, Y_VQRS_CUPY, **_vqrs_kwargs(), engine="dense")
    gpu = compute_approximations(
        X_VQRS_CUPY,
        Y_VQRS_CUPY,
        **_vqrs_kwargs(),
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert gpu.backend == "cupy"
    assert gpu.used_gpu_approximation_accumulators is True
    _assert_same_vqrs_outputs(gpu, dense)
