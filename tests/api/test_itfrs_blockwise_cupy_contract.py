# SPDX-License-Identifier: BSD-3-Clause
"""CuPy and blockwise execution-contract tests for public ITFRS."""

import numpy as np
import pytest

from frsutils import compute_approximations
from tests._fake_cupy_backend import FakeCupyArray, install_fake_cupy_module


X_ITFRS_PHASE4 = np.array(
    [
        [0.00, 0.00],
        [0.05, 0.20],
        [0.20, 0.10],
        [0.55, 0.65],
        [0.70, 0.75],
        [0.95, 0.90],
        [1.00, 0.85],
    ],
    dtype=float,
)
Y_ITFRS_PHASE4 = np.array(["cold", "cold", "cold", "warm", "warm", "hot", "hot"], dtype=object)


def _assert_same_approximations(actual, expected, *, atol=1e-12):
    """Assert equality of all public approximation vectors."""
    np.testing.assert_allclose(actual.lower, expected.lower, atol=atol)
    np.testing.assert_allclose(actual.upper, expected.upper, atol=atol)
    np.testing.assert_allclose(actual.boundary, expected.boundary, atol=atol)
    np.testing.assert_allclose(actual.positive_region, expected.positive_region, atol=atol)


@pytest.mark.parametrize("block_size", [1, 2, 3, 20])
def test_itfrs_blockwise_numpy_matches_dense_for_object_labels(block_size):
    """Blockwise ITFRS should equal dense ITFRS for non-numeric labels."""
    dense = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="dense",
    )
    blockwise = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=block_size,
    )

    _assert_same_approximations(blockwise, dense)
    assert blockwise.used_blockwise is True
    assert blockwise.backend == "numpy"
    assert blockwise.used_gpu_similarity_blocks is False
    assert blockwise.used_gpu_approximation_accumulators is False
    assert blockwise.similarity_matrix is None


def test_itfrs_fake_cupy_blockwise_matches_dense_with_strict_backend_boundary(monkeypatch):
    """Fake-CuPy ITFRS must match dense while returning only public NumPy arrays."""
    install_fake_cupy_module(monkeypatch)

    dense = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="dense",
    )
    gpu_like = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=3,
    )

    _assert_same_approximations(gpu_like, dense)
    assert gpu_like.backend == "cupy"
    assert gpu_like.used_blockwise is True
    assert gpu_like.used_gpu_similarity_blocks is True
    assert gpu_like.used_gpu_approximation_accumulators is True
    assert gpu_like.similarity_matrix is None
    assert isinstance(gpu_like.lower, np.ndarray)
    assert isinstance(gpu_like.upper, np.ndarray)
    assert isinstance(gpu_like.boundary, np.ndarray)
    assert isinstance(gpu_like.positive_region, np.ndarray)
    assert not isinstance(gpu_like.lower, FakeCupyArray)
    assert not isinstance(gpu_like.upper, FakeCupyArray)
    assert not isinstance(gpu_like.boundary, FakeCupyArray)
    assert not isinstance(gpu_like.positive_region, FakeCupyArray)


def test_itfrs_fake_cupy_finalizes_only_lower_and_upper(monkeypatch):
    """ITFRS should derive boundary/positive after final lower/upper conversion."""
    fake_cupy = install_fake_cupy_module(monkeypatch)
    asnumpy_calls = []
    original_asnumpy = fake_cupy.asnumpy

    def counted_asnumpy(value):
        asnumpy_calls.append(value)
        return original_asnumpy(value)

    fake_cupy.asnumpy = counted_asnumpy

    result = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=3,
    )

    assert len(asnumpy_calls) == 2
    assert all(isinstance(value, FakeCupyArray) for value in asnumpy_calls)
    assert all(value.dtype == np.float64 for value in asnumpy_calls)
    assert all(
        value.dtype == np.float64
        for value in (result.lower, result.upper, result.boundary, result.positive_region)
    )
    np.testing.assert_allclose(result.boundary, result.upper - result.lower, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def test_itfrs_fake_cupy_blockwise_matches_dense_with_parameterized_tnorm(monkeypatch):
    """CuPy-backed blockwise ITFRS should share dense config behavior."""
    install_fake_cupy_module(monkeypatch)

    dense = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="dense",
        ub_tnorm_name="yager",
        ub_tnorm_p=0.83,
        lb_implicator_name="lukasiewicz",
    )
    gpu_like = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
        ub_tnorm_name="yager",
        ub_tnorm_p=0.83,
        lb_implicator_name="lukasiewicz",
    )

    _assert_same_approximations(gpu_like, dense)
    assert gpu_like.used_gpu_approximation_accumulators is True


def test_itfrs_fake_cupy_result_dictionary_preserves_gpu_contract_metadata(monkeypatch):
    """Serialized public result metadata should report the CuPy blockwise contract."""
    install_fake_cupy_module(monkeypatch)

    result = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=4,
    )
    payload = result.as_dict(include_similarity_matrix=True)

    assert payload["model"] == "itfrs"
    assert payload["engine"] == "blockwise"
    assert payload["backend"] == "cupy"
    assert payload["used_blockwise"] is True
    assert payload["used_gpu_similarity_blocks"] is True
    assert payload["used_gpu_approximation_accumulators"] is True
    assert payload["similarity_matrix"] is None
    np.testing.assert_allclose(payload["positive_region"], payload["lower"], atol=1e-12)
    np.testing.assert_allclose(
        payload["boundary"],
        np.asarray(payload["upper"]) - np.asarray(payload["lower"]),
        atol=1e-12,
    )


def _require_cupy_device():
    """Return CuPy or skip when CuPy/CUDA is unavailable."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available.")
    except Exception as exc:  # pragma: no cover - environment-specific CUDA path
        pytest.skip(f"CuPy CUDA device is not available: {exc}")
    return cp


def test_itfrs_real_cupy_blockwise_matches_dense_when_cuda_is_available():
    """Real CuPy ITFRS should match dense NumPy when CUDA is available."""
    _require_cupy_device()

    dense = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="dense",
    )
    gpu = compute_approximations(
        X_ITFRS_PHASE4,
        Y_ITFRS_PHASE4,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=3,
    )

    _assert_same_approximations(gpu, dense)
    assert gpu.backend == "cupy"
    assert gpu.used_gpu_similarity_blocks is True
    assert gpu.used_gpu_approximation_accumulators is True
