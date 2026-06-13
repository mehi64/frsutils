# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6 contract tests for optional CuPy similarity-block backend."""

import numpy as np
import pytest

from FRsutils.api import build_similarity_engine, build_similarity_matrix, compute_approximations
from FRsutils.core.backends import build_array_backend


X_GPU = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.2, 0.1],
        [0.75, 0.8],
        [0.85, 0.75],
        [0.95, 0.9],
    ],
    dtype=float,
)
Y_GPU = np.array([0, 0, 0, 1, 1, 1])


def _require_cupy_device():
    """Return CuPy or skip when CuPy/CUDA is unavailable."""
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available.")
    except Exception as exc:  # pragma: no cover - environment-specific CUDA path
        pytest.skip(f"CuPy CUDA device is not available: {exc}")
    return cp


def test_cupy_backend_optional_dependency_boundary_is_explicit():
    """Requesting CuPy either resolves a CuPy backend or raises a clear ImportError."""
    try:
        backend = build_array_backend("cupy")
    except ImportError as exc:
        assert "backend='cupy' requires" in str(exc)
    else:
        assert backend.name == "cupy"


def test_cupy_similarity_engine_matches_dense_numpy_when_cuda_is_available():
    """CuPy blockwise similarity blocks must materialize to the same dense matrix."""
    _require_cupy_device()
    expected = build_similarity_matrix(X_GPU, similarity="linear", similarity_tnorm="minimum")
    engine = build_similarity_engine(
        X_GPU,
        engine="blockwise",
        backend="cupy",
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize("model", ["itfrs", "vqrs", "owafrs"])
def test_cupy_blockwise_approximations_match_dense_when_cuda_is_available(model):
    """CuPy-backed blockwise approximations must equal the dense CPU reference."""
    _require_cupy_device()
    dense = compute_approximations(
        X_GPU,
        Y_GPU,
        model=model,
        similarity="linear",
        engine="dense",
    )
    gpu = compute_approximations(
        X_GPU,
        Y_GPU,
        model=model,
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    np.testing.assert_allclose(gpu.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(gpu.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(gpu.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(gpu.positive_region, dense.positive_region, atol=1e-12)
