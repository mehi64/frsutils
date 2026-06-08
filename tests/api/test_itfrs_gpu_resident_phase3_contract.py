"""
@file test_itfrs_gpu_resident_phase3_contract.py
@brief Phase 3 contract tests for GPU-resident ITFRS blockwise accumulators.

These tests use a small fake CuPy namespace backed by NumPy so the Phase 3
execution contract can be validated in CPU-only CI. The optional real-CUDA
contract remains covered by test_cupy_backend_phase6_contract.py.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# fake CuPy backend                    Exercises backend='cupy' without CUDA hardware
# ITFRS GPU metadata                   Confirms ITFRS uses backend-resident accumulators
# OWAFRS boundary                      Confirms OWAFRS is not falsely marked GPU-resident

# ✅ Design Patterns & Clean Code Notes
# - Optional Dependency Testing: avoids requiring CuPy in normal CI
# - Contract Testing: public metadata must describe actual execution boundaries
# - Regression Testing: CPU dense and fake-CuPy blockwise values remain equivalent
##############################################
"""

import sys
import types

import numpy as np

from FRsutils.api import compute_approximations
from FRsutils.api.similarity import build_similarity_engine


class _FakeCupy(types.SimpleNamespace):
    """@brief Minimal NumPy-backed CuPy stand-in for contract tests."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(value):
        """@brief Mirror cupy.asnumpy with NumPy conversion."""
        return np.asarray(value)


def _install_fake_cupy(monkeypatch):
    """@brief Install the fake CuPy module for the duration of one test."""
    fake_cupy = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    return fake_cupy


X_PHASE3 = np.array(
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
Y_PHASE3 = np.array([0, 0, 0, 1, 1, 1])


def test_itfrs_fake_cupy_path_uses_gpu_resident_accumulator_metadata(monkeypatch):
    """@brief ITFRS marks both similarity blocks and approximation accumulators as GPU-backed."""
    _install_fake_cupy(monkeypatch)

    dense = compute_approximations(X_PHASE3, Y_PHASE3, model="itfrs", similarity="linear", engine="dense")
    gpu_like = compute_approximations(
        X_PHASE3,
        Y_PHASE3,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert gpu_like.backend == "cupy"
    assert gpu_like.used_blockwise is True
    assert gpu_like.used_gpu_similarity_blocks is True
    assert gpu_like.used_gpu_approximation_accumulators is True
    assert gpu_like.as_dict()["used_gpu_approximation_accumulators"] is True
    np.testing.assert_allclose(gpu_like.lower, dense.lower, atol=1e-12)
    np.testing.assert_allclose(gpu_like.upper, dense.upper, atol=1e-12)
    np.testing.assert_allclose(gpu_like.boundary, dense.boundary, atol=1e-12)
    np.testing.assert_allclose(gpu_like.positive_region, dense.positive_region, atol=1e-12)


def test_backend_blocks_can_remain_backend_resident_with_cupy_alias(monkeypatch):
    """@brief Blockwise engines expose backend-resident blocks separately from NumPy iter_blocks()."""
    _install_fake_cupy(monkeypatch)

    engine = build_similarity_engine(
        X_PHASE3,
        engine="blockwise",
        backend="cupy",
        block_size=3,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    backend_block = next(engine.iter_backend_blocks())
    public_block = next(engine.iter_blocks())

    assert backend_block.values_backend == "cupy"
    assert backend_block.values_are_backend_resident is True
    assert public_block.values_backend == "numpy"
    assert public_block.values_are_backend_resident is False
    np.testing.assert_allclose(public_block.values, backend_block.values, atol=1e-12)


def test_owafrs_fake_cupy_path_does_not_claim_gpu_approximation_accumulators(monkeypatch):
    """@brief GPU-resident approximation metadata is still not claimed for OWAFRS."""
    _install_fake_cupy(monkeypatch)

    owafrs = compute_approximations(
        X_PHASE3,
        Y_PHASE3,
        model="owafrs",
        similarity="linear",
        engine="blockwise",
        backend="cupy",
        block_size=2,
    )

    assert owafrs.backend == "cupy"
    assert owafrs.used_gpu_similarity_blocks is True
    assert owafrs.used_gpu_approximation_accumulators is False
