# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for tests that require a usable real CuPy/CUDA device."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def require_usable_cupy() -> Any:
    """Return CuPy after a real device allocation and kernel smoke test.

    Returns
    -------
    module
        Imported CuPy module backed by at least one usable CUDA device.

    Notes
    -----
    Importing CuPy is not sufficient evidence that CUDA is usable. This helper
    skips real-CUDA tests when device discovery, allocation, kernel execution,
    synchronization, or host conversion fails.
    """
    cp = pytest.importorskip("cupy", reason="CuPy is not installed in this test environment.")

    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            pytest.skip("CuPy is installed but no CUDA device is available.")

        probe = cp.asarray([0.0, 1.0, 2.0, 3.0], dtype=cp.float64)
        squared_sum = cp.sum(probe * probe)
        cp.cuda.Stream.null.synchronize()
        np.testing.assert_allclose(cp.asnumpy(squared_sum), np.asarray(14.0), atol=0.0)
    except Exception as exc:  # pragma: no cover - depends on the local CUDA stack.
        pytest.skip(f"CuPy is importable but CUDA is unusable in this environment: {exc}")

    return cp
