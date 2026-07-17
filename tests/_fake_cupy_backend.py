# SPDX-License-Identifier: BSD-3-Clause
"""Fake CuPy module used by tests for optional backend contracts."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


class FakeCupyArray(np.ndarray):
    """NumPy ndarray subclass that marks values as fake CuPy-resident."""

    __array_priority__ = 1000

    def __array_finalize__(self, obj: Any) -> None:
        """Preserve ndarray subclass metadata during NumPy view operations."""
        return None

    def get(self) -> np.ndarray:
        """Return a NumPy copy, matching the common CuPy array API."""
        return np.asarray(self).copy()


def as_fake_cupy_array(value: Any, dtype: Any = None) -> FakeCupyArray:
    """Convert a value to a fake CuPy-resident ndarray subclass."""
    return np.asarray(value, dtype=dtype).view(FakeCupyArray)


class FakeCupyModule:
    """Small NumPy-backed subset of the CuPy API used by similarity tests."""

    ndarray = FakeCupyArray
    float64 = np.float64

    def __init__(self) -> None:
        """Initialize call counters used by backend contract assertions."""
        self.asarray_calls = []
        self.host_to_device_calls = []
        self.asnumpy_calls = []
        self.fill_diagonal_calls = []

    def asarray(self, value: Any, dtype: Any = None) -> FakeCupyArray:
        """Return a fake CuPy array and record host-to-device boundaries."""
        self.asarray_calls.append(value)
        if not isinstance(value, FakeCupyArray):
            self.host_to_device_calls.append(value)
        return as_fake_cupy_array(value, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = np.float64) -> FakeCupyArray:
        """Return fake CuPy zeros."""
        return as_fake_cupy_array(np.zeros(shape, dtype=dtype))

    def ones(self, shape: Any, dtype: Any = np.float64) -> FakeCupyArray:
        """Return fake CuPy ones."""
        return as_fake_cupy_array(np.ones(shape, dtype=dtype))

    def asnumpy(self, value: Any) -> np.ndarray:
        """Convert a fake CuPy array back to NumPy and record the boundary."""
        if not isinstance(value, FakeCupyArray):
            raise AssertionError("asnumpy must receive a fake CuPy-resident array.")
        self.asnumpy_calls.append(value)
        return np.asarray(value).copy()

    def fill_diagonal(self, value: Any, scalar: float) -> None:
        """Fill the diagonal of a fake CuPy-resident array."""
        if not isinstance(value, FakeCupyArray):
            raise AssertionError("cupy.fill_diagonal must not be called on NumPy arrays.")
        self.fill_diagonal_calls.append(value)
        np.fill_diagonal(value, scalar)

    def minimum(self, a: Any, b: Any) -> FakeCupyArray:
        """Apply NumPy-backed minimum and keep fake CuPy residency."""
        return as_fake_cupy_array(np.minimum(np.asarray(a), np.asarray(b)))

    def maximum(self, a: Any, b: Any) -> FakeCupyArray:
        """Apply NumPy-backed maximum and keep fake CuPy residency."""
        return as_fake_cupy_array(np.maximum(np.asarray(a), np.asarray(b)))

    def abs(self, value: Any) -> FakeCupyArray:
        """Apply NumPy-backed absolute value and keep fake CuPy residency."""
        return as_fake_cupy_array(np.abs(np.asarray(value)))

    def exp(self, value: Any) -> FakeCupyArray:
        """Apply NumPy-backed exponential and keep fake CuPy residency."""
        return as_fake_cupy_array(np.exp(np.asarray(value)))

    def where(self, condition: Any, x: Any, y: Any) -> FakeCupyArray:
        """Apply NumPy-backed where and keep fake CuPy residency."""
        return as_fake_cupy_array(np.where(condition, np.asarray(x), np.asarray(y)))

    def min(self, value: Any, axis: Any = None) -> FakeCupyArray:
        """Apply NumPy-backed minimum reduction and keep fake CuPy residency."""
        return as_fake_cupy_array(np.min(np.asarray(value), axis=axis))

    def max(self, value: Any, axis: Any = None) -> FakeCupyArray:
        """Apply NumPy-backed maximum reduction and keep fake CuPy residency."""
        return as_fake_cupy_array(np.max(np.asarray(value), axis=axis))

    def any(self, value: Any, axis: Any = None) -> FakeCupyArray:
        """Apply NumPy-backed any reduction and keep fake CuPy residency."""
        return as_fake_cupy_array(np.any(np.asarray(value), axis=axis))

    def isfinite(self, value: Any) -> FakeCupyArray:
        """Apply NumPy-backed finite check and keep fake CuPy residency."""
        return as_fake_cupy_array(np.isfinite(np.asarray(value)))

    def prod(self, value: Any, axis: Any = None) -> FakeCupyArray:
        """Apply NumPy-backed product reduction and keep fake CuPy residency."""
        return as_fake_cupy_array(np.prod(np.asarray(value), axis=axis))

    def sum(self, value: Any, axis: Any = None) -> FakeCupyArray:
        """Apply NumPy-backed sum reduction and keep fake CuPy residency."""
        return as_fake_cupy_array(np.sum(np.asarray(value), axis=axis))


def install_fake_cupy_module(monkeypatch: Any) -> FakeCupyModule:
    """Install and return a fake CuPy module through pytest monkeypatching."""
    fake_cupy = FakeCupyModule()
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    return fake_cupy
