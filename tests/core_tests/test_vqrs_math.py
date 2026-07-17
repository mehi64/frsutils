# SPDX-License-Identifier: BSD-3-Clause
"""Numerical contract tests for shared VQRS ratio calculations."""

import numpy as np

from frsutils.core.models.vqrs_math import compute_vqrs_interim_ratio
from tests._fake_cupy_backend import FakeCupyArray, install_fake_cupy_module


def test_vqrs_interim_ratio_assigns_zero_without_nonself_mass():
    """Zero denominators represent absence of non-self evidence."""
    numerator = np.array([0.0, 0.0], dtype=float)
    denominator = np.array([0.0, 0.0], dtype=float)

    result = compute_vqrs_interim_ratio(numerator, denominator)

    np.testing.assert_allclose(result, np.zeros(2), atol=0.0)


def test_vqrs_interim_ratio_clips_floating_point_excursions():
    """Round-off outside the mathematical ratio interval is clipped safely."""
    numerator = np.array([np.nextafter(1.0, 2.0), -np.finfo(float).eps], dtype=float)
    denominator = np.ones(2, dtype=float)

    result = compute_vqrs_interim_ratio(numerator, denominator)

    np.testing.assert_allclose(result, np.array([1.0, 0.0]), atol=0.0)


def test_vqrs_interim_ratio_preserves_fake_cupy_residency(monkeypatch):
    """Backend-aware ratio stabilization stays resident until public conversion."""
    fake_cupy = install_fake_cupy_module(monkeypatch)
    numerator = fake_cupy.asarray([0.0, 0.4, np.nextafter(1.0, 2.0)])
    denominator = fake_cupy.asarray([0.0, 0.8, 1.0])

    result = compute_vqrs_interim_ratio(
        numerator,
        denominator,
        xp=fake_cupy,
    )

    assert isinstance(result, FakeCupyArray)
    np.testing.assert_allclose(np.asarray(result), np.array([0.0, 0.5, 1.0]), atol=0.0)
