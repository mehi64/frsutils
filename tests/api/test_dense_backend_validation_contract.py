# SPDX-License-Identifier: BSD-3-Clause
"""Backend-selection contracts for the public approximation API."""

from __future__ import annotations

import numpy as np
import pytest

from frsutils import compute_approximations


X = np.array(
    [
        [0.0],
        [0.2],
        [0.8],
        [1.0],
    ],
    dtype=float,
)
Y = np.array([0, 0, 1, 1], dtype=int)


def test_dense_auto_backend_normalizes_to_numpy() -> None:
    """Dense auto selection resolves deterministically to the NumPy backend."""
    result = compute_approximations(
        X,
        Y,
        model="itfrs",
        similarity="linear",
        engine="dense",
        backend="auto",
    )

    assert result.engine == "dense"
    assert result.backend == "numpy"


def test_dense_cupy_backend_is_rejected_explicitly() -> None:
    """Dense approximation execution does not silently ignore a CuPy request."""
    with pytest.raises(ValueError, match="dense.*NumPy-only"):
        compute_approximations(
            X,
            Y,
            model="itfrs",
            similarity="linear",
            engine="dense",
            backend="cupy",
        )


@pytest.mark.parametrize("engine", ["dense", "blockwise"])
def test_unknown_approximation_backend_is_rejected(engine: str) -> None:
    """Unknown backend aliases fail before entering either execution path."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        compute_approximations(
            X,
            Y,
            model="itfrs",
            similarity="linear",
            engine=engine,
            backend="not-a-backend",
        )


@pytest.mark.parametrize("backend", [None, "", "   ", 123])
def test_invalid_approximation_backend_type_is_rejected(backend: object) -> None:
    """Backend selectors must be non-empty strings."""
    with pytest.raises(TypeError, match="backend must be a non-empty string"):
        compute_approximations(
            X,
            Y,
            model="itfrs",
            similarity="linear",
            backend=backend,
        )
