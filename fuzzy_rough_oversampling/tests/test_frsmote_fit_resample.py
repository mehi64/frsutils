"""
@file test_frsmote_fit_resample.py
@brief Basic fit_resample behavior tests for the standalone FRSMOTE.

These tests verify that the migrated FRSMOTE implementation can be used as a
real imbalanced-learn style sampler, not only imported or registered. The input
features are already in [0, 1] because fuzzy-rough similarities in FRsutils
expect normalized numeric feature values.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from fuzzy_rough_oversampling import FRSMOTE


def build_small_imbalanced_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Build a deterministic normalized binary imbalanced dataset.

    @return: Tuple of X/y arrays suitable for FRSMOTE smoke tests.
    """
    rng = np.random.RandomState(123)
    majority = rng.uniform(0.05, 0.55, size=(24, 4))
    minority = rng.uniform(0.45, 0.95, size=(8, 4))
    X = np.vstack([majority, minority]).astype(float)
    y = np.array([0] * len(majority) + [1] * len(minority))
    return X, y


def test_frsmote_fit_resample_balances_binary_dataset() -> None:
    """@brief Verify that FRSMOTE balances a simple binary dataset by default."""
    X, y = build_small_imbalanced_dataset()

    sampler = FRSMOTE(
        type="itfrs",
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="minimum",
        k_neighbors=3,
        random_state=42,
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    counts = Counter(y_resampled)
    assert X_resampled.shape[0] == y_resampled.shape[0]
    assert X_resampled.shape[1] == X.shape[1]
    assert counts[0] == counts[1]
    assert counts[0] == 24
    assert sampler.is_built
    assert sampler.n_features_in_ == X.shape[1]


@pytest.mark.parametrize("model_type", ["itfrs", "owafrs", "vqrs"])
def test_frsmote_fit_resample_supports_registered_model_types(model_type: str) -> None:
    """@brief Verify basic fit_resample compatibility for ITFRS/OWAFRS/VQRS."""
    X, y = build_small_imbalanced_dataset()

    sampler = FRSMOTE(
        type=model_type,
        similarity="gaussian",
        similarity_sigma=0.4,
        similarity_tnorm="minimum",
        k_neighbors=3,
        random_state=7,
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    assert X_resampled.shape == (48, X.shape[1])
    assert Counter(y_resampled) == {0: 24, 1: 24}
