"""
@file _sampling_utils.py
@brief Internal random-selection helpers for oversampling algorithms.

This module hosts small sampling helpers copied into the standalone oversampling
package to avoid depending on FRsutils internal utility paths.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# weighted_random_choice               Roulette-wheel selection with safe fallback

# ✅ Design Patterns & Clean Code Notes
# - SRP: random sampling helpers only
# - Determinism: accepts sklearn-compatible random-state objects
##############################################
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np


def weighted_random_choice(items_with_weights: Sequence[Tuple[Any, float]], random_state):
    """
    @brief Select one item based on positive weights.

    @param items_with_weights: Sequence of (item, weight) pairs.
    @param random_state: NumPy/sklearn random-state object.
    @return: Tuple of selected item and original index.
    """
    if not items_with_weights:
        return None, -1

    items, weights = zip(*items_with_weights)
    weights_array = np.asarray(weights, dtype=float)

    valid_mask = weights_array > 0
    if not np.any(valid_mask):
        idx = random_state.randint(len(items))
        return items[idx], idx

    valid_weights = weights_array[valid_mask]
    valid_items = [item for item, keep in zip(items, valid_mask) if keep]
    original_indices = [idx for idx, keep in enumerate(valid_mask) if keep]

    random_value = random_state.uniform(0, np.sum(valid_weights))
    cumulative_weight = np.cumsum(valid_weights)
    selected_idx = np.searchsorted(cumulative_weight, random_value, side="left")
    selected_idx = min(selected_idx, len(valid_items) - 1)

    return valid_items[selected_idx], original_indices[selected_idx]


__all__ = ["weighted_random_choice"]
