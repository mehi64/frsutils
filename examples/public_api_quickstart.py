# SPDX-License-Identifier: BSD-3-Clause
"""Quickstart example for the canonical FRsutils public API.

This script demonstrates the stable ``FRsutils.api`` import path. It is intended
for examples and smoke tests, not as part of the stable Python API.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np

# Allow direct execution from a source checkout without requiring editable
# installation first. Installed-package usage is unchanged.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from FRsutils.api import (  # noqa: E402
    FuzzyRoughPositiveRegionScorer,
    compute_approximations,
    compute_positive_region,
)


def make_demo_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Return a tiny normalized binary-class dataset for examples.

    Returns
    -------
    X : ndarray of shape (6, 2)
        Normalized feature matrix.
    y : ndarray of shape (6,)
        Binary class labels.
    """
    X = np.array(
        [
            [0.00, 0.10],
            [0.08, 0.18],
            [0.15, 0.12],
            [0.80, 0.82],
            [0.88, 0.90],
            [0.95, 0.86],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    return X, y


def run_approximation_example() -> np.ndarray:
    """Compute positive-region scores with the functional public API.

    Returns
    -------
    scores : ndarray of shape (6,)
        Positive-region scores from ``compute_approximations``.
    """
    X, y = make_demo_dataset()
    result = compute_approximations(
        X,
        y,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=3,
        backend="numpy",
    )
    shortcut = compute_positive_region(
        X,
        y,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=3,
        backend="numpy",
    )
    assert np.allclose(result.positive_region, shortcut)
    return result.positive_region


def run_scorer_example() -> np.ndarray:
    """Compute positive-region scores with the sklearn-style scorer.

    Returns
    -------
    scores : ndarray of shape (6,)
        Positive-region scores from ``FuzzyRoughPositiveRegionScorer``.
    """
    X, y = make_demo_dataset()
    scorer = FuzzyRoughPositiveRegionScorer(
        model="owafrs",
        similarity="linear",
        engine="dense",
    )
    return scorer.fit_score(X, y)


def main() -> int:
    """Run both quickstart examples from the command line.

    Returns
    -------
    int
        Process exit code.
    """
    approximation_scores = run_approximation_example()
    scorer_scores = run_scorer_example()

    print("FRsutils public API quickstart")
    print("compute_approximations positive region:", approximation_scores)
    print("scorer positive region:", scorer_scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
