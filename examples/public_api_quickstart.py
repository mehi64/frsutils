# SPDX-License-Identifier: BSD-3-Clause
"""Quickstart example for the canonical frsutils public API.

This script demonstrates the stable ``frsutils`` root import path. It is intended
for examples and smoke tests, not as part of the stable Python API.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

# Allow direct execution from a source checkout without requiring editable
# installation first. Installed-package usage is unchanged.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from frsutils import (  # noqa: E402
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


def _format_array(values: np.ndarray) -> str:
    """Format an array compactly for console output.

    Parameters
    ----------
    values : ndarray
        Array to format.

    Returns
    -------
    text : str
        Compact string representation with stable precision.
    """
    return np.array2string(values, precision=4, suppress_small=True)


def run_approximation_example() -> np.ndarray:
    """Compute positive-region scores with functional public API calls.

    Returns
    -------
    scores : ndarray of shape (6,)
        Positive-region scores from blockwise ITFRS execution.
    """
    X, y = make_demo_dataset()

    dense = compute_approximations(
        X,
        y,
        model="itfrs",
        similarity="linear",
        engine="dense",
        backend="numpy",
    )
    blockwise = compute_approximations(
        X,
        y,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=3,
        backend="numpy",
    )
    shortcut_scores = compute_positive_region(
        X,
        y,
        model="itfrs",
        similarity="linear",
        engine="blockwise",
        block_size=3,
        backend="numpy",
    )

    assert np.allclose(dense.positive_region, blockwise.positive_region)
    assert np.allclose(blockwise.positive_region, shortcut_scores)

    print("Functional API example")
    print("  dense positive region:    ", _format_array(dense.positive_region))
    print("  blockwise positive region:", _format_array(blockwise.positive_region))
    print("  blockwise metadata:")
    print(f"    engine={blockwise.engine!r}")
    print(f"    backend={blockwise.backend!r}")
    print(f"    block_size={blockwise.block_size!r}")
    print(f"    used_blockwise={blockwise.used_blockwise!r}")
    print(f"    used_gpu_similarity_blocks={blockwise.used_gpu_similarity_blocks!r}")
    print(
        "    used_gpu_approximation_accumulators="
        f"{blockwise.used_gpu_approximation_accumulators!r}"
    )
    print("  dense/blockwise equivalence: OK")

    return blockwise.positive_region


def run_model_examples() -> Dict[str, np.ndarray]:
    """Run representative ITFRS, OWAFRS, and VQRS public API calls.

    Returns
    -------
    scores_by_model : dict of str to ndarray
        Positive-region scores produced by each public model alias.
    """
    X, y = make_demo_dataset()
    results = {
        "itfrs": compute_approximations(
            X,
            y,
            model="itfrs",
            similarity="gaussian",
            similarity_sigma=0.4,
            similarity_tnorm="yager",
            similarity_tnorm_p=2.0,
            ub_tnorm_name="yager",
            ub_tnorm_p=1.7,
            lb_implicator_name="goguen",
        ),
        "owafrs": compute_approximations(
            X,
            y,
            model="owafrs",
            similarity="linear",
            ub_tnorm_name="minimum",
            lb_implicator_name="lukasiewicz",
            ub_owa_method_name="exponential",
            ub_owa_method_base=2.5,
            lb_owa_method_name="harmonic",
        ),
        "vqrs": compute_approximations(
            X,
            y,
            model="vqrs",
            similarity="linear",
            lb_fuzzy_quantifier_name="linear",
            lb_fuzzy_quantifier_alpha=0.0,
            lb_fuzzy_quantifier_beta=0.5,
            ub_fuzzy_quantifier_name="quadratic",
            ub_fuzzy_quantifier_alpha=0.1,
            ub_fuzzy_quantifier_beta=0.8,
        ),
    }
    scores_by_model = {
        model: result.positive_region for model, result in results.items()
    }

    print("Model configuration examples")
    for model, scores in scores_by_model.items():
        print(f"  {model}: {_format_array(scores)}")
    return scores_by_model


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
    scores = scorer.fit_score(X, y)

    print("Scorer API example")
    print("  OWAFRS positive region:", _format_array(scores))
    return scores


def main() -> int:
    """Run the public API quickstart from the command line.

    Returns
    -------
    exit_code : int
        Process exit code.
    """
    print("frsutils public API quickstart")
    run_approximation_example()
    run_model_examples()
    run_scorer_example()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
