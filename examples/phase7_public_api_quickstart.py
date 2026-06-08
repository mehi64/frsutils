"""
@file phase7_public_api_quickstart.py
@brief Phase 7 release-ready public API example for FRsutils.

This example is intentionally small and deterministic. It demonstrates the
public FRsutils approximation facade, dense vs exact blockwise execution,
execution metadata, and the conservative GPU claim boundary. The example uses
only `FRsutils.api`, so it is safe for README, paper artifact, and downstream
package documentation.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# main                                 Run a small dense/blockwise approximation demo
# make_demo_dataset                    Build normalized toy data
# _format_array                        Create compact display strings

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: imports only from FRsutils.api
# - Reproducibility: fixed toy data with deterministic labels
# - Release Artifact Readiness: prints result metadata and equivalence checks
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# python examples/phase7_public_api_quickstart.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np

# Allow direct execution from a source checkout without requiring editable
# installation first. Installed-package usage still works unchanged.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from FRsutils.api import compute_approximations, compute_positive_region  # noqa: E402


def make_demo_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Build a tiny normalized numeric dataset for public API examples.

    @return: Tuple `(X, y)` with values already in a comparable `[0, 1]` range.
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
    """
    @brief Format NumPy arrays compactly for console examples.

    @param values: Array to display.
    @return: Compact string with stable precision.
    """
    return np.array2string(values, precision=4, suppress_small=True)


def main() -> int:
    """
    @brief Run the Phase 7 public API quickstart.

    @return: Process exit code.
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

    print("FRsutils Phase 7 public API quickstart")
    print("dense positive region:   ", _format_array(dense.positive_region))
    print("blockwise positive region:", _format_array(blockwise.positive_region))
    print("blockwise metadata:")
    print(f"  engine={blockwise.engine!r}")
    print(f"  backend={blockwise.backend!r}")
    print(f"  block_size={blockwise.block_size!r}")
    print(f"  used_blockwise={blockwise.used_blockwise!r}")
    print(f"  used_gpu_similarity_blocks={blockwise.used_gpu_similarity_blocks!r}")
    print(
        "  used_gpu_approximation_accumulators="
        f"{blockwise.used_gpu_approximation_accumulators!r}"
    )
    print("dense/blockwise equivalence: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
