# SPDX-License-Identifier: BSD-3-Clause
"""Numerically stable VQRS ratio calculations.

This module centralizes the leave-one-out VQRS ratio contract shared by the
NumPy dense reference model and NumPy/CuPy-aware blockwise engines.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_vqrs_interim_ratio(
    numerator: Any,
    denominator: Any,
    *,
    xp: Any = np,
):
    """Compute a finite VQRS support ratio in the closed interval [0, 1].

    Parameters
    ----------
    numerator : array-like
        Same-class similarity mass after excluding self-comparisons.
    denominator : array-like
        Total non-self similarity mass for each sample.
    xp : array namespace, default=numpy
        NumPy- or CuPy-compatible array namespace owning the inputs.

    Returns
    -------
    array-like
        Backend-resident VQRS ratios. Rows without non-self similarity mass are
        assigned zero, and round-off excursions are clipped to ``[0, 1]``.

    Notes
    -----
    The zero-denominator convention is conservative: an isolated sample has no
    non-self evidence supporting its class and therefore receives ratio zero.
    Clipping is mathematically safe because the same-class numerator cannot
    exceed the total non-self similarity denominator in exact arithmetic.
    """
    numerator_array = xp.asarray(numerator, dtype=np.float64)
    denominator_array = xp.asarray(denominator, dtype=np.float64)

    errstate = getattr(xp, "errstate", np.errstate)
    with errstate(divide="ignore", invalid="ignore"):
        raw_ratio = xp.where(
            denominator_array > 0.0,
            numerator_array / denominator_array,
            0.0,
        )

    return xp.minimum(xp.maximum(raw_ratio, 0.0), 1.0)
