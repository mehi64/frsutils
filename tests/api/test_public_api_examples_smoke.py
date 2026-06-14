# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for canonical FRsutils public API examples."""

from __future__ import annotations

import numpy as np

from examples.public_api_quickstart import (
    main,
    run_approximation_example,
    run_scorer_example,
)


def test_public_api_quickstart_functional_example_returns_scores():
    scores = run_approximation_example()

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_public_api_quickstart_scorer_example_returns_scores():
    scores = run_scorer_example()

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_public_api_quickstart_main_runs(capsys):
    assert main() == 0
    captured = capsys.readouterr()
    assert "FRsutils public API quickstart" in captured.out
