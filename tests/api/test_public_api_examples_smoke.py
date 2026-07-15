# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for canonical frsutils public API examples."""

from __future__ import annotations

import numpy as np

from examples.public_api_quickstart import (
    main,
    run_approximation_example,
    run_model_examples,
    run_scorer_example,
)


def test_public_api_quickstart_functional_example_returns_scores():
    """The functional quickstart returns finite per-sample scores."""
    scores = run_approximation_example()

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_public_api_quickstart_scorer_example_returns_scores():
    """The scorer quickstart returns finite per-sample scores."""
    scores = run_scorer_example()

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_public_api_quickstart_runs_all_public_model_examples():
    """The runnable quickstart covers ITFRS, OWAFRS, and VQRS."""
    scores_by_model = run_model_examples()

    assert set(scores_by_model) == {"itfrs", "owafrs", "vqrs"}
    for scores in scores_by_model.values():
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (6,)
        assert np.all(np.isfinite(scores))


def test_public_api_quickstart_main_runs(capsys):
    """The command-line quickstart completes and prints its heading."""
    assert main() == 0
    captured = capsys.readouterr()
    assert "frsutils public API quickstart" in captured.out
