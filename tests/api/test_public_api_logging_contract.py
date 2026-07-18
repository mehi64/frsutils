# SPDX-License-Identifier: BSD-3-Clause
"""Logging side-effect contracts for the public FRsutils API."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import frsutils.utils.logger.logger_util as logger_util
from frsutils import FuzzyRoughPositiveRegionScorer, compute_approximations
from frsutils.utils.logger.logger_util import get_logger


X_SMALL = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.1],
        [0.8, 0.8],
        [0.9, 0.9],
    ],
    dtype=float,
)
Y_SMALL = np.array([0, 0, 1, 1])


@pytest.mark.parametrize("endpoint", ["function", "scorer"])
def test_public_api_default_logging_is_silent_and_write_free(
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default public calls must work when logging writes are forbidden."""
    monkeypatch.chdir(tmp_path)

    def fail_on_log_write(*args, **kwargs) -> None:
        """Fail if default API execution attempts to create a log destination."""
        raise AssertionError("Default public API execution attempted a logging write.")

    monkeypatch.setattr(logger_util, "_ensure_log_file_parent", fail_on_log_write)

    if endpoint == "function":
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            similarity="linear",
        )
    else:
        scorer = FuzzyRoughPositiveRegionScorer(
            model="itfrs",
            similarity="linear",
        )
        scorer.fit(X_SMALL, Y_SMALL)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("endpoint", ["function", "scorer"])
def test_public_api_honors_explicit_console_logging_opt_in(
    endpoint: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicitly supplied logger remains observable through public calls."""
    logger = get_logger(
        name=f"frsutils.tests.public-api-console-opt-in.{endpoint}",
        log_to_console=True,
        level=logging.DEBUG,
    )

    if endpoint == "function":
        compute_approximations(
            X_SMALL,
            Y_SMALL,
            model="itfrs",
            similarity="linear",
            logger=logger,
        )
    else:
        scorer = FuzzyRoughPositiveRegionScorer(
            model="itfrs",
            similarity="linear",
            extra_params={"logger": logger},
        )
        scorer.fit(X_SMALL, Y_SMALL)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ITFRS initialized." in captured.err
