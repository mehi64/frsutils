# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for silent-by-default and opt-in FRsutils logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from frsutils.utils.base_component_with_logger import BaseComponentWithLogger
from frsutils.utils.logger.logger_util import get_logger


def test_default_logger_is_silent_and_creates_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default logging must not print or create filesystem artifacts."""
    monkeypatch.chdir(tmp_path)

    logger = get_logger()
    logger.debug("hidden debug")
    logger.info("hidden info")
    logger.warning("hidden warning")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logger.logger.handlers
    )


@pytest.mark.parametrize("env", ["runtime", "debug", "test"])
def test_named_environment_profiles_remain_silent(
    env: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Environment selection alone must not opt into any output destination."""
    monkeypatch.chdir(tmp_path)

    logger = get_logger(env=env, experiment_name="contract-test")
    logger.info("must remain silent")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []


def test_base_component_uses_silent_default_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Components must inherit the no-output default contract."""
    monkeypatch.chdir(tmp_path)

    component = BaseComponentWithLogger()
    component.logger.debug("component initialization")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert list(tmp_path.iterdir()) == []


def test_console_logging_requires_explicit_opt_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Console output must appear only after explicit enablement."""
    logger = get_logger(
        env="runtime",
        name="frsutils.tests.console-opt-in",
        log_to_console=True,
        level=logging.INFO,
    )

    logger.info("visible message")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "visible message" in captured.err


def test_file_logging_requires_explicit_opt_in_and_uses_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit file logging must resolve relative paths from the caller's CWD."""
    monkeypatch.chdir(tmp_path)
    relative_path = Path("artifacts") / "research-run.json"

    logger = get_logger(
        env="runtime",
        name="frsutils.tests.file-opt-in",
        log_to_file=True,
        file_path=relative_path,
        experiment_name="experiment-a",
        run_id="run-7",
    )

    assert list(tmp_path.iterdir()) == []
    logger.info("recorded message")

    output_path = tmp_path / relative_path
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["message"] == "recorded message"
    assert payload["experiment"] == "experiment-a"
    assert payload["run_id"] == "run-7"


def test_file_path_is_ignored_when_file_logging_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Providing a path alone must not create or validate file destinations."""
    monkeypatch.chdir(tmp_path)

    logger = get_logger(
        file_path=Path("forbidden") / "unused.invalid",
        log_file_extension="invalid",
    )
    logger.error("still silent")

    assert list(tmp_path.iterdir()) == []


def test_invalid_structured_format_is_rejected_only_when_enabled(
    tmp_path: Path,
) -> None:
    """Unsupported structured formats must fail after explicit file opt-in."""
    with pytest.raises(ValueError, match="Unsupported log file extension"):
        get_logger(
            log_to_file=True,
            file_path=tmp_path / "run.log",
            log_file_extension="yaml",
        )
