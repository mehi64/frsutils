# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in logging helpers for FRsutils components.

The default logger is silent and never creates files. Console and structured-file
logging are enabled only when explicitly requested by the caller.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import colorlog
except ImportError:  # pragma: no cover - optional presentation dependency
    colorlog = None

__all__ = ["get_logger"]

_SUPPORTED_STRUCTURED_FORMATS = frozenset({"csv", "json"})
_LOGGER_NAMES = {
    "debug": "frsutils.debug",
    "runtime": "frsutils.runtime",
    "test": "frsutils.test",
}


def _detect_env() -> str:
    """Return the current execution profile without enabling any handlers."""
    if "PYTEST_CURRENT_TEST" in os.environ:
        return "test"

    try:
        if sys.gettrace():
            return "debug"
        if any(name in sys.modules for name in ("pydevd", "debugpy", "pdb")):
            return "debug"
    except Exception:
        pass

    return "runtime"


def _resolve_log_path(file_path: str | os.PathLike[str]) -> Path:
    """Resolve a log path relative to the caller's current working directory."""
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _ensure_log_file_parent(file_path: Path) -> Path:
    """Create the parent directory for an explicitly requested log file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _resolve_structured_format(
    file_path: Path,
    log_file_extension: str | None,
) -> str:
    """Return and validate the structured log format."""
    extension = log_file_extension
    if extension is None:
        suffix = file_path.suffix.lower().lstrip(".")
        extension = suffix if suffix in _SUPPORTED_STRUCTURED_FORMATS else "json"

    normalized = str(extension).lower().lstrip(".")
    if normalized not in _SUPPORTED_STRUCTURED_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_STRUCTURED_FORMATS))
        raise ValueError(
            f"Unsupported log file extension {log_file_extension!r}. "
            f"Expected one of: {supported}."
        )
    return normalized


def _build_console_formatter() -> logging.Formatter:
    """Build the console formatter, with optional color support."""
    pattern = (
        "%(asctime)s [%(levelname)s] "
        "%(filename)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    if colorlog is None:
        return logging.Formatter(pattern, datefmt="%Y-%m-%d %H:%M:%S")

    return colorlog.ColoredFormatter(
        f"%(log_color)s{pattern}",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def _ensure_null_handler(logger: logging.Logger) -> None:
    """Attach one ``NullHandler`` so silent logging follows library practice."""
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())


def _ensure_console_handler(logger: logging.Logger) -> None:
    """Attach one FRsutils-owned console handler to ``logger``."""
    for handler in logger.handlers:
        if getattr(handler, "_frsutils_console_handler", False):
            return

    handler = logging.StreamHandler()
    handler.setFormatter(_build_console_formatter())
    handler._frsutils_console_handler = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


class _TinyLogger:
    """Small logger facade supporting silent, console, and structured output."""

    def __init__(
        self,
        *,
        name: str,
        log_to_console: bool = False,
        log_to_file: bool = False,
        file_path: str | os.PathLike[str] | None = None,
        log_file_extension: str | None = None,
        level: int = logging.DEBUG,
        run_id: int | str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        """Initialize an opt-in logger facade.

        Parameters
        ----------
        name : str
            Name of the underlying standard-library logger.
        log_to_console : bool, default=False
            Whether to emit messages to the console.
        log_to_file : bool, default=False
            Whether to write structured log records.
        file_path : path-like or None, default=None
            Structured output path. Relative paths are resolved from the current
            working directory, never from the package installation directory.
        log_file_extension : {"csv", "json"} or None, default=None
            Structured output format. When omitted, the suffix is inferred and
            falls back to JSON.
        level : int, default=logging.DEBUG
            Minimum accepted logging level.
        run_id : int, str, or None, default=None
            Optional run identifier for structured records.
        experiment_name : str or None, default=None
            Optional experiment label for structured records.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        _ensure_null_handler(self.logger)
        if log_to_console:
            _ensure_console_handler(self.logger)

        self.log_to_console = bool(log_to_console)
        self.log_to_file = bool(log_to_file)
        self.run_id = run_id if run_id is not None else int(time.time() * 1000)
        self.experiment_name = experiment_name or "default_experiment"

        self.file_path: Path | None = None
        self.structured_output: str | None = None
        self.structured_path: Path | None = None
        if self.log_to_file:
            requested_path = file_path or Path("logs") / "frsutils_log.json"
            resolved_path = _resolve_log_path(requested_path)
            structured_format = _resolve_structured_format(
                resolved_path,
                log_file_extension,
            )
            self.file_path = resolved_path
            self.structured_output = structured_format
            self.structured_path = resolved_path.with_suffix(f".{structured_format}")

    def _structured_log(
        self,
        level_name: str,
        message: str,
        record: logging.LogRecord,
    ) -> None:
        """Append one structured record when file logging is enabled."""
        if self.structured_path is None or self.structured_output is None:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "level": level_name,
            "filename": record.filename,
            "function": record.funcName,
            "line": record.lineno,
            "message": message,
        }

        path = _ensure_log_file_parent(self.structured_path)
        if self.structured_output == "json":
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            return

        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=log_entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(log_entry)

    def _log(
        self,
        level: int,
        message: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Dispatch a message to explicitly enabled logging destinations."""
        if not self.logger.isEnabledFor(level):
            return

        if self.log_to_console:
            console_kwargs = dict(kwargs)
            console_kwargs.setdefault("stacklevel", 3)
            self.logger.log(level, message, *args, **console_kwargs)

        if not self.log_to_file:
            return

        rendered_message = str(message)
        if args:
            try:
                rendered_message = rendered_message % args
            except (TypeError, ValueError):
                rendered_message = " ".join(
                    [rendered_message, *(str(value) for value in args)]
                )

        frame = sys._getframe(2)
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname=frame.f_code.co_filename,
            lineno=frame.f_lineno,
            msg=rendered_message,
            args=(),
            exc_info=kwargs.get("exc_info"),
            func=frame.f_code.co_name,
        )
        self._structured_log(
            logging.getLevelName(level),
            rendered_message,
            record,
        )

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Log a debug message to explicitly enabled destinations."""
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Log an informational message to explicitly enabled destinations."""
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Log a warning message to explicitly enabled destinations."""
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Log an error message to explicitly enabled destinations."""
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """Log a critical message to explicitly enabled destinations."""
        self._log(logging.CRITICAL, message, *args, **kwargs)

    def set_run(
        self,
        run_id: int | str,
        experiment_name: str | None = None,
    ) -> None:
        """Update the structured-log run identifier and experiment label."""
        self.run_id = run_id
        if experiment_name is not None:
            self.experiment_name = experiment_name

    def attach_exception_hook(self) -> None:
        """Attach a global hook that logs uncaught exceptions when enabled."""

        def handle_exception(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: Any,
        ) -> None:
            """Log an uncaught exception or delegate keyboard interrupts."""
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.critical(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = handle_exception


def get_logger(
    env: str | None = None,
    experiment_name: str | None = None,
    *,
    name: str | None = None,
    log_to_console: bool = False,
    log_to_file: bool = False,
    file_path: str | os.PathLike[str] | None = None,
    log_file_extension: str | None = None,
    level: int = logging.DEBUG,
    run_id: int | str | None = None,
) -> _TinyLogger:
    """Build an FRsutils logger with no output side effects by default.

    Parameters
    ----------
    env : {"runtime", "debug", "test"} or None, default=None
        Execution profile used only to choose a logger name. It does not enable
        console or file output.
    experiment_name : str or None, default=None
        Optional experiment label for structured records.
    name : str or None, default=None
        Explicit standard-library logger name.
    log_to_console : bool, default=False
        Enable console logging explicitly.
    log_to_file : bool, default=False
        Enable structured file logging explicitly.
    file_path : path-like or None, default=None
        Output path used only when ``log_to_file=True``. Relative paths are
        resolved from the current working directory.
    log_file_extension : {"csv", "json"} or None, default=None
        Structured output format.
    level : int, default=logging.DEBUG
        Minimum accepted logging level.
    run_id : int, str, or None, default=None
        Optional run identifier for structured records.

    Returns
    -------
    _TinyLogger
        Logger facade configured for the requested destinations.

    Notes
    -----
    Calling ``get_logger()`` never prints output or creates directories/files.
    """
    resolved_env = env or _detect_env()
    if resolved_env not in _LOGGER_NAMES:
        supported = ", ".join(sorted(_LOGGER_NAMES))
        raise ValueError(
            f"Unknown environment {resolved_env!r}. Expected one of: {supported}."
        )

    return _TinyLogger(
        name=name or _LOGGER_NAMES[resolved_env],
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        file_path=file_path,
        log_file_extension=log_file_extension,
        level=level,
        run_id=run_id,
        experiment_name=experiment_name,
    )
