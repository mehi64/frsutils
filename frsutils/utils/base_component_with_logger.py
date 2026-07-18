# SPDX-License-Identifier: BSD-3-Clause
"""Base component with dependency-injected, silent-by-default logging."""

from __future__ import annotations

from typing import Any

from frsutils.utils.logger.logger_util import get_logger


class BaseComponentWithLogger:
    """Provide logger injection and silent default logging for components."""

    def __init__(self, logger: Any | None = None) -> None:
        """Store an injected logger or create a silent FRsutils logger."""
        self._logger = logger if logger is not None else get_logger()

    @property
    def logger(self) -> Any:
        """Return the logger associated with this component."""
        return self._logger

    @classmethod
    def get_logger(cls) -> Any:
        """Return a silent default FRsutils logger."""
        return get_logger()

    @staticmethod
    def get_silent_logger() -> Any:
        """Return a minimal logger that discards all messages."""

        class SilentLogger:
            """Discard messages for all standard logging levels."""

            def debug(self, *args: Any, **kwargs: Any) -> None:
                """Discard a debug message."""

            def info(self, *args: Any, **kwargs: Any) -> None:
                """Discard an informational message."""

            def warning(self, *args: Any, **kwargs: Any) -> None:
                """Discard a warning message."""

            def error(self, *args: Any, **kwargs: Any) -> None:
                """Discard an error message."""

            def critical(self, *args: Any, **kwargs: Any) -> None:
                """Discard a critical message."""

        return SilentLogger()
