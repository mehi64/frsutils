# SPDX-License-Identifier: BSD-3-Clause
"""Base component helper with logger initialization support.

This module provides shared utility behavior used by FRsutils components.
"""

from FRsutils.utils.logger.logger_util import get_logger

class BaseComponentWithLogger:
    """Base class to provide logger injection and access.
    
    Use this in any class that requires a logger. You can inject a logger explicitly
    or default to the framework's standard TinyLogger.
    """

    def __init__(self, logger=None):
        """Initialize the BaseComponentWithLogger instance."""
        self._logger = logger or get_logger()

    @property
    def logger(self):
        """Property accessor for the logger.
        
        Returns
        -------
        object
            A logger instance (either injected or default).
        """
        return self._logger
    
    @classmethod
    def get_logger(cls):
        """Return the logger associated with this component."""
        from FRsutils.utils.logger.logger_util import get_logger
        return get_logger()

    @staticmethod
    def get_silent_logger():
        """Returns a silent logger that discards all messages.
        
        Useful in unit tests or when logging needs to be suppressed.
        """
        class SilentLogger:
            def debug(self, *a, **k): pass
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
            def critical(self, *a, **k): pass
        return SilentLogger()
