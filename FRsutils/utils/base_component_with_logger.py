"""
@file base_component_with_logger.py
@brief Lightweight base class for optional logger injection.

Provides a reusable logger interface via dependency injection or defaults.
All logger-aware classes should inherit from this base to ensure a consistent
logging API without mixin bloat.

##############################################
# ✅ Summary of Design Principles
# - SRP: Handles only logging responsibility
# - Composition: Enables consistent logger usage across components
# - Clean Code: Optional dependency, safe fallback, centralized pattern
##############################################
# usage in different contexts:
# - Inside instance methods: 
#       self.logger.info("Instance log")
# - Inside classmethods:
#       cls.get_logger().debug("Validation started")
# - Inside staticmethod:
#       BaseComponentWithLogger.get_logger().debug("Static log")
"""

from FRsutils.utils.logger.logger_util import get_logger

class BaseComponentWithLogger:
    """
    @brief Base class to provide logger injection and access.

    Use this in any class that requires a logger. You can inject a logger explicitly
    or default to the framework's standard TinyLogger.
    """

    def __init__(self, logger=None):
        self._logger = logger or get_logger()

    @property
    def logger(self):
        """
        @brief Property accessor for the logger.

        @return: A logger instance (either injected or default).
        """
        return self._logger
    
    @classmethod
    def get_logger(cls):
        from FRsutils.utils.logger.logger_util import get_logger
        return get_logger()

    @staticmethod
    def get_silent_logger():
        """
        @brief Returns a silent logger that discards all messages.

        Useful in unit tests or when logging needs to be suppressed.
        """
        class SilentLogger:
            def debug(self, *a, **k): pass
            def info(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
            def critical(self, *a, **k): pass
        return SilentLogger()
