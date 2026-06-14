# SPDX-License-Identifier: BSD-3-Clause
"""Project logger factory and logging-directory setup helpers.

This module provides shared utility behavior used by FRsutils components.
"""

# only get_logger is available for import, not _TinyLogger.
__all__ = ["get_logger"]

import inspect
import logging
import os
import sys
from pathlib import Path
import json
import csv
from datetime import datetime
import time
from contextlib import contextmanager
from functools import wraps


try:
    import colorlog
except ImportError:  # pragma: no cover - exercised only when optional colorlog is absent
    colorlog = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"


def _ensure_project_logs_dir():
    """Ensures that the project-level logs directory exists.
    
    Returns
    -------
    object
        Path to the project-level logs directory.
    """

    DEFAULT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_LOGS_DIR


def _resolve_project_log_path(file_path):
    """Resolves relative log file paths from project root and ensures parent directory exists.
    
    Parameters
    ----------
    file_path : object
        Relative or absolute log file path.
    
    Returns
    -------
    object
        Absolute Path object for the log file.
    """

    _ensure_project_logs_dir()
    path = Path(file_path)
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _ensure_log_file_parent(file_path):
    """Ensures the parent directory for a log file exists.
    
    Parameters
    ----------
    file_path : object
        Relative or absolute log file path.
    
    Returns
    -------
    object
        Absolute Path object for the log file.
    """

    return _resolve_project_log_path(file_path)


class _TinyLogger:
    """tiny logger"""

    def __init__(
        self,
        name=__name__,
        log_to_console=True,
        log_to_file=False,
        file_path="logs/ml_logs.log",
        log_file_extension=None,  # 'csv' or 'json'
        level=logging.DEBUG, 
        run_id=None,
        experiment_name=None
    ):
        """Initializes the logger.
        
        Parameters
        ----------
        name : object
            Logger name. (unique name gives singleton logger)
        log_to_console : object
            Enable terminal output. possible to use with file output.
        log_to_file : object
            Enable logging to file. possible to use with console output.
        file_path : object
            Path to the human-readable log file.
        structured_output Optional structured format : object
            'csv' or 'json'.
        level : object
            Logging level. Default is logging.DEBUG which prints all. With selecting one of the levels, the logger will only print the messages of the specified level and above. The order of the levels is: DEBUG < INFO < WARNING < ERROR < CRITICAL. The lower the level, the more verbose the output. e.g. by selecting DEBUG, the logger will print all the messages.
        run_id : object
            Unique identifier for this experiment run. Uses current timestamp to generate unique run ids.
        experiment_name : object
            Optional experiment group label.
        """
        
        # find out who is calling this __init__function
        # Look 2 frames up the call stack
        stack = inspect.stack()
        caller = stack[1].function

        if caller != 'get_logger':
            raise RuntimeError("Direct instantiation is not allowed. Use get_logger() instead.")

        # Always create the project-level logs directory before configuring logging.
        _ensure_project_logs_dir()
        self.file_path = _resolve_project_log_path(file_path)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        self.log_to_console = log_to_console
        self.log_to_file = log_to_file

        self.structured_output = log_file_extension

        # *1000 ensures we get timstamp in milliseconds which is important for the fast computers to
        # generate unique values
        self.run_id = run_id or int(time.time() * 1000)
        self.experiment_name = experiment_name or "default_experiment"

        # removes the file extension and adds the structured output extension to the file path
        # e.g. if:
        # structured_output='csv',  # or "json" or None
        # file_path="E:/tst/some_folder/log_output.json",
        # then self.structured_path="E:/tst/some_folder/log_output.csv"
        if log_file_extension:
            base, _ = os.path.splitext(str(self.file_path))
            self.structured_path = _ensure_log_file_parent(base + f".{log_file_extension}")
        else:
            self.structured_path = None

      
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Prefer colorized console output when colorlog is installed, but keep
            # FRsutils importable in minimal environments used by downstream packages.
            if colorlog is not None:
                console_formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors={
                        'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'
                    }
                )
            else:
                console_formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

            if log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)

            # There is no need for adding a handler for file. because the logger.log() functions 
            # which is called internally, gets the handlers attached to the logger. Then the log file will
            # contain duplicated records. Instead, we have _structured_log which logs into csv file

    def _structured_log(self, level_name, message, record):
        """Writes a structured log entry to a separate structured file.
        
        Parameters
        ----------
        level_name : object
            Log level as string.
        message : object
            Log message.
        record : object
            A LogRecord instance containing caller context.
        """

        # This checks if the log level is less than the logger's level
        # If this is the case, then we do not want to log the message.
        # This is done to avoid logging messages that are less important than the logger's level.
        message_level_value = logging.getLevelName(level_name)
        if (message_level_value <= self.logger.level):
            return


        log_entry = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "level": '[' + level_name + ']',
            "filename": record.filename,
            "function": record.funcName,
            "line": record.lineno,
            "message": message
        }

        if self.structured_output == "json":
            _ensure_log_file_parent(self.structured_path)
            with open(self.structured_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        elif self.structured_output == "csv":
            _ensure_log_file_parent(self.structured_path)
            write_header = not os.path.exists(self.structured_path) or os.stat(self.structured_path).st_size == 0
            with open(self.structured_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(log_entry)

    def _log(self, level, message):
        """Core log dispatcher. Writes to standard logger and structured file (if enabled).
        
        Parameters
        ----------
        level : object
            Logging level (e.g., logging.INFO).
        message : object
            Message to log.
        """
       
        if self.log_to_console:
            # stacklevel-3 is necessary since this is called from a wrapper function
            # otherwiase, it would log to the wrapper function's logger and all
            # output information of the caller will be logger.util.py which is nonsense.
            # stacklevel=3 skips one level up from the current frame and returns the caller's frame.
            # this is necessary since the logger is called from a wrapper function (e.g., log_time)
            self.logger.log(level, message, stacklevel=3)

        # 
        if self.log_to_file:
            frame = sys._getframe(2)
            fake_record = logging.LogRecord(
                name=self.logger.name, level=level,
                pathname=frame.f_code.co_filename,
                lineno=frame.f_lineno,
                msg=message, args=(), exc_info=None,
                func=frame.f_code.co_name
            )
            self._structured_log(logging.getLevelName(level), message, fake_record)

    # Basic log level methods
    def debug(self, msg):
        """Log a debug message."""
        self._log(logging.DEBUG, msg)

    def info(self, msg):
        """Log an informational message."""
        self._log(logging.INFO, msg)

    def warning(self, msg):
        """Log a warning message."""
        self._log(logging.WARNING, msg)

    def error(self, msg):
        """Log an error message."""
        self._log(logging.ERROR, msg)

    def critical(self, msg):
        """Log a critical message."""
        self._log(logging.CRITICAL, msg)

    def set_run(self, run_id, experiment_name=None):
        """Update experiment run ID and optionally name."""
        self.run_id = run_id
        if experiment_name:
            self.experiment_name = experiment_name


    def attach_exception_hook(self):
        """Attach global exception hook to capture uncaught errors."""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception


    # def log_config(self, config):
    #     """Log configuration parameters (e.g., model/training settings)."""
    #     self.info("Experiment config: " + json.dumps(config, indent=2))

    # def log_metric(self, name, value, step=None):
    #     """Log a scalar metric.
    #     Parameters: name is the metric name.
    #     Parameters: value is the metric value.
    #     Parameters: step is an optional step or epoch.
    #     """
    #     msg = f"Metric [{name}] = {value}" + (f" @ step {step}" if step else "")
    #     self.info(msg)

    #     if self.structured_output == "json":
    #         metric_record = {
    #             "timestamp": datetime.utcnow().isoformat(),
    #             "experiment": self.experiment_name,
    #             "run_id": self.run_id,
    #             "type": "metric",
    #             "metric_name": name,
    #             "value": value,
    #             "step": step
    #         }
    #         with open(self.structured_path, "a") as f:
    #             f.write(json.dumps(metric_record) + "\n")

    # def log_git_info(self):
    #     """Log the current Git commit hash and dirty status."""
    #     try:
    #         import subprocess
    #         commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    #         dirty = subprocess.call(['git', 'diff', '--quiet']) != 0
    #         self.info(f"Git commit: {commit} | Dirty: {dirty}")
    #     except Exception as e:
    #         self.warning(f"Could not retrieve Git info: {e}")

    # def log_system_info(self):
    #     """Log system specs such as platform, CPU, memory, CUDA availability."""
    #     try:
    #         import platform, psutil
    #         info = {
    #             "platform": platform.platform(),
    #             "cpu": platform.processor(),
    #             "memory_gb": round(psutil.virtual_memory().total / 1e9, 2),
    #         }
    #         try:
    #             import torch
    #             info["cuda_available"] = torch.cuda.is_available()
    #         except ImportError:
    #             info["cuda_available"] = False
    #         self.info("System Info: " + json.dumps(info))
    #     except Exception as e:
    #         self.warning(f"Could not get system info: {e}")



    # @contextmanager
    # def log_time(self, step_name):
    #     """Context manager for timing a code block.
    #     Parameters: step_name describes the timed step.
    #     """
    #     start = time.time()
    #     self.info(f"Started: {step_name}")
    #     yield
    #     end = time.time()
    #     self.info(f"Finished: {step_name} in {end - start:.2f}s")

    # def log_step(self, step_name):
    #     """Decorator to log the entry and exit of a function, with duration."""
    #     def decorator(func):
    #         @wraps(func)
    #         def wrapper(*args, **kwargs):
    #             self.info(f"[STEP] Starting: {step_name}")
    #             start = time.time()
    #             try:
    #                 return func(*args, **kwargs)
    #             finally:
    #                 duration = time.time() - start
    #                 self.info(f"[STEP] Finished: {step_name} in {duration:.2f}s")
    #         return wrapper
    #     return decorator


def _detect_env():
    """Auto-detect if we're in testing, debugging, or runtime mode."""

    if "PYTEST_CURRENT_TEST" in os.environ:
        return "test"
    try:
        # sys.gettrace is None when no debugger is attached
        if sys.gettrace():
            return "debug"

        # Additional check for common debuggers
        for name in ("pydevd", "debugpy", "pdb"):
            if name in sys.modules:
                return "debug"

    except Exception:
        pass
    
    return "runtime"

def get_logger(env=None, experiment_name=None):
    """Return the logger associated with this component."""
    env = env or _detect_env()
    # print("logger type: ", env)
    if env == "debug":
        return _TinyLogger(name="logs/debug_logger", 
                           log_to_console=True, 
                           log_to_file=False, 
                           level=logging.DEBUG)
    elif env == "test":
        return _TinyLogger(name="test_logger", 
                           log_to_console=False, 
                           log_to_file=True,
                           file_path='logs/test_logs.csv',
                           log_file_extension='csv',
                           experiment_name=experiment_name)
    elif env == "runtime":
        return _TinyLogger(name="runtime_logger", 
                           log_to_console=True, 
                           log_to_file=True, 
                           file_path="logs/run_log.json", 
                           log_file_extension="json")
        
    
    raise ValueError(f"Unknown environment: {env}")
