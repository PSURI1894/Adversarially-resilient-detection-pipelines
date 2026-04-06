"""
================================================================================
UTILITIES — LOGGING, DIRECTORIES & PERFORMANCE MONITORING
================================================================================
Features:
    - Structured JSON logging (for log aggregation tools)
    - Traditional human-readable console logging
    - Correlation ID propagation for distributed tracing
    - Timing decorators for performance monitoring
    - Directory management
================================================================================
"""

import os
import sys
import json
import time
import uuid
import logging
import functools
from pathlib import Path
from typing import Optional
from contextvars import ContextVar

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Correlation ID for distributed tracing
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


# ==============================================================================
# CORRELATION ID
# ==============================================================================

def get_correlation_id() -> str:
    """Get the current correlation ID, generating one if absent."""
    cid = _correlation_id.get()
    if not cid:
        cid = uuid.uuid4().hex[:12]
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str):
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


# ==============================================================================
# JSON LOG FORMATTER
# ==============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for machine-parseable log output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields if present
        for key in ("stage", "metric", "duration_ms"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry)


# ==============================================================================
# LOGGING
# ==============================================================================

def get_logger(name: str, json_logs: bool = False) -> logging.Logger:
    """
    Configures a professional multi-handler logger.

    Parameters
    ----------
    name : str
        Logger name.
    json_logs : bool
        If True, file handler uses structured JSON format.
        Console always uses human-readable format.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.DEBUG)

    # Human-readable formatter for console
    console_fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    # Ensure log directory exists
    log_dir = PROJECT_ROOT / 'reports' / 'audit_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(console_fmt)
    logger.addHandler(sh)

    # File handler (optionally JSON-formatted)
    log_path = log_dir / 'system.log'
    fh = logging.FileHandler(log_path)
    if json_logs:
        fh.setFormatter(JSONFormatter())
    else:
        fh.setFormatter(console_fmt)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# TIMING DECORATORS
# ==============================================================================

def timed(logger_name: Optional[str] = None):
    """
    Decorator that logs execution time of a function.

    Usage:
        @timed("MyModule")
        def train_model(...):
            ...

    Logs: "train_model completed in 1234.5ms"
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger(logger_name or func.__module__)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.info(f"{func.__name__} completed in {elapsed_ms:.1f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                log.error(f"{func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
                raise
        return wrapper
    return decorator


def timed_stage(stage_name: str, logger_name: str = "Pipeline"):
    """
    Context manager for timing pipeline stages.

    Usage:
        with timed_stage("conformal_calibration"):
            conformal.calibrate(model, X_cal, y_cal)
    """
    class _Timer:
        def __init__(self):
            self.elapsed_ms = 0.0
            self._log = logging.getLogger(logger_name)

        def __enter__(self):
            self._t0 = time.perf_counter()
            self._log.info(f"[STAGE] {stage_name} started")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
            if exc_type is None:
                self._log.info(
                    f"[STAGE] {stage_name} completed in {self.elapsed_ms:.1f}ms"
                )
            else:
                self._log.error(
                    f"[STAGE] {stage_name} failed after {self.elapsed_ms:.1f}ms: {exc_val}"
                )
            return False

    return _Timer()


# ==============================================================================
# DIRECTORY MANAGEMENT
# ==============================================================================

def ensure_directories():
    """
    Builds the folder architecture automatically.
    Safe to call multiple times.
    """
    dirs = [
        PROJECT_ROOT / 'data' / 'raw',
        PROJECT_ROOT / 'data' / 'processed',
        PROJECT_ROOT / 'reports' / 'figures',
        PROJECT_ROOT / 'reports' / 'audit_logs',
        PROJECT_ROOT / 'reports' / 'experiments',
        PROJECT_ROOT / 'reports' / 'dashboards',
        PROJECT_ROOT / 'tests',
        PROJECT_ROOT / 'models',
        PROJECT_ROOT / 'models' / 'registry',
        PROJECT_ROOT / 'configs',
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
