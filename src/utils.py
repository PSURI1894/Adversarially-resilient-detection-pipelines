import logging
import sys
import os
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_logger(name: str) -> logging.Logger:
    """
    Configures a professional multi-handler logger.
    Safe to call from any module.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Prevent duplicate handlers

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    # Ensure log directory exists
    log_dir = PROJECT_ROOT / 'reports' / 'audit_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File handler
    log_path = log_dir / 'system.log'
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


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
        PROJECT_ROOT / 'tests',
        PROJECT_ROOT / 'models'
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)