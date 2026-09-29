"""
Utility functions and logging helpers for the Aircraft Electrical System Health Monitor.

Provides centralized logging configuration, random seed management, directory
creation utilities, and formatting helpers.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def setup_logger(
    name: str = "aircraft_health_monitor",
    log_level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    log_file : str or Path, optional
        Path to an optional file destination for log entries.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility across random and numpy.

    Parameters
    ----------
    seed : int
        The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directories(*paths: str | Path) -> None:
    """Ensure that specified directory paths exist, creating them if necessary.

    Parameters
    ----------
    *paths : str or Path
        Variable number of directory paths to create.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
