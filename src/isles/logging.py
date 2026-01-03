"""
Logging utilities
"""

import logging
from typing import Generator
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def operation_logger(
    name: str, log_file: str | Path | None = None, level: int = logging.INFO
) -> Generator[logging.Logger, None, None]:
    """
    Context manager to setup a logger with console output and optional file output.

    Parameters
    ----------
    name : str
        The name for the logger.
    log_file : str | Path | None
        Path to the file where logs should be saved. If None, the logger will only
        print to console.
    level : int, optional
        The logging level (default is logging.INFO).

    Yields
    ------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    try:
        yield logger
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
