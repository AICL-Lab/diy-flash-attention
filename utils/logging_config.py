"""
Logging configuration for DIY FlashAttention.

This module provides centralized logging setup for consistent log formatting
across all components.

Usage:
    from utils.logging_config import setup_logging, get_logger

    # Setup logging at application start
    setup_logging(level="DEBUG")

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Operation completed successfully")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from utils.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging(
    level: Optional[str] = None,
    format_str: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to LOG_LEVEL from config.
        format_str: Format string for log messages.
                    Defaults to LOG_FORMAT from config.
        date_format: Format string for timestamps.
                     Defaults to LOG_DATE_FORMAT from config.
    """
    if level is None:
        level = LOG_LEVEL
    if format_str is None:
        format_str = LOG_FORMAT
    if date_format is None:
        date_format = LOG_DATE_FORMAT

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_str,
        datefmt=date_format,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Name of the logger, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporarily changing log level.

    Usage:
        with LoggingContext(level="DEBUG"):
            # Debug logs will be shown here
            logger.debug("Detailed information")
    """

    def __init__(self, level: str = "DEBUG"):
        self.level = level
        self.old_level: Optional[int] = None

    def __enter__(self) -> "LoggingContext":
        root_logger = logging.getLogger()
        self.old_level = root_logger.level
        root_logger.setLevel(getattr(logging, self.level.upper(), logging.DEBUG))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.old_level is not None:
            logging.getLogger().setLevel(self.old_level)
