"""Logging helpers for sqdkit modules."""

import logging

logger = logging.getLogger(__name__)


def get_logger(name: str) -> logging.Logger:
    """Return the named logger without configuring global logging."""
    return logging.getLogger(name)
