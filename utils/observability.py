"""
Structured pipeline logging.

Lightweight observability for indexing, retrieval, and generation — a stepping
stone before OpenTelemetry or full APM in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config.settings import LOGS_DIR

_CONFIGURED = False


def setup_logging(name: str = "rag_lab", level: int = logging.INFO) -> logging.Logger:
    """Configure file + console logging once per process."""
    global _CONFIGURED
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "rag_lab") -> logging.Logger:
    return setup_logging(name)
