"""Loguru-based structured logging setup."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


_configured = False


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>")
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    logger.add(log_path / "privateye_{time:YYYY-MM-DD}.log",
               level="DEBUG", rotation="00:00", retention="30 days",
               serialize=True, enqueue=True)
    _configured = True


def get_logger():
    return logger
