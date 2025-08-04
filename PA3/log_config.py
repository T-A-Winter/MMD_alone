import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from pathlib import Path

# Setup
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "kmeans.log")
MAX_BYTES = 1_000_000  # 1 MB
BACKUP_COUNT = 9       # total of 10 files: 1 current + 9 backups

# ANSI color codes
LOG_COLORS = {
    logging.DEBUG: "\033[94m",     # Blue
    logging.INFO: "\033[92m",      # Green
    logging.WARNING: "\033[93m",   # Yellow
    logging.ERROR: "\033[91m",     # Red
    logging.CRITICAL: "\033[91m",  # Red
}
RESET_COLOR = "\033[0m"

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

_shared_handler = None  # shared rotating file handler

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{RESET_COLOR}"

class ColorizingStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.setFormatter(ColoredFormatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
        ))

def get_logger(name: str,
               log_level: int = logging.DEBUG,
               path_to_log: Optional[Path] = None,
               enabled: bool = True) -> logging.Logger:
    """
    Returns a logger:
    - If enabled is False: returns a disabled logger (NullHandler, no logging).
    - If path_to_log is given: logs to that file (no rotation).
    - Else: logs to shared rotating file (logs/kmeans.log).
    Console output is always colorized.
    """
    global _shared_handler

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not enabled:
        logger.disabled = True
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    if not logger.handlers:
        # File handler
        if path_to_log is not None:
            path_to_log.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(path_to_log))
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
            ))
        else:
            if _shared_handler is None:
                _shared_handler = RotatingFileHandler(
                    LOG_FILE,
                    maxBytes=MAX_BYTES,
                    backupCount=BACKUP_COUNT
                )
                _shared_handler.setFormatter(logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s"
                ))
            file_handler = _shared_handler

        console_handler = ColorizingStreamHandler()

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
