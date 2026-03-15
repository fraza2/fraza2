import logging
import os
from bot.config import Config


def get_logger(name: str) -> logging.Logger:
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(Config.LOG_FILE)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
