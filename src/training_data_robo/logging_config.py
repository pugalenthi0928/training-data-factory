# src/training_data_robo/logging_config.py
import logging
from logging import Logger

def get_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    logger.propagate = False
    return logger
