"""Logging utilities."""
import os
import logging
from datetime import datetime
from typing import Optional


class Logger:
    def __init__(self, log_dir: str, name: str = "light_rl"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"))
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(ch)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        # Placeholder for tensorboard integration
        pass


def setup_logger(log_dir: str, name: str = "light_rl") -> Logger:
    return Logger(log_dir, name)
