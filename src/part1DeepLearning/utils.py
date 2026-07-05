# src/part1DeepLearning/utils.py
# --------------------------------------------------
# Small shared helpers: reproducibility (seeding) and consistent logging.
# Every script previously used bare print() statements with no way to persist
# run output; this routes everything through the standard logging module so
# runs can be redirected to a log file (see train.py --log-file).
# --------------------------------------------------

import logging
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("vehicle_detector")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
