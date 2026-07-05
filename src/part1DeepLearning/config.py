# src/part1DeepLearning/config.py
# --------------------------------------------------
# Centralizes paths and hyperparameters that used to be hardcoded (and Windows-only,
# via raw backslash strings) inside each script's __main__ block. Every value here
# can be overridden from the command line by the scripts that import it.
# --------------------------------------------------

import os
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_LABELS_CSV = os.path.join(REPO_ROOT, "Datasets & Problem Statement", "Part 1", "labels.csv")
IMAGE_DIR = os.path.join(REPO_ROOT, "Datasets & Problem Statement", "Part 1", "Images")
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
CLASSES_JSON = os.path.join(DATA_DIR, "classes.json")
MODEL_PATH = os.path.join(MODELS_DIR, "vehicle_detector.pth")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
METRICS_PATH = os.path.join(MODELS_DIR, "eval_metrics.json")


@dataclass
class TrainConfig:
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    max_train_images: int | None = 500  # keeps CPU training time reasonable; None = use all
    seed: int = 42


@dataclass
class EvalConfig:
    iou_threshold: float = 0.5
    score_threshold: float = 0.5
    max_eval_images: int | None = None
