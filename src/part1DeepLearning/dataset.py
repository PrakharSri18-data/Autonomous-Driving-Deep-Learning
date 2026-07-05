# src/part1DeepLearning/dataset.py
# --------------------------------------------------
# Shared VehicleDataset used by train.py, evaluation.py, and inference.py.
# Previously this class was copy-pasted into three files, and each one recomputed
# class_to_idx from whatever CSV it happened to be given. If train/val/inference
# CSVs contained different subsets of labels, the class index for a given label
# could silently differ between training and evaluation, corrupting metrics.
# This module fixes that by persisting the class list to disk (see save_classes /
# load_classes) so every stage of the pipeline shares one source of truth.
# --------------------------------------------------

import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES_FILENAME = "classes.json"


def build_class_mapping(labels: list[str]) -> dict[str, int]:
    classes = sorted(set(labels))
    return {cls: idx + 1 for idx, cls in enumerate(classes)}  # 0 is reserved for background


def save_classes(class_to_idx: dict[str, int], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CLASSES_FILENAME)
    with open(path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    return path


def load_classes(classes_path: str) -> dict[str, int]:
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            f"Class mapping not found at {classes_path}. "
            "Run data_ingestion.py first to generate it."
        )
    with open(classes_path) as f:
        return json.load(f)


class VehicleDataset(Dataset):
    """Bounding-box detection dataset for the MIO-TCD-style labels CSV.

    class_to_idx must be passed in (via load_classes) rather than derived from
    this split's CSV, so train/val/inference always agree on label indices.
    """

    def __init__(self, csv_file, image_dir, class_to_idx: dict[str, int], transform=None, max_images: int | None = None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

        all_image_ids = self.data["image_id"].unique()

        available_ids = {
            int(os.path.splitext(fname)[0])
            for fname in os.listdir(self.image_dir)
        }

        self.image_ids = [img_id for img_id in all_image_ids if int(img_id) in available_ids]

        if max_images is not None:
            self.image_ids = self.image_ids[:max_images]

        if not self.image_ids:
            raise ValueError(
                f"No images from {csv_file} were found in {image_dir}. "
                "Check that the Images/ folder is populated (it is gitignored, "
                "see README 'Dataset' section)."
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.data[self.data["image_id"] == image_id]

        image_filename = f"{int(image_id):08d}.jpg"
        img_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(img_path).convert("RGB")

        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values.copy()
        labels = records["label"].map(self.class_to_idx).values.copy()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target
