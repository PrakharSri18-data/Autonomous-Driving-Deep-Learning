import csv
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "part1DeepLearning"))


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Creates a tiny on-disk labels CSV + matching JPEG images for pipeline tests.

    3 images, 2 classes ("car", "bicycle"), one image has no bicycle at all —
    this is what previously could desync class_to_idx between train/val splits.
    """
    image_dir = tmp_path / "Images"
    image_dir.mkdir()

    rows = [
        ("00000000", "car", 10, 10, 50, 50),
        ("00000000", "car", 60, 60, 90, 90),
        ("00000001", "bicycle", 5, 5, 25, 25),
        ("00000002", "car", 0, 0, 20, 20),
    ]

    for image_id in ("00000000", "00000001", "00000002"):
        Image.new("RGB", (100, 100), color=(120, 120, 120)).save(image_dir / f"{image_id}.jpg")

    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return {"csv_path": str(csv_path), "image_dir": str(image_dir), "tmp_path": tmp_path}
