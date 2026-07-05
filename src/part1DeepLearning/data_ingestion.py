# src/part1DeepLearning/data_ingestion.py
# --------------------------------------------------
# Reads the raw labels CSV, splits it into train/val by unique image_id (so no
# image leaks across the split), and persists a single class_to_idx mapping
# (classes.json) derived from the FULL label set. Train/eval/inference all load
# this same file so class indices can never drift between pipeline stages.
# --------------------------------------------------

import argparse
import os

import config
import pandas as pd
from dataset import build_class_mapping, save_classes
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, csv_path: str, output_dir: str, val_size: float = 0.2, seed: int = 42):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.val_size = val_size
        self.seed = seed

    def initiate_data_ingestion(self):
        df = pd.read_csv(self.csv_path, header=None)
        df.columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]

        unique_images = df["image_id"].unique()
        train_ids, val_ids = train_test_split(
            unique_images, test_size=self.val_size, random_state=self.seed
        )

        train_df = df[df["image_id"].isin(train_ids)]
        val_df = df[df["image_id"].isin(val_ids)]

        os.makedirs(self.output_dir, exist_ok=True)

        train_csv_path = os.path.join(self.output_dir, "train.csv")
        val_csv_path = os.path.join(self.output_dir, "val.csv")
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        # Class mapping is built from the FULL label set (not just train_df), so a
        # label that only appears in val/inference data still gets a stable index.
        class_to_idx = build_class_mapping(df["label"].tolist())
        classes_path = save_classes(class_to_idx, self.output_dir)

        print(f"Train/val CSVs written to {self.output_dir}")
        print(f"Class mapping ({len(class_to_idx)} classes) written to {classes_path}")

        return train_csv_path, val_csv_path, classes_path


def main():
    parser = argparse.ArgumentParser(description="Split raw labels CSV into train/val and persist class mapping.")
    parser.add_argument("--csv-path", default=config.RAW_LABELS_CSV)
    parser.add_argument("--output-dir", default=config.DATA_DIR)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ingestion = DataIngestion(args.csv_path, args.output_dir, args.val_size, args.seed)
    ingestion.initiate_data_ingestion()


if __name__ == "__main__":
    main()
