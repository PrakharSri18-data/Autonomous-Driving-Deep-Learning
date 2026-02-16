# src/part1DeepLearning/data_ingestion.py
# --------------------------------------------------
# data_ingestion.py is responsible for reading the dataset from a CSV file & assign column names.
# It splits the dataset into training and validation sets based on unique image IDs to ensure that images from the same ID do not end up in both sets.
# The split datasets are then saved as separate CSV files for use in training and evaluation.
# --------------------------------------------------


# ==================================================
# Importing necessary libraries
# ------------------------------------------------
# OS Library: For handling file paths and directory operations.
# Pandas: For data manipulation and loading the dataset from CSV files.
# Sklearn's train_test_split: For splitting the dataset into training and validation sets.
# ==================================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================
# DataIngestion Class
# ------------------------------------------------
# __init__: Initializes the class with the path to the CSV file and the output directory where the split CSV files will be saved.
# initiate_data_ingestion: Reads the CSV file, splits the dataset into training and validation sets based on unique image IDs, saves the split datasets as separate CSV files, and returns the paths to the created CSV files.
# if __name__ == "__main__": This block allows the script to be run directly, creating an instance of DataIngestion with the path to the CSV file and calling the initiate_data_ingestion method.
# ==================================================
class DataIngestion:
    def __init__(self, csv_path, output_dir="data"):
        self.csv_path = csv_path
        self.output_dir = output_dir

    def initiate_data_ingestion(self):

        # Read CSV without header
        df = pd.read_csv(self.csv_path, header=None)
        df.columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]

        # Split based on unique image IDs
        unique_images = df["image_id"].unique()

        train_ids, val_ids = train_test_split(
            unique_images,
            test_size=0.2,
            random_state=42
        )

        train_df = df[df["image_id"].isin(train_ids)]
        val_df = df[df["image_id"].isin(val_ids)]

        os.makedirs(self.output_dir, exist_ok=True)

        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.output_dir, "val.csv"), index=False)

        print("Train and validation CSV files created successfully.")

        return (
            os.path.join(self.output_dir, "train.csv"),
            os.path.join(self.output_dir, "val.csv"),
        )


if __name__ == "__main__":
    ingestion = DataIngestion(
        csv_path=r"Datasets & Problem Statement\Part 1\labels.csv"
    )
    ingestion.initiate_data_ingestion()
