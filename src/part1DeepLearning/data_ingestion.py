# src/part1DeepLearning/data_ingestion.py
# --------------------------------------------------
# Data Ingestion model loads the dataset & unzips it.
# Organizes it into training and validation directories. 
# It also handles any necessary preprocessing steps to ensure the data is ready for model training.
# --------------------------------------------------


# ==================================================
# Importing necessary libraries
# ------------------------------------------------
# OS Library: For handling file paths and directory operations.
# Logging: For logging the data ingestion process.
# Pathlib: For handling file paths in a more elegant way.
# Sklearn.model_selection: For splitting the dataset into training and validation sets.
# Shutil: For copying files from one directory to another.
# ==================================================
import os
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


# ==================================================
# Setting up logging configuration
# ------------------------------------------------
# Logging is configured to display the timestamp, log level, and message for each log entry.
# This helps in tracking the progress and debugging the data ingestion process.
# ==================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==================================================
# DataIngestion Class
# ------------------------------------------------
# __init__: Initializes the class with parameters for zip file path, directories for extraction and processed data, train-validation split ratio, and random state for reproducibility.
# create_directories: Creates necessary directories for raw and processed data.
# get_image_paths: Retrieves all image file paths from the extracted dataset.
# split_dataset: Splits the dataset into training and validation sets based on the specified split ratio.
# copy_files: Copies files from the source paths to the destination directory.
# initiate_data_ingestion: Orchestrates the entire data ingestion process by calling the above methods in sequence and returns the paths to the training and validation directories.
# if __name__ == "__main__": This block allows the script to be run directly, initiating the data ingestion process with a specified zip file path.
# ==================================================
class DataIngestion:
    def __init__(
        self,
        dataset_dir: str,
        processed_dir: str = "data/processed",
        train_split: float = 0.8,
        random_state: int = 42,
    ):
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        self.train_split = train_split
        self.random_state = random_state

    def create_directories(self):
        os.makedirs(os.path.join(self.processed_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_dir, "val"), exist_ok=True)
        logging.info("Directories created successfully.")

    def get_image_paths(self):
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []

        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            raise ValueError("No images found in dataset directory.")

        logging.info(f"Total images found: {len(image_paths)}")
        return image_paths

    def split_dataset(self, image_paths):
        train_paths, val_paths = train_test_split(
            image_paths,
            train_size=self.train_split,
            random_state=self.random_state,
            shuffle=True,
        )

        logging.info(f"Training samples: {len(train_paths)}")
        logging.info(f"Validation samples: {len(val_paths)}")

        return train_paths, val_paths

    def copy_files(self, file_paths, destination):
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(destination, filename))

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")

        self.create_directories()

        image_paths = self.get_image_paths()
        train_paths, val_paths = self.split_dataset(image_paths)

        self.copy_files(train_paths, os.path.join(self.processed_dir, "train"))
        self.copy_files(val_paths, os.path.join(self.processed_dir, "val"))

        logging.info("Data ingestion completed successfully.")

        return {
            "train_dir": os.path.join(self.processed_dir, "train"),
            "val_dir": os.path.join(self.processed_dir, "val"),
        }


if __name__ == "__main__":
    ingestion = DataIngestion(
        dataset_dir=r"Datasets & Problem Statement\Part 1\Images"
    )
    ingestion.initiate_data_ingestion()
