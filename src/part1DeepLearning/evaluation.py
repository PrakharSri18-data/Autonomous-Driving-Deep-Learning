# src/part1DeepLearning/evaluation.py
# --------------------------------------------------
# evaluation.py is reponsible for getting predicted boxes and trained boxes and calculating precision and recall.
# It also computes the Intersection over Union (IoU) between predicted and true bounding boxes to determine true positives, false positives, and false negatives.
# Finally, it prints the calculated precision and recall values.
# --------------------------------------------------


# ==================================================
# Importing necessary libraries
# ------------------------------------------------
# OS Library: For handling file paths and directory operations.
# Torch: For building and training the neural network model.
# Pandas: For data manipulation and loading the dataset from CSV files.
# PIL: For image processing and loading images.
# Torchvision: For data transformations and loading pretrained models.
# model.py: For importing the VehicleDetector class which defines the model architecture.
# ==================================================
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from model import VehicleDetector


# ==================================================
# VehicleDataset Class (Custom Detection Dataset)
# ------------------------------------------------
# __init__: Initializes the dataset with the path to the CSV file containing annotations, the directory containing images, and any transformations to be applied to the images.
# __len__: Returns the total number of unique images in the dataset.
# __getitem__: Loads an image and its corresponding annotations (bounding boxes and labels) based on the index, applies transformations, and returns the image and target in the format required by the model.
# ==================================================
class VehicleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.classes = sorted(self.data["label"].unique())
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}

        # Filter image IDs to only those that actually exist in folder
        all_image_ids = self.data["image_id"].unique()
        available_ids = set(
            int(file.split(".")[0])
            for file in os.listdir(self.image_dir)
        )

        self.image_ids = [
            img_id for img_id in all_image_ids
            if int(img_id) in available_ids
        ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.data[self.data["image_id"] == image_id]

        # Zero-pad image ID to match dataset format (e.g., 00008602.jpg)
        image_filename = f"{int(image_id):08d}.jpg"
        img_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(img_path).convert("RGB")

        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        labels = records["label"].map(self.class_to_idx).values

        # Copy arrays to avoid non-writable NumPy warning
        boxes = torch.tensor(boxes.copy(), dtype=torch.float32)
        labels = torch.tensor(labels.copy(), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


# ==================================================
# IoU Calculation Function
# ------------------------------------------------
# calculate_iou: This function takes two bounding boxes as input and calculates the Intersection over Union (IoU) between them.
# It computes the area of intersection and the area of union of the two boxes and returns the IoU value.
# ==================================================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ==================================================
# evaluate_model Function
# ------------------------------------------------
# evaluate_model: This function takes the path to the validation CSV file, the directory containing images, the path to the trained model, and an optional IoU threshold for determining true positives.
# It sets up the device, data transformations, dataset, and data loader.
# It loads the trained model, runs inference on the validation dataset, and compares the predicted bounding boxes with the true bounding boxes using the IoU metric to calculate true positives, false positives, and false negatives.
# Finally, it calculates and prints the precision and recall values.
# if __name__ == "__main__": This block allows the script to be run directly, calling the evaluate_model function with the appropriate paths and parameters.
# ==================================================
def evaluate_model(val_csv, image_dir, model_path, iou_threshold=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = VehicleDataset(val_csv, image_dir, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    num_classes = len(dataset.classes) + 1
    detector = VehicleDetector(num_classes)
    model = detector.get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("Evaluating model...")

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)

            outputs = model(images)

            pred_boxes = outputs[0]["boxes"].cpu()
            pred_labels = outputs[0]["labels"].cpu()
            pred_scores = outputs[0]["scores"].cpu()

            true_boxes = targets[0]["boxes"]
            true_labels = targets[0]["labels"]

            matched = []

            for i, true_box in enumerate(true_boxes):
                found_match = False
                for j, pred_box in enumerate(pred_boxes):

                    # Ignore low confidence predictions
                    if pred_scores[j] < 0.5:
                        continue

                    iou = calculate_iou(true_box, pred_box)

                    if (
                        iou >= iou_threshold
                        and true_labels[i] == pred_labels[j]
                        and j not in matched
                    ):
                        total_tp += 1
                        matched.append(j)
                        found_match = True
                        break

                if not found_match:
                    total_fn += 1

            total_fp += len(pred_boxes) - len(matched)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    evaluate_model(
        val_csv="data/val.csv",
        image_dir=r"Datasets & Problem Statement\Part 1\Images",
        model_path="models/vehicle_detector.pth"
    )
