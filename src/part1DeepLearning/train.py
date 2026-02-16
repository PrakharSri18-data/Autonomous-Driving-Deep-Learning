# src/part1DeepLearning/train.py
# --------------------------------------------------
# train.py is responsible for training the neural network model for vehicle detection.
# It loads the training dataset & images, bounding box annotations, and labels.
# It converts the annotations into PyTorch tensors and feeds them into the model for training.
# It loads model from model.py & sends model to GPU(if available) for faster training.
# Then runs the training loop for a specified number of epochs, calculating the loss and updating the model weights using an optimizer.
# Finally, it saves the trained model to disk for later use in evaluation and inference.
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
import os
import torch
import pandas as pd
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

        self.image_ids = self.data["image_id"].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.data[self.data["image_id"] == image_id]

        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        labels = records["label"].map(self.class_to_idx).values

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target


# ==================================================
# train_model Function(Training Function)
# -------------------------------------------------
# train_model: This function takes the path to the training CSV file, the directory containing images and the number of epochs for training. 
# It sets up the device, data transformations, dataset, and data loader. It initializes the model using the VehicleDetector class, defines the optimizer, and runs the training loop for the specified number of epochs. After training, it saves the trained model to disk.
# if __name__ == "__main__": This block allows the script to be run directly, calling the train_model function with the appropriate paths and parameters.
# ==================================================
def train_model(train_csv, image_dir, num_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = VehicleDataset(train_csv, image_dir, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    num_classes = len(dataset.classes) + 1  # + background

    detector = VehicleDetector(num_classes)
    model = detector.get_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/vehicle_detector.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    train_model(
        train_csv="data/train.csv",
        image_dir=r"Datasets & Problem Statement\Part 1\Images",
        num_epochs=5
    )
