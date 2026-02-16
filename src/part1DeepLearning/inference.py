# src/part1DeepLearning/inference.py
# --------------------------------------------------
# inference.py is responsible for running inference on new images using the trained model.
# It loads the trained model, processes the input image, and outputs the predicted bounding boxes and class labels.
# It also includes a function to visualize the predictions by drawing bounding boxes and labels on the input image and saving the output image.
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
import os
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms
from model import VehicleDetector


# ==================================================
# get_class_mapping Function
# ------------------------------------------------
# get_class_mapping: This function takes the path to a CSV file containing the dataset annotations and creates a mapping of class labels to indices and vice versa.
# It reads the CSV file, extracts the unique class labels, and creates two dictionaries: one for mapping class names to indices (class_to_idx) and another for mapping indices back to class names (idx_to_class). 
# The function returns the idx_to_class dictionary for use in inference.
# ==================================================
def get_class_mapping(csv_path):
    # CSV has no header → define column names manually
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]

    classes = sorted(df["label"].unique())
    class_to_idx = {cls: idx + 1 for idx, cls in enumerate(classes)}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return idx_to_class


# ==================================================
# run_inference Function
# ------------------------------------------------
# run_inference: This function takes the path to an input image, the path to the trained model, the path to the CSV file for class mapping, and an optional score threshold for filtering predictions.
# It sets up the device, loads the class mapping, loads the trained model, and processes the input image. 
# It runs inference using the model and filters the predictions based on the score threshold.
# It then draws bounding boxes and class labels on the input image for the predictions that meet the threshold and saves the output image with the predictions visualized.
# if __name__ == "__main__": This block allows the script to be run directly, calling the run_inference function with the appropriate paths and parameters.
# ==================================================
def run_inference(image_path, model_path, csv_path, score_threshold=0.5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get class names
    idx_to_class = get_class_mapping(csv_path)
    num_classes = len(idx_to_class) + 1

    # Load model
    detector = VehicleDetector(num_classes)
    model = detector.get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = outputs[0]["boxes"].cpu()
    labels = outputs[0]["labels"].cpu()
    scores = outputs[0]["scores"].cpu()

    top_k = 5
    scores, indices = scores.sort(descending=True)
    indices = indices[:top_k]

    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[:top_k]

    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box.tolist()
            class_name = idx_to_class.get(label.item(), "Unknown")

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw label text
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, max(0, y1 - 15)), text, fill="red")

    os.makedirs("outputs", exist_ok=True)

    # Save output with same name as input
    output_filename = os.path.basename(image_path)
    output_path = os.path.join("outputs", output_filename)

    image.save(output_path)

    print(f"Inference completed. Output saved at {output_path}")


if __name__ == "__main__":
    run_inference(
        image_path=r"Datasets & Problem Statement\Part 1\Images\00000001.jpg",  # use zero-padded name
        model_path="models/vehicle_detector.pth",
        csv_path=r"Datasets & Problem Statement\Part 1\labels.csv",  # use original CSV
        score_threshold=0.1
    )


