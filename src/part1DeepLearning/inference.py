# src/part1DeepLearning/inference.py
# --------------------------------------------------
# Runs the trained model on a single image and draws the top-k predictions.
# Loads idx_to_class from the persisted classes.json (see dataset.py) instead
# of recomputing it from an arbitrary CSV, so labels always match training.
# --------------------------------------------------

import argparse
import os

import config
import torch
from dataset import load_classes
from model import VehicleDetector
from PIL import Image, ImageDraw
from torchvision import transforms


def run_inference(image_path: str, model_path: str, classes_path: str, output_dir: str, score_threshold: float = 0.5, top_k: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_to_idx = load_classes(classes_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class) + 1

    model = VehicleDetector(num_classes).get_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = outputs[0]["boxes"].cpu()
    labels = outputs[0]["labels"].cpu()
    scores = outputs[0]["scores"].cpu()

    scores, indices = scores.sort(descending=True)
    indices = indices[:top_k]
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[:top_k]

    draw = ImageDraw.Draw(image)
    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box.tolist()
            class_name = idx_to_class.get(label.item(), "Unknown")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 15)), f"{class_name}: {score:.2f}", fill="red")
            kept += 1

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    image.save(output_path)

    print(f"Inference completed: {kept} detection(s) above threshold. Output saved at {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--classes-path", default=config.CLASSES_JSON)
    parser.add_argument("--output-dir", default=config.OUTPUTS_DIR)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_inference(
        image_path=args.image_path,
        model_path=args.model_path,
        classes_path=args.classes_path,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
