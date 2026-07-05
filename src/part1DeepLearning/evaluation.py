# src/part1DeepLearning/evaluation.py
# --------------------------------------------------
# Runs the trained model against the validation set and reports both a
# single-threshold precision/recall (score>=0.5, IoU>=0.5 — what the original
# script computed) and per-class Average Precision + mAP@0.5 (the standard
# object-detection benchmark metric). Results are written to eval_metrics.json
# so the README's Results table can be generated from real numbers.
# --------------------------------------------------

import argparse
import json

import config
import torch
from dataset import VehicleDataset, load_classes
from metrics import mean_average_precision, precision_recall_at_threshold
from model import VehicleDetector
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import setup_logging


def evaluate_model(
    val_csv: str,
    image_dir: str,
    classes_path: str,
    model_path: str,
    metrics_path: str,
    eval_cfg: config.EvalConfig,
    logger,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = load_classes(classes_path)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = VehicleDataset(
        val_csv, image_dir, class_to_idx, transform=transform, max_images=eval_cfg.max_eval_images
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = len(class_to_idx) + 1
    model = VehicleDetector(num_classes).get_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    logger.info(f"Evaluating on {len(dataset)} validation images...")

    image_results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            image_results.append(
                {
                    "pred_boxes": outputs[0]["boxes"].cpu().tolist(),
                    "pred_labels": outputs[0]["labels"].cpu().tolist(),
                    "pred_scores": outputs[0]["scores"].cpu().tolist(),
                    "true_boxes": targets[0]["boxes"].tolist(),
                    "true_labels": targets[0]["labels"].tolist(),
                }
            )

    pr = precision_recall_at_threshold(image_results, eval_cfg.iou_threshold, eval_cfg.score_threshold)
    map_result = mean_average_precision(image_results, iou_threshold=eval_cfg.iou_threshold)

    per_class_ap_named = {
        idx_to_class.get(label, str(label)): ap for label, ap in map_result["per_class_AP"].items()
    }

    results = {
        "num_images": len(dataset),
        "iou_threshold": eval_cfg.iou_threshold,
        "score_threshold": eval_cfg.score_threshold,
        "precision": pr["precision"],
        "recall": pr["recall"],
        "true_positives": pr["tp"],
        "false_positives": pr["fp"],
        "false_negatives": pr["fn"],
        "mAP@0.5": map_result["mAP"],
        "per_class_AP": per_class_ap_named,
    }

    logger.info(f"Precision: {pr['precision']:.4f}  Recall: {pr['recall']:.4f}  mAP@0.5: {map_result['mAP']:.4f}")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics written to {metrics_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the vehicle detector on the validation set.")
    parser.add_argument("--val-csv", default=config.VAL_CSV)
    parser.add_argument("--image-dir", default=config.IMAGE_DIR)
    parser.add_argument("--classes-path", default=config.CLASSES_JSON)
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--metrics-path", default=config.METRICS_PATH)
    parser.add_argument("--iou-threshold", type=float, default=config.EvalConfig.iou_threshold)
    parser.add_argument("--score-threshold", type=float, default=config.EvalConfig.score_threshold)
    parser.add_argument("--max-eval-images", type=int, default=config.EvalConfig.max_eval_images)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    eval_cfg = config.EvalConfig(
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_eval_images=args.max_eval_images,
    )

    evaluate_model(
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        classes_path=args.classes_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        eval_cfg=eval_cfg,
        logger=logger,
    )


if __name__ == "__main__":
    main()
