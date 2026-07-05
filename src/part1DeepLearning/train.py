# src/part1DeepLearning/train.py
# --------------------------------------------------
# Trains the Faster R-CNN / MobileNetV3 vehicle detector. Loads the shared
# VehicleDataset with the persisted class mapping (see dataset.py), runs the
# training loop, tracks per-epoch loss + a held-out validation loss, saves the
# best checkpoint (by val loss, not just "whatever the last epoch produced"),
# and writes training_history.json so loss curves can be plotted later.
# --------------------------------------------------

import argparse
import json
import os

import config
import torch
from dataset import VehicleDataset, load_classes
from model import VehicleDetector
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import seed_everything, setup_logging


def _compute_val_loss(model, data_loader, device) -> float:
    """Faster R-CNN only returns a loss dict in train() mode, so we keep the model
    in train mode for this forward pass but disable gradient tracking — this is
    the standard workaround for torchvision detection models, not a bug."""
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


def train_model(
    train_csv: str,
    val_csv: str,
    image_dir: str,
    classes_path: str,
    train_cfg: config.TrainConfig,
    model_path: str,
    history_path: str,
    logger,
):
    seed_everything(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    class_to_idx = load_classes(classes_path)
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = VehicleDataset(
        train_csv, image_dir, class_to_idx, transform=transform, max_images=train_cfg.max_train_images
    )
    val_dataset = VehicleDataset(val_csv, image_dir, class_to_idx, transform=transform)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = len(class_to_idx) + 1  # + background
    model = VehicleDetector(num_classes).get_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=train_cfg.learning_rate, momentum=train_cfg.momentum, weight_decay=train_cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    logger.info(f"Starting training for {train_cfg.num_epochs} epoch(s) on {len(train_dataset)} images...")

    for epoch in range(train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = _compute_val_loss(model, val_loader, device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        logger.info(f"Epoch [{epoch + 1}/{train_cfg.num_epochs}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best val_loss {val_loss:.4f} -> saved checkpoint to {model_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history written to {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the vehicle detector.")
    parser.add_argument("--train-csv", default=config.TRAIN_CSV)
    parser.add_argument("--val-csv", default=config.VAL_CSV)
    parser.add_argument("--image-dir", default=config.IMAGE_DIR)
    parser.add_argument("--classes-path", default=config.CLASSES_JSON)
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--history-path", default=config.HISTORY_PATH)
    parser.add_argument("--epochs", type=int, default=config.TrainConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=config.TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=config.TrainConfig.learning_rate)
    parser.add_argument("--max-train-images", type=int, default=config.TrainConfig.max_train_images)
    parser.add_argument("--seed", type=int, default=config.TrainConfig.seed)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log_file)

    train_cfg = config.TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_images=args.max_train_images,
        seed=args.seed,
    )

    train_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_dir=args.image_dir,
        classes_path=args.classes_path,
        train_cfg=train_cfg,
        model_path=args.model_path,
        history_path=args.history_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
