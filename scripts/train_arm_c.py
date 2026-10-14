"""Training Script for Arm C (ResNet-50 End-to-End Ergonomic Scoring).

Supports CPU/CUDA execution, multi-task MSE + Cross-Entropy loss, and Colab GPU training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.arm_c_resnet_e2e import ResNet50ErgoMultiTask
from data.dataset import ErgonomicsDataset, generate_canonical_reference_dataset


def train_arm_c(
    data_csv: str = "data/canonical_validation.csv",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    device_str: str = "cpu",
    output_weights: str = "models/weights/arm_c_best.pt",
    smoke_test: bool = False,
):
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Training Arm C (ResNet-50 E2E) on device: {device}")

    if not os.path.exists(data_csv):
        print(f"Dataset '{data_csv}' missing. Generating canonical reference dataset...")
        data_csv = generate_canonical_reference_dataset()

    train_ds = ErgonomicsDataset(csv_path=data_csv, split="train", seed=42)
    val_ds = ErgonomicsDataset(csv_path=data_csv, split="val", seed=42)

    if len(train_ds) == 0:
        train_ds = ErgonomicsDataset(csv_path=data_csv, split=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(train_ds)} train samples.")

    # Initialize model
    model = ResNet50ErgoMultiTask(pretrained=True)
    model.to(device)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    max_epochs = 1 if smoke_test else epochs

    print(f"Beginning training for {max_epochs} epoch(s)...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            cfg_vec = batch["config_features"].to(device)

            target_rula_score = batch["rula_score"].to(device)
            target_rula_cls = batch["rula_class"].to(device)
            target_reba_score = batch["reba_score"].to(device)
            target_reba_cls = batch["reba_class"].to(device)

            optimizer.zero_grad()
            preds = model(images, cfg_vec)

            # Multitask loss calculation
            loss_rula_reg = mse_loss(preds["rula_score"], target_rula_score)
            loss_rula_cls = ce_loss(preds["rula_logits"], target_rula_cls)
            loss_reba_reg = mse_loss(preds["reba_score"], target_reba_score)
            loss_reba_cls = ce_loss(preds["reba_logits"], target_reba_cls)

            total_loss = (
                loss_rula_reg + 0.5 * loss_rula_cls +
                0.5 * loss_reba_reg + 0.5 * loss_reba_cls
            )

            total_loss.backward()
            optimizer.step()

            total_epoch_loss += total_loss.item()
            batches += 1

            if smoke_test:
                break

        avg_loss = total_epoch_loss / max(1, batches)
        print(f"Epoch [{epoch}/{max_epochs}] - Multitask Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), output_weights)
    print(f"Saved trained Arm C model weights to '{output_weights}'")


def main():
    parser = argparse.ArgumentParser(description="Train Arm C ResNet-50 E2E model.")
    parser.add_argument("--data-csv", type=str, default="data/canonical_validation.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-weights", type=str, default="models/weights/arm_c_best.pt")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1-step verification.")
    args = parser.parse_args()

    train_arm_c(
        data_csv=args.data_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        output_weights=args.output_weights,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
