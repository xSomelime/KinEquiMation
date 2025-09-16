# model_pipeline/training/train_gait.py

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.dat a import DataLoader

from model_pipeline.models.gait_classifier import GaitClassifier
from model_pipeline.datasets.gait_dataset import GaitDataset


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, log_f, args):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["keypoints"].to(device)   # [B,T,K,3]
        y = batch["label"].to(device)       # [B]


        preds = model(x)                   # [B,num_classes]
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)

    log_entry = {
        "epoch": epoch,
        "loss_cls": avg_loss,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "timestamp": time.strftime("%y%m%d_%H%M"),
    }
    log_f.write(json.dumps(log_entry) + "\n")
    log_f.flush()

    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str,
        default="dataset_pipeline/data/dataset_exports/lifter_dataset.json",
        help="Path till lifter_dataset.json (med 3D keypoints + action labels)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="outputs/gait_runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Dataset (nu med riktiga 3D-keypoints + 'action')
    dataset = GaitDataset(
        ann_file=args.data,
        seq_len=args.seq_len,
        num_joints=68,
        in_features=3,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print(f"[INFO] Hittade {len(dataset)} sekvenser")
    print(f"[INFO] Gångart-mapping: {dataset.action_to_idx}")

    # Modell
    model = GaitClassifier(
        num_joints=68,
        in_features=3,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(dataset.action_to_idx),
    )
    model = model.to(device)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Run directory
    run_dir = Path(args.save_dir) / time.strftime("%y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training_log.json"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            avg_loss = train_one_epoch(
                model, dataloader, optimizer, criterion, device, epoch, log_f, args
            )
            print(f"[Epoch {epoch}/{args.epochs}] LossCLS: {avg_loss:.6f}")

            # Spara checkpoint var 5:e epoch + sista
            if epoch % 5 == 0 or epoch == args.epochs:
                ckpt_file = ckpt_dir / f"gait_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "label_mapping": dataset.action_to_idx,
                    },
                    ckpt_file,
                )
                print(f"[OK] Sparade checkpoint: {ckpt_file}")


if __name__ == "__main__":
    main()
