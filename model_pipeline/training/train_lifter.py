# model_pipeline/training/train_lifter.py

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_pipeline.models.lifter_3d import TemporalLifter
from model_pipeline.datasets.lifter_dataset import LifterDataset


class MPJPELoss(nn.Module):
    """Mean Per Joint Position Error (L2)."""
    def forward(self, pred, target, mask=None):
        # pred, target: [B,T,K,3]
        error = torch.norm(pred - target, dim=3)  # [B,T,K]
        if mask is not None:
            error = error * mask
            return (error.sum() / mask.sum()).mean()
        else:
            return error.mean()


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, log_f, args):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # Flytta till GPU om möjligt
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # inputs = 2D keypoints (u,v,vis), targets = 3D keypoints
        x = batch["keypoints_2d"]  # [B,T,K,3]
        y = batch["keypoints_3d"]  # [B,T,K,3]

        B, T, K, C = x.shape
        x = x.reshape(B, T, -1)  # [B,T,K*3]

        preds = model(x)         # [B,T,K,3]

        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)

    # Logga som JSONL
    log_entry = {
        "epoch": epoch,
        "loss_3d": avg_loss,
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
        help="Path till dataset med 2D & 3D keypoints"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=27)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="outputs/lifter_runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Dataset
    dataset = LifterDataset(
        ann_file=args.data,
        seq_len=args.seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True)

    # Modell (nu in_features=3 eftersom vi har u,v,vis)
    model = TemporalLifter(num_joints=68, in_features=3,
                           hidden_dim=1024, num_blocks=3)
    model = model.to(device)

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = MPJPELoss()

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
            print(f"[Epoch {epoch}/{args.epochs}] Loss3D: {avg_loss:.6f}")

            # Spara checkpoint var 10:e epoch + sista
            if epoch % 10 == 0 or epoch == args.epochs:
                ckpt_file = ckpt_dir / f"lifter_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    },
                    ckpt_file,
                )
                print(f"[OK] Sparade checkpoint: {ckpt_file}")


if __name__ == "__main__":
    main()
