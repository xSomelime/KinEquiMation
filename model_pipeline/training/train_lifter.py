# model_pipeline/training/train_lifter.py

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        x = batch["keypoints_2d"]  # [B,T,K,3]
        y = batch["keypoints_3d"]  # [B,T,K,3]

        B, T, K, C = x.shape
        x = x.reshape(B, T, -1)  # [B,T,K*3]

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        x = batch["keypoints_2d"]
        y = batch["keypoints_3d"]

        B, T, K, C = x.shape
        x = x.reshape(B, T, -1)

        preds = model(x)
        loss = criterion(preds, y)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str,
        default="dataset_pipeline/data/dataset_exports/lifter_dataset_train.json",
        help="Path till dataset (träning)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=None,
                        help="Sekvenslängd. Om None används datasetets default.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="outputs/lifter_runs")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Andel av datasetet som används för validering")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Dataset och split
    full_dataset = LifterDataset(ann_file=args.data, seq_len=args.seq_len, full_seq=False)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"[INFO] Laddade dataset: {len(full_dataset)} sekvenser "
          f"({train_size} train / {val_size} val)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=False)

    # Modell – num_joints tas direkt från datasetets bone_names
    num_joints = len(full_dataset.bone_names)
    print(f"[INFO] Antal leder i dataset: {num_joints}")
    model = TemporalLifter(num_joints=num_joints, in_features=3,
                           hidden_dim=1024, num_blocks=3)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = MPJPELoss()

    # Run directory
    run_dir = Path(args.save_dir) / time.strftime("%y%m%d_%H%M")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "training_log.json"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    with open(log_file, "w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_one_epoch(model, val_loader, criterion, device)

            print(f"[Epoch {epoch}/{args.epochs}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len or full_dataset.default_seq_len,
                "lr": args.lr,
                "timestamp": time.strftime("%y%m%d_%H%M"),
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()

            # 🔥 Spara bästa modellen
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_file = ckpt_dir / "best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    ckpt_file,
                )
                print(f"[BEST] Ny bästa modell sparad (val_loss={val_loss:.6f})")

            # 🔁 Spara var 10:e epoch + sista
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
