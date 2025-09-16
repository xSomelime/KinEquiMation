# model_pipeline/evaluation/eval_gait.py
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_pipeline.models.gait_classifier import GaitClassifier
from model_pipeline.datasets.gait_dataset import GaitDataset


def find_latest_run(save_dir: Path) -> Path | None:
    """Hitta senaste run-mappen baserat på timestamp i namnet."""
    runs = [p for p in save_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    latest = max(runs, key=lambda p: p.name)
    ckpts = sorted((latest / "checkpoints").glob("*.pth"))
    if not ckpts:
        return None
    return ckpts[-1]


def evaluate(model, dataloader, device, idx_to_action, num_samples=10):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["keypoints"].to(device)  # [B,T,K*3]
            y_true = batch["label"]            # [B]

            logits = model(x)                  # [B,num_classes]
            preds = torch.argmax(logits, dim=1).cpu()

            for yt, yp in zip(y_true, preds):
                gt_label = idx_to_action[int(yt)]
                pred_label = idx_to_action[int(yp)]
                print(f"GT: {gt_label:<8} | Pred: {pred_label:<8}")
                total += 1
                if yt.item() == yp.item():
                    correct += 1

            if total >= num_samples:
                break

    acc = correct / max(1, total)
    print(f"[RESULT] Accuracy on {total} samples: {acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="dataset_pipeline/data/dataset_exports/lifter_dataset.json",
                        help="Path till lifter_dataset.json (med 3D keypoints + action labels)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path till tränad checkpoint (valfritt, annars används senaste körningen)")
    parser.add_argument("--runs_dir", type=str, default="outputs/gait_runs",
                        help="Bas-mapp för träningskörningar")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Antal sekvenser att visa")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    dataset = GaitDataset(
        ann_file=args.data,
        seq_len=args.seq_len,
        num_joints=68,
        in_features=3,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Ladda senaste checkpoint om ingen fil angavs
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_run(Path(args.runs_dir))
        if ckpt_path is None:
            raise RuntimeError(f"Hittar ingen checkpoint i {args.runs_dir}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = GaitClassifier(
        num_joints=68,
        in_features=3,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(dataset.action_to_idx),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    print(f"[INFO] Utvärderar med checkpoint: {ckpt_path}")
    print(f"[INFO] Label mapping: {ckpt.get('label_mapping')}")

    evaluate(model, dataloader, device, dataset.idx_to_action, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
