import json
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def load_train_metrics(scalars_file):
    """Läs träningens loss och acc_pose från scalars.json (JSON lines)."""
    train_epochs, train_loss, train_acc = [], [], []

    with open(scalars_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            epoch = obj.get("epoch")
            if "loss" in obj:
                train_epochs.append(epoch)
                train_loss.append(obj["loss"])
                train_acc.append(obj.get("acc_pose"))
    return train_epochs, train_loss, train_acc


def load_val_metrics(log_file):
    """Läs valideringsresultat (coco/AP) från .log-filen."""
    val_epochs, val_ap = [], []
    regex = re.compile(r"Epoch\(val\)\s+\[(\d+)\].*coco/AP:\s+([0-9.]+)")
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = regex.search(line)
            if match:
                epoch = int(match.group(1))
                ap = float(match.group(2))
                val_epochs.append(epoch)
                val_ap.append(ap)
    return val_epochs, val_ap


def plot_curves(train_epochs, train_loss, train_acc, val_epochs, val_ap, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train loss
    if train_loss:
        plt.figure()
        plt.plot(train_epochs, train_loss, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(out_dir / "train_loss.png")
        plt.close()

    # Train acc_pose
    if any(a is not None for a in train_acc):
        plt.figure()
        plt.plot(
            [e for e, a in zip(train_epochs, train_acc) if a is not None],
            [a for a in train_acc if a is not None],
            label="Train acc_pose"
        )
        plt.xlabel("Epoch")
        plt.ylabel("acc_pose")
        plt.title("Training Accuracy (acc_pose)")
        plt.legend()
        plt.savefig(out_dir / "train_acc.png")
        plt.close()

    # Validation AP
    if val_epochs:
        plt.figure()
        plt.plot(val_epochs, val_ap, label="Val coco/AP")
        plt.xlabel("Epoch")
        plt.ylabel("AP")
        plt.title("Validation AP")
        plt.legend()
        plt.savefig(out_dir / "val_ap.png")
        plt.close()

    print(f"[OK] Sparade grafer i: {out_dir}")


def find_latest_run(base_dir: Path, prefix="horse_hrnet_"):
    """Hitta senaste run och returnera scalars.json + .log"""
    scalars = list(base_dir.glob(f"{prefix}*/**/vis_data/scalars.json"))
    logs = list(base_dir.glob(f"{prefix}*/**/*.log"))
    if not scalars or not logs:
        raise FileNotFoundError("Hittade inte både scalars.json och .log")
    latest_scalars = max(scalars, key=lambda p: p.stat().st_mtime)
    latest_log = max(logs, key=lambda p: p.stat().st_mtime)
    return latest_scalars, latest_log, latest_scalars.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="outputs/checkpoints",
                        help="Basfolder där horse_hrnet_* run-mappar ligger")
    args = parser.parse_args()

    scalars_file, log_file, out_dir = find_latest_run(Path(args.base_dir))

    print(f"[INFO] Läser träning från: {scalars_file}")
    print(f"[INFO] Läser validering från: {log_file}")

    train_epochs, train_loss, train_acc = load_train_metrics(scalars_file)
    val_epochs, val_ap = load_val_metrics(log_file)

    plot_curves(train_epochs, train_loss, train_acc, val_epochs, val_ap, out_dir)
