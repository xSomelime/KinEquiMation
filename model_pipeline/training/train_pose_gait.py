# ai/models/train_pose_gait.py - Training loop
import os, argparse, random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm

from .dataset import PoseGaitSequenceDataset
from .tiny_model import PoseGaitTinyNet

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pose_loss(pred, target):
    # pred/target: [B,T,K,3]
    return nn.SmoothL1Loss()(pred, target)

def gait_acc(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def move_to_device(batch, device, non_blocking=False):
    """Flytta alla tensorer i batch till device."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

def main(args):
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"[device] Using {device}")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    ds = PoseGaitSequenceDataset(
        project_root=args.project_root,
        bones_txt=args.bones_txt,
        image_size=args.image_size,
        seq_len=args.seq_len,
        stride=args.stride
    )
    K = ds.K
    num_gaits = ds.num_gaits

    if len(ds) == 0 or ds.K == 0 or ds.num_gaits == 0:
        raise RuntimeError(
            "Dataset tomt eller omatchade labels.\n"
            "- Kontrollera att images-filerna och per-frame JSON har matchande frame-id (t.ex. ..._000123.png ↔ 000123.json).\n"
            "- Vi stödjer både .png och .PNG.\n"
            "- DEF-benen hittas i JSON under bones[DEF-*].head.\n"
            "- (Valfritt) skapa ai/data/def_bones.txt för att låsa ordningen."
        )

    # Split
    n_total = len(ds)
    n_val = max(1, int(n_total * args.val_split))
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        drop_last=True
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        drop_last=False
    )

    model = PoseGaitTinyNet(K=K, num_gaits=num_gaits, feat_dim=args.feat_dim).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    gait_criterion = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_pose, ep_gacc, seen = 0.0, 0.0, 0
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = move_to_device(batch, device, non_blocking=use_cuda)
            imgs, y_pose, y_gait = batch["images"], batch["pose"], batch["gait"]

            with torch.cuda.amp.autocast(enabled=use_cuda):
                pred_pose, logits = model(imgs)
                lp = pose_loss(pred_pose, y_pose)
                lg = gait_criterion(logits, y_gait)
                loss = lp + args.lambda_gait * lg

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = imgs.size(0)
            ep_pose += lp.item() * bs
            ep_gacc += gait_acc(logits.detach(), y_gait) * bs
            seen += bs
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        train_pose = ep_pose / max(1, seen)
        train_gacc = ep_gacc / max(1, seen)

        # Validering
        model.eval()
        val_pose_sum, val_gacc_sum, val_seen = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_ld:
                batch = move_to_device(batch, device, non_blocking=use_cuda)
                imgs, y_pose, y_gait = batch["images"], batch["pose"], batch["gait"]
                pred_pose, logits = model(imgs)
                val_pose_sum += pose_loss(pred_pose, y_pose).item() * imgs.size(0)
                val_gacc_sum += gait_acc(logits, y_gait) * imgs.size(0)
                val_seen += imgs.size(0)
        val_pose = val_pose_sum / max(1, val_seen)
        val_gacc = val_gacc_sum / max(1, val_seen)

        print(f"[epoch {epoch}] train_pose={train_pose:.4f} train_gait_acc={train_gacc:.3f} | "
              f"val_pose={val_pose:.4f} val_gait_acc={val_gacc:.3f}")

        # Spara bäst på pose-loss
        if val_pose < best_val:
            best_val = val_pose
            ckpt = os.path.join(args.out_dir, "tiny_posegait_best.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "bones": ds.bones,
                "K": K, "num_gaits": num_gaits,
                "args": vars(args),
            }, ckpt)
            print(f"[ckpt] saved {ckpt}")

    # Sista vikt
    last = os.path.join(args.out_dir, "tiny_posegait_last.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "bones": ds.bones,
        "K": K, "num_gaits": num_gaits,
        "args": vars(args),
    }, last)
    print(f"[ckpt] saved {last}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".", help="repo-root (innehåller data/dataset/final)")
    ap.add_argument("--bones_txt", type=str, default="ai/data/def_bones.txt")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--feat_dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambda_gait", type=float, default=0.5)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--out_dir", type=str, default="ai/models/checkpoints")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
