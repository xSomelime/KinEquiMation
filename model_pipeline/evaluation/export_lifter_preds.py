# model_pipeline/evaluation/export_lifter_preds.py
# Kombinerad eval + export: Visualisera GT + Pred samt exportera predictions till JSON (Blender)

import os, argparse, json, cv2, random, time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_pipeline.models.lifter_3d import TemporalLifter
from model_pipeline.datasets.lifter_dataset import LifterDataset


def load_def_bones(def_bones_path):
    with open(def_bones_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_skeleton_edges(edges_path, bones_order):
    if not os.path.exists(edges_path):
        print(f"[warn] Hittar ingen skeleton_edges.json ({edges_path}) – ritar inga linjer.")
        return []
    with open(edges_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    name_to_idx = {name: i for i, name in enumerate(bones_order)}
    edges = []
    if "edges" in raw:
        for parent, child in raw["edges"]:
            if parent in name_to_idx and child in name_to_idx:
                edges.append((name_to_idx[parent], name_to_idx[child]))

    print(f"[viz] Laddade {len(edges)} edges från {edges_path}")
    return edges


def draw_points_and_edges(img, pts, edges, point_color=(0, 255, 0), edge_color=(255, 0, 0), radius=3):
    for (i1, i2) in edges:
        if i1 < len(pts) and i2 < len(pts):
            p1, p2 = pts[i1], pts[i2]
            if p1[0] is not None and p2[0] is not None:
                cv2.line(img, p1, p2, edge_color, 2)
    for (x, y) in pts:
        if x is None or y is None:
            continue
        cv2.circle(img, (x, y), radius, point_color, -1)
    return img


def to_xy_tuples(array, W, H):
    out = []
    for p in array:
        if p is None or len(p) < 2:
            out.append((None, None))
        else:
            x, y = float(p[0]), float(p[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                x, y = int(x * W), int(y * H)
            else:
                x, y = int(x), int(y)
            out.append((x, y))
    return out


def find_latest_run(save_dir: Path) -> Path:
    runs = [p for p in save_dir.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"Inga run-mappar i {save_dir}")
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    ckpts = sorted((latest / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"Inga checkpoints i {latest}/checkpoints")
    return ckpts[-1]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_run(Path(args.save_dir))
    print(f"[viz+export] Använder checkpoint: {ckpt_path}")

    # Modell
    model = TemporalLifter(num_joints=68, in_features=3, hidden_dim=1024, num_blocks=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Dataset
    dataset = LifterDataset(ann_file=args.data, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Skelett + bones
    kp_names = load_def_bones(args.def_bones)
    edges = load_skeleton_edges(args.skeleton, kp_names)

    # Output directories
    viz_out = Path(args.out_dir)
    viz_out.mkdir(parents=True, exist_ok=True)
    export_root = Path("outputs/lifter_preds")
    export_root.mkdir(parents=True, exist_ok=True)

    chosen = random.sample(list(dataloader), min(args.num_samples, len(dataset)))

    for idx, batch in enumerate(chosen):
        x = batch["keypoints_2d"].to(device)   # [1,T,K,3]
        y = batch["keypoints_3d"].to(device)   # [1,T,K,3]
        img_seqs = batch.get("image_paths", None)
        labels_seqs = batch.get("labels", None)
        action = batch.get("action", "Unknown")

        B, T, K, C = x.shape
        preds = model(x.view(B, T, -1))  # [1,T,K,3]

        # Export: ny mapp för denna action
        timestamp = time.strftime("%y%m%d_%H%M%S")
        seq_dir = export_root / f"action_{timestamp}"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for t in range(T):
            if img_seqs is None:
                continue

            frame_item = img_seqs[t]
            if "materials" in frame_item and frame_item["materials"]:
                img_path = Path(args.dataset_root) / frame_item["materials"][0]
            elif "flat" in frame_item and frame_item["flat"]:
                img_path = Path(args.dataset_root) / frame_item["flat"][0]
            else:
                print(f"[WARN] Dict utan materials/flat: {frame_item}")
                continue

            # GT-label
            label_item = labels_seqs[t]
            if isinstance(label_item, (list, tuple)):
                label_item = label_item[0]
            json_path = Path(args.dataset_root) / label_item

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # GT från JSON
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bones = meta.get("bones", {})
            gt_pts = []
            for name in kp_names:
                b = bones.get(name, {})
                if "uv" in b and b.get("in_frame", True):
                    u, v = b["uv"]
                    gt_pts.append((int(float(u)), int(float(v))))
                else:
                    gt_pts.append((None, None))

            pred_frame = preds[0, t].detach().cpu().numpy()  # [K,3]

            # --- Visualisering (kan hoppas över) ---
            if not args.no_viz:
                intr = meta.get("camera_intrinsics", None)
                extr = meta.get("camera_extrinsics", None)

                pred_pts = []
                if intr and extr:
                    K = np.array(intr["K"])
                    M = np.array(extr["matrix_world"])
                    M_inv = np.linalg.inv(M)

                    for (X, Y, Z) in pred_frame:
                        Pw = np.array([X, Y, Z, 1.0])
                        Pc = M_inv @ Pw

                        Xc, Yc, Zc = Pc[0], Pc[1], Pc[2]
                        Xc_cam = Xc
                        Yc_cam = Yc
                        Zc_cam = -Zc

                        if Zc_cam <= 0:
                            pred_pts.append((None, None))
                            continue

                        u = (K[0, 0] * (Xc_cam / Zc_cam)) + K[0, 2]
                        v = (K[1, 1] * (Yc_cam / Zc_cam)) + K[1, 2]
                        v = H - v
                        pred_pts.append((int(u), int(v)))
                else:
                    print("[WARN] Ingen kamerainfo i label – hoppar över reprojektion")
                    pred_pts = to_xy_tuples(pred_frame, W, H)

                img_out = img.copy()
                img_out = draw_points_and_edges(img_out, gt_pts, edges,
                                                point_color=(0, 255, 0), edge_color=(0, 200, 0))
                img_out = draw_points_and_edges(img_out, pred_pts, edges,
                                                point_color=(0, 0, 255), edge_color=(255, 0, 0))

                viz_path = viz_out / f"overlay_{idx:03d}_f{t:03d}.png"
                cv2.imwrite(str(viz_path), img_out)
                print(f"[viz] sparade {viz_path}")

            # --- Export ---
            frame_data = {"bones": {}}
            for j, coords in enumerate(pred_frame):
                bone_name = kp_names[j] if j < len(kp_names) else f"DEF-{j}"
                pos = coords.tolist()
                rot = [0.0, 0.0, 0.0, 1.0]  # placeholder
                frame_data["bones"][bone_name] = {"pos": pos, "rot": rot}

            frame_path = seq_dir / f"frame_{t:03d}.json"
            with open(frame_path, "w", encoding="utf-8") as f:
                json.dump(frame_data, f, indent=2)

        # meta.json för denna sekvens
        meta_out = {
            "action": action,
            "num_frames": T,
            "timestamp": timestamp,
            "bone_names": kp_names,
            "format": "pos+rot"
        }
        with open(seq_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[export] Sparade {T} frames + meta → {seq_dir}")
        time.sleep(1)  # säkerställ unik timestamp per action



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path till lifter_dataset.json")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="outputs/lifter_runs")
    ap.add_argument("--seq_len", type=int, default=27)
    ap.add_argument("--skeleton", type=str, default="dataset_pipeline/data/dataset_exports/skeleton_edges.json")
    ap.add_argument("--def_bones", type=str, default="dataset_pipeline/data/dataset_exports/def_bones.txt")
    ap.add_argument("--dataset_root", type=str, default="dataset_pipeline/data/dataset/final")
    ap.add_argument("--out_dir", type=str, default="outputs/overlays_lifter")
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--no-viz", action="store_true",
                help="Hoppa över visualisering, exportera endast JSON")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    main(args)
