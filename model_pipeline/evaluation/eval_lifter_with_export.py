# model_pipeline/evaluation/eval_lifter_with_export.py
# IDENTISK overlay som eval_lifter.py (GT grönt, Pred rött).
# Exporterar dessutom keypoints (pos+rot) till JSON för Blender.
# - Läser hela klipp från original-datasetet (inte LifterDataset).
# - Tar EN label + EN bild per frame (vald kamera, default: c00).
# - Exporterar frame_XXX.json för varje frame.
# - Skriver meta.json per sekvens.
# - Väntar 1 sekund mellan varje action för att undvika map-krockar.

import os, argparse, json, cv2, time
import numpy as np
from pathlib import Path
import torch

from model_pipeline.models.lifter_3d import TemporalLifter


def load_def_bones(def_bones_path):
    with open(def_bones_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_skeleton_edges(edges_path, bones_order):
    if not os.path.exists(edges_path):
        print(f"[warn] Hittar ingen skeleton_edges.json ({edges_path}) – ritar inga linjer.")
        return [], []
    with open(edges_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    name_to_idx = {name: i for i, name in enumerate(bones_order)}
    edges = []
    for parent, child in raw.get("edges", []):
        if parent in name_to_idx and child in name_to_idx:
            edges.append((name_to_idx[parent], name_to_idx[child]))
    return edges, raw.get("edges", [])


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


def align_pred_to_gt(gt_2d, pred_3d):
    """Aligna pred (3D → 2D) mot GT 2D med centroid + skala (samma som eval_lifter.py)."""
    pred_2d = pred_3d[:, :2]
    gt_center = gt_2d.mean(0)
    pred_center = pred_2d.mean(0)
    pred_2d = pred_2d - pred_center
    gt_scale = np.linalg.norm(gt_2d.max(0) - gt_2d.min(0))
    pred_scale = np.linalg.norm(pred_2d.max(0) - pred_2d.min(0))
    if pred_scale > 1e-6:
        pred_2d *= (gt_scale / pred_scale)
    pred_2d = pred_2d + gt_center
    return pred_2d


def bone_rotations(pred_3d, edges, kp_names):
    rots = {}
    for (i1, i2) in edges:
        v = pred_3d[i2] - pred_3d[i1]
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            v = v / norm
        rots[kp_names[i2]] = v.tolist()
    return rots


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
    print(f"[viz/export] Använder checkpoint: {ckpt_path}")

    # Modell
    model = TemporalLifter(num_joints=68, in_features=3, hidden_dim=1024, num_blocks=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    # Skelett + bones
    kp_names = load_def_bones(args.def_bones)
    edges, edges_names = load_skeleton_edges(args.skeleton, kp_names)

    # Iterera över klipp i dataset_root
    clips = [p for p in Path(args.dataset_root).iterdir() if p.is_dir()]
    for clip in clips:
        labels_dir = clip / "labels"
        images_dir = clip / "images" / "flat"
        if not labels_dir.exists() or not images_dir.exists():
            continue

        # Filtrera labels för vald kamera
        all_labels = sorted(labels_dir.glob("*.json"))
        frame_map = {}
        for lf in all_labels:
            parts = lf.stem.split("_")  # ex: f00001_c00
            if len(parts) != 2:
                continue
            frame_id, cam_id = parts
            if cam_id == args.camera:
                frame_map[frame_id] = lf
        label_files = [frame_map[k] for k in sorted(frame_map.keys())]
        if not label_files:
            continue

        # Action-namn från första label
        with open(label_files[0], "r", encoding="utf-8") as f:
            first_meta = json.load(f)
        action_name = first_meta.get("action", "Unknown")

        # Hämta keypoints GT
        all_keypoints = []
        metas = []
        for lf in label_files:
            with open(lf, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bones = meta.get("bones", {})
            frame_kps = []
            for name in kp_names:
                b = bones.get(name, {})
                if "uv" in b:
                    u, v = b["uv"]
                    frame_kps.append([float(u), float(v), 1.0])
                else:
                    frame_kps.append([0.0, 0.0, 0.0])
            all_keypoints.append(frame_kps)
            metas.append(meta)

        # Modell-prediktion
        x = torch.tensor(all_keypoints, dtype=torch.float32).unsqueeze(0).to(device)  # [1,T,K,3]
        preds = model(x.view(1, len(label_files), -1))  # [1,T,K,3]

        # Skapa mappar
        timestamp = time.strftime("%y%m%d_%H%M%S")
        overlay_dir = Path(args.out_dir) / f"action_{timestamp}"
        export_dir = Path(args.export_dir) / f"action_{timestamp}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Iterera frames
        for t, lf in enumerate(label_files):
            pred_frame = preds[0, t].detach().cpu().numpy()

            # Bild för overlay
            frame_id, cam_id = lf.stem.split("_")
            img_path = images_dir / f"{frame_id}_{cam_id}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            # GT från label
            meta = metas[t]
            bones = meta.get("bones", {})
            gt_pts = []
            for name in kp_names:
                b = bones.get(name, {})
                if "uv" in b and b.get("in_frame", True):
                    u, v = b["uv"]
                    gt_pts.append((int(float(u)), int(float(v))))
                else:
                    gt_pts.append((None, None))

            # --- PROJICERING ---
            intr = meta.get("camera_intrinsics", None)
            extr = meta.get("camera_extrinsics", None)
            pred_pts = []

            if intr and extr:
                K = np.array(intr["K"])
                M = np.array(extr["matrix_world"])
                M_inv = np.linalg.inv(M)

                for (X, Y, Z) in pred_frame:
                    Pw = np.array([X, Y, Z, 1.0])
                    Pc = M_inv @ Pw  # world → camera

                    # Blender's camera convention: -Z forward, +Y up
                    Xc, Yc, Zc = Pc[0], Pc[1], Pc[2]
                    Xc_cam = Xc
                    Yc_cam = Yc       # behåll original, ingen flip här
                    Zc_cam = -Zc      # vi vänder bara Z (framåt/bakåt)

                    if Zc_cam <= 0:
                        pred_pts.append((None, None))
                        continue

                    u = (K[0, 0] * (Xc_cam / Zc_cam)) + K[0, 2]
                    v = (K[1, 1] * (Yc_cam / Zc_cam)) + K[1, 2]

                    # flip v så (0,0) är uppe i vänstra hörnet, som i OpenCV
                    v = H - v

                    pred_pts.append((int(u), int(v)))

            # Fallback om reprojektion failar
            if not pred_pts or all(pt == (None, None) for pt in pred_pts):
                print(f"[WARN] Frame {t}: reprojektion misslyckades, använder align_pred_to_gt")
                valid_gt = np.array([p for p in gt_pts if p[0] is not None], dtype=np.float32)
                if len(valid_gt) > 0:
                    pred_2d = align_pred_to_gt(valid_gt, pred_frame)
                    pred_pts = to_xy_tuples(pred_2d, W, H)
                else:
                    pred_pts = [(None, None)] * len(pred_frame)

            # Rita overlay
            img_out = img.copy()
            img_out = draw_points_and_edges(img_out, gt_pts, edges,
                                            point_color=(0, 255, 0), edge_color=(0, 200, 0))
            img_out = draw_points_and_edges(img_out, pred_pts, edges,
                                            point_color=(0, 0, 255), edge_color=(255, 0, 0))
            out_path = overlay_dir / f"overlay_f{t:03d}.png"
            cv2.imwrite(str(out_path), img_out)
            print(f"[viz] sparade {out_path}")

            # Export JSON (3D)
            rots = bone_rotations(pred_frame, edges, kp_names)
            frame_data = {
                "frame_idx": t,
                "keypoints": {
                    name: {"location": pred_frame[i].tolist(),
                           "rotation": rots.get(name, [0, 0, 0])}
                    for i, name in enumerate(kp_names)
                }
            }
            with open(export_dir / f"frame_{t:03d}.json", "w", encoding="utf-8") as f:
                json.dump(frame_data, f, indent=2)

        # Meta per action
        meta_out = {
            "action": action_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_frames": len(label_files),
            "fps": 30,
            "bones_order": kp_names,
            "skeleton_edges": edges_names,
            "camera": args.camera
        }
        with open(export_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[export] Sparade {len(label_files)} frames för {action_name} → {export_dir}")
        time.sleep(1)  # unik timestamp




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="outputs/lifter_runs")
    ap.add_argument("--skeleton", type=str,
                    default="dataset_pipeline/data/dataset_exports/skeleton_edges.json")
    ap.add_argument("--def_bones", type=str,
                    default="dataset_pipeline/data/dataset_exports/def_bones.txt")
    ap.add_argument("--dataset_root", type=str,
                    default="dataset_pipeline/data/dataset/final")
    ap.add_argument("--out_dir", type=str, default="outputs/overlays_lifter")
    ap.add_argument("--export_dir", type=str, default="outputs/lifter_preds")
    ap.add_argument("--camera", type=str, default="c00", help="Vilken kamera som ska användas för overlays")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    main(args)
