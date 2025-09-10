# model_pipeline/evaluation/viz_overlay_predictions.py
# Visualisera 2D: GT (grön) + Pred (röd) + skeleton-linjer med MMPose HRNet-W32 (68kp)
# Automatisk: hittar senaste run i --runs_root och tar dess "best*.pth".

import os, argparse, json, cv2, random
import numpy as np
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples


def load_gt_uv(json_path, bones_order):
    """Läs ground truth (2D) från exporterad JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bones = obj.get("bones", {})
    pts = []
    for bname in bones_order:
        b = bones.get(bname, {})
        if b and "uv" in b and b.get("in_frame", True):
            u, v = b["uv"]
            pts.append((int(u), int(v)))
        else:
            pts.append((None, None))
    return pts


def load_skeleton_edges(edges_path, bones_order):
    """Ladda skeleton_edges.json och mappa bone-namn till index i bones_order."""
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


def to_xy_tuples(pts):
    """Konvertera array eller lista till [(x,y), ...] med ints."""
    out = []
    for p in pts:
        if p is None:
            out.append((None, None))
        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
            x, y = p[0], p[1]
            if x is None or y is None:
                out.append((None, None))
            else:
                out.append((int(x), int(y)))
        else:
            out.append((None, None))
    return out


def draw_points_and_edges(img, pts, edges, point_color=(0, 255, 0), edge_color=(255, 0, 0), radius=3):
    """Ritar keypoints (punkter) och skelettlänkar (linjer)."""
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


def collect_all_frames(root_dir):
    """Hämta alla (bild, json)-par från datasetet."""
    samples = []
    for clip in os.listdir(root_dir):
        clip_dir = os.path.join(root_dir, clip)
        if not os.path.isdir(clip_dir):
            continue

        images_root = os.path.join(clip_dir, "images")
        labels_dir = os.path.join(clip_dir, "labels")
        if not os.path.isdir(images_root) or not os.path.isdir(labels_dir):
            continue

        for variant in os.listdir(images_root):
            variant_dir = os.path.join(images_root, variant)
            if not os.path.isdir(variant_dir):
                continue
            for f in os.listdir(variant_dir):
                if not f.endswith(".png"):
                    continue
                base_name = os.path.splitext(f)[0]
                img_path = os.path.join(variant_dir, f)
                json_path = os.path.join(labels_dir, base_name + ".json")
                if os.path.exists(json_path):
                    samples.append((img_path, json_path))
                else:
                    print(f"[warn] Hittade ingen GT för {img_path}")
    return samples


def find_latest_run_and_best(runs_root):
    """Hitta senaste run-mapp och dess best*.pth."""
    run_dirs = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]
    if not run_dirs:
        raise FileNotFoundError(f"Inga run-mappar i {runs_root}")
    latest_run = max(run_dirs, key=os.path.getmtime)

    ckpt_files = [f for f in os.listdir(latest_run) if f.startswith("best") and f.endswith(".pth")]
    if not ckpt_files:
        raise FileNotFoundError(f"Ingen 'best*.pth' i {latest_run}")
    ckpt_file = ckpt_files[0]
    ckpt_path = os.path.join(latest_run, ckpt_file)
    return latest_run, ckpt_file, ckpt_path


def main(args):
    device = 'cuda' if (not args.cpu) else 'cpu'

    run_dir, ckpt_file, ckpt_path = find_latest_run_and_best(args.runs_root)

    run_name = os.path.basename(run_dir.rstrip("/"))
    timestamp = run_name.split("_")[-2] + "_" + run_name.split("_")[-1]

    out_dir = os.path.join(args.out_dir, f"horse_hrnet_viz_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[viz] Senaste run: {run_dir}")
    print(f"[viz] Använder checkpoint: {ckpt_path}")
    print(f"[viz] Sparar i: {out_dir}")

    model = init_model(args.config, ckpt_path, device=device)

    with open(args.bones_txt, "r", encoding="utf-8") as f:
        bones_order = [line.strip() for line in f if line.strip()]

    edges = load_skeleton_edges(args.edges_json, bones_order)

    all_samples = collect_all_frames(args.img_root)
    print(f"[viz] Hittade totalt {len(all_samples)} frames")
    if not all_samples:
        print("[viz] Inga frames hittades, avbryt.")
        return

    chosen = random.sample(all_samples, min(args.num_samples, len(all_samples)))

    for idx, (img_path, json_path) in enumerate(chosen):
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        full_bbox = np.array([0, 0, W, H])
        results = inference_topdown(model, img, [full_bbox], bbox_format='xyxy')
        if not results:
            print(f"[viz] Ingen prediktion för {img_path}")
            continue

        pred = merge_data_samples(results)
        pred_pts = to_xy_tuples(pred.pred_instances.keypoints[0])
        gt_pts = load_gt_uv(json_path, bones_order)

        img_out = img.copy()
        img_out = draw_points_and_edges(img_out, gt_pts, edges, point_color=(0, 255, 0), edge_color=(0, 200, 0))
        img_out = draw_points_and_edges(img_out, pred_pts, edges, point_color=(0, 0, 255), edge_color=(255, 0, 0))

        out_path = os.path.join(out_dir, f"{timestamp}_overlay_{idx:03d}.png")
        cv2.imwrite(out_path, img_out)
        print(f"[viz] sparade {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Path till MMPose config (t.ex. model_pipeline/configs/hrnet_w32_horse68_256x256.py)")
    ap.add_argument("--runs_root", type=str, required=True,
                    help="Root till alla run-mappar (t.ex. outputs/checkpoints)")
    ap.add_argument("--img_root", type=str, required=True,
                    help="Rotmapp till dataset/final (innehåller walk/, trot/, ...)")
    ap.add_argument("--bones_txt", type=str, default="dataset_pipeline/data/dataset_exports/def_bones.txt")
    ap.add_argument("--edges_json", type=str,
                    default="dataset_pipeline/data/dataset_exports/skeleton_edges.json",
                    help="Path till skeleton_edges.json")
    ap.add_argument("--out_dir", type=str, default="outputs/overlays_hrnet")
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    main(args)
