# model_pipeline/evaluation/viz_3d_skeleton.py
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_points(json_path, bones_order):
    """Läs in 3D head-punkter för varje ben."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bones = obj.get("bones", {})
    pts = []
    for bname in bones_order:
        b = bones.get(bname, {})
        if "head" in b:
            x, y, z = b["head"]
            pts.append((float(x), float(y), float(z)))
        else:
            pts.append((None, None, None))
    return pts

def load_edges(edges_json, bones_order):
    with open(edges_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    name_to_idx = {name: i for i, name in enumerate(bones_order)}
    edges = []
    for parent, child in raw["edges"]:
        if parent in name_to_idx and child in name_to_idx:
            edges.append((name_to_idx[parent], name_to_idx[child]))
    return edges

def plot_skeleton(ax, pts, edges, point_color, edge_color, label=None):
    xs, ys, zs = [], [], []
    for (x, y, z) in pts:
        if x is not None:
            xs.append(x); ys.append(y); zs.append(z)
    ax.scatter(xs, ys, zs, c=[point_color], s=15, label=label)

    for (i1, i2) in edges:
        if pts[i1][0] is not None and pts[i2][0] is not None:
            ax.plot([pts[i1][0], pts[i2][0]],
                    [pts[i1][1], pts[i2][1]],
                    [pts[i1][2], pts[i2][2]], color=edge_color, linewidth=1)

def set_axes_equal(ax):
    """Se till att proportionerna blir rätt i 3D."""
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0
    mid_x = np.mean(x_limits); mid_y = np.mean(y_limits); mid_z = np.mean(z_limits)
    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

def iterate_frames(img_root, lifter_dir):
    """Matcha GT och Pred JSON för samma frame i valt action (endast cam c00).
       img_root ska vara direkt t.ex. .../final/walk eller .../final/trot"""
    img_dir = os.path.join(img_root, "images/flat")
    label_dir = os.path.join(img_root, "labels")

    if not os.path.exists(img_dir):
        raise RuntimeError(f"Hittar inte {img_dir}")

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png") and "_c00" in f])
    for img_file in images:
        base = os.path.splitext(img_file)[0]      # f00001_c00
        frame_id = base.split("_")[0][1:]         # f00001 -> 00001
        frame_id = frame_id.zfill(6)

        json_gt = os.path.join(label_dir, base + ".json")
        json_pred = os.path.join(lifter_dir, frame_id + ".json")

        if os.path.exists(json_gt) and os.path.exists(json_pred):
            yield base, json_gt, json_pred

def main(args):
    with open(args.bones_txt, "r", encoding="utf-8") as f:
        bones_order = [line.strip() for line in f if line.strip()]
    edges = load_edges(args.edges_json, bones_order)

    os.makedirs(args.out_dir, exist_ok=True)

    # 👉 Hämta första frame med både GT och Pred
    frames = list(iterate_frames(args.img_root, args.lifter_dir))
    if not frames:
        raise RuntimeError("Hittade inga matchande frames mellan GT och Preds")

    base, json_gt, json_pred = frames[0]
    gt_pts = load_points(json_gt, bones_order)
    pred_pts = load_points(json_pred, bones_order)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_skeleton(ax, gt_pts, edges, point_color="green", edge_color="green", label="GT")
    plot_skeleton(ax, pred_pts, edges, point_color="red", edge_color="blue", label="Pred")

    # extrahera actionnamn (sista mappen i img_root)
    action_name = os.path.basename(args.img_root.rstrip("/\\"))
    ax.set_title(f"{action_name.upper()} – Frame {base}", fontsize=14)
    set_axes_equal(ax)
    ax.legend()

    out_path = os.path.join(args.out_dir, f"{action_name}_{base}_3d.png")
    plt.savefig(out_path, dpi=150)
    print(f"[INFO] Sparade {out_path}")

    plt.show()  # 🔹 visar interaktivt fönster
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True,
                    help="Rotmapp till en action i datasetet (t.ex. .../final/walk)")
    ap.add_argument("--lifter_dir", type=str, required=True,
                    help="Mapp med lifter_preds/*.json för samma action")
    ap.add_argument("--bones_txt", type=str, default="dataset_pipeline/data/dataset_exports/def_bones.txt")
    ap.add_argument("--edges_json", type=str, default="dataset_pipeline/data/dataset_exports/skeleton_edges.json")
    ap.add_argument("--out_dir", type=str, default="outputs/overlays_lifter3d")
    args = ap.parse_args()
    main(args)
