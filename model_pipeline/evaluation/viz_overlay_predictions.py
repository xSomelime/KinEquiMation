# ai/models/viz_overlay_predictions.py - Visualisera GT + Pred overlay
import os, argparse, cv2, torch, json
import numpy as np

from .dataset import PoseGaitSequenceDataset
from .tiny_model import PoseGaitTinyNet


def move_to_device(batch, device, non_blocking=False):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def load_gt_uv(json_path, bones_order):
    """
    Läser ground truth UV från JSON-exporten (uv).
    Returnerar [K,2] array i samma ordning som bones-listan.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bones = obj.get("bones", {})
    pts = []
    for bname in bones_order:
        b = bones.get(bname, {})
        if b.get("in_frame") and "uv" in b:
            u, v = b["uv"]
            pts.append([u, v])
        else:
            pts.append([np.nan, np.nan])
    return np.array(pts, dtype=float)


def project_points(xyz, intr, extr, target_size=None):
    """
    Projektera 3D-punkter till 2D pixelkoordinater.
    xyz: [K,3] torch eller numpy (världs-koordinater)
    intr: dict med fx, fy, cx, cy
    extr: [4,4] numpy (kamera matrix_world från Blender)
    """
    if torch.is_tensor(xyz):
        pts = xyz.detach().cpu().numpy()
    else:
        pts = np.array(xyz)

    # Homogena koordinater [4,K]
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T  

    # Blender → CV fix (Z-up, -Y forward → Z-forward, Y-down)
    blender_to_cv = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ], dtype=float)

    cam_T = np.linalg.inv(extr) @ blender_to_cv
    cam = cam_T @ pts_h
    X, Y, Z = cam[0], cam[1], cam[2]

    Z[Z == 0] = 1e-6  # undvik division med noll

    u = intr["fx"] * (X / Z) + intr["cx"]
    v = intr["fy"] * (Y / Z) + intr["cy"]

    if target_size is not None:
        H, W = target_size
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

    return np.stack([u, v], axis=1)


def draw_points(img, pts, color=(0,0,255), radius=3):
    for (x, y) in pts:
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[viz] using device {device}")

    ds = PoseGaitSequenceDataset(
        project_root=args.project_root,
        bones_txt=args.bones_txt,
        image_size=args.image_size,
        seq_len=args.seq_len,
        stride=args.stride
    )
    print(f"[viz] dataset with {len(ds)} windows, {ds.K} keypoints")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = PoseGaitTinyNet(
        K=ckpt["K"],
        num_gaits=ckpt["num_gaits"],
        feat_dim=ckpt["args"]["feat_dim"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    for idx in range(min(args.num_samples, len(ds))):
        sample = ds[idx]
        batch = {"images": sample["images"].unsqueeze(0)}
        batch = move_to_device(batch, device)

        with torch.no_grad():
            pred_pose, _ = model(batch["images"])  # [1,T,K,3]

        pred_pose = pred_pose[0]   # [T,K,3]
        img_paths = sample["meta"]["img_paths"]
        intr = sample["meta"]["intrinsics"]
        extr = sample["meta"].get("extrinsics", np.eye(4))

        for t in range(pred_pose.shape[0]):
            img_path = img_paths[t]
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]

            # Hämta JSON som har samma basename som bilden (inkl. _c00)
            base_name = os.path.splitext(os.path.basename(img_path))[0]  # t.ex. "f00001_c00"
            clip_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))  # .../final/<action>
            json_path = os.path.join(clip_dir, "labels", f"{base_name}.json")

            # GT från uv
            gt_2d = load_gt_uv(json_path, ds.bones)

            # Pred → 2D
            pred_2d = project_points(pred_pose[t], intr, extr, target_size=(H, W))

            # Rita
            img_out = img.copy()
            img_out = draw_points(img_out, gt_2d, color=(0,255,0))   # GT = grönt
            img_out = draw_points(img_out, pred_2d, color=(0,0,255)) # Pred = rött

            out_path = os.path.join(args.out_dir, f"overlay{idx:03d}_frame{t:03d}.png")
            cv2.imwrite(out_path, img_out)
            print(f"[viz] saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--bones_txt", type=str, default="ai/data/def_bones.txt")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="ai/outputs/overlays_pred")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
