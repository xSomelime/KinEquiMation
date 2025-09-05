# ai/models/viz_overlay_predictions.py
import os, argparse, cv2, torch
import numpy as np

from .dataset import PoseGaitSequenceDataset
from .tiny_model import PoseGaitTinyNet

# --- hjälpfunktion för att flytta batch till GPU/CPU ---
def move_to_device(batch, device, non_blocking=False):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

# --- Rita skeleton på en bild ---
def draw_skeleton(img, joints, bones, color=(0,0,255), radius=3):
    """
    img: numpy array [H,W,3]
    joints: [K,3] tensor eller array (x,y,z)
    bones: lista med bone-namn i ordning (DEF_*), används för att koppla ihop.
    """
    h, w, _ = img.shape
    pts = joints[:, :2].cpu().numpy().astype(int)

    # Rita punkter
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)

    # Exempel: enkel parent→child-ritning om namnen har hierarki
    # Här använder vi samma logik som i viz_overlay_skeleton (förenklad)
    for i in range(len(bones)-1):
        p1, p2 = pts[i], pts[i+1]
        cv2.line(img, tuple(p1), tuple(p2), color, 1, cv2.LINE_AA)

    return img

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[viz] using device {device}")

    # --- Ladda dataset ---
    ds = PoseGaitSequenceDataset(
        project_root=args.project_root,
        bones_txt=args.bones_txt,
        image_size=args.image_size,
        seq_len=args.seq_len,
        stride=args.stride
    )
    print(f"[viz] dataset with {len(ds)} windows, {ds.K} keypoints")

    # --- Ladda modell + checkpoint ---
    ckpt = torch.load(args.ckpt, map_location=device)
    model = PoseGaitTinyNet(K=ckpt["K"], num_gaits=ckpt["num_gaits"], feat_dim=ckpt["args"]["feat_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Ta några sekvenser och rita ---
    for idx in range(min(args.num_samples, len(ds))):
        sample = ds[idx]
        batch = {"images": sample["images"].unsqueeze(0)}  # [1,T,C,H,W]
        batch = move_to_device(batch, device)
        with torch.no_grad():
            pred_pose, _ = model(batch["images"])  # [1,T,K,3]

        pred_pose = pred_pose[0]   # [T,K,3]
        gt_pose   = sample["pose"] # [T,K,3]

        for t in range(pred_pose.shape[0]):
            # ladda originalbild
            img_path = sample["meta"].get("img_path") if "img_path" in sample["meta"] else sample["images"][t]
            if isinstance(img_path, str) and os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                # fallback: använd den redan transformerade tensorbilden
                img = (sample["images"][t].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

            # Rita GT (grön) och pred (röd)
            img = draw_skeleton(img.copy(), gt_pose[t], ds.bones, color=(0,255,0))
            img = draw_skeleton(img, pred_pose[t], ds.bones, color=(0,0,255))

            out_path = os.path.join(args.out_dir, f"sample{idx:03d}_frame{t:03d}.png")
            cv2.imwrite(out_path, img)
            print(f"[viz] saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--bones_txt", type=str, default="ai/data/def_bones.txt")
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--ckpt", type=str, required=True, help="Path till modellens checkpoint (.pt)")
    ap.add_argument("--out_dir", type=str, default="ai/outputs/overlays_pred")
    ap.add_argument("--num_samples", type=int, default=5, help="Antal sekvenser att visualisera")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
