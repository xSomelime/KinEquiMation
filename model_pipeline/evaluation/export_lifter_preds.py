# model_pipeline/evaluation/export_lifter_preds.py
# Exporterar 3D-predictions till JSON (rig_action_exports-kompatibelt format)
# - Root kopplas till ML-root (som i sanity checken)
# - Varje ben byggs från pred_head och pred_tail
# - Orientering beräknas från vektorn (tail - head), justerad med riggens restpose

import argparse
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from model_pipeline.models.lifter_3d import TemporalLifter
from model_pipeline.datasets.lifter_dataset import LifterDataset


DEBUG_BONES = {"DEF-spine.001", "DEF-forearm.L", "DEF-r_hoof.R"}


def run_model_on_full_seq(model, x):
    """Kör modellen på HELA sekvensen utan att dela upp i seq_len."""
    B, T, K, C = x.shape
    with torch.no_grad():
        preds = model(x.view(B, T, -1))  # [B,T,K,3]
    return preds


def make_matrix_from_head_tail(head, tail, rest_axes=None):
    """
    Bygg en 4x4 matrix som placerar bone vid head och pekar mot tail.
    - Y-axeln: riktning från head→tail (Blender-konvention)
    - X/Z-axlar: hämtas från restpose om tillgängligt, projiceras så de blir ortogonala
    """
    v = tail - head
    v = v / (np.linalg.norm(v) + 1e-8)   # Y-axeln (forward i Blender)

    if rest_axes is not None:
        # Använd restpose-X som referens, projicera bort komponent längs Y
        x_axis = rest_axes[:, 0]
        x_axis = x_axis - np.dot(x_axis, v) * v
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0], dtype=np.float32)  # fallback
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, v)
    else:
        # fallback (utan riggdata)
        up = np.array([0, 0, 1], dtype=np.float32)
        if abs(np.dot(v, up)) > 0.9:
            up = np.array([1, 0, 0], dtype=np.float32)
        x_axis = np.cross(v, up); x_axis /= (np.linalg.norm(x_axis) + 1e-8)
        z_axis = np.cross(x_axis, v)

    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = x_axis   # X
    M[:3, 1] = v        # Y = head→tail
    M[:3, 2] = z_axis   # Z
    M[:3, 3] = head
    return M



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # === Hitta checkpoint ===
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.save_dir).rglob("best.pth").__next__()
    print(f"[export] Använder checkpoint: {ckpt_path}")

    # === Hämta timestamp från run-mappen ===
    run_dir = ckpt_path.parent.parent
    timestamp = run_dir.name

    # === Dataset ===
    dataset = LifterDataset(ann_file=args.data, full_seq=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    bone_names = dataset.bone_names
    print(f"[export] Dataset innehåller {len(bone_names)} joints (head/tail)")

    # === Ladda rig-metadata (för restpose-orientering) ===
    rig_meta_path = Path("outputs/animation_pipeline/skeleton_edges_from_rig.json")
    with open(rig_meta_path, "r", encoding="utf-8") as f:
        rig_meta_full = json.load(f)
    rig_meta = rig_meta_full["bones"]

    # === Skapa bone-par (basename -> (head_idx, tail_idx)) ===
    bone_pairs = {}
    for i, name in enumerate(bone_names):
        if name.endswith("_head"):
            base = name[:-5]
            tail_name = base + "_tail"
            if tail_name in bone_names:
                bone_pairs[base] = (i, bone_names.index(tail_name))

    print(f"[export] Bygger {len(bone_pairs)} bone-par")

    # === Modell ===
    model = TemporalLifter(num_joints=len(bone_names), in_features=3, hidden_dim=1024, num_blocks=3)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # === Export root ===
    export_root = Path("outputs/lifter_preds")
    export_root.mkdir(parents=True, exist_ok=True)

    # === Exportera sekvenser ===
    for idx, batch in enumerate(dataloader):
        x = batch["keypoints_2d"].to(device)   # [1,T,K,3]
        action = batch.get("action", f"seq{idx}")
        if isinstance(action, (list, tuple)):
            action = action[0]

        B, T, K, C = x.shape
        preds = run_model_on_full_seq(model, x)[0].cpu().numpy()  # [T,K,3]

        pred_name = f"pred_{action}_{timestamp}"
        seq_dir = export_root / pred_name
        seq_dir.mkdir(parents=True, exist_ok=True)

        print(f"[export] Exporterar {T} frames för action '{action}' → {seq_dir}")

        # === Frame loop ===
        for t in range(T):
            frame_idx = t + 1
            bones_payload = {}

            # Root från ML-preds om finns
            if "root_head" in bone_names:
                root_idx = bone_names.index("root_head")
                root_pos = preds[t, root_idx]
            else:
                root_pos = np.zeros(3, dtype=np.float32)

            root_matrix_world = np.eye(4, dtype=np.float32)
            root_matrix_world[:3, 3] = root_pos

            for base, (h_idx, t_idx) in bone_pairs.items():
                pred_head = preds[t, h_idx]
                pred_tail = preds[t, t_idx]

                # Hämta restpose-orientering om finns
                static = rig_meta.get(base, {})
                rest = np.array(static.get("matrix_local", np.eye(4)), dtype=np.float32)
                rest_axes = rest[:3, :3]

                mat_world = make_matrix_from_head_tail(pred_head, pred_tail, rest_axes)
                mat_rel_root = np.linalg.inv(root_matrix_world) @ mat_world

                bones_payload[base] = {
                    "head": pred_head.tolist(),
                    "tail": pred_tail.tolist(),
                    "matrix_world": mat_world.tolist(),
                    "matrix_rel_root": mat_rel_root.tolist(),
                }

                if base in DEBUG_BONES and frame_idx <= 2:
                    print(f"\n[DEBUG] Frame {frame_idx} Bone {base}")
                    print("  pred_head:", pred_head)
                    print("  pred_tail:", pred_tail)
                    print("  mat_world:\n", mat_world)

            payload = {
                "action": f"{action}_rigexport_rootrel",
                "frame": frame_idx,
                "armature": "ML_rig",
                "root_matrix_world": root_matrix_world.tolist(),
                "bones": bones_payload,
            }

            outpath = seq_dir / f"{frame_idx:06d}.json"
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        # === Meta-fil ===
        meta_out = {
            "action": action,
            "num_frames": T,
            "bone_pairs": list(bone_pairs.keys()),
            "format": "rig_action_exports-compatible",
            "timestamp": timestamp,
        }
        with open(seq_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2)

        print(f"[export] Sparade {T} frames + meta → {seq_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    default="dataset_pipeline/data/dataset_exports/lifter_dataset_full.json")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="outputs/lifter_runs")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    main(args)
