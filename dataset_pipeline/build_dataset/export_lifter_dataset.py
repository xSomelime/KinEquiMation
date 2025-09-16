# dataset_pipeline/build_dataset/export_lifter_dataset.py
# Bygger träningsdataset för 3D-liftern:
# Tar in frame-JSONs (med bones.uv och bones.head) och exporterar sekvenser [T,K,3] (2D: u,v,vis) + [T,K,3] (3D: x,y,z) + action + image_paths + labels.

import os, json, glob, argparse
from tqdm import tqdm


def load_def_bones(def_bones_path):
    with open(def_bones_path, "r", encoding="utf-8") as f:
        kp_names = [ln.strip() for ln in f if ln.strip()]
    return kp_names


def collect_sequences(dataset_root, kp_names, seq_len=27, render_width=512, render_height=512):
    """
    Loopar igenom alla klipp → bygger sekvenser av längd T=seq_len.
    Varje sekvens innehåller:
        - keypoints_2d [T,K,3]  (u,v,vis), normaliserat till [0,1]
        - keypoints_3d [T,K,3]  (x,y,z)
        - action (gångart/etikett)
        - image_paths [T] (lista med dicts {"materials":..., "flat":...})
        - labels [T] (relativ path till label JSON)
    """
    sequences = []
    clips = [d for d in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(d)]
    if not clips:
        raise RuntimeError(f"Inga klipp hittades i {dataset_root}")

    for clip_dir in tqdm(sorted(clips), desc="Clips"):
        labels_dir = os.path.join(clip_dir, "labels")
        jfiles = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
        if not jfiles:
            continue

        frames_2d, frames_3d, frames_img, frames_labels = [], [], [], []
        current_action = None
        clip_name = os.path.basename(clip_dir)

        for jp in jfiles:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bones = meta.get("bones", {})

            # Hämta action (från metadata eller clip-namn)
            action = meta.get("action", clip_name)
            if current_action is None:
                current_action = action

            # Bildnamn (matchar JSON-frame)
            base = os.path.splitext(os.path.basename(jp))[0]  # t.ex. f00001_c00
            img_name = base + ".png"
            img_rel = {
                "materials": os.path.join(clip_name, "images", "materials", img_name),
                "flat": os.path.join(clip_name, "images", "flat", img_name),
            }
            label_rel = os.path.join(clip_name, "labels", base + ".json")

            # Extrahera keypoints
            kps2d, kps3d = [], []
            for name in kp_names:
                b = bones.get(name, {})
                # 2D keypoints med visibility
                if "uv" in b:
                    u = float(b["uv"][0]) / render_width
                    v = float(b["uv"][1]) / render_height
                    vis = 2 if b.get("in_frame", True) else 0
                else:
                    u, v, vis = 0.0, 0.0, 0
                kps2d.append([u, v, vis])
                # 3D keypoints
                if "head" in b:
                    x, y, z = map(float, b["head"])
                else:
                    x, y, z = 0.0, 0.0, 0.0
                kps3d.append([x, y, z])

            frames_2d.append(kps2d)
            frames_3d.append(kps3d)
            frames_img.append(img_rel)
            frames_labels.append(label_rel)

            # När vi nått en full sekvens → spara
            if len(frames_2d) == seq_len:
                sequences.append({
                    "keypoints_2d": list(frames_2d),
                    "keypoints_3d": list(frames_3d),
                    "action": current_action,
                    "image_paths": list(frames_img),
                    "labels": list(frames_labels)   # <-- NYTT
                })
                # Sliding window
                frames_2d = frames_2d[1:]
                frames_3d = frames_3d[1:]
                frames_img = frames_img[1:]
                frames_labels = frames_labels[1:]
                current_action = action  # sätt om för nästa sekvens

    return sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Path till dataset/final directory (med action/clip/labels)")
    ap.add_argument("--def_bones", type=str, required=True,
                    help="Path till def_bones.txt (ordning på keypoints)")
    ap.add_argument("--out", type=str, required=True,
                    help="Path till output JSON (t.ex. dataset_pipeline/data/dataset_exports/lifter_dataset.json)")
    ap.add_argument("--seq_len", type=int, default=27, help="Sekvenslängd")
    args = ap.parse_args()

    kp_names = load_def_bones(args.def_bones)
    sequences = collect_sequences(args.dataset_root, kp_names, seq_len=args.seq_len)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sequences, f)

    print(f"[OK] Exporterat {len(sequences)} sekvenser till {args.out}")


if __name__ == "__main__":
    main()
