# dataset_pipeline/build_dataset/export_lifter_dataset.py
# Bygger träningsdataset för 3D-liftern:
# Tar in frame-JSONs (med bones.uv och bones.head) och exporterar sekvenser [T,K,2] + [T,K,3].

import os, json, glob, argparse
from tqdm import tqdm

def load_def_bones(def_bones_path):
    with open(def_bones_path, "r", encoding="utf-8") as f:
        kp_names = [ln.strip() for ln in f if ln.strip()]
    return kp_names

def collect_sequences(dataset_root, kp_names, seq_len=27):
    """Loopar igenom alla clips → gör sekvenser av längd T=seq_len"""
    sequences = []
    clips = [d for d in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(d)]
    if not clips:
        raise RuntimeError(f"Inga clips hittades i {dataset_root}")

    for clip_dir in tqdm(sorted(clips), desc="Clips"):
        labels_dir = os.path.join(clip_dir, "labels")
        jfiles = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
        if not jfiles:
            continue

        frames_2d, frames_3d = [], []

        for jp in jfiles:
            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            bones = meta.get("bones", {})

            kps2d, kps3d = [], []
            for name in kp_names:
                b = bones.get(name, {})
                # 2D keypoints
                if "uv" in b:
                    u, v = float(b["uv"][0]), float(b["uv"][1])
                else:
                    u, v = 0.0, 0.0
                kps2d.append([u, v])
                # 3D keypoints
                if "head" in b:
                    x, y, z = map(float, b["head"])
                else:
                    x, y, z = 0.0, 0.0, 0.0
                kps3d.append([x, y, z])

            frames_2d.append(kps2d)
            frames_3d.append(kps3d)

            # När vi nått en full sekvens → spara
            if len(frames_2d) == seq_len:
                sequences.append({
                    "keypoints_2d": frames_2d,
                    "keypoints_3d": frames_3d
                })
                # Skjut fönstret 1 frame framåt (overlap sliding window)
                frames_2d = frames_2d[1:]
                frames_3d = frames_3d[1:]

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
