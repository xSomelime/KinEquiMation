# dataset_pipeline/build_dataset/export_lifter_dataset.py
# Bygger träningsdataset för 3D-liftern (head+tail)

import os, json, glob, argparse
from tqdm import tqdm


def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def norm_path(path: str) -> str:
    return path.replace("\\", "/")


def collect_sequences(dataset_root, def_bones, bone_order,
                      seq_len=27, render_width=512, render_height=512,
                      mode="train"):
    # mapping def_index -> bone_order_index
    name_to_def_idx = {name: i for i, name in enumerate(def_bones)}
    index_map = [name_to_def_idx[name] for name in bone_order if name in name_to_def_idx]

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

            if mode == "full" and not str(meta.get("camera", "")).endswith("Cam_00"):
                continue

            bones = meta.get("bones", {})

            # Action
            action = meta.get("action", clip_name)
            if current_action is None:
                current_action = action

            # Bildnamn
            base = os.path.splitext(os.path.basename(jp))[0]
            img_name = base + ".png"
            img_rel = {
                "materials": norm_path(os.path.join(clip_name, "images", "materials", img_name)),
                "flat": norm_path(os.path.join(clip_name, "images", "flat", img_name)),
            }
            label_rel = norm_path(os.path.join(clip_name, "labels", base + ".json"))

            # Extrahera head+tail i def_bones-ordning
            kps2d_def, kps3d_def = [], []
            for name in def_bones:
                b = bones.get(name, {})

                # Head
                if "uv" in b:
                    u = float(b["uv"][0]) / render_width
                    v = float(b["uv"][1]) / render_height
                    vis = 2 if b.get("in_frame", True) else 0
                else:
                    u, v, vis = 0.0, 0.0, 0
                if "head" in b:
                    x, y, z = map(float, b["head"])
                else:
                    x, y, z = 0.0, 0.0, 0.0
                kps2d_def.append([u, v, vis])
                kps3d_def.append([x, y, z])

                # Tail
                if "uv_tail" in b:
                    u = float(b["uv_tail"][0]) / render_width
                    v = float(b["uv_tail"][1]) / render_height
                    vis = 2 if b.get("in_frame_tail", True) else 0
                else:
                    u, v, vis = 0.0, 0.0, 0
                if "tail" in b:
                    x, y, z = map(float, b["tail"])
                else:
                    x, y, z = 0.0, 0.0, 0.0
                kps2d_def.append([u, v, vis])
                kps3d_def.append([x, y, z])

            # Mappa om (nu head+tail per bone)
            kps2d = [kps2d_def[idx*2 + off] for idx in index_map for off in (0,1)]
            kps3d = [kps3d_def[idx*2 + off] for idx in index_map for off in (0,1)]

            frames_2d.append(kps2d)
            frames_3d.append(kps3d)
            frames_img.append(img_rel)
            frames_labels.append(label_rel)

            if mode == "train" and len(frames_2d) == seq_len:
                sequences.append({
                    "keypoints_2d": list(frames_2d),
                    "keypoints_3d": list(frames_3d),
                    "action": current_action,
                    "image_paths": list(frames_img),
                    "labels": list(frames_labels),
                })
                frames_2d, frames_3d, frames_img, frames_labels = \
                    frames_2d[1:], frames_3d[1:], frames_img[1:], frames_labels[1:]
                current_action = action

        if mode == "full" and len(frames_2d) > 0:
            print(f"[INFO] Exporterar {len(frames_2d)} frames för '{clip_name}' med kamera *Cam_00")
            sequences.append({
                "keypoints_2d": list(frames_2d),
                "keypoints_3d": list(frames_3d),
                "action": current_action,
                "image_paths": list(frames_img),
                "labels": list(frames_labels),
            })

    return sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--def_bones", type=str, required=True)
    ap.add_argument("--bone_order", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=27)
    ap.add_argument("--mode", type=str, choices=["train", "full"], default="train")
    args = ap.parse_args()

    def_bones = load_list(args.def_bones)
    bone_order = load_list(args.bone_order)
    sequences = collect_sequences(args.dataset_root, def_bones, bone_order,
                                  seq_len=args.seq_len, mode=args.mode)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_obj = {
        "bone_names": [bn + suf for bn in bone_order for suf in ("_head","_tail")],
        "seq_len": args.seq_len,
        "mode": args.mode,
        "sequences": sequences,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[OK] Exporterat {len(sequences)} sekvenser (mode={args.mode}) till {args.out}")


if __name__ == "__main__":
    main()
