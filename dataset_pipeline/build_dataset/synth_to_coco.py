# tools/synth_to_coco.py
import os, json, glob, argparse
from collections import defaultdict

def find_image(clip_dir, base, prefer="materials"):
    # 1) prefererad variant
    p1 = os.path.join(clip_dir, "images", prefer, base + ".png")
    if os.path.exists(p1): return p1
    # 2) alternativ variant
    alt = "materials" if prefer=="flat" else "flat"
    p2 = os.path.join(clip_dir, "images", alt, base + ".png")
    if os.path.exists(p2): return p2
    # 3) fallback: sök djupare
    hits = glob.glob(os.path.join(clip_dir, "images", "**", base + ".png"), recursive=True)
    return hits[0] if hits else None

def bbox_from_kpts(kpts_xyv):
    xs, ys = [], []
    for i in range(0, len(kpts_xyv), 3):
        x, y, v = kpts_xyv[i], kpts_xyv[i+1], kpts_xyv[i+2]
        if v > 0 and x > 0 and y > 0:
            xs.append(x); ys.append(y)
    if not xs:
        return [0, 0, 0, 0], 0.0
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    # liten padding
    pad = 0.05
    x0 = x0 - pad*w; y0 = y0 - pad*h
    w = w * (1 + 2*pad); h = h * (1 + 2*pad)
    return [float(x0), float(y0), float(w), float(h)], float(w*h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--def_bones", type=str, required=True)
    ap.add_argument("--skeleton", type=str, required=True)
    ap.add_argument("--prefer", type=str, default="materials", choices=["materials","flat"])
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # Läs ordning (keypoint-namn)
    with open(args.def_bones, "r", encoding="utf-8") as f:
        kp_names = [ln.strip() for ln in f if ln.strip()]
    K = len(kp_names)
    name_to_idx = {n:i for i,n in enumerate(kp_names)}

    # Läs kanter (parent->child) och mappa till 1-baserade index för COCO
    with open(args.skeleton, "r", encoding="utf-8") as f:
        skel = json.load(f)
    edges_names = skel.get("edges", [])
    skeleton = []
    for pa, ch in edges_names:
        if pa in name_to_idx and ch in name_to_idx:
            skeleton.append([name_to_idx[pa]+1, name_to_idx[ch]+1])

    # Skanna dataset
    search_root = os.path.join(args.project_root, "data", "dataset", "final")
    clips = [d for d in glob.glob(os.path.join(search_root, "*")) if os.path.isdir(d)]
    assert clips, f"Inga clips i {search_root}"

    images, annotations = [], []
    img_id = 1
    ann_id = 1

    for clip_dir in sorted(clips):
        labels_dir = os.path.join(clip_dir, "labels")
        jfiles = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
        if not jfiles: 
            continue

        for jp in jfiles:
            base = os.path.splitext(os.path.basename(jp))[0]  # t.ex. 'f00012_c03'
            imgp = find_image(clip_dir, base, prefer=args.prefer)
            if not imgp:
                continue

            with open(jp, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Bildstorlek
            if "image_size" in meta and isinstance(meta["image_size"], list):
                w, h = meta["image_size"][0], meta["image_size"][1]
            else:
                intr = meta.get("camera_intrinsics", {})
                w, h = intr.get("width", 0), intr.get("height", 0)

            # Keypoints i K-ordningen
            bones = meta.get("bones", {})
            kpts = []
            for name in kp_names:
                b = bones.get(name)
                if b and "uv" in b and isinstance(b["uv"], list) and len(b["uv"]) >= 2:
                    u, v = float(b["uv"][0]), float(b["uv"][1])
                    vis = 2 if bool(b.get("in_frame", True)) else 0
                    if vis == 0:
                        u, v = 0.0, 0.0
                else:
                    u, v, vis = 0.0, 0.0, 0
                kpts.extend([u, v, vis])

            bbox, area = bbox_from_kpts(kpts)

            images.append({
                "id": img_id,
                "file_name": os.path.abspath(imgp),
                "width": int(w), "height": int(h)
            })
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "num_keypoints": int(sum(1 for i in range(2, 3*K, 3) if kpts[i] > 0)),
                "keypoints": kpts,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            img_id += 1
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "horse",
            "supercategory": "animal",
            "keypoints": kp_names,
            "skeleton": skeleton
        }]
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[OK] COCO sparad: {args.out}")
    print(f"images={len(images)} annotations={len(annotations)} K={K} edges={len(skeleton)}")

if __name__ == "__main__":
    main()
