# dataset_pipeline/build_dataset/export_dataset.py
# Exporterar dataset av syntetiska frames (flat + materials) till CSV och COCO JSON
import os
import json
import argparse
import csv
from glob import glob


def collect_data(root: str):
    """Samla ihop alla frames och annotationer från datasetets actions.
    Exporterar både flat- och materials-bilder för varje label.
    """
    dataset = []
    for action in sorted(os.listdir(root)):
        labels_dir = os.path.join(root, action, "labels")
        if not os.path.isdir(labels_dir):
            continue

        for jp in sorted(glob(os.path.join(labels_dir, "*.json"))):
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            frame = data.get("frame")
            cam = data.get("camera")
            img_w, img_h = data.get("image_size", [0, 0])
            img_name = os.path.basename(jp)[:-5] + ".png"  # t.ex. f00017_c33.png

            # Exportera för både "flat" och "materials"
            for variant in ["flat", "materials"]:
                image_file_abs = os.path.join(root, action, "images", variant, img_name)
                image_file_rel = os.path.relpath(image_file_abs, root).replace("\\", "/")

                dataset.append(dict(
                    action=action,
                    variant=variant,
                    frame=frame,
                    camera=cam,
                    image_path=image_file_rel,
                    image_width=img_w,
                    image_height=img_h,
                    bones=data.get("bones", {})
                ))
    return dataset


def export_csv(dataset, out_path: str):
    """Exportera dataset till CSV (en rad per bild)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = list(dataset[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"[export_dataset] CSV exporterat till {out_path}, {len(dataset)} rader")


def export_coco(dataset, def_bones_path: str, out_path: str):
    """Exportera dataset till COCO-format JSON (alla keypoints per bild)."""
    with open(def_bones_path, "r", encoding="utf-8") as f:
        kp_names = [ln.strip() for ln in f if ln.strip()]
    K = len(kp_names)

    images, annotations = [], []
    img_id_counter = 1
    ann_id = 1

    for row in dataset:
        image_id = img_id_counter
        images.append({
            "id": image_id,
            "file_name": row["image_path"],
            "width": int(row.get("image_width", 0)),
            "height": int(row.get("image_height", 0)),
            "action": row["action"],
            "variant": row["variant"],
            "frame": row["frame"],
            "camera": row["camera"]
        })
        img_id_counter += 1

        # Samla keypoints i def_bones-ordning
        kpts = []
        num_kpts = 0
        bones = row["bones"]
        for name in kp_names:
            b = bones.get(name)
            if b and "uv" in b:
                u, v = float(b["uv"][0]), float(b["uv"][1])
                vis = 2 if b.get("in_frame", True) else 0
                if vis == 0:
                    u, v = 0.0, 0.0
            else:
                u, v, vis = 0.0, 0.0, 0
            kpts.extend([u, v, vis])
            if vis > 0:
                num_kpts += 1

        # Bbox
        xs = [kpts[i] for i in range(0, len(kpts), 3) if kpts[i+2] > 0]
        ys = [kpts[i+1] for i in range(0, len(kpts), 3) if kpts[i+2] > 0]
        if xs and ys:
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
            bbox = [float(x0), float(y0), float(w), float(h)]
            area = float(w * h)
        else:
            bbox, area = [0, 0, 0, 0], 0.0

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": kpts,
            "num_keypoints": num_kpts,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "horse",
            "supercategory": "animal",
            "keypoints": kp_names,
            "skeleton": []
        }]
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[export_dataset] COCO JSON exporterat till {out_path}, "
          f"{len(images)} bilder, {len(annotations)} annoteringar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Rotmapp till datasetet (t.ex. dataset_pipeline/data/dataset/final)")
    ap.add_argument("--format", choices=["csv", "coco", "both"], default="both",
                    help="Exportformat (csv, coco eller both)")
    ap.add_argument("--out", default=None,
                    help="Basnamn för utfil (utan ändelse). Om None -> sparas i <root>/coco_files/coco_synth_68")
    ap.add_argument("--def_bones", type=str,
                    default="dataset_pipeline/data/dataset_exports/def_bones.txt",
                    help="Path till def_bones.txt")
    args = ap.parse_args()

    dataset = collect_data(args.root)
    if not dataset:
        print(f"[export_dataset] Inga data hittades i {args.root}")
        return

    # Standard: spara i <root>/coco_files/coco_synth_68
    if args.out is None:
        out_base = os.path.join(args.root, "coco_files", "coco_synth_68")
    else:
        out_base = args.out

    if args.format in ["csv", "both"]:
        export_csv(dataset, f"{out_base}.csv")
    if args.format in ["coco", "both"]:
        export_coco(dataset, args.def_bones, f"{out_base}.json")


if __name__ == "__main__":
    main()
