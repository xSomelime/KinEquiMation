# tools/export_dataset.py - Export dataset of synthetic frames to csv and json
import os
import json
import argparse
import csv
from glob import glob

def collect_data(root):
    """Samla ihop all data från alla actions och frames"""
    dataset = []
    for action in sorted(os.listdir(root)):
        labels_dir = os.path.join(root, action, "labels")
        if not os.path.isdir(labels_dir):
            continue

        for jp in sorted(glob(os.path.join(labels_dir, "*.json"))):
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_file = jp.replace(os.sep + "labels" + os.sep,
                                    os.sep + "images" + os.sep)[:-5] + ".png"

            frame = data["frame"]
            cam = data["camera"]
            img_w, img_h = data.get("image_size", [None, None])

            for bone, info in data.get("bones", {}).items():
                row = {
                    "action": action,
                    "frame": frame,
                    "camera": cam,
                    "image_path": image_file,
                    "image_width": img_w,
                    "image_height": img_h,
                    "bone": bone,
                    "parent": info.get("parent"),
                    # 2D
                    "u": info.get("uv", [None, None])[0],
                    "v": info.get("uv", [None, None])[1],
                    "in_frame": int(info.get("in_frame", False)),
                    "u_tail": info.get("uv_tail", [None, None])[0],
                    "v_tail": info.get("uv_tail", [None, None])[1],
                    "in_frame_tail": int(info.get("in_frame_tail", False)),
                    "depth_cam": info.get("depth_cam", None),
                    # 3D
                    "head_x": info.get("head", [None, None, None])[0] if "head" in info else None,
                    "head_y": info.get("head", [None, None, None])[1] if "head" in info else None,
                    "head_z": info.get("head", [None, None, None])[2] if "head" in info else None,
                    "tail_x": info.get("tail", [None, None, None])[0] if "tail" in info else None,
                    "tail_y": info.get("tail", [None, None, None])[1] if "tail" in info else None,
                    "tail_z": info.get("tail", [None, None, None])[2] if "tail" in info else None,
                    # Rotation
                    "quat_w": info.get("quat", [None, None, None, None])[0] if "quat" in info else None,
                    "quat_x": info.get("quat", [None, None, None, None])[1] if "quat" in info else None,
                    "quat_y": info.get("quat", [None, None, None, None])[2] if "quat" in info else None,
                    "quat_z": info.get("quat", [None, None, None, None])[3] if "quat" in info else None,
                }
                dataset.append(row)
    return dataset

def export_csv(dataset, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = list(dataset[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"[export_dataset] CSV exporterat till {out_path}, {len(dataset)} rader")

def export_coco(dataset, out_path):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "horse", "supercategory": "animal"}]

    img_id_map = {}
    ann_id = 1
    img_id_counter = 1

    for row in dataset:
        image_path = row["image_path"]
        if image_path not in img_id_map:
            img_id_map[image_path] = img_id_counter
            images.append({
                "id": img_id_counter,
                "file_name": image_path,
                "width": row.get("image_width", 0),
                "height": row.get("image_height", 0),
                "action": row["action"],
                "frame": row["frame"],
                "camera": row["camera"]
            })
            img_id_counter += 1

        keypoints = [
            row["u"], row["v"], row["in_frame"]
        ]
        extra = {
            "bone": row["bone"],
            "parent": row["parent"],
            "u_tail": row["u_tail"], "v_tail": row["v_tail"], "in_frame_tail": row["in_frame_tail"],
            "depth_cam": row["depth_cam"],
            "head": [row["head_x"], row["head_y"], row["head_z"]],
            "tail": [row["tail_x"], row["tail_y"], row["tail_z"]],
            "quat": [row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]],
        }

        annotations.append({
            "id": ann_id,
            "image_id": img_id_map[image_path],
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": 1,
            "extra": extra
        })
        ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[export_dataset] COCO JSON exporterat till {out_path}, {len(images)} bilder, {len(annotations)} annoteringar")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Rotmapp till datasetet (t.ex. data/dataset/final)")
    ap.add_argument("--format", choices=["csv", "coco", "both"], default="csv",
                    help="Exportformat (csv, coco eller both)")
    ap.add_argument("--out", default="data/exports/dataset.csv",
                    help="Basnamn för utfil (utan ändelse vid --format both)")
    args = ap.parse_args()

    dataset = collect_data(args.root)
    if not dataset:
        print("[export_dataset] Inga data hittades i", args.root)
        return

    if args.format == "csv":
        export_csv(dataset, args.out)
    elif args.format == "coco":
        export_coco(dataset, args.out)
    elif args.format == "both":
        base, _ = os.path.splitext(args.out)
        export_csv(dataset, base + ".csv")
        export_coco(dataset, base + ".json")

if __name__ == "__main__":
    main()
