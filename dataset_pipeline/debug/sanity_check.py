# dataset_pipeline/debug/sanity_check.py
# Check exported datasets (CSV, dataset.json, COCO JSON)

import csv
import json
import os
import argparse
from collections import Counter

def check_csv(path):
    print(f"\n=== Checking CSV: {path} ===")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Rows in CSV: {len(rows)}")

    # actions
    actions = Counter(r.get("action") for r in rows)
    print("Actions in CSV:", dict(actions))

    # key coordinates present?
    missing = [r for r in rows if not r.get("u") or not r.get("v") or not r.get("head_x")]
    if missing:
        print(f"⚠️ {len(missing)} rows in CSV are missing coordinates")
    else:
        print("✔ All CSV rows have coordinates")


def check_dataset_json(path):
    print(f"\n=== Checking dataset JSON: {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Top-level keys:", list(data.keys()))

    if isinstance(data, list):
        print(f"Contains {len(data)} frame entries")
        # look for actions
        actions = Counter(d.get("action") for d in data if "action" in d)
        print("Actions in dataset JSON:", dict(actions))
    else:
        print("Not a list of frames, might be another format.")


def check_coco_json(path):
    print(f"\n=== Checking COCO JSON: {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    cats = coco.get("categories", [])
    print(f"Images in JSON: {len(images)}")
    print(f"Annotations in JSON: {len(annotations)}")
    print(f"Categories: {[c.get('name') for c in cats]}")

    # action stats (if present in image metadata)
    actions = Counter(img.get("action") for img in images if "action" in img)
    if actions:
        print("Actions in COCO JSON:", dict(actions))
    else:
        print("ℹ️ No action labels in COCO JSON (normal for MMPose).")

    # check annotations
    bad_anns = [a for a in annotations if "keypoints" not in a]
    if bad_anns:
        print(f"⚠️ {len(bad_anns)} annotations missing keypoints")
    else:
        print("✔ All annotations have keypoints")

    # check images
    bad_imgs = [img for img in images if not img.get("width") or not img.get("height")]
    if bad_imgs:
        print(f"⚠️ {len(bad_imgs)} images missing width/height")
    else:
        print("✔ All images have width/height")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to dataset.csv")
    ap.add_argument("--dataset_json", type=str, help="Path to dataset.json (detailed format)")
    ap.add_argument("--coco_json", type=str, help="Path to coco_synth_XX.json (COCO format)")
    args = ap.parse_args()

    if args.csv and os.path.exists(args.csv):
        check_csv(args.csv)
    if args.dataset_json and os.path.exists(args.dataset_json):
        check_dataset_json(args.dataset_json)
    if args.coco_json and os.path.exists(args.coco_json):
        check_coco_json(args.coco_json)

    print("\n✔ Finished checking datasets")


if __name__ == "__main__":
    main()
