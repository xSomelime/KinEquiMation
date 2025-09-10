# dataset_pipeline/build_dataset/build_metainfo.py
# Kombinerar def_bones.txt och skeleton_edges.json till både JSON och Python metainfo för MMPose

import os
import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def_bones", type=str, required=True, help="Path to def_bones.txt")
    ap.add_argument("--skeleton", type=str, required=True, help="Path to skeleton_edges.json")
    ap.add_argument("--out_json", type=str, required=True, help="Output JSON path (metainfo)")
    ap.add_argument("--out_py", type=str, required=True, help="Output Python path (dataset_info)")
    args = ap.parse_args()

    # ---- Läs in nyckelpunkternas namn ----
    with open(args.def_bones, "r", encoding="utf-8") as f:
        keypoint_names = [ln.strip() for ln in f if ln.strip()]
    name_to_idx = {n: i for i, n in enumerate(keypoint_names)}

    # ---- Läs in skeleton edges (parent → child) ----
    with open(args.skeleton, "r", encoding="utf-8") as f:
        edges_names = json.load(f).get("edges", [])

    skeleton = []
    for pa, ch in edges_names:
        if pa in name_to_idx and ch in name_to_idx:
            skeleton.append([name_to_idx[pa], name_to_idx[ch]])  


    # ---- JSON-metainfo (för coco_synth_68.json) ----
    metainfo_json = {
        "dataset_name": "horse68",
        "keypoint_names": keypoint_names,
        "skeleton": skeleton,
        "joint_weights": [1.0] * len(keypoint_names),
        "flip_pairs": []  # TODO: kan fyllas i med speglande ben om du vill
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(metainfo_json, f, ensure_ascii=False, indent=2)

    print(f"[OK] Skrev JSON-metainfo: {args.out_json}")
    print(f"- {len(keypoint_names)} keypoints")
    print(f"- {len(skeleton)} edges")

    # ---- Python-metainfo (för MMPose horse68.py) ----
    keypoint_info = {}
    for i, name in enumerate(keypoint_names):
        side = "upper"
        if any(s in name for s in [".L", ".R"]):
            side = "lower" if "leg" in name or "hoof" in name or "toe" in name or "pelvis" in name else "upper"
        swap = ""
        if name.endswith(".L"):
            swap = name.replace(".L", ".R")
        elif name.endswith(".R"):
            swap = name.replace(".R", ".L")
        elif ".L." in name:
            swap = name.replace(".L.", ".R.")
        elif ".R." in name:
            swap = name.replace(".R.", ".L.")
        keypoint_info[i] = dict(
            name=name,
            id=i,
            color=[0, 255, 0],
            type=side,
            swap=swap
        )

    skeleton_info = {i: {"link": [pa, ch], "id": i, "color": [255, 0, 0]}
                     for i, (pa, ch) in enumerate(skeleton)}

    metainfo_py = f"""# AUTO-GENERATED FILE: horse68.py
# Correct metainfo for horse68 (from def_bones + skeleton_edges)

dataset_info = dict(
    dataset_name='horse68',
    paper_info=dict(
        author='KinEquiMation',
        title='Horse Pose Estimation (68 keypoints)',
        year=2025,
        homepage='https://github.com/xSomelime/KinEquiMation',
    ),
    keypoint_info={json.dumps(keypoint_info, indent=4)},
    skeleton_info={json.dumps(skeleton_info, indent=4)},
    joint_weights={[1.0] * len(keypoint_names)},
    sigmas={[0.025] * len(keypoint_names)},
)
"""

    os.makedirs(os.path.dirname(args.out_py), exist_ok=True)
    with open(args.out_py, "w", encoding="utf-8") as f:
        f.write(metainfo_py)

    print(f"[OK] Skrev Python-metainfo: {args.out_py}")

if __name__ == "__main__":
    main()
