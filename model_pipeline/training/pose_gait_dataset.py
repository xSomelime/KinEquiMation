# ai/models/dataset.py - Load dataset from files
import os, glob, json
from typing import List, Dict, Optional, Tuple, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

GAIT_CANON = [
    "idlebase","idlerest","idle","walk","trot",
    "leftcanter","rightcanter","leftgallop","rightgallop","jump","base"
]

# ----------------- helpers -----------------
def _load_def_bones(def_path: str) -> Optional[List[str]]:
    if os.path.isfile(def_path):
        with open(def_path, "r", encoding="utf-8") as f:
            bones = [ln.strip() for ln in f if ln.strip()]
        bones = [b for b in bones if b.startswith("DEF-")]
        return bones or None
    return None

def _extract_def_heads(bones_block: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, info in bones_block.items():
        if not name.startswith("DEF-"):
            continue
        head = info.get("head")
        if isinstance(head, (list, tuple)) and len(head) >= 3:
            out[name] = [float(head[0]), float(head[1]), float(head[2])]
    return out

def _read_json_one(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    action = obj.get("action")
    bones  = obj.get("bones", {})
    xyz    = _extract_def_heads(bones)
    intr   = obj.get("camera_intrinsics", {})
    extr   = obj.get("camera_extrinsics", {}).get("matrix_world", None)
    if extr is None:
        extr = np.eye(4)
    else:
        extr = np.array(extr, dtype=float)

    frame   = int(obj.get("frame", 0))
    camera  = obj.get("camera", "")
    return action, xyz, intr, extr, frame, camera

def _make_img_filename(frame: int, camera: str) -> str:
    """
    Bygger filnamn för en given frame + kamera.
    Exempel: frame=1, camera='Walk_Cam_09' -> 'f00001_c09.png'
    """
    cam_id = 0
    if "_" in camera:
        try:
            cam_id = int(camera.split("_")[-1])
        except:
            cam_id = 0
    return f"f{frame:05d}_c{cam_id:02d}.png"

def _find_image_for_label(clip_dir: str, fname: str) -> Optional[str]:
    # slumpa mellan materials och flat
    variants = ["materials", "flat"]
    random.shuffle(variants)  # blanda ordningen
    for v in variants:
        cand = os.path.join(clip_dir, "images", v, fname)
        if os.path.exists(cand):
            return cand
    return None


# ----------------- Dataset -----------------
class PoseGaitSequenceDataset(Dataset):
    """
    Läser sekvenser från:
      data/dataset/final/<clip>/{images/{materials|flat}/, labels/*.json}
    Matchning: <clip>/labels/frame.json ↔ <clip>/images/.../f00001_c00.png
    Return:
      images: [T,C,H,W]
      pose:   [T,K,3]
      gait:   int
      meta:   dict med img_paths, intrinsics, extrinsics
    """
    def __init__(self, project_root: str, bones_txt: str = "ai/data/def_bones.txt",
                 image_size: int = 384, seq_len: int = 8, stride: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.stride  = stride
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        search_root = os.path.join(project_root, "data", "dataset", "final")
        # tuple = (image_path, clip_dir, per_bone_xyz, gait_label_str, intrinsics, extrinsics)
        self.samples: List[Tuple[str, str, Dict[str, List[float]], str, Dict[str,float], np.ndarray]] = []

        total_pairs = 0
        for clip_dir in sorted(glob.glob(os.path.join(search_root, "*"))):
            if not os.path.isdir(clip_dir): 
                continue
            gait_from_dir = os.path.basename(clip_dir).lower()
            labels_dir = os.path.join(clip_dir, "labels")
            if not os.path.isdir(labels_dir):
                continue

            json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
            if not json_files:
                continue

            paired = 0
            for jp in json_files:
                action, xyz, intr, extr, frame, camera = _read_json_one(jp)
                if not xyz: 
                    continue
                fname = _make_img_filename(frame, camera)
                imgp = _find_image_for_label(clip_dir, fname)
                if not imgp:
                    continue
                gait = action.lower() if isinstance(action, str) else gait_from_dir
                self.samples.append((imgp, clip_dir, xyz, gait, intr, extr))
                paired += 1
                total_pairs += 1

            print(f"[dataset] {os.path.basename(clip_dir)}: matched {paired} image/label pairs")

        bones_path = bones_txt if os.path.isabs(bones_txt) else os.path.join(project_root, bones_txt)
        bones = _load_def_bones(bones_path)
        if bones is None:
            inferred = set()
            for _img, _cd, xyz, _g, _intr, _extr in self.samples:
                inferred |= set([k for k in xyz.keys() if k.startswith("DEF-")])
                if inferred: 
                    break
            bones = sorted(inferred)
            print(f"[dataset] Ingen def_bones.txt – infererar DEF-bensordning ({len(bones)}).")
        self.bones = bones
        self.K = len(bones)

        per_clip: Dict[str, List[int]] = {}
        for i, (imgp, clip_dir, _xyz, _g, _intr, _extr) in enumerate(self.samples):
            per_clip.setdefault(clip_dir, []).append(i)

        for clip_dir, idxs in per_clip.items():
            idxs.sort(key=lambda i: os.path.basename(self.samples[i][0]))
            per_clip[clip_dir] = idxs

        self.index: List[List[int]] = []
        for idxs in per_clip.values():
            n = len(idxs)
            if n < self.seq_len: 
                continue
            for s in range(0, n - self.seq_len + 1, self.stride):
                self.index.append(idxs[s:s+self.seq_len])

        uniq_gaits = []
        for _img, _cd, _xyz, gait, _intr, _extr in self.samples:
            if gait not in uniq_gaits:
                uniq_gaits.append(gait)

        ordered = [g for g in GAIT_CANON if g in uniq_gaits] + [g for g in uniq_gaits if g not in GAIT_CANON]
        self.gait_to_id = {g:i for i,g in enumerate(ordered)}
        self.num_gaits  = len(self.gait_to_id)

        print(f"[dataset] Total matched pairs: {total_pairs}")
        print(f"[dataset] Gaits: {self.gait_to_id} | K={self.K} (DEF bones) | windows={len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        window = self.index[i]
        X, Y = [], []
        first_imgp, _cd, _xyz, gait, intr, extr = self.samples[window[0]]
        img_paths = []
        for j in window:
            imgp, _clip_dir, xyz_map, _g, _intr, _extr = self.samples[j]
            img_paths.append(imgp)
            im = Image.open(imgp).convert("RGB")
            X.append(self.tf(im))
            xyz = torch.zeros(self.K, 3, dtype=torch.float32)
            for k, bone in enumerate(self.bones):
                v = xyz_map.get(bone)
                if v and len(v) >= 3:
                    xyz[k,0] = float(v[0]); xyz[k,1] = float(v[1]); xyz[k,2] = float(v[2])
            Y.append(xyz)

        images = torch.stack(X, dim=0)     # [T,C,H,W]
        pose   = torch.stack(Y, dim=0)     # [T,K,3]
        gait_id = torch.tensor(self.gait_to_id[gait], dtype=torch.long)

        return {
            "images": images,
            "pose": pose,
            "gait": gait_id,
            "meta": {
                "img_paths": img_paths,
                "intrinsics": intr,
                "extrinsics": extr
            }
        }
