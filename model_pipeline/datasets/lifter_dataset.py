# model_pipeline/datasets/lifter_dataset.py

import json
import torch
from torch.utils.data import Dataset


class LifterDataset(Dataset):
    """
    Dataset för 3D-lifter.
    Läser sekvenser från lifter_dataset.json och returnerar:
        - keypoints_2d: [T,K,3]
        - keypoints_3d: [T,K,3]
        - action: gångart (om finns)
        - image_paths: lista med paths för sekvensens frames (om finns)
        - labels: paths till JSON-labels (för kamera/GT-lookup i viz)

    Parametrar:
        ann_file: path till lifter_dataset.json
        seq_len: fönsterlängd (vid träning). Om None → använd default från datasetet.
        full_seq: om True → returnera hela sekvensen (ignorerar seq_len).
    """

    def __init__(self, ann_file, seq_len=None, full_seq=False):
        super().__init__()
        with open(ann_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Nyare format
        if "sequences" in raw:
            self.bone_names = raw.get("bone_names", [])
            self.default_seq_len = raw.get("seq_len", 27)
            self.data = raw["sequences"]
        else:
            # Bakåtkompatibilitet
            print("[WARN] Laddade dataset utan 'sequences'-nyckel – använder legacy-format.")
            self.bone_names = []
            self.default_seq_len = 27
            self.data = raw

        self.seq_len = seq_len if seq_len is not None else self.default_seq_len
        self.full_seq = full_seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        kps2d = torch.tensor(item["keypoints_2d"], dtype=torch.float32)  # [T,K,3]
        kps3d = torch.tensor(item["keypoints_3d"], dtype=torch.float32)  # [T,K,3]

        if not self.full_seq and self.seq_len and kps2d.shape[0] > self.seq_len:
            # Trimma/sliding window
            start = torch.randint(0, kps2d.shape[0] - self.seq_len + 1, (1,)).item()
            end = start + self.seq_len
            kps2d = kps2d[start:end]
            kps3d = kps3d[start:end]
            image_paths = item.get("image_paths", [])[start:end]
            labels = item.get("labels", [])[start:end]
        else:
            # Ta hela sekvensen
            image_paths = item.get("image_paths", [])
            labels = item.get("labels", [])

        out = {
            "keypoints_2d": kps2d,
            "keypoints_3d": kps3d,
            "action": item.get("action", ""),
            "image_paths": image_paths,
            "labels": labels,
        }

        return out
