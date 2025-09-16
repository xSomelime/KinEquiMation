# model_pipeline/datasets/lifter_dataset.py

import json
import torch
from torch.utils.data import Dataset


class LifterDataset(Dataset):
    """
    Dataset för 3D-lifter-träning.
    Läser sekvenser från lifter_dataset.json och returnerar:
        - keypoints_2d: [T,K,2]
        - keypoints_3d: [T,K,3]
        - action: gångart (om finns)
        - image_paths: lista med paths för sekvensens frames (om finns)
    """

    def __init__(self, ann_file, seq_len=27):
        super().__init__()
        with open(ann_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 2D och 3D keypoints
        kps2d = torch.tensor(item["keypoints_2d"], dtype=torch.float32)  # [T,K,2]
        kps3d = torch.tensor(item["keypoints_3d"], dtype=torch.float32)  # [T,K,3]

        out = {
            "keypoints_2d": kps2d,
            "keypoints_3d": kps3d,
        }

        # Metadata (om tillgängligt)
        if "action" in item:
            out["action"] = item["action"]

        if "image_paths" in item:
            out["image_paths"] = item["image_paths"]

        if "labels" in item:
            out["labels"] = item["labels"]


        return out
