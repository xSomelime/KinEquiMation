# model_pipeline/datasets/gait_dataset.py

import json
import torch
from torch.utils.data import Dataset


class GaitDataset(Dataset):
    """
    Dataset för gångartsigenkänning.
    Läser sekvenser från lifter_dataset.json och returnerar 3D-keypoints + gångartslabel.
    """

    def __init__(self, ann_file, seq_len=30, num_joints=68, in_features=3):
        super().__init__()
        self.ann_file = ann_file
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.in_features = in_features

        with open(ann_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Mappa actions (gångarter) till heltalsklasser
        actions = sorted(set(item["action"] for item in self.data))
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        self.idx_to_action = {i: a for a, i in self.action_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        kpts_3d = torch.tensor(item["keypoints_3d"], dtype=torch.float32)  # [T,K,3]

        # Klipp/padda till seq_len
        T, K, C = kpts_3d.shape
        if T >= self.seq_len:
            kpts_3d = kpts_3d[: self.seq_len]
        else:
            pad = torch.zeros(self.seq_len - T, K, C)
            kpts_3d = torch.cat([kpts_3d, pad], dim=0)

        # Flatten joints: [T, K*3]
        x = kpts_3d.view(self.seq_len, self.num_joints * self.in_features)

        # Label som int
        label = torch.tensor(self.action_to_idx[item["action"]], dtype=torch.long)

        return {
            "keypoints": x,   # [T, K*3]
            "label": label
        }

