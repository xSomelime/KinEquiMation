# model_pipeline/models/lifter_3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class TemporalLifter(nn.Module):
    def __init__(self, num_joints=80, in_features=3,
                 hidden_dim=1024, num_blocks=3, kernel_size=3, dropout=0.25):
        super().__init__()
        self.input_dim = num_joints * in_features
        self.output_dim = num_joints * 3

        # Input projection (per frame → feature space)
        self.input_proj = nn.Conv1d(self.input_dim, hidden_dim, kernel_size=1)

        # Stack of temporal residual blocks with increasing dilation
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(TemporalBlock(hidden_dim, hidden_dim,
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        dropout=dropout))
        self.blocks = nn.Sequential(*blocks)

        # Output projection: hidden → 3D keypoints
        self.output_layer = nn.Conv1d(hidden_dim, self.output_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, T, K*in_features]  (t.ex. 2D keypoints över tid)
        Returns:
            y: [B, T, K, 3]  (predikterade 3D-koordinater)
        """
        B, T, D = x.shape
        x = x.permute(0, 2, 1)         # [B, D, T]
        x = self.input_proj(x)
        x = self.blocks(x)             # [B, hidden_dim, T]
        y = self.output_layer(x)       # [B, K*3, T]
        y = y.permute(0, 2, 1)         # [B, T, K*3]
        y = y.view(B, T, -1, 3)        # [B, T, K, 3]
        return y



if __name__ == "__main__":
    # Snabb testkörning
    model = TemporalLifter(num_joints=80, in_features=2, hidden_dim=256, num_blocks=2)
    dummy = torch.randn(2, 27, 80 * 2)  # B=2, T=27 frames, 80 keypoints × 2D
    out = model(dummy)
    print("Output shape:", out.shape)  # [2, 27, 80*3]
