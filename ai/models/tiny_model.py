# ai/models/tiny_model.py
import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """Liten, snabb encoder för enskilda frames."""
    def __init__(self, in_ch: int = 3, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, x):  # x: [B,C,H,W]
        f = self.net(x).flatten(1)   # [B,256]
        f = self.proj(f)             # [B,feat_dim]
        return f

class PoseHead(nn.Module):
    """Regresserar K×3 från en frame-feature."""
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, K * 3),
        )
    def forward(self, f):  # [B*T, F]
        out = self.mlp(f)  # [B*T, K*3]
        return out.view(out.size(0), self.K, 3)

class GaitHead(nn.Module):
    """Sekvensklassificering över frame-features."""
    def __init__(self, in_dim: int, num_gaits: int, hidden: int = 256, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                           num_layers=num_layers, batch_first=True, bidirectional=False)
        self.cls = nn.Linear(hidden, num_gaits)

    def forward(self, seq_f):  # [B,T,F]
        h, _ = self.rnn(seq_f)       # [B,T,H]
        logits = self.cls(h[:, -1])  # sista tidssteg
        return logits

class PoseGaitTinyNet(nn.Module):
    """
    Delar encoder: Pose-head körs per frame, Gait-head kör sekvens.
    """
    def __init__(self, K: int, num_gaits: int, feat_dim: int = 256):
        super().__init__()
        self.encoder = TinyCNN(in_ch=3, feat_dim=feat_dim)
        self.pose_head = PoseHead(feat_dim, K)
        self.gait_head = GaitHead(feat_dim, num_gaits)

    def forward(self, images):  # images: [B,T,C,H,W]
        B, T, C, H, W = images.shape
        flat = images.view(B * T, C, H, W)              # [B*T,C,H,W]
        feats = self.encoder(flat)                       # [B*T,F]
        pose = self.pose_head(feats).view(B, T, -1, 3)  # [B,T,K,3]
        seq_feats = feats.view(B, T, -1)                # [B,T,F]
        gait_logits = self.gait_head(seq_feats)         # [B,num_gaits]
        return pose, gait_logits
