# model_pipeline/models/gait_classifier.py

import torch
import torch.nn as nn


class GaitClassifier(nn.Module):
    """
    Minimal LSTM-baserad gångarts-klassificerare.
    Tar en sekvens av keypoints (2D eller 3D) och förutspår gångartsklass.
    """

    def __init__(
        self,
        num_joints: int = 68,
        in_features: int = 3,      # 2 för 2D, 3 för 3D
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,      # t.ex. skritt, trav, galopp, stilla
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = num_joints * in_features

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Eftersom vi använder bidirectional LSTM blir output-dim 2*hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, T, K*in_features]  (B=batch, T=antal frames, K=antal leder)
        Returns:
            logits: [B, num_classes] (klass-sannolikheter innan softmax)
        """
        # LSTM behöver (B, T, input_dim)
        out, _ = self.lstm(x)              # out: [B, T, 2*hidden_dim]
        last_hidden = out[:, -1]           # ta sista tidssteget
        logits = self.fc(last_hidden)      # [B, num_classes]
        return logits


if __name__ == "__main__":
    model = GaitClassifier()
    dummy = torch.randn(4, 30, 68 * 3)  # batch=4, sekvens=30 frames, 68 leder × 3D
    out = model(dummy)
    print("Output shape:", out.shape)  # [4, num_classes]
