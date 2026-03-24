from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_features: int = 5, num_severity_classes: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.risk_head = nn.Linear(16, 1)
        self.severity_head = nn.Linear(16, num_severity_classes)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(features)
        return {
            "risk_score": self.risk_head(hidden).squeeze(1),
            "severity_logits": self.severity_head(hidden),
        }

