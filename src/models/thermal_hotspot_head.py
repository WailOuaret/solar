from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rgb_backbone import RGBBackbone


class ThermalHotspotModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = RGBBackbone(backbone_name=backbone_name, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        return {"hotspot_logits": self.classifier(features).squeeze(1)}

