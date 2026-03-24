from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rgb_backbone import RGBBackbone


class PowerLossModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_tabular_features: int = 0,
        num_severity_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = RGBBackbone(backbone_name=backbone_name, pretrained=pretrained)
        self.num_tabular_features = num_tabular_features

        tabular_dim = 0
        if num_tabular_features > 0:
            self.tabular_mlp = nn.Sequential(
                nn.Linear(num_tabular_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            tabular_dim = 32
        else:
            self.tabular_mlp = None

        fused_dim = self.encoder.feature_dim + tabular_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(256, 1)
        self.severity_head = nn.Linear(256, num_severity_classes)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        image_features = self.encoder(image)
        if self.tabular_mlp is not None and tabular is not None and tabular.numel() > 0:
            tabular_features = self.tabular_mlp(tabular)
            features = torch.cat([image_features, tabular_features], dim=1)
        else:
            features = image_features
        fused = self.fusion(features)
        return {
            "power_loss": self.regression_head(fused).squeeze(1),
            "severity_logits": self.severity_head(fused),
        }

