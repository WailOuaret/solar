from __future__ import annotations

import torch
import torch.nn as nn

from src.models.rgb_backbone import RGBBackbone


class ElectricalModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_weather_features: int = 0,
        num_targets: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = RGBBackbone(backbone_name=backbone_name, pretrained=pretrained)
        weather_dim = 0
        if num_weather_features > 0:
            self.weather_mlp = nn.Sequential(
                nn.Linear(num_weather_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            weather_dim = 32
        else:
            self.weather_mlp = None

        fused_dim = self.encoder.feature_dim + weather_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_targets),
        )

    def forward(self, image: torch.Tensor, weather: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        image_features = self.encoder(image)
        if self.weather_mlp is not None and weather is not None and weather.numel() > 0:
            weather_features = self.weather_mlp(weather)
            features = torch.cat([image_features, weather_features], dim=1)
        else:
            features = image_features
        return {"electrical": self.regression_head(features)}

