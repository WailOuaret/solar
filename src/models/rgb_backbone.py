from __future__ import annotations

import torch
import torch.nn as nn


class _TorchvisionBackbone(nn.Module):
    def __init__(self, model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RGBBackbone(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.model, self.feature_dim = self._build_backbone(backbone_name, pretrained)

    def _build_backbone(self, backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
        try:
            import timm

            try:
                model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            except Exception:
                model = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
            feature_dim = getattr(model, "num_features", None)
            if feature_dim is None:
                raise ValueError(f"Unable to infer feature dim for timm backbone: {backbone_name}")
            return model, int(feature_dim)
        except Exception:
            from torchvision import models

            if backbone_name == "resnet18":
                try:
                    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                    model = models.resnet18(weights=weights)
                except Exception:
                    model = models.resnet18(weights=None)
                feature_dim = model.fc.in_features
                model.fc = nn.Identity()
                return model, feature_dim

            if backbone_name == "resnet34":
                try:
                    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
                    model = models.resnet34(weights=weights)
                except Exception:
                    model = models.resnet34(weights=None)
                feature_dim = model.fc.in_features
                model.fc = nn.Identity()
                return model, feature_dim

            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                model = models.efficientnet_b0(weights=weights)
            except Exception:
                model = models.efficientnet_b0(weights=None)
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
