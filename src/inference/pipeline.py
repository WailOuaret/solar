from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.transforms_rgb import build_rgb_transform
from src.data.transforms_thermal import build_thermal_transform
from src.inference.decision_rules import recommend_action
from src.inference.severity import hotspot_probability_to_severity, power_loss_to_severity
from src.models.electrical_head import ElectricalModel
from src.models.powerloss_head import PowerLossModel
from src.models.thermal_hotspot_head import ThermalHotspotModel
from src.utils.io import load_yaml
from src.utils.paths import resolve_project_path


class InferencePipeline:
    def __init__(self, config_path: str | Path = "configs/inference.yaml") -> None:
        self.config = load_yaml(resolve_project_path(config_path))
        self.device = self._select_device(self.config.get("device", "auto"))
        self.rgb_transform = build_rgb_transform(224, is_train=False)
        self.thermal_transform = build_thermal_transform(256, is_train=False)
        self.powerloss_model = self._load_powerloss_model()
        self.electrical_model = self._load_electrical_model()
        self.thermal_model = self._load_thermal_model()

    def _select_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str | None) -> torch.nn.Module:
        model.to(self.device)
        if checkpoint_path:
            path = resolve_project_path(checkpoint_path)
            if path.exists():
                state = torch.load(path, map_location=self.device)
                model.load_state_dict(state["model_state_dict"], strict=False)
        model.eval()
        return model

    def _load_powerloss_model(self) -> PowerLossModel:
        model_cfg = self.config["models"]["powerloss"]
        checkpoint = self.config["checkpoints"].get("powerloss")
        model = PowerLossModel(
            backbone_name=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            num_tabular_features=model_cfg.get("num_tabular_features", 0),
            num_severity_classes=model_cfg.get("num_severity_classes", 4),
        )
        return self._load_checkpoint(model, checkpoint)

    def _load_electrical_model(self) -> ElectricalModel:
        model_cfg = self.config["models"]["electrical"]
        checkpoint = self.config["checkpoints"].get("electrical")
        model = ElectricalModel(
            backbone_name=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
            num_weather_features=model_cfg.get("num_weather_features", 0),
            num_targets=len(model_cfg.get("target_names", ["pmpp", "isc", "ff"])),
        )
        return self._load_checkpoint(model, checkpoint)

    def _load_thermal_model(self) -> ThermalHotspotModel:
        model_cfg = self.config["models"]["thermal"]
        checkpoint = self.config["checkpoints"].get("thermal")
        model = ThermalHotspotModel(
            backbone_name=model_cfg["backbone"],
            pretrained=model_cfg["pretrained"],
        )
        return self._load_checkpoint(model, checkpoint)

    def _image_tensor(self, image_path: str | Path, modality: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        transform = self.thermal_transform if modality == "thermal" else self.rgb_transform
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor

    def predict_powerloss(self, image_path: str | Path, tabular_features: list[float] | None = None) -> dict[str, Any]:
        image = self._image_tensor(image_path, modality="rgb")
        tabular = tabular_features or [0.0] * self.config["models"]["powerloss"].get("num_tabular_features", 0)
        tabular_tensor = torch.tensor([tabular], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.powerloss_model(image, tabular_tensor)
            power_loss = float(outputs["power_loss"].cpu().item())
            severity_idx = int(outputs["severity_logits"].argmax(dim=1).cpu().item())
        return {
            "power_loss_pct": power_loss,
            "predicted_severity_index": severity_idx,
            "severity_rule": power_loss_to_severity(power_loss),
        }

    def predict_electrical(self, image_path: str | Path, weather_features: list[float] | None = None) -> dict[str, Any]:
        image = self._image_tensor(image_path, modality="rgb")
        weather = weather_features or [0.0] * self.config["models"]["electrical"].get("num_weather_features", 0)
        weather_tensor = torch.tensor([weather], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.electrical_model(image, weather_tensor)
            values = outputs["electrical"].cpu().numpy().reshape(-1).tolist()
        target_names = self.config["models"]["electrical"].get("target_names", ["pmpp", "isc", "ff"])
        predictions = dict(zip(target_names, values))
        scale = max(abs(values[0]) if values else 0.0, abs(values[1]) if len(values) > 1 else 0.0, 1.0)
        predictions["electrical_score"] = min(max(sum(abs(v) for v in values) / (len(values) * scale), 0.0), 1.0)
        return predictions

    def predict_thermal(self, image_path: str | Path) -> dict[str, Any]:
        image = self._image_tensor(image_path, modality="thermal")
        with torch.no_grad():
            outputs = self.thermal_model(image)
            probability = float(torch.sigmoid(outputs["hotspot_logits"]).cpu().item())
        return {
            "hotspot_probability": probability,
            "hotspot_severity": hotspot_probability_to_severity(probability),
        }

    def predict_sample(
        self,
        image_path: str | Path,
        modality: str,
        tabular_features: list[float] | None = None,
        weather_features: list[float] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "image_path": str(image_path),
            "modality": modality,
        }

        power_loss_pct = None
        electrical_score = None
        hotspot_probability = None

        if modality == "rgb":
            powerloss_result = self.predict_powerloss(image_path, tabular_features=tabular_features)
            electrical_result = self.predict_electrical(image_path, weather_features=weather_features)
            power_loss_pct = powerloss_result["power_loss_pct"]
            electrical_score = electrical_result["electrical_score"]
            result.update(powerloss_result)
            result.update(electrical_result)
        elif modality == "thermal":
            thermal_result = self.predict_thermal(image_path)
            hotspot_probability = thermal_result["hotspot_probability"]
            result.update(thermal_result)

        decision = recommend_action(
            power_loss_pct=power_loss_pct,
            electrical_score=electrical_score,
            hotspot_probability=hotspot_probability,
        )
        result.update(decision)
        return result
