from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torch is required for datasets") from exc

from src.data.schema import SEVERITY_TO_INT


def _load_image(image_path: str | Path, modality: str) -> Image.Image:
    image = Image.open(image_path)
    if modality == "thermal":
        return image.convert("RGB")
    return image.convert("RGB")


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


class BasePVDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        transform: Any = None,
        modality: str = "rgb",
        image_column: str = "image_path",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.modality = modality
        self.image_column = image_column

    def __len__(self) -> int:
        return len(self.frame)

    def _prepare_image(self, path: str | Path):
        image = _load_image(path, self.modality)
        if self.transform is not None:
            return self.transform(image)
        return image


class DeepSolarEyeDataset(BasePVDataset):
    def __init__(self, frame: pd.DataFrame, transform: Any = None, tabular_features: list[str] | None = None) -> None:
        super().__init__(frame, transform=transform, modality="rgb")
        self.tabular_features = tabular_features or []

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = self._prepare_image(row[self.image_column])
        power_loss = _safe_float(row.get("power_loss_pct"))
        severity_label = _safe_str(row.get("severity_label"))
        severity_idx = SEVERITY_TO_INT.get(severity_label, 0)
        tabular = np.array([_safe_float(row.get(feature)) for feature in self.tabular_features], dtype=np.float32)
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "power_loss_pct": torch.tensor(power_loss, dtype=torch.float32),
            "severity_label": torch.tensor(severity_idx, dtype=torch.long),
        }


class VillegasDataset(BasePVDataset):
    def __init__(self, frame: pd.DataFrame, transform: Any = None, weather_features: list[str] | None = None) -> None:
        super().__init__(frame, transform=transform, modality="rgb")
        self.weather_features = weather_features or []

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = self._prepare_image(row[self.image_column])
        weather = np.array([_safe_float(row.get(feature)) for feature in self.weather_features], dtype=np.float32)
        targets = np.array(
            [
                _safe_float(row.get("pmpp")),
                _safe_float(row.get("isc")),
                _safe_float(row.get("ff")),
            ],
            dtype=np.float32,
        )
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "weather": torch.tensor(weather, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
        }


class TRSAIThermalDataset(BasePVDataset):
    def __init__(self, frame: pd.DataFrame, transform: Any = None) -> None:
        super().__init__(frame, transform=transform, modality="thermal")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = self._prepare_image(row[self.image_column])
        label = _safe_int(row.get("hotspot_label"))
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "hotspot_label": torch.tensor(label, dtype=torch.float32),
        }


class UnifiedPVDataset(BasePVDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = self._prepare_image(row[self.image_column])
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "dataset_name": row.get("dataset_name", ""),
            "modality": row.get("modality", ""),
        }
