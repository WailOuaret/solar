from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SEVERITY_TO_INT = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "urgent": 3,
}

INT_TO_SEVERITY = {value: key for key, value in SEVERITY_TO_INT.items()}


@dataclass(slots=True)
class UnifiedSample:
    sample_id: str
    dataset_name: str
    modality: str
    image_path: str
    split: str = ""
    timestamp: str = ""
    timestamp_date: str = ""
    source_panel_id: str = ""
    panel_id: str = ""
    session_id: str = ""
    dataset_version: str = ""
    irradiance: float | None = None
    temperature: float | None = None
    azimuth: float | None = None
    zenith: float | None = None
    albedo: float | None = None
    power_loss_pct: float | None = None
    pmpp: float | None = None
    isc: float | None = None
    ff: float | None = None
    soiling_type: str = ""
    hotspot_label: int | None = None
    severity_label: str = ""
    augmentation_flag: bool = False
    source_metadata_file: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "UnifiedSample":
        clean = {key: record.get(key) for key in cls.__dataclass_fields__}
        return cls(**clean)


def empty_metadata_frame() -> pd.DataFrame:
    sample = UnifiedSample(
        sample_id="",
        dataset_name="",
        modality="",
        image_path="",
    )
    return pd.DataFrame(columns=list(sample.to_record().keys()))


def ensure_required_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = list(empty_metadata_frame().columns)
    for column in required_columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[required_columns]


def build_metadata_frame(samples: list[UnifiedSample]) -> pd.DataFrame:
    if not samples:
        return empty_metadata_frame()
    frame = pd.DataFrame([sample.to_record() for sample in samples])
    return ensure_required_columns(frame)


def infer_severity_from_power_loss(power_loss_pct: float | None) -> str:
    if power_loss_pct is None:
        return ""
    if power_loss_pct < 5:
        return "low"
    if power_loss_pct < 12:
        return "medium"
    if power_loss_pct < 20:
        return "high"
    return "urgent"


def safe_sample_id(dataset_name: str, image_path: str | Path) -> str:
    image_path = Path(image_path)
    return f"{dataset_name}:{image_path.stem}"

