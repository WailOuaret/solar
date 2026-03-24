from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml
from src.utils.paths import resolve_project_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(path: str | Path) -> dict:
    return load_yaml(resolve_project_path(path))


def load_metadata_frame(path: str | Path) -> pd.DataFrame:
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def list_images(root: str | Path) -> list[Path]:
    root_path = resolve_project_path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_frame(frame: pd.DataFrame, dataset_name: str | None = None, split: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    if dataset_name is not None:
        result = result[result["dataset_name"] == dataset_name]
    if split is not None and "split" in result.columns:
        result = result[result["split"] == split]
    return result.reset_index(drop=True)


def limit_frame(frame: pd.DataFrame, max_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    if max_samples is None:
        return frame.reset_index(drop=True)

    max_samples = int(max_samples)
    if max_samples <= 0 or len(frame) <= max_samples:
        return frame.reset_index(drop=True)

    return frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)
