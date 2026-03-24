from __future__ import annotations

import argparse
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.loaders import list_images, load_config
from src.data.schema import UnifiedSample, build_metadata_frame, infer_severity_from_power_loss, safe_sample_id
from src.utils.io import ensure_dir
from src.utils.paths import resolve_project_path


DEEPSOLAREYE_PATTERN = re.compile(
    r"^solar_(?P<weekday>[A-Za-z]+)_(?P<month>[A-Za-z]+)_(?P<day>\d+)_(?P<hour>\d+)__(?P<minute>\d+)__(?P<second>\d+)_(?P<year>\d+)_L_(?P<loss>[-0-9.]+)_I_(?P<irradiance>[-0-9.]+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def _safe_float(value) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _extract_zip_if_needed(zip_path: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.rglob("*")):
        return output_dir
    ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    return output_dir


def _find_first(raw_dir: Path, filename: str) -> Path | None:
    matches = list(raw_dir.rglob(filename))
    return matches[0] if matches else None


def _build_image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            index[path.stem.lower()] = path
    return index


def _first_present(row: pd.Series, candidates: list[str]) -> float | None:
    for candidate in candidates:
        if candidate in row.index:
            value = _safe_float(row.get(candidate))
            if value is not None:
                return value
    return None


def _first_by_substring(row: pd.Series, substring: str) -> float | None:
    substring = substring.lower()
    for column in row.index:
        if substring in str(column).lower():
            value = _safe_float(row.get(column))
            if value is not None:
                return value
    return None


def _harmonize_villegas_pmpp(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1000.0 if value > 100.0 else value


def _harmonize_villegas_isc(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 1000.0 if value > 10.0 else value


def _harmonize_villegas_ff(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 2.0:
        return None
    return value


def parse_deepsolareye(raw_dir: Path) -> pd.DataFrame:
    images = list_images(raw_dir)
    samples: list[UnifiedSample] = []
    readme_path = _find_first(raw_dir, "README.md")

    for image_path in images:
        match = DEEPSOLAREYE_PATTERN.match(image_path.name)
        timestamp = ""
        timestamp_date = ""
        power_loss_pct = None
        irradiance = None
        session_id = "unknown_session"

        if match:
            parts = match.groupdict()
            dt = datetime.strptime(
                f"{parts['weekday']}_{parts['month']}_{parts['day']}_{parts['hour']}_{parts['minute']}_{parts['second']}_{parts['year']}",
                "%a_%b_%d_%H_%M_%S_%Y",
            )
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_date = dt.strftime("%Y-%m-%d")
            session_id = dt.strftime("%Y-%m-%d")
            power_loss_pct = float(parts["loss"]) * 100.0
            irradiance = float(parts["irradiance"])

        relative_parts = image_path.relative_to(raw_dir).parts
        soiling_group = relative_parts[-2] if len(relative_parts) >= 2 else ""

        samples.append(
            UnifiedSample(
                sample_id=safe_sample_id("deepsolareye", image_path),
                dataset_name="deepsolareye",
                modality="rgb",
                image_path=str(image_path),
                timestamp=timestamp,
                timestamp_date=timestamp_date,
                source_panel_id="panel_1",
                panel_id="panel_1",
                session_id=session_id,
                irradiance=irradiance,
                power_loss_pct=power_loss_pct,
                severity_label=infer_severity_from_power_loss(power_loss_pct),
                soiling_type=soiling_group,
                source_metadata_file=str(readme_path or ""),
            )
        )

    return build_metadata_frame(samples)


def parse_villegas(raw_dir: Path) -> pd.DataFrame:
    features_path = _find_first(raw_dir, "Features.xlsx")
    if features_path is None:
        return build_metadata_frame([])

    images_zip = _find_first(raw_dir, "Images.zip")
    electrical_zip = _find_first(raw_dir, "Electrical_data.zip")
    images_dir = resolve_project_path("data/interim/villegas/images")

    if images_zip is not None:
        _extract_zip_if_needed(images_zip, images_dir)

    image_index = _build_image_index(images_dir)
    features = _normalize_columns(pd.read_excel(features_path))
    samples: list[UnifiedSample] = []

    for _, row in features.iterrows():
        record = str(row.get("record", "")).strip()
        if not record:
            continue
        image_path = image_index.get(record.lower())
        timestamp = ""
        timestamp_date = ""
        session_id = ""
        try:
            dt = datetime.strptime(record, "%Y_%m_%d_%H_%M")
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_date = dt.strftime("%Y-%m-%d")
            session_id = dt.strftime("%Y-%m-%d")
        except Exception:
            session_id = record[:10]

        pmpp = _harmonize_villegas_pmpp(_first_by_substring(row, "pmpp"))
        isc = _harmonize_villegas_isc(_first_by_substring(row, "isc"))
        ff = _harmonize_villegas_ff(_first_by_substring(row, "fill factor"))

        samples.append(
            UnifiedSample(
                sample_id=record,
                dataset_name="villegas",
                modality="rgb",
                image_path=str(image_path) if image_path is not None else "",
                timestamp=timestamp,
                timestamp_date=timestamp_date,
                source_panel_id="panel_1",
                panel_id="panel_1",
                session_id=session_id,
                irradiance=_first_by_substring(row, "irradiance"),
                temperature=_first_by_substring(row, "temperature"),
                azimuth=_first_by_substring(row, "azimuth"),
                zenith=_first_by_substring(row, "zenith"),
                albedo=_first_by_substring(row, "albedo"),
                pmpp=pmpp,
                isc=isc,
                ff=ff,
                source_metadata_file=str(electrical_zip or features_path),
            )
        )

    return build_metadata_frame(samples)


def parse_trsai(raw_dir: Path) -> pd.DataFrame:
    samples: list[UnifiedSample] = []
    zip_paths = sorted(raw_dir.rglob("TRSAI.v*i*.zip"))

    for zip_path in zip_paths:
        version_match = re.search(r"TRSAI\.(v\d+)", zip_path.name, flags=re.IGNORECASE)
        dataset_version = version_match.group(1).lower() if version_match else zip_path.stem.lower()
        extract_dir = resolve_project_path(f"data/interim/trsai/{dataset_version}")
        _extract_zip_if_needed(zip_path, extract_dir)

        image_paths = [
            path
            for path in extract_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]

        for image_path in image_paths:
            parts_lower = [part.lower() for part in image_path.parts]
            if "train" in parts_lower:
                split = "train"
            elif "valid" in parts_lower or "val" in parts_lower:
                split = "val"
            elif "test" in parts_lower:
                split = "test"
            else:
                split = ""

            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            hotspot_label = 1 if label_path.exists() and label_path.stat().st_size > 0 else 0

            samples.append(
                UnifiedSample(
                    sample_id=f"{dataset_version}:{image_path.stem}",
                    dataset_name="trsai",
                    modality="thermal",
                    image_path=str(image_path),
                    split=split,
                    session_id=f"{dataset_version}_{split}" if split else dataset_version,
                    source_panel_id="panel_thermal",
                    panel_id="panel_thermal",
                    dataset_version=dataset_version,
                    hotspot_label=hotspot_label,
                    source_metadata_file=str(zip_path),
                )
            )

    return build_metadata_frame(samples)


def enrich_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour_sin"] = 0.0
    frame["hour_cos"] = 0.0
    if "timestamp" not in frame.columns:
        return frame

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    hours = timestamps.dt.hour.fillna(0) + timestamps.dt.minute.fillna(0) / 60.0
    radians = (hours / 24.0) * (2 * math.pi)
    frame["hour_sin"] = radians.apply(math.sin)
    frame["hour_cos"] = radians.apply(math.cos)
    return frame


def validate_image_paths(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_exists"] = frame["image_path"].fillna("").apply(lambda path: Path(path).exists() if path else False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified metadata for all project datasets.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config.")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Optional fraction for fast metadata/debug runs.")
    parser.add_argument("--max-samples-per-dataset", type=int, default=0, help="Optional cap per dataset for fast metadata/debug runs.")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = []

    for dataset_name, dataset_cfg in config["datasets"].items():
        raw_dir = resolve_project_path(dataset_cfg["raw_dir"])
        if dataset_name == "deepsolareye":
            frame = parse_deepsolareye(raw_dir)
        elif dataset_name == "villegas":
            frame = parse_villegas(raw_dir)
        elif dataset_name == "trsai":
            frame = parse_trsai(raw_dir)
        else:
            frame = build_metadata_frame([])

        if 0 < args.sample_fraction < 1.0 and not frame.empty:
            frame = frame.sample(frac=args.sample_fraction, random_state=42).reset_index(drop=True)
        if args.max_samples_per_dataset > 0 and len(frame) > args.max_samples_per_dataset:
            frame = frame.sample(n=args.max_samples_per_dataset, random_state=42).reset_index(drop=True)

        frame = enrich_time_features(frame)
        frame = validate_image_paths(frame)
        dataset_csv = resolve_project_path(dataset_cfg["metadata_csv"])
        ensure_dir(dataset_csv.parent)
        frame.to_csv(dataset_csv, index=False)
        outputs.append(frame)

    master = pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()
    master_csv = resolve_project_path(config["unified_metadata"]["csv"])
    master_parquet = resolve_project_path(config["unified_metadata"]["parquet"])
    ensure_dir(master_csv.parent)
    master.to_csv(master_csv, index=False)
    try:
        master.to_parquet(master_parquet, index=False)
    except Exception:
        pass

    print(f"Saved unified metadata to {master_csv}")


if __name__ == "__main__":
    main()
