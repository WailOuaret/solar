from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.loaders import load_config, load_metadata_frame
from src.utils.io import ensure_dir


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, size: int = 8) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(image, dtype=np.float32)
    return arr > arr.mean()


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_hist(values: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    values.dropna().plot(kind="hist", bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_line(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    series.plot(kind="line")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def image_size_summary(frame: pd.DataFrame, sample_n: int = 300) -> pd.DataFrame:
    rows = []
    for dataset_name, group in frame.groupby("dataset_name"):
        sample = group.sample(min(sample_n, len(group)), random_state=42) if len(group) > sample_n else group
        for _, row in sample.iterrows():
            path = Path(row["image_path"])
            if not path.exists():
                continue
            try:
                with Image.open(path) as image:
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "width": image.width,
                            "height": image.height,
                        }
                    )
            except Exception:
                continue
    return pd.DataFrame(rows)


def exact_duplicate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for dataset_name, group in frame.groupby("dataset_name"):
        hashes = []
        for path_str in group["image_path"].dropna():
            path = Path(path_str)
            if not path.exists():
                continue
            hashes.append(file_sha1(path))
        counter = Counter(hashes)
        duplicate_files = sum(count for count in counter.values() if count > 1)
        duplicate_groups = sum(1 for count in counter.values() if count > 1)
        records.append(
            {
                "dataset_name": dataset_name,
                "files_hashed": len(hashes),
                "duplicate_groups": duplicate_groups,
                "duplicate_files": duplicate_files,
            }
        )
    return pd.DataFrame(records)


def limit_group(group: pd.DataFrame, max_files: int | None, seed: int = 42) -> pd.DataFrame:
    if max_files is None or len(group) <= max_files:
        return group
    return group.sample(n=max_files, random_state=seed)


def deepsolareye_near_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["dataset_name"] == "deepsolareye"].copy()
    subset["timestamp_dt"] = pd.to_datetime(subset["timestamp"], errors="coerce")
    subset = subset.sort_values(["session_id", "timestamp_dt", "image_path"]).reset_index(drop=True)
    hashes: dict[str, np.ndarray] = {}
    rows = []
    previous_row = None

    for _, row in subset.iterrows():
        path = Path(row["image_path"])
        if not path.exists() or pd.isna(row["timestamp_dt"]):
            previous_row = row
            continue
        try:
            current_hash = hashes.setdefault(str(path), average_hash(path))
        except Exception:
            previous_row = row
            continue

        if previous_row is not None and previous_row["session_id"] == row["session_id"] and pd.notna(previous_row["timestamp_dt"]):
            time_delta = (row["timestamp_dt"] - previous_row["timestamp_dt"]).total_seconds()
            if 0 <= time_delta <= 10:
                prev_path = Path(previous_row["image_path"])
                try:
                    prev_hash = hashes.setdefault(str(prev_path), average_hash(prev_path))
                    distance = hamming_distance(prev_hash, current_hash)
                    rows.append(
                        {
                            "sample_id_prev": previous_row["sample_id"],
                            "sample_id_curr": row["sample_id"],
                            "time_delta_seconds": time_delta,
                            "hash_distance": distance,
                            "near_duplicate": distance <= 5,
                        }
                    )
                except Exception:
                    pass

        previous_row = row

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audit outputs for the three PV datasets.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-hash-files-per-dataset", type=int, default=None)
    parser.add_argument("--skip-exact-duplicates", action="store_true")
    parser.add_argument("--skip-near-duplicates", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frame = load_metadata_frame(cfg["unified_metadata"]["csv"])
    figures_dir = ensure_dir("outputs/figures")
    tables_dir = ensure_dir("outputs/tables")

    dataset_counts = frame["dataset_name"].value_counts().sort_index()
    save_bar(dataset_counts, "Samples per Dataset", "Dataset", "Samples", figures_dir / "sample_counts_by_dataset.png")

    split_counts = frame.groupby(["dataset_name", "split"]).size().unstack(fill_value=0)
    split_counts.plot(kind="bar", figsize=(10, 5), title="Split Counts by Dataset")
    plt.tight_layout()
    plt.savefig(figures_dir / "split_counts_by_dataset.png")
    plt.close()

    deepsolareye = frame[frame["dataset_name"] == "deepsolareye"].copy()
    if not deepsolareye.empty:
        save_hist(deepsolareye["power_loss_pct"], "DeepSolarEye Power Loss Distribution", "power_loss_pct", figures_dir / "deepsolareye_power_loss_hist.png")
        by_day = deepsolareye["timestamp_date"].fillna("missing").value_counts().sort_index()
        save_line(by_day, "DeepSolarEye Samples by Day", "Day", "Samples", figures_dir / "deepsolareye_samples_by_day.png")

    villegas = frame[frame["dataset_name"] == "villegas"].copy()
    if not villegas.empty:
        missing_series = pd.Series(
            {
                "pmpp_missing": int(villegas["pmpp"].isna().sum()),
                "isc_missing": int(villegas["isc"].isna().sum()),
                "ff_missing": int(villegas["ff"].isna().sum()),
                "irradiance_missing": int(villegas["irradiance"].isna().sum()),
                "temperature_missing": int(villegas["temperature"].isna().sum()),
                "azimuth_missing": int(villegas["azimuth"].isna().sum()),
                "zenith_missing": int(villegas["zenith"].isna().sum()),
                "albedo_missing": int(villegas["albedo"].isna().sum()),
            }
        )
        save_bar(missing_series, "Villegas Missing Metadata", "Field", "Missing rows", figures_dir / "villegas_missing_metadata.png")
        by_day = villegas["timestamp_date"].fillna("missing").value_counts().sort_index()
        save_line(by_day, "Villegas Samples by Day", "Day", "Samples", figures_dir / "villegas_samples_by_day.png")

    trsai = frame[frame["dataset_name"] == "trsai"].copy()
    if not trsai.empty:
        class_balance = trsai["hotspot_label"].fillna(-1).astype(int).value_counts().sort_index()
        save_bar(class_balance, "TRSAI Hotspot Label Balance", "Hotspot label", "Samples", figures_dir / "trsai_class_balance.png")

    size_frame = image_size_summary(frame)
    if not size_frame.empty:
        size_summary = size_frame.groupby("dataset_name")[["width", "height"]].median().round(0)
        size_summary.to_csv(tables_dir / "image_size_summary.csv")

    hash_frame = frame.groupby("dataset_name", group_keys=False).apply(
        lambda group: limit_group(group, args.max_hash_files_per_dataset)
    ).reset_index(drop=True)

    if args.skip_exact_duplicates:
        duplicate_frame = pd.DataFrame(
            [{"dataset_name": name, "files_hashed": 0, "duplicate_groups": 0, "duplicate_files": 0} for name in sorted(frame["dataset_name"].dropna().unique())]
        )
    else:
        duplicate_frame = exact_duplicate_summary(hash_frame)
    duplicate_frame.to_csv(tables_dir / "duplicate_summary.csv", index=False)

    if args.skip_near_duplicates:
        near_duplicates = pd.DataFrame(columns=["sample_id_prev", "sample_id_curr", "time_delta_seconds", "hash_distance", "near_duplicate"])
    else:
        deep_frame = hash_frame[hash_frame["dataset_name"] == "deepsolareye"].copy()
        near_duplicates = deepsolareye_near_duplicates(deep_frame)
    near_duplicates.to_csv(tables_dir / "deepsolareye_near_duplicates.csv", index=False)

    audit_summary = {
        "row_count": int(len(frame)),
        "dataset_counts": dataset_counts.to_dict(),
        "split_counts": {dataset: values for dataset, values in split_counts.astype(int).to_dict(orient="index").items()},
        "image_exists_by_dataset": frame.groupby("dataset_name")["image_exists"].mean().round(4).to_dict(),
        "deepsolareye": {
            "power_loss_missing": int(deepsolareye["power_loss_pct"].isna().sum()) if not deepsolareye.empty else 0,
            "session_count": int(deepsolareye["session_id"].nunique()) if not deepsolareye.empty else 0,
            "near_duplicate_pairs": int(len(near_duplicates)),
            "near_duplicate_pairs_hash_le_5": int(near_duplicates["near_duplicate"].sum()) if not near_duplicates.empty else 0,
        },
        "villegas": {
            "pmpp_missing": int(villegas["pmpp"].isna().sum()) if not villegas.empty else 0,
            "isc_missing": int(villegas["isc"].isna().sum()) if not villegas.empty else 0,
            "ff_missing": int(villegas["ff"].isna().sum()) if not villegas.empty else 0,
            "weather_present_fraction": {
                "irradiance": float(villegas["irradiance"].notna().mean()) if not villegas.empty else 0.0,
                "temperature": float(villegas["temperature"].notna().mean()) if not villegas.empty else 0.0,
                "azimuth": float(villegas["azimuth"].notna().mean()) if not villegas.empty else 0.0,
                "zenith": float(villegas["zenith"].notna().mean()) if not villegas.empty else 0.0,
                "albedo": float(villegas["albedo"].notna().mean()) if not villegas.empty else 0.0,
            },
        },
        "trsai": {
            "hotspot_balance": trsai["hotspot_label"].fillna(-1).astype(int).value_counts().to_dict() if not trsai.empty else {},
        },
        "exact_duplicates": duplicate_frame.to_dict(orient="records"),
        "audit_options": {
            "max_hash_files_per_dataset": args.max_hash_files_per_dataset,
            "skip_exact_duplicates": args.skip_exact_duplicates,
            "skip_near_duplicates": args.skip_near_duplicates,
        },
    }

    with (tables_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_summary, handle, indent=2)

    print(json.dumps(audit_summary, indent=2))


if __name__ == "__main__":
    main()
