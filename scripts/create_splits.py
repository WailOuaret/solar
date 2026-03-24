from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.loaders import load_config, load_metadata_frame
from src.utils.io import dump_json
from src.utils.paths import resolve_project_path

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    GroupShuffleSplit = None


def select_group_column(frame: pd.DataFrame) -> str:
    for column in ("session_id", "timestamp_date", "source_panel_id", "dataset_version"):
        if column in frame.columns and frame[column].fillna("").astype(str).str.len().gt(0).any():
            return column
    return "sample_id"


def grouped_split(frame: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy(), frame.copy()

    group_col = select_group_column(frame)
    groups = frame[group_col].fillna("missing_group").astype(str)

    unique_groups = groups.nunique()
    if unique_groups < 3:
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]

    if GroupShuffleSplit is None:
        group_sizes = frame.groupby(group_col).size().sort_values(ascending=False)
        ordered_groups = list(group_sizes.index)
        rng = np.random.default_rng(seed)
        rng.shuffle(ordered_groups)

        target_train = len(frame) * train_ratio
        target_val = len(frame) * val_ratio

        train_groups: list[str] = []
        val_groups: list[str] = []
        test_groups: list[str] = []
        train_count = 0
        val_count = 0

        for group in ordered_groups:
            size = int(group_sizes[group])
            if train_count < target_train:
                train_groups.append(group)
                train_count += size
            elif val_count < target_val:
                val_groups.append(group)
                val_count += size
            else:
                test_groups.append(group)

        train = frame[groups.isin(train_groups)].copy()
        val = frame[groups.isin(val_groups)].copy()
        test = frame[groups.isin(test_groups)].copy()
        return train, val, test

    splitter_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(splitter_train.split(frame, groups=groups))
    train = frame.iloc[train_idx].copy()
    temp = frame.iloc[temp_idx].copy()

    temp_groups = temp[group_col].fillna("missing_group").astype(str)
    relative_val_size = val_ratio / max(1e-6, (1.0 - train_ratio))
    splitter_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=seed)
    val_idx, test_idx = next(splitter_val.split(temp, groups=temp_groups))
    val = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()
    return train, val, test


def leakage_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    group_col = select_group_column(pd.concat([train, val, test], ignore_index=True))
    train_groups = set(train[group_col].astype(str))
    val_groups = set(val[group_col].astype(str))
    test_groups = set(test[group_col].astype(str))
    return {
        "group_column": group_col,
        "train_val_overlap": sorted(train_groups & val_groups),
        "train_test_overlap": sorted(train_groups & test_groups),
        "val_test_overlap": sorted(val_groups & test_groups),
    }


def assign_split(frame: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = ""
    result.loc[train.index, "split"] = "train"
    result.loc[val.index, "split"] = "val"
    result.loc[test.index, "split"] = "test"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware train/val/test splits.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config.")
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_metadata_frame(config["unified_metadata"]["csv"])

    train_ratio = config["splits"]["train_ratio"]
    val_ratio = config["splits"]["val_ratio"]
    seed = config["splits"]["random_seed"]

    updated_frames = []

    for dataset_name, dataset_cfg in config["datasets"].items():
        subset = frame[frame["dataset_name"] == dataset_name].copy().reset_index(drop=True)
        if dataset_name == "trsai" and subset["split"].isin(["train", "val", "test"]).any():
            subset = subset.copy()
            subset["split"] = subset["split"].replace({"valid": "val"})
        else:
            subset["split"] = ""
            train, val, test = grouped_split(subset, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)
            subset = assign_split(subset, train, val, test)
        updated_frames.append(subset)

        split_summary = {
            "dataset_name": dataset_name,
            "counts": {
                "train": int((subset["split"] == "train").sum()),
                "val": int((subset["split"] == "val").sum()),
                "test": int((subset["split"] == "test").sum()),
            },
            "leakage_check": leakage_summary(
                subset[subset["split"] == "train"],
                subset[subset["split"] == "val"],
                subset[subset["split"] == "test"],
            ),
        }
        dump_json(split_summary, resolve_project_path(dataset_cfg["split_json"]))

    updated = pd.concat(updated_frames, ignore_index=True, sort=False) if updated_frames else frame
    updated.to_csv(resolve_project_path(config["unified_metadata"]["csv"]), index=False)
    try:
        updated.to_parquet(resolve_project_path(config["unified_metadata"]["parquet"]), index=False)
    except Exception:
        pass

    dump_json(
        {
            "datasets": list(config["datasets"].keys()),
            "metadata_csv": config["unified_metadata"]["csv"],
            "notes": "Splits are grouped to reduce temporal or session leakage.",
        },
        resolve_project_path("data/splits/cross_dataset_eval.json"),
    )

    print("Split files created.")


if __name__ == "__main__":
    main()
