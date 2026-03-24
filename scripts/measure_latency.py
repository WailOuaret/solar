from __future__ import annotations

import argparse
import statistics
import time

import torch

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.datasets import DeepSolarEyeDataset, TRSAIThermalDataset, VillegasDataset
from src.data.loaders import load_config, load_metadata_frame
from src.data.transforms_rgb import build_rgb_transform
from src.data.transforms_thermal import build_thermal_transform
from src.models.electrical_head import ElectricalModel
from src.models.powerloss_head import PowerLossModel
from src.models.thermal_hotspot_head import ThermalHotspotModel
from src.utils.io import dump_json
from src.utils.paths import resolve_project_path


def _time_forward(callable_fn, warmup: int = 2, repeats: int = 10) -> dict[str, float]:
    for _ in range(warmup):
        callable_fn()

    durations_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_fn()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": float(statistics.mean(durations_ms)),
        "median_ms": float(statistics.median(durations_ms)),
        "min_ms": float(min(durations_ms)),
        "max_ms": float(max(durations_ms)),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure single-sample inference latency for trained branch checkpoints.")
    parser.add_argument("--rgb-config", default="configs/train_rgb.yaml")
    parser.add_argument("--regression-config", default="configs/train_regression_transfer.yaml")
    parser.add_argument("--thermal-config", default="configs/train_thermal.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, float] | str] = {"device": str(device)}

    rgb_cfg = load_config(args.rgb_config)
    metadata = load_metadata_frame(rgb_cfg["metadata_csv"])
    rgb_frame = metadata[(metadata["dataset_name"] == rgb_cfg["dataset_name"]) & (metadata["split"] == args.split) & metadata["power_loss_pct"].notna()].head(1)
    if not rgb_frame.empty:
        rgb_dataset = DeepSolarEyeDataset(
            rgb_frame,
            transform=build_rgb_transform(rgb_cfg["image_size"], is_train=False),
            tabular_features=rgb_cfg.get("tabular_features", []),
        )
        batch = rgb_dataset[0]
        image = batch["image"].unsqueeze(0).to(device)
        tabular = batch["tabular"].unsqueeze(0).to(device)
        model = PowerLossModel(
            backbone_name=rgb_cfg["backbone"],
            pretrained=False,
            num_tabular_features=len(rgb_cfg.get("tabular_features", [])) if rgb_cfg.get("use_tabular_features", False) else 0,
            num_severity_classes=4,
        ).to(device)
        checkpoint = torch.load(resolve_project_path(rgb_cfg["output_dir"]) / rgb_cfg["save_name"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        with torch.no_grad():
            results["deepsolareye_rgb_powerloss"] = _time_forward(lambda: model(image, tabular), repeats=args.repeats)

    regression_cfg = load_config(args.regression_config)
    regression_frame = metadata[
        (metadata["dataset_name"] == regression_cfg["dataset_name"])
        & (metadata["split"] == args.split)
        & metadata["pmpp"].notna()
        & metadata["isc"].notna()
        & metadata["ff"].notna()
    ].head(1)
    if not regression_frame.empty:
        weather_features = regression_cfg.get("weather_features", []) if regression_cfg.get("use_weather_features", False) else []
        regression_dataset = VillegasDataset(
            regression_frame,
            transform=build_rgb_transform(regression_cfg["image_size"], is_train=False),
            weather_features=weather_features,
        )
        batch = regression_dataset[0]
        image = batch["image"].unsqueeze(0).to(device)
        weather = batch["weather"].unsqueeze(0).to(device)
        model = ElectricalModel(
            backbone_name=regression_cfg["backbone"],
            pretrained=False,
            num_weather_features=len(weather_features),
            num_targets=len(regression_cfg["targets"]),
        ).to(device)
        checkpoint = torch.load(resolve_project_path(regression_cfg["output_dir"]) / regression_cfg["save_name"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        with torch.no_grad():
            results["villegas_rgb_electrical_transfer"] = _time_forward(lambda: model(image, weather), repeats=args.repeats)

    thermal_cfg = load_config(args.thermal_config)
    thermal_frame = metadata[(metadata["dataset_name"] == thermal_cfg["dataset_name"]) & (metadata["split"] == args.split)].head(1)
    if not thermal_frame.empty:
        thermal_dataset = TRSAIThermalDataset(
            thermal_frame,
            transform=build_thermal_transform(thermal_cfg["image_size"], is_train=False),
        )
        batch = thermal_dataset[0]
        image = batch["image"].unsqueeze(0).to(device)
        model = ThermalHotspotModel(backbone_name=thermal_cfg["backbone"], pretrained=False).to(device)
        checkpoint = torch.load(resolve_project_path(thermal_cfg["output_dir"]) / thermal_cfg["save_name"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()
        with torch.no_grad():
            results["trsai_thermal_hotspot"] = _time_forward(lambda: model(image), repeats=args.repeats)

    dump_json(results, resolve_project_path("outputs/tables/latency_summary.json"))
    print(results)


if __name__ == "__main__":
    main()
