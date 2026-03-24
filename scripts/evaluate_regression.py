from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.datasets import VillegasDataset
from src.data.loaders import limit_frame, load_config, load_metadata_frame
from src.data.transforms_rgb import build_rgb_transform
from src.models.electrical_head import ElectricalModel
from src.training.metrics import multioutput_regression_metrics
from src.utils.io import dump_json, ensure_dir
from src.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Villegas electrical regression branch.")
    parser.add_argument("--config", default="configs/train_regression.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment_name", "villegas_rgb_electrical")
    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[(frame["dataset_name"] == cfg["dataset_name"]) & (frame["split"] == args.split)].copy()
    frame = frame[frame["pmpp"].notna() & frame["isc"].notna() & frame["ff"].notna()].copy()
    frame = limit_frame(frame, cfg.get("max_test_samples"), seed=cfg.get("random_seed", 42))

    weather_features = cfg.get("weather_features", []) if cfg.get("use_weather_features", False) else []
    dataset = VillegasDataset(frame, transform=build_rgb_transform(cfg["image_size"], is_train=False), weather_features=weather_features)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElectricalModel(
        backbone_name=cfg["backbone"],
        pretrained=False,
        num_weather_features=len(weather_features),
        num_targets=len(cfg["targets"]),
    ).to(device)

    output_dir = resolve_project_path(cfg["output_dir"])
    checkpoint = torch.load(output_dir / cfg["save_name"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            weather = batch["weather"].to(device)
            outputs = model(image, weather)
            preds = outputs["electrical"].cpu().numpy()
            targets = batch["targets"].cpu().numpy()
            for i, sample_id in enumerate(batch["sample_id"]):
                row = {"sample_id": sample_id}
                for j, name in enumerate(cfg["targets"]):
                    row[f"target_{name}"] = float(targets[i][j])
                    row[f"pred_{name}"] = float(preds[i][j])
                rows.append(row)

    pred_frame = pd.DataFrame(rows)
    metrics = {}
    if not pred_frame.empty:
        target_array = pred_frame[[f"target_{name}" for name in cfg["targets"]]].to_numpy()
        pred_array = pred_frame[[f"pred_{name}" for name in cfg["targets"]]].to_numpy()
        metrics.update(multioutput_regression_metrics(target_array, pred_array, cfg["targets"]))

    predictions_dir = resolve_project_path("outputs/predictions")
    tables_dir = resolve_project_path("outputs/tables")
    ensure_dir(predictions_dir)
    ensure_dir(tables_dir)
    pred_frame.to_csv(predictions_dir / f"{experiment_name}_predictions.csv", index=False)
    dump_json(metrics, tables_dir / f"{experiment_name}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
