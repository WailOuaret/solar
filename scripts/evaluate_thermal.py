from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.datasets import TRSAIThermalDataset
from src.data.loaders import limit_frame, load_config, load_metadata_frame
from src.data.transforms_thermal import build_thermal_transform
from src.models.thermal_hotspot_head import ThermalHotspotModel
from src.training.metrics import classification_metrics
from src.utils.io import dump_json, ensure_dir
from src.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TRSAI thermal branch.")
    parser.add_argument("--config", default="configs/train_thermal.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment_name", "trsai_thermal_hotspot")
    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[(frame["dataset_name"] == cfg["dataset_name"]) & (frame["split"] == args.split)].copy()
    frame = limit_frame(frame, cfg.get("max_test_samples"), seed=cfg.get("random_seed", 42))

    dataset = TRSAIThermalDataset(frame, transform=build_thermal_transform(cfg["image_size"], is_train=False))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalHotspotModel(backbone_name=cfg["backbone"], pretrained=False).to(device)

    output_dir = resolve_project_path(cfg["output_dir"])
    checkpoint = torch.load(output_dir / cfg["save_name"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            outputs = model(image)
            probabilities = torch.sigmoid(outputs["hotspot_logits"]).cpu().numpy()
            targets = batch["hotspot_label"].cpu().numpy()
            preds = (probabilities >= cfg.get("threshold", 0.5)).astype(int)
            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target_hotspot": int(targets[i]),
                        "pred_hotspot": int(preds[i]),
                        "hotspot_probability": float(probabilities[i]),
                    }
                )

    pred_frame = pd.DataFrame(rows)
    metrics = {}
    if not pred_frame.empty:
        metrics.update(classification_metrics(pred_frame["target_hotspot"].to_numpy(), pred_frame["pred_hotspot"].to_numpy()))

    predictions_dir = resolve_project_path("outputs/predictions")
    tables_dir = resolve_project_path("outputs/tables")
    ensure_dir(predictions_dir)
    ensure_dir(tables_dir)
    pred_frame.to_csv(predictions_dir / f"{experiment_name}_predictions.csv", index=False)
    dump_json(metrics, tables_dir / f"{experiment_name}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
