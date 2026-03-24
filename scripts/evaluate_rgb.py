from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.datasets import DeepSolarEyeDataset
from src.data.loaders import limit_frame, load_config, load_metadata_frame
from src.data.transforms_rgb import build_rgb_transform
from src.models.powerloss_head import PowerLossModel
from src.training.metrics import classification_metrics, regression_metrics
from src.utils.io import dump_json, ensure_dir
from src.utils.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepSolarEye RGB branch.")
    parser.add_argument("--config", default="configs/train_rgb.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_name = cfg.get("experiment_name", "deepsolareye_rgb")
    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[(frame["dataset_name"] == cfg["dataset_name"]) & (frame["split"] == args.split)].copy()
    frame = frame[frame["power_loss_pct"].notna()].copy()
    frame = limit_frame(frame, cfg.get("max_test_samples"), seed=cfg.get("random_seed", 42))

    dataset = DeepSolarEyeDataset(
        frame,
        transform=build_rgb_transform(cfg["image_size"], is_train=False),
        tabular_features=cfg.get("tabular_features", []),
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PowerLossModel(
        backbone_name=cfg["backbone"],
        pretrained=False,
        num_tabular_features=len(cfg.get("tabular_features", [])) if cfg.get("use_tabular_features", False) else 0,
        num_severity_classes=4,
    ).to(device)

    output_dir = resolve_project_path(cfg["output_dir"])
    checkpoint = torch.load(output_dir / cfg["save_name"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            outputs = model(image, tabular)
            preds = outputs["power_loss"].cpu().numpy()
            targets = batch["power_loss_pct"].cpu().numpy()
            labels = batch["severity_label"].cpu().numpy()
            pred_severity = outputs["severity_logits"].argmax(dim=1).cpu().numpy() if cfg.get("classification_weight", 0.0) > 0 else None
            for i, sample_id in enumerate(batch["sample_id"]):
                row = {
                    "sample_id": sample_id,
                    "target_power_loss_pct": float(targets[i]),
                    "pred_power_loss_pct": float(preds[i]),
                }
                if pred_severity is not None:
                    row["target_severity"] = int(labels[i])
                    row["pred_severity"] = int(pred_severity[i])
                rows.append(row)

    pred_frame = pd.DataFrame(rows)
    metrics = {}
    if not pred_frame.empty:
        metrics.update(regression_metrics(pred_frame["target_power_loss_pct"].to_numpy(), pred_frame["pred_power_loss_pct"].to_numpy()))
        if {"target_severity", "pred_severity"}.issubset(pred_frame.columns):
            metrics.update(classification_metrics(pred_frame["target_severity"].to_numpy(), pred_frame["pred_severity"].to_numpy()))

    predictions_dir = resolve_project_path("outputs/predictions")
    tables_dir = resolve_project_path("outputs/tables")
    ensure_dir(predictions_dir)
    ensure_dir(tables_dir)
    pred_frame.to_csv(predictions_dir / f"{experiment_name}_predictions.csv", index=False)
    dump_json(metrics, tables_dir / f"{experiment_name}_metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
