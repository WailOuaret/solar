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
from src.training.engine import Trainer
from src.training.losses import thermal_loss
from src.utils.io import ensure_dir
from src.utils.paths import resolve_project_path
from src.utils.seed import seed_everything
from src.visualization.plots import save_history_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRSAI thermal hotspot baseline.")
    parser.add_argument("--config", default="configs/train_thermal.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_seed", 42))

    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[frame["dataset_name"] == cfg["dataset_name"]].copy()
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    train_frame = limit_frame(train_frame, cfg.get("max_train_samples"), seed=cfg.get("random_seed", 42))
    val_frame = limit_frame(val_frame, cfg.get("max_val_samples"), seed=cfg.get("random_seed", 42))

    train_ds = TRSAIThermalDataset(train_frame, transform=build_thermal_transform(cfg["image_size"], is_train=True))
    val_ds = TRSAIThermalDataset(val_frame, transform=build_thermal_transform(cfg["image_size"], is_train=False))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = ThermalHotspotModel(backbone_name=cfg["backbone"], pretrained=cfg["pretrained"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    trainer = Trainer(device="cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve_project_path(cfg["output_dir"])
    checkpoint_path = output_dir / cfg["save_name"]
    print(f"Train samples: {len(train_frame)} | Val samples: {len(val_frame)}")
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=thermal_loss,
        epochs=cfg["epochs"],
        checkpoint_path=str(checkpoint_path),
    )

    ensure_dir(output_dir)
    history_path = output_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    save_history_plot(history, output_dir / "training_curve.png")
    print(f"Training finished. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
