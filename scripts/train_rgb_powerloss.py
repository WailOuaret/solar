from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.data.datasets import DeepSolarEyeDataset
from src.data.loaders import limit_frame, load_config, load_metadata_frame
from src.data.transforms_rgb import build_rgb_transform
from src.models.powerloss_head import PowerLossModel
from src.training.engine import Trainer
from src.training.losses import powerloss_loss
from src.utils.io import ensure_dir
from src.utils.paths import resolve_project_path
from src.utils.seed import seed_everything
from src.visualization.plots import save_history_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepSolarEye RGB power-loss baseline.")
    parser.add_argument("--config", default="configs/train_rgb.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_seed", 42))

    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[frame["dataset_name"] == cfg["dataset_name"]].copy()
    frame = frame[frame["power_loss_pct"].notna()].copy()
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    train_frame = limit_frame(train_frame, cfg.get("max_train_samples"), seed=cfg.get("random_seed", 42))
    val_frame = limit_frame(val_frame, cfg.get("max_val_samples"), seed=cfg.get("random_seed", 42))

    train_ds = DeepSolarEyeDataset(
        train_frame,
        transform=build_rgb_transform(cfg["image_size"], is_train=True),
        tabular_features=cfg.get("tabular_features", []),
    )
    val_ds = DeepSolarEyeDataset(
        val_frame,
        transform=build_rgb_transform(cfg["image_size"], is_train=False),
        tabular_features=cfg.get("tabular_features", []),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = PowerLossModel(
        backbone_name=cfg["backbone"],
        pretrained=cfg["pretrained"],
        num_tabular_features=len(cfg.get("tabular_features", [])) if cfg.get("use_tabular_features", False) else 0,
        num_severity_classes=4,
    )

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
        loss_fn=lambda outputs, batch: powerloss_loss(
            outputs,
            batch,
            regression_weight=cfg.get("regression_weight", 1.0),
            classification_weight=cfg.get("classification_weight", 0.3),
        ),
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
