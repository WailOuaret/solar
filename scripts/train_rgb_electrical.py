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
from src.training.engine import Trainer
from src.training.losses import electrical_loss
from src.utils.io import ensure_dir
from src.utils.paths import resolve_project_path
from src.utils.seed import seed_everything
from src.visualization.plots import save_history_plot


def load_matching_checkpoint(model: torch.nn.Module, checkpoint_path: str, encoder_only: bool = False) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint)
    target_state = model.state_dict()
    matched = {}

    for key, value in source_state.items():
        if encoder_only and not key.startswith("encoder."):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value

    target_state.update(matched)
    model.load_state_dict(target_state, strict=False)
    return len(matched)


def filter_valid_villegas_rows(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.copy()
    valid = valid[valid["pmpp"].notna() & valid["isc"].notna() & valid["ff"].notna()].copy()
    valid = valid[(valid["pmpp"] >= 0.0) & (valid["isc"] >= 0.0)].copy()
    # Fill factor is physically bounded near [0, 1]. Keep a small margin for noise and drop obvious metadata corruption.
    valid = valid[(valid["ff"] >= 0.0) & (valid["ff"] <= 1.5)].copy()
    return valid


def compute_target_scales(frame: pd.DataFrame, target_names: list[str]) -> list[float]:
    quantiles = frame[target_names].abs().quantile(0.95)
    return [max(float(value), 1e-3) for value in quantiles.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Villegas RGB electrical regression baseline.")
    parser.add_argument("--config", default="configs/train_regression.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_seed", 42))

    frame = load_metadata_frame(cfg["metadata_csv"])
    frame = frame[frame["dataset_name"] == cfg["dataset_name"]].copy()
    before_count = len(frame)
    frame = filter_valid_villegas_rows(frame)
    print(f"Filtered Villegas rows: kept {len(frame)} / {before_count} after target validity checks")
    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "val"].copy()
    train_frame = limit_frame(train_frame, cfg.get("max_train_samples"), seed=cfg.get("random_seed", 42))
    val_frame = limit_frame(val_frame, cfg.get("max_val_samples"), seed=cfg.get("random_seed", 42))
    target_scales = compute_target_scales(train_frame, cfg["targets"])
    print(f"Target scales for robust loss: {dict(zip(cfg['targets'], target_scales))}")

    weather_features = cfg.get("weather_features", []) if cfg.get("use_weather_features", False) else []
    train_ds = VillegasDataset(train_frame, transform=build_rgb_transform(cfg["image_size"], is_train=True), weather_features=weather_features)
    val_ds = VillegasDataset(val_frame, transform=build_rgb_transform(cfg["image_size"], is_train=False), weather_features=weather_features)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = ElectricalModel(
        backbone_name=cfg["backbone"],
        pretrained=cfg["pretrained"],
        num_weather_features=len(weather_features),
        num_targets=len(cfg["targets"]),
    )
    if cfg.get("pretrained_checkpoint"):
        matched = load_matching_checkpoint(
            model,
            str(resolve_project_path(cfg["pretrained_checkpoint"])),
            encoder_only=bool(cfg.get("load_encoder_only", False)),
        )
        print(f"Loaded {matched} matching parameters from {resolve_project_path(cfg['pretrained_checkpoint'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    trainer = Trainer(device="cuda" if torch.cuda.is_available() else "cpu", mixed_precision=False)
    output_dir = resolve_project_path(cfg["output_dir"])
    checkpoint_path = output_dir / cfg["save_name"]
    print(f"Train samples: {len(train_frame)} | Val samples: {len(val_frame)}")
    history = trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda outputs, batch: electrical_loss(outputs, batch, target_scales=target_scales),
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
