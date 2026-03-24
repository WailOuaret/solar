from __future__ import annotations

from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import EarlyStopping, save_checkpoint


class Trainer:
    def __init__(self, device: str | torch.device, mixed_precision: bool = True) -> None:
        self.device = torch.device(device)
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move_batch(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
    ) -> dict[str, float]:
        model.train()
        running = defaultdict(float)

        for batch in tqdm(loader, desc="train", leave=False):
            batch = self._move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self._forward(model, batch)
                loss, logs = loss_fn(outputs, batch)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss detected. Check metadata values and feature preprocessing.")

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            for key, value in logs.items():
                running[key] += value

        return {key: value / max(1, len(loader)) for key, value in running.items()}

    def evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        loss_fn: Callable,
    ) -> dict[str, float]:
        model.eval()
        running = defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(loader, desc="val", leave=False):
                batch = self._move_batch(batch)
                outputs = self._forward(model, batch)
                loss, logs = loss_fn(outputs, batch)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation loss detected. Check metadata values and feature preprocessing.")
                for key, value in logs.items():
                    running[key] += value

        return {key: value / max(1, len(loader)) for key, value in running.items()}

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        epochs: int,
        checkpoint_path: str,
        patience: int = 5,
    ) -> list[dict[str, float]]:
        model.to(self.device)
        early_stopper = EarlyStopping(patience=patience)
        history: list[dict[str, float]] = []
        best_metric = float("inf")

        for epoch in range(1, epochs + 1):
            train_logs = self.train_one_epoch(model, train_loader, optimizer, loss_fn)
            val_logs = self.evaluate(model, val_loader, loss_fn)
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}, **{f"val_{k}": v for k, v in val_logs.items()}}
            history.append(record)

            current_metric = val_logs.get("loss", 0.0)
            if isfinite(current_metric) and current_metric < best_metric:
                best_metric = current_metric
                save_checkpoint(model, optimizer, epoch, current_metric, checkpoint_path)

            if early_stopper.step(current_metric):
                break

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Training completed without producing a checkpoint: {checkpoint_path}")

        return history

    def _forward(self, model: torch.nn.Module, batch: dict) -> dict:
        if "tabular" in batch:
            return model(batch["image"], batch["tabular"])
        if "weather" in batch:
            return model(batch["image"], batch["weather"])
        return model(batch["image"])
