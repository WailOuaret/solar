from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.utils.io import ensure_dir


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best: float | None = None
    counter: int = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metric: float, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

