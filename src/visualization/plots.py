from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_history_plot(history: list[dict], output_path: str | Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    if "train_loss" in frame.columns:
        plt.plot(frame["epoch"], frame["train_loss"], label="train_loss")
    if "val_loss" in frame.columns:
        plt.plot(frame["epoch"], frame["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

