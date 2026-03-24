from __future__ import annotations

import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.models.fusion_head import FusionHead
from src.training.engine import Trainer
from src.training.losses import fusion_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned fusion head from branch predictions.")
    parser.add_argument("--features-csv", default="outputs/predictions/fusion_features.csv")
    parser.add_argument("--output", default="outputs/models/fusion/fusion_head.pt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    frame = pd.read_csv(args.features_csv)
    feature_cols = ["power_loss_score", "electrical_score", "hotspot_probability", "pmpp_norm", "isc_norm"]
    target_cols = ["risk_score", "severity_label"]
    features = torch.tensor(frame[feature_cols].fillna(0.0).to_numpy(), dtype=torch.float32)
    risk_score = torch.tensor(frame["risk_score"].fillna(0.0).to_numpy(), dtype=torch.float32)
    severity = torch.tensor(frame["severity_label"].fillna(0).to_numpy(), dtype=torch.long)

    dataset = TensorDataset(features, risk_score, severity)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class _FusionWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = FusionHead(in_features=len(feature_cols))

        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return self.head(image)

    model = _FusionWrapper()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = Trainer(device="cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(batch):
        x = torch.stack([item[0] for item in batch], dim=0)
        risk = torch.stack([item[1] for item in batch], dim=0)
        sev = torch.stack([item[2] for item in batch], dim=0)
        return {"image": x, "risk_score": risk, "severity_label": sev}

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
    trainer.fit(model, loader, loader, optimizer, lambda outputs, batch: fusion_loss(outputs, batch), args.epochs, args.output)
    print(f"Fusion model saved to {args.output}")


if __name__ == "__main__":
    main()
