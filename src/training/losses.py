from __future__ import annotations

import torch
import torch.nn.functional as F


def powerloss_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    regression_weight: float = 1.0,
    classification_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    regression = F.mse_loss(outputs["power_loss"], batch["power_loss_pct"])
    classification = F.cross_entropy(outputs["severity_logits"], batch["severity_label"])
    loss = regression_weight * regression + classification_weight * classification
    return loss, {
        "loss": float(loss.detach().cpu()),
        "regression_loss": float(regression.detach().cpu()),
        "classification_loss": float(classification.detach().cpu()),
    }


def electrical_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_scales: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions = outputs["electrical"]
    targets = batch["targets"]

    if target_scales is not None:
        scales = torch.as_tensor(target_scales, dtype=predictions.dtype, device=predictions.device).view(1, -1).clamp_min(1e-6)
        predictions = predictions / scales
        targets = targets / scales

    loss = F.smooth_l1_loss(predictions, targets)
    return loss, {"loss": float(loss.detach().cpu())}


def thermal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.binary_cross_entropy_with_logits(outputs["hotspot_logits"], batch["hotspot_label"])
    return loss, {"loss": float(loss.detach().cpu())}


def fusion_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    risk_loss = F.mse_loss(outputs["risk_score"], targets["risk_score"])
    severity_loss = F.cross_entropy(outputs["severity_logits"], targets["severity_label"])
    loss = risk_loss + 0.5 * severity_loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "risk_loss": float(risk_loss.detach().cpu()),
        "severity_loss": float(severity_loss.detach().cpu()),
    }
