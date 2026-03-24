from __future__ import annotations


def power_loss_to_severity(power_loss_pct: float) -> str:
    if power_loss_pct < 5.0:
        return "low"
    if power_loss_pct < 12.0:
        return "medium"
    if power_loss_pct < 20.0:
        return "high"
    return "urgent"


def hotspot_probability_to_severity(probability: float) -> str:
    if probability < 0.2:
        return "low"
    if probability < 0.5:
        return "medium"
    if probability < 0.8:
        return "high"
    return "urgent"


def scalar_to_priority(score: float) -> str:
    if score < 0.25:
        return "monitor"
    if score < 0.5:
        return "schedule_check"
    if score < 0.8:
        return "high_priority"
    return "urgent"

