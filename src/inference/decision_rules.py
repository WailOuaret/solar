from __future__ import annotations

import math

from src.inference.severity import power_loss_to_severity, scalar_to_priority

DEFAULT_ELECTRICAL_REFERENCE = {
    "pmpp": 60.4712,
    "isc": 5.1519,
    "ff": 0.792574,
}


def _coerce_optional_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def electrical_targets_to_risk_score(
    pmpp: float | None = None,
    isc: float | None = None,
    ff: float | None = None,
    references: dict[str, float] | None = None,
) -> float | None:
    """Convert electrical performance targets into a degradation-oriented risk score."""

    refs = {**DEFAULT_ELECTRICAL_REFERENCE, **(references or {})}
    components: list[float] = []

    for key, value in {"pmpp": pmpp, "isc": isc, "ff": ff}.items():
        value = _coerce_optional_float(value)
        if value is None:
            continue
        reference = max(float(refs[key]), 1e-6)
        normalized_performance = min(max(value, 0.0) / reference, 1.0)
        components.append(1.0 - normalized_performance)

    if not components:
        return None
    return sum(components) / len(components)


def build_final_risk_score(
    power_loss_pct: float | None = None,
    electrical_score: float | None = None,
    hotspot_probability: float | None = None,
) -> float:
    score = 0.0
    if power_loss_pct is not None:
        score += min(max(power_loss_pct / 25.0, 0.0), 1.0) * 0.45
    if electrical_score is not None:
        score += min(max(electrical_score, 0.0), 1.0) * 0.3
    if hotspot_probability is not None:
        score += min(max(hotspot_probability, 0.0), 1.0) * 0.6
    return min(score, 1.0)


def recommend_action(
    power_loss_pct: float | None = None,
    electrical_score: float | None = None,
    hotspot_probability: float | None = None,
) -> dict[str, str | float]:
    final_score = build_final_risk_score(
        power_loss_pct=power_loss_pct,
        electrical_score=electrical_score,
        hotspot_probability=hotspot_probability,
    )

    if hotspot_probability is not None and hotspot_probability >= 0.8:
        return {
            "final_severity": "urgent",
            "final_score": final_score,
            "recommended_action": "Immediate inspection for hotspot risk",
            "priority": scalar_to_priority(final_score),
        }

    if power_loss_pct is not None and power_loss_pct >= 12.0:
        return {
            "final_severity": power_loss_to_severity(power_loss_pct),
            "final_score": final_score,
            "recommended_action": "Schedule cleaning and performance inspection",
            "priority": scalar_to_priority(final_score),
        }

    if electrical_score is not None and electrical_score >= 0.5:
        return {
            "final_severity": "medium",
            "final_score": final_score,
            "recommended_action": "Check shading conditions and electrical behavior",
            "priority": scalar_to_priority(final_score),
        }

    return {
        "final_severity": "low",
        "final_score": final_score,
        "recommended_action": "Monitor panel condition",
        "priority": scalar_to_priority(final_score),
    }
