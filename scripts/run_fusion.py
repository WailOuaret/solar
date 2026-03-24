from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.inference.decision_rules import electrical_targets_to_risk_score, recommend_action
from src.utils.io import ensure_dir, load_yaml
from src.utils.paths import resolve_project_path


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse branch outputs into a maintenance report.")
    parser.add_argument("--config", default="configs/fusion.yaml")
    args = parser.parse_args()

    cfg = load_yaml(resolve_project_path(args.config))
    electrical_reference = cfg.get("electrical_reference", {})

    powerloss = _safe_read_csv(resolve_project_path(cfg["powerloss_predictions"]))
    electrical = _safe_read_csv(resolve_project_path(cfg["electrical_predictions"]))
    thermal = _safe_read_csv(resolve_project_path(cfg["thermal_predictions"]))

    base = powerloss.copy() if not powerloss.empty else pd.DataFrame(columns=["sample_id"])
    if base.empty and not electrical.empty:
        base = electrical[["sample_id"]].copy()
    if base.empty and not thermal.empty:
        base = thermal[["sample_id"]].copy()

    if not electrical.empty:
        base = base.merge(electrical, on="sample_id", how="outer")
    if not thermal.empty:
        base = base.merge(thermal, on="sample_id", how="outer")

    rows = []
    for _, row in base.iterrows():
        power_loss_pct = row.get("pred_power_loss_pct")
        electrical_score = electrical_targets_to_risk_score(
            pmpp=row.get("pred_pmpp"),
            isc=row.get("pred_isc"),
            ff=row.get("pred_ff"),
            references=electrical_reference,
        )
        hotspot_probability = row.get("hotspot_probability") if pd.notna(row.get("hotspot_probability")) else None
        decision = recommend_action(
            power_loss_pct=float(power_loss_pct) if pd.notna(power_loss_pct) else None,
            electrical_score=float(electrical_score) if electrical_score is not None else None,
            hotspot_probability=float(hotspot_probability) if hotspot_probability is not None else None,
        )
        rows.append(
            {
                "sample_id": row["sample_id"],
                "power_loss_pct": power_loss_pct,
                "electrical_score": electrical_score,
                "hotspot_probability": hotspot_probability,
                **decision,
            }
        )

    fusion_frame = pd.DataFrame(rows)
    output_predictions = resolve_project_path(cfg["output_predictions"])
    output_report = resolve_project_path(cfg["output_report"])
    ensure_dir(output_predictions.parent)
    ensure_dir(output_report.parent)
    fusion_frame.to_csv(output_predictions, index=False)

    with output_report.open("w", encoding="utf-8") as handle:
        handle.write("# Fusion Report\n\n")
        handle.write(f"Samples fused: {len(fusion_frame)}\n\n")
        if not fusion_frame.empty:
            handle.write("## Priority Counts\n\n")
            for priority, count in fusion_frame["priority"].value_counts().items():
                handle.write(f"- {priority}: {count}\n")

    print(f"Fusion outputs saved to {output_predictions}")


if __name__ == "__main__":
    main()
