from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import math

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.inference.decision_rules import electrical_targets_to_risk_score, recommend_action
from src.utils.io import ensure_dir, load_yaml
from src.utils.paths import resolve_project_path


@dataclass
class DemoCase:
    case_id: str
    dataset_name: str
    sample_id: str
    image_path: str
    summary: str
    predicted_primary_value: float | None
    predicted_secondary_value: float | None
    predicted_tertiary_value: float | None
    derived_score: float | None
    final_severity: str
    recommended_action: str
    priority: str


def _select_quantile_rows(frame: pd.DataFrame, column: str, quantiles: list[float], prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    selected_rows: list[pd.Series] = []
    used_ids: set[str] = set()
    for quantile in quantiles:
        target = frame[column].astype(float).quantile(quantile)
        ordered = frame.assign(_distance=(frame[column].astype(float) - target).abs()).sort_values("_distance")
        for _, row in ordered.iterrows():
            sample_id = str(row["sample_id"])
            if sample_id not in used_ids:
                used_ids.add(sample_id)
                selected_rows.append(row)
                break

    if not selected_rows:
        return frame.head(0).copy()

    selected = pd.DataFrame(selected_rows).copy()
    selected.insert(0, "case_id", [f"{prefix}_{idx + 1}" for idx in range(len(selected))])
    return selected


def _electrical_score(row: pd.Series, electrical_reference: dict[str, float]) -> float:
    score = electrical_targets_to_risk_score(
        pmpp=row.get("pred_pmpp"),
        isc=row.get("pred_isc"),
        ff=row.get("pred_ff"),
        references=electrical_reference,
    )
    return float(score or 0.0)


def _build_powerloss_cases(metadata: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(resolve_project_path("outputs/predictions/deepsolareye_rgb_powerloss_multitask_predictions.csv"))
    frame = predictions.merge(metadata[["sample_id", "image_path"]], on="sample_id", how="left")
    cases = _select_quantile_rows(frame, "pred_power_loss_pct", [0.0, 0.25, 0.5, 0.75, 1.0], "deepsolareye")

    rows = []
    for _, row in cases.iterrows():
        decision = recommend_action(power_loss_pct=float(row["pred_power_loss_pct"]))
        rows.append(
            asdict(
                DemoCase(
                    case_id=row["case_id"],
                    dataset_name="deepsolareye",
                    sample_id=str(row["sample_id"]),
                    image_path=str(row["image_path"]),
                    summary=f"Predicted power loss {float(row['pred_power_loss_pct']):.2f}%",
                    predicted_primary_value=float(row["pred_power_loss_pct"]),
                    predicted_secondary_value=float(row["target_power_loss_pct"]),
                    predicted_tertiary_value=float(row["pred_severity"]) if "pred_severity" in row else None,
                    derived_score=None,
                    final_severity=str(decision["final_severity"]),
                    recommended_action=str(decision["recommended_action"]),
                    priority=str(decision["priority"]),
                )
            )
        )
    return pd.DataFrame(rows)


def _build_villegas_cases(
    metadata: pd.DataFrame,
    predictions_path: str,
    electrical_reference: dict[str, float],
) -> pd.DataFrame:
    predictions = pd.read_csv(resolve_project_path(predictions_path))
    frame = predictions.merge(metadata[["sample_id", "image_path"]], on="sample_id", how="left")
    frame["electrical_score"] = frame.apply(_electrical_score, axis=1, electrical_reference=electrical_reference)
    cases = _select_quantile_rows(frame, "electrical_score", [0.0, 0.5, 1.0], "villegas")

    rows = []
    for _, row in cases.iterrows():
        decision = recommend_action(electrical_score=float(row["electrical_score"]))
        rows.append(
            asdict(
                DemoCase(
                    case_id=row["case_id"],
                    dataset_name="villegas",
                    sample_id=str(row["sample_id"]),
                    image_path=str(row["image_path"]),
                    summary=f"Predicted pmpp {float(row['pred_pmpp']):.4f}, isc {float(row['pred_isc']):.4f}, ff {float(row['pred_ff']):.4f}",
                    predicted_primary_value=float(row["pred_pmpp"]),
                    predicted_secondary_value=float(row["pred_isc"]),
                    predicted_tertiary_value=float(row["pred_ff"]),
                    derived_score=float(row["electrical_score"]),
                    final_severity=str(decision["final_severity"]),
                    recommended_action=str(decision["recommended_action"]),
                    priority=str(decision["priority"]),
                )
            )
        )
    return pd.DataFrame(rows)


def _build_thermal_cases(metadata: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(resolve_project_path("outputs/predictions/trsai_thermal_hotspot_predictions.csv"))
    frame = predictions.merge(metadata[["sample_id", "image_path"]], on="sample_id", how="left")
    cases = _select_quantile_rows(frame, "hotspot_probability", [0.0, 0.5, 1.0], "trsai")

    rows = []
    for _, row in cases.iterrows():
        decision = recommend_action(hotspot_probability=float(row["hotspot_probability"]))
        rows.append(
            asdict(
                DemoCase(
                    case_id=row["case_id"],
                    dataset_name="trsai",
                    sample_id=str(row["sample_id"]),
                    image_path=str(row["image_path"]),
                    summary=f"Predicted hotspot probability {float(row['hotspot_probability']):.4f}",
                    predicted_primary_value=float(row["hotspot_probability"]),
                    predicted_secondary_value=float(row["pred_hotspot"]),
                    predicted_tertiary_value=float(row["target_hotspot"]),
                    derived_score=None,
                    final_severity=str(decision["final_severity"]),
                    recommended_action=str(decision["recommended_action"]),
                    priority=str(decision["priority"]),
                )
            )
        )
    return pd.DataFrame(rows)


def _build_manual_fusion_cases(
    powerloss_cases: pd.DataFrame,
    villegas_cases: pd.DataFrame,
    thermal_cases: pd.DataFrame,
) -> pd.DataFrame:
    low_power = powerloss_cases.iloc[0]
    high_power = powerloss_cases.iloc[-1]
    mid_villegas = villegas_cases.iloc[len(villegas_cases) // 2]
    high_thermal = thermal_cases.iloc[-1]

    manual_cases = [
        {
            "scenario_id": "fusion_monitor_case",
            "power_loss_pct": float(low_power["predicted_primary_value"]),
            "electrical_score": None,
            "hotspot_probability": None,
            "source_notes": f"Low-power-loss DeepSolarEye sample {low_power['sample_id']}",
        },
        {
            "scenario_id": "fusion_cleaning_case",
            "power_loss_pct": float(high_power["predicted_primary_value"]),
            "electrical_score": None,
            "hotspot_probability": None,
            "source_notes": f"High-power-loss DeepSolarEye sample {high_power['sample_id']}",
        },
        {
            "scenario_id": "fusion_electrical_case",
            "power_loss_pct": None,
            "electrical_score": float(mid_villegas["derived_score"]),
            "hotspot_probability": None,
            "source_notes": f"Representative Villegas sample {mid_villegas['sample_id']}",
        },
        {
            "scenario_id": "fusion_hotspot_case",
            "power_loss_pct": None,
            "electrical_score": None,
            "hotspot_probability": float(high_thermal["predicted_primary_value"]),
            "source_notes": f"High-probability TRSAI sample {high_thermal['sample_id']}",
        },
        {
            "scenario_id": "fusion_combined_urgent_case",
            "power_loss_pct": float(high_power["predicted_primary_value"]),
            "electrical_score": float(mid_villegas["derived_score"]),
            "hotspot_probability": float(high_thermal["predicted_primary_value"]),
            "source_notes": (
                f"Combined showcase from DeepSolarEye {high_power['sample_id']}, "
                f"Villegas {mid_villegas['sample_id']}, TRSAI {high_thermal['sample_id']}"
            ),
        },
    ]

    output_rows = []
    for case in manual_cases:
        decision = recommend_action(
            power_loss_pct=case["power_loss_pct"],
            electrical_score=case["electrical_score"],
            hotspot_probability=case["hotspot_probability"],
        )
        output_rows.append({**case, **decision})
    return pd.DataFrame(output_rows)


def _write_markdown(
    powerloss_cases: pd.DataFrame,
    villegas_cases: pd.DataFrame,
    thermal_cases: pd.DataFrame,
    fusion_cases: pd.DataFrame,
) -> None:
    def _display(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float) and math.isnan(value):
            return "N/A"
        return str(value)

    output_path = resolve_project_path("outputs/reports/demo_showcase.md")
    ensure_dir(output_path.parent)

    lines = ["# Demo Showcase", "", "## DeepSolarEye Cases", ""]
    for section_name, frame in [
        ("DeepSolarEye Cases", powerloss_cases),
        ("Villegas Cases", villegas_cases),
        ("TRSAI Cases", thermal_cases),
    ]:
        if section_name != "DeepSolarEye Cases":
            lines.extend(["", f"## {section_name}", ""])
        for _, row in frame.iterrows():
            lines.append(f"### {row['case_id']}")
            lines.append(f"- sample_id: {row['sample_id']}")
            lines.append(f"- image_path: {row['image_path']}")
            lines.append(f"- summary: {row['summary']}")
            lines.append(f"- final_severity: {row['final_severity']}")
            lines.append(f"- recommended_action: {row['recommended_action']}")
            lines.append(f"- priority: {row['priority']}")
            lines.append("")

    lines.extend(["## Manual Fusion Showcase", ""])
    for _, row in fusion_cases.iterrows():
        lines.append(f"### {row['scenario_id']}")
        lines.append(f"- power_loss_pct: {_display(row['power_loss_pct'])}")
        lines.append(f"- electrical_score: {_display(row['electrical_score'])}")
        lines.append(f"- hotspot_probability: {_display(row['hotspot_probability'])}")
        lines.append(f"- final_severity: {row['final_severity']}")
        lines.append(f"- final_score: {float(row['final_score']):.4f}")
        lines.append(f"- recommended_action: {row['recommended_action']}")
        lines.append(f"- priority: {row['priority']}")
        lines.append(f"- source_notes: {row['source_notes']}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo-ready prediction bundles and a manual fusion showcase.")
    parser.add_argument("--metadata", default="data/processed/unified_metadata/metadata_master.csv")
    parser.add_argument("--fusion-config", default="configs/fusion.yaml")
    args = parser.parse_args()

    metadata = pd.read_csv(resolve_project_path(args.metadata))
    fusion_cfg = load_yaml(resolve_project_path(args.fusion_config))
    predictions_dir = resolve_project_path("outputs/predictions")
    ensure_dir(predictions_dir)

    powerloss_cases = _build_powerloss_cases(metadata)
    villegas_cases = _build_villegas_cases(
        metadata,
        predictions_path=fusion_cfg["electrical_predictions"],
        electrical_reference=fusion_cfg.get("electrical_reference", {}),
    )
    thermal_cases = _build_thermal_cases(metadata)
    fusion_cases = _build_manual_fusion_cases(powerloss_cases, villegas_cases, thermal_cases)

    powerloss_cases.to_csv(predictions_dir / "demo_deepsolareye_cases.csv", index=False)
    villegas_cases.to_csv(predictions_dir / "demo_villegas_cases.csv", index=False)
    thermal_cases.to_csv(predictions_dir / "demo_trsai_cases.csv", index=False)
    fusion_cases.to_csv(predictions_dir / "demo_fusion_cases.csv", index=False)
    _write_markdown(powerloss_cases, villegas_cases, thermal_cases, fusion_cases)
    print("Demo outputs written to outputs/predictions and outputs/reports/demo_showcase.md")


if __name__ == "__main__":
    main()
