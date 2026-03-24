from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import load_json


def _load_optional_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return load_json(file_path)


def _format_metric_lines(metrics: dict) -> list[str]:
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return lines or ["- Metrics not available."]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile experiment outputs into a report-friendly markdown summary.")
    parser.add_argument("--output", default="outputs/reports/final_report.md")
    args = parser.parse_args()

    audit = _load_optional_json("outputs/tables/audit_summary.json")
    deepsolareye_baseline = _load_optional_json("outputs/tables/deepsolareye_rgb_powerloss_metrics.json")
    deepsolareye_multitask = _load_optional_json("outputs/tables/deepsolareye_rgb_powerloss_multitask_metrics.json")
    villegas_image_only = _load_optional_json("outputs/tables/villegas_rgb_electrical_image_only_metrics.json")
    villegas_weather = _load_optional_json("outputs/tables/villegas_rgb_electrical_metrics.json")
    villegas_transfer = _load_optional_json("outputs/tables/villegas_rgb_electrical_transfer_metrics.json")
    trsai = _load_optional_json("outputs/tables/trsai_thermal_hotspot_metrics.json")
    latency = _load_optional_json("outputs/tables/latency_summary.json")

    lines = [
        "# Final Report",
        "",
        "## Dataset Summary",
        "",
    ]

    if audit:
        lines.append(f"- Total metadata rows: {audit.get('row_count', 0)}")
        dataset_counts = audit.get("dataset_counts", {})
        for dataset_name, count in dataset_counts.items():
            lines.append(f"- {dataset_name}: {count} samples")
        lines.append("")
        lines.append("## Split Summary")
        lines.append("")
        for dataset_name, split_counts in audit.get("split_counts", {}).items():
            counts = ", ".join(f"{split}: {count}" for split, count in split_counts.items())
            lines.append(f"- {dataset_name}: {counts}")
        lines.append("")
    else:
        lines.extend(["- Audit summary not available.", ""])

    lines.extend(["## DeepSolarEye Results", ""])
    lines.append("### Regression Baseline")
    lines.extend(_format_metric_lines(deepsolareye_baseline))
    lines.append("")
    lines.append("### Multitask Regression + Severity")
    lines.extend(_format_metric_lines(deepsolareye_multitask))
    lines.append("")

    lines.extend(["## Villegas Results", ""])
    lines.append("### Image-Only Baseline")
    lines.extend(_format_metric_lines(villegas_image_only))
    lines.append("")
    lines.append("### Image + Weather")
    lines.extend(_format_metric_lines(villegas_weather))
    lines.append("")
    lines.append("### DeepSolarEye Transfer + Weather")
    lines.extend(_format_metric_lines(villegas_transfer))
    lines.append("")

    lines.extend(["## TRSAI Results", ""])
    lines.extend(_format_metric_lines(trsai))
    lines.append("")

    lines.extend(["## Limitations", ""])
    lines.append("- TRSAI metadata currently resolves to a single positive class, so its thermal results are one-class hotspot scoring only.")
    lines.append("- DeepSolarEye contains many near-duplicate temporal neighbors; leakage-safe splits were required and random image splits would be misleading.")
    lines.append("- The reported model runs use reduced sample caps for CPU feasibility and should be treated as prototype baselines, not final full-dataset numbers.")
    lines.append("- Cross-branch fusion is implemented in code, but a fully aligned fused evaluation is still pending because the three datasets do not share common sample IDs.")
    lines.append("")

    lines.extend(["## Deployment Notes", ""])
    lines.append("- Raw datasets are frozen under data/raw and parsed into unified metadata without modifying the original files.")
    lines.append("- Fast audit options are available in scripts/run_data_audit.py to cap hashing work on large rebuilds.")
    lines.append("- Transfer learning from DeepSolarEye to Villegas is now supported through configs/train_regression_transfer.yaml.")
    if latency:
        lines.append("")
        lines.append("## Latency Summary")
        lines.append("")
        for branch_name, branch_metrics in latency.items():
            if not isinstance(branch_metrics, dict):
                lines.append(f"- {branch_name}: {branch_metrics}")
                continue
            lines.append(
                f"- {branch_name}: mean {branch_metrics.get('mean_ms', 0.0):.2f} ms, median {branch_metrics.get('median_ms', 0.0):.2f} ms"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report summary written to {args.output}")


if __name__ == "__main__":
    main()
