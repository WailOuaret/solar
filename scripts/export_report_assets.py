from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile metrics into a report-friendly markdown summary.")
    parser.add_argument("--output", default="outputs/reports/final_report.md")
    args = parser.parse_args()

    metrics_files = {
        "DeepSolarEye": Path("outputs/tables/deepsolareye_metrics.json"),
        "Villegas": Path("outputs/tables/villegas_metrics.json"),
        "TRSAI": Path("outputs/tables/trsai_metrics.json"),
    }

    lines = ["# Final Report", "", "## Experiment Summary", ""]
    for name, path in metrics_files.items():
        lines.append(f"### {name}")
        if path.exists():
            metrics = load_json(path)
            if metrics:
                for key, value in metrics.items():
                    lines.append(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")
            else:
                lines.append("- Metrics file exists but is empty.")
        else:
            lines.append("- Metrics not available yet.")
        lines.append("")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Report summary written to {args.output}")


if __name__ == "__main__":
    main()
