from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import load_json
from src.utils.paths import resolve_project_path

PRE_UNIT_FIX_VILLEGAS = {
    "image_only": {
        "pmpp_mae": 17168.55,
        "pmpp_rmse": 26690.7547,
        "pmpp_r2": -0.7058,
        "isc_mae": 1211.2347,
        "isc_rmse": 2237.1591,
        "isc_r2": -0.4129,
        "ff_mae": 0.0350,
        "ff_rmse": 0.0502,
        "ff_r2": 0.3585,
    },
    "weather": {
        "pmpp_mae": 17167.6587,
        "pmpp_rmse": 26690.2856,
        "pmpp_r2": -0.7057,
        "isc_mae": 1211.9464,
        "isc_rmse": 2237.1285,
        "isc_r2": -0.4128,
        "ff_mae": 0.0553,
        "ff_rmse": 0.0703,
        "ff_r2": -0.2556,
    },
    "transfer": {
        "pmpp_mae": 17167.4288,
        "pmpp_rmse": 26689.9844,
        "pmpp_r2": -0.7057,
        "isc_mae": 1212.0139,
        "isc_rmse": 2236.6532,
        "isc_r2": -0.4122,
        "ff_mae": 0.0589,
        "ff_rmse": 0.0750,
        "ff_r2": -0.4291,
    },
}

PROPOSAL_ALIGNMENT = [
    (
        "Public-dataset-only workflow",
        "Implemented",
        "The project uses frozen datasets under data/raw and does not attempt drone or robot communication.",
    ),
    (
        "RGB-based anomaly analysis",
        "Implemented",
        "DeepSolarEye and Villegas branches cover RGB power-loss estimation and RGB-to-electrical estimation.",
    ),
    (
        "Severity scoring and maintenance recommendation",
        "Implemented",
        "The pipeline exports severity labels, fused risk scores, and operator-facing recommendation text.",
    ),
    (
        "Thermal hotspot branch",
        "Partially implemented",
        "TRSAI runs end-to-end, but the current labels collapse to a single positive class, so evaluation is only a prototype.",
    ),
    (
        "Detection / localization",
        "Partially implemented",
        "The system supports severity and demo overlays, but no dedicated YOLO-style detection model or bounding-box benchmark was completed.",
    ),
    (
        "Segmentation stretch goal",
        "Not implemented",
        "No U-Net or Mask R-CNN style segmentation branch was added in the final project state.",
    ),
    (
        "SCADA / dashboard integration",
        "Not implemented",
        "The report includes deployment notes, but no live dashboard or SCADA connector was built.",
    ),
    (
        "Validation and deployment timing",
        "Implemented",
        "Held-out evaluation, GPU latency measurement, and branch-wise runtime reporting were completed.",
    ),
    (
        "Controlled laboratory captures",
        "Not implemented",
        "The final system relies on public datasets only, which is consistent with the proposal scope reduction.",
    ),
]


def _load_optional_json(path: str) -> dict:
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return {}
    return load_json(file_path)


def _safe_read_csv(path: str) -> pd.DataFrame:
    file_path = resolve_project_path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if pd.isna(value):
        return "N/A"
    if value.is_integer():
        return f"{int(value):,}"
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    return f"{seconds:.1f} s ({seconds / 60.0:.2f} min)"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    if not rows:
        return ["No data available.", ""]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    lines.append("")
    return lines


def _value_counts_series(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _group_counts(frame: pd.DataFrame, group_columns: list[str]) -> dict[str, dict[str, int]]:
    if frame.empty:
        return {}
    grouped = frame.groupby(group_columns).size()
    output: dict[str, dict[str, int]] = {}
    for keys, value in grouped.items():
        if not isinstance(keys, tuple):
            keys = (keys,)
        outer = str(keys[0])
        inner = str(keys[1])
        output.setdefault(outer, {})[inner] = int(value)
    return output


def _dataset_image_sizes(image_sizes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if image_sizes.empty:
        return {}
    output: dict[str, tuple[float, float]] = {}
    for _, row in image_sizes.iterrows():
        output[str(row["dataset_name"])] = (float(row["width"]), float(row["height"]))
    return output


def _build_deepsolareye_table(baseline: dict, multitask: dict) -> list[list[object]]:
    return [
        [
            "Baseline regression",
            baseline.get("mae"),
            baseline.get("rmse"),
            baseline.get("r2"),
            baseline.get("spearman"),
            "N/A",
            "N/A",
        ],
        [
            "Multitask regression + severity",
            multitask.get("mae"),
            multitask.get("rmse"),
            multitask.get("r2"),
            multitask.get("spearman"),
            multitask.get("accuracy"),
            multitask.get("f1_macro"),
        ],
    ]


def _build_villegas_table(image_only: dict, weather: dict, transfer: dict) -> list[list[object]]:
    return [
        [
            "Image only",
            image_only.get("pmpp_rmse"),
            image_only.get("pmpp_r2"),
            image_only.get("isc_rmse"),
            image_only.get("isc_r2"),
            image_only.get("ff_rmse"),
            image_only.get("ff_r2"),
        ],
        [
            "Image + weather",
            weather.get("pmpp_rmse"),
            weather.get("pmpp_r2"),
            weather.get("isc_rmse"),
            weather.get("isc_r2"),
            weather.get("ff_rmse"),
            weather.get("ff_r2"),
        ],
        [
            "DeepSolarEye transfer + weather",
            transfer.get("pmpp_rmse"),
            transfer.get("pmpp_r2"),
            transfer.get("isc_rmse"),
            transfer.get("isc_r2"),
            transfer.get("ff_rmse"),
            transfer.get("ff_r2"),
        ],
    ]


def _build_villegas_improvement_table(weather: dict, transfer: dict) -> list[list[object]]:
    return [
        [
            "Image + weather",
            PRE_UNIT_FIX_VILLEGAS["weather"]["pmpp_rmse"],
            weather.get("pmpp_rmse"),
            PRE_UNIT_FIX_VILLEGAS["weather"]["isc_rmse"],
            weather.get("isc_rmse"),
            PRE_UNIT_FIX_VILLEGAS["weather"]["pmpp_r2"],
            weather.get("pmpp_r2"),
            PRE_UNIT_FIX_VILLEGAS["weather"]["isc_r2"],
            weather.get("isc_r2"),
        ],
        [
            "Transfer + weather",
            PRE_UNIT_FIX_VILLEGAS["transfer"]["pmpp_rmse"],
            transfer.get("pmpp_rmse"),
            PRE_UNIT_FIX_VILLEGAS["transfer"]["isc_rmse"],
            transfer.get("isc_rmse"),
            PRE_UNIT_FIX_VILLEGAS["transfer"]["pmpp_r2"],
            transfer.get("pmpp_r2"),
            PRE_UNIT_FIX_VILLEGAS["transfer"]["isc_r2"],
            transfer.get("isc_r2"),
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile final teacher-ready report assets.")
    parser.add_argument("--output", default="outputs/reports/final_report.md")
    args = parser.parse_args()

    metadata = _safe_read_csv("data/processed/unified_metadata/metadata_master.csv")
    audit = _load_optional_json("outputs/tables/audit_summary.json")
    image_sizes = _safe_read_csv("outputs/tables/image_size_summary.csv")
    duplicate_summary = _safe_read_csv("outputs/tables/duplicate_summary.csv")
    fusion_predictions = _safe_read_csv("outputs/predictions/fusion_predictions.csv")
    runtime = _load_optional_json("outputs/tables/runtime_summary.json")
    latency = _load_optional_json("outputs/tables/latency_summary.json")

    deepsolareye_baseline = _load_optional_json("outputs/tables/deepsolareye_rgb_powerloss_metrics.json")
    deepsolareye_multitask = _load_optional_json("outputs/tables/deepsolareye_rgb_powerloss_multitask_metrics.json")
    villegas_image_only = _load_optional_json("outputs/tables/villegas_rgb_electrical_image_only_metrics.json")
    villegas_weather = _load_optional_json("outputs/tables/villegas_rgb_electrical_metrics.json")
    villegas_transfer = _load_optional_json("outputs/tables/villegas_rgb_electrical_transfer_metrics.json")
    trsai_metrics = _load_optional_json("outputs/tables/trsai_thermal_hotspot_metrics.json")

    dataset_counts = _value_counts_series(metadata, "dataset_name")
    split_counts = _group_counts(metadata, ["dataset_name", "split"])
    usable_counts = {
        "deepsolareye": _value_counts_series(
            metadata[(metadata["dataset_name"] == "deepsolareye") & metadata["power_loss_pct"].notna()],
            "split",
        ),
        "villegas": _value_counts_series(
            metadata[
                (metadata["dataset_name"] == "villegas")
                & metadata["pmpp"].notna()
                & metadata["isc"].notna()
                & metadata["ff"].notna()
            ],
            "split",
        ),
        "trsai": _value_counts_series(metadata[metadata["dataset_name"] == "trsai"], "split"),
    }
    image_size_map = _dataset_image_sizes(image_sizes)

    hardware = runtime.get("hardware", {})
    data_prep = runtime.get("data_preparation_seconds", {})
    training_seconds = runtime.get("training_seconds", {})
    evaluation_seconds = runtime.get("evaluation_seconds", {})
    priority_counts = _value_counts_series(fusion_predictions, "priority")

    lines: list[str] = [
        "# Final Report",
        "",
        "## Executive Summary",
        "",
        (
            "This report documents the final state of the PV monitoring internship project after reviewing "
            "the proposal (`proposal_last_version.docx`), rebuilding the data pipeline on the real datasets, "
            "training the branch models on GPU, correcting a Villegas metadata unit issue, rerunning the "
            "electrical experiments, and regenerating the fusion and demo artifacts."
        ),
        "",
        (
            "The strongest final results are the DeepSolarEye RGB branch and the Villegas transfer + weather "
            "electrical branch. The thermal TRSAI branch runs end-to-end but remains a prototype because the "
            "available parsed labels currently collapse to a single positive class."
        ),
        "",
        "## What Was Implemented",
        "",
        "- Modular branch architecture: DeepSolarEye RGB power-loss branch, Villegas RGB-electrical branch, TRSAI thermal branch, and a fused maintenance-decision layer.",
        "- Frozen raw datasets under `data/raw/` with unified metadata, leakage-aware splits, and audit outputs.",
        "- Full training and evaluation scripts for RGB regression, multitask severity prediction, electrical regression, transfer learning, thermal hotspot scoring, fusion, latency measurement, and demo generation.",
        "- Teacher-facing artifacts including prediction CSVs, training curves, demo bundles, a fusion report, and this final report.",
        "",
        "## Proposal Alignment Review",
        "",
    ]

    lines.extend(_markdown_table(["Proposal Area", "Status", "Comment"], [list(row) for row in PROPOSAL_ALIGNMENT]))

    lines.extend(
        [
            "## Workflow Executed",
            "",
            "```powershell",
            "python scripts/build_metadata.py --config configs/data.yaml",
            "python scripts/create_splits.py --config configs/data.yaml",
            "python scripts/run_data_audit.py --config configs/data.yaml --max-hash-files-per-dataset 4000",
            "python scripts/train_rgb_powerloss.py --config configs/train_rgb_full.yaml",
            "python scripts/train_rgb_powerloss.py --config configs/train_rgb_multitask_full.yaml",
            "python scripts/train_rgb_electrical.py --config configs/train_regression_image_only_full.yaml",
            "python scripts/train_rgb_electrical.py --config configs/train_regression_full.yaml",
            "python scripts/train_rgb_electrical.py --config configs/train_regression_transfer_full.yaml",
            "python scripts/train_thermal_hotspot.py --config configs/train_thermal_full.yaml",
            "python scripts/evaluate_rgb.py --config configs/train_rgb_full.yaml --split test",
            "python scripts/evaluate_rgb.py --config configs/train_rgb_multitask_full.yaml --split test",
            "python scripts/evaluate_regression.py --config configs/train_regression_image_only_full.yaml --split test",
            "python scripts/evaluate_regression.py --config configs/train_regression_full.yaml --split test",
            "python scripts/evaluate_regression.py --config configs/train_regression_transfer_full.yaml --split test",
            "python scripts/evaluate_thermal.py --config configs/train_thermal_full.yaml --split test",
            "python scripts/run_fusion.py --config configs/fusion.yaml",
            "python scripts/generate_demo_outputs.py --metadata data/processed/unified_metadata/metadata_master.csv --fusion-config configs/fusion.yaml",
            "python scripts/measure_latency.py --rgb-config configs/train_rgb_multitask_full.yaml --regression-config configs/train_regression_transfer_full.yaml --thermal-config configs/train_thermal_full.yaml --repeats 20",
            "```",
            "",
            "## Dataset Summary",
            "",
        ]
    )

    dataset_rows = []
    for dataset_name in ["deepsolareye", "villegas", "trsai"]:
        dataset_rows.append(
            [
                dataset_name,
                dataset_counts.get(dataset_name, 0),
                split_counts.get(dataset_name, {}).get("train", 0),
                split_counts.get(dataset_name, {}).get("val", 0),
                split_counts.get(dataset_name, {}).get("test", 0),
                image_size_map.get(dataset_name, ("N/A", "N/A"))[0],
                image_size_map.get(dataset_name, ("N/A", "N/A"))[1],
            ]
        )
    lines.extend(
        _markdown_table(
            ["Dataset", "Total Rows", "Train", "Val", "Test", "Width", "Height"],
            dataset_rows,
        )
    )

    lines.extend(["### Effective Rows Used For Training / Evaluation", ""])
    usable_rows = []
    for dataset_name in ["deepsolareye", "villegas", "trsai"]:
        usable_rows.append(
            [
                dataset_name,
                usable_counts.get(dataset_name, {}).get("train", 0),
                usable_counts.get(dataset_name, {}).get("val", 0),
                usable_counts.get(dataset_name, {}).get("test", 0),
            ]
        )
    lines.extend(_markdown_table(["Dataset", "Usable Train", "Usable Val", "Usable Test"], usable_rows))

    lines.extend(
        [
            "### Audit Highlights",
            "",
            f"- DeepSolarEye missing `power_loss_pct` rows: {audit.get('deepsolareye', {}).get('power_loss_missing', 'N/A')}",
            f"- DeepSolarEye session count: {audit.get('deepsolareye', {}).get('session_count', 'N/A')}",
            f"- DeepSolarEye near-duplicate temporal pairs: {audit.get('deepsolareye', {}).get('near_duplicate_pairs', 'N/A')}",
            f"- DeepSolarEye extremely close near-duplicates (hash <= 5): {audit.get('deepsolareye', {}).get('near_duplicate_pairs_hash_le_5', 'N/A')}",
            f"- Villegas missing `ff` rows after parsing: {audit.get('villegas', {}).get('ff_missing', 'N/A')}",
            (
                "- Villegas weather coverage: "
                + ", ".join(
                    f"{name}={fraction:.2f}"
                    for name, fraction in audit.get("villegas", {}).get("weather_present_fraction", {}).items()
                )
            ),
            (
                "- TRSAI hotspot balance: "
                + ", ".join(
                    f"class {label}={count}"
                    for label, count in audit.get("trsai", {}).get("hotspot_balance", {}).items()
                )
            ),
            "",
            "### Exact Duplicate Scan",
            "",
        ]
    )

    duplicate_rows = []
    for _, row in duplicate_summary.iterrows():
        duplicate_rows.append(
            [
                row.get("dataset_name"),
                row.get("files_hashed"),
                row.get("duplicate_groups"),
                row.get("duplicate_files"),
            ]
        )
    lines.extend(_markdown_table(["Dataset", "Files Hashed", "Duplicate Groups", "Duplicate Files"], duplicate_rows))

    lines.extend(
        [
            "## Data Preparation Timing",
            "",
            "The first full data rebuild was measured before the Villegas parser correction. "
            "After identifying mixed electrical units in Villegas, the parser was fixed and the full rebuild was rerun.",
            "",
        ]
    )

    initial_prep = data_prep.get("initial_full_rebuild", {})
    post_fix_prep = data_prep.get("post_villegas_unit_fix_rebuild", {})
    lines.extend(
        _markdown_table(
            ["Step", "Initial Full Rebuild", "Post-Fix Rebuild"],
            [
                ["build_metadata.py", _fmt_seconds(initial_prep.get("build_metadata")), _fmt_seconds(post_fix_prep.get("build_metadata"))],
                ["create_splits.py", _fmt_seconds(initial_prep.get("create_splits")), _fmt_seconds(post_fix_prep.get("create_splits"))],
                [
                    "run_data_audit.py --max-hash-files-per-dataset 4000",
                    _fmt_seconds(initial_prep.get("run_data_audit_hash4000")),
                    _fmt_seconds(post_fix_prep.get("run_data_audit_hash4000")),
                ],
                ["Total", _fmt_seconds(initial_prep.get("total")), _fmt_seconds(post_fix_prep.get("total"))],
            ],
        )
    )

    lines.extend(
        [
            "## Improvement Performed After The Initial Full Run",
            "",
            "The main corrective improvement was applied to the Villegas metadata parser in `scripts/build_metadata.py`.",
            "",
            "- Problem found: `Pmpp` and `Isc` values were mixed across different unit scales (`mW` vs `W`, `mA` vs `A`), and a few `FF` entries were invalid outliers.",
            "- Fix applied: harmonize large `Pmpp` values by dividing by `1000` when needed, harmonize large `Isc` values by dividing by `1000` when needed, and null out implausible `FF` values above `2`.",
            "- Result: the electrical targets became physically consistent enough to rerun Villegas experiments meaningfully.",
            "",
            "### Villegas Before / After Unit Harmonization",
            "",
        ]
    )

    lines.extend(
        _markdown_table(
            [
                "Run",
                "Pmpp RMSE Before",
                "Pmpp RMSE After",
                "Isc RMSE Before",
                "Isc RMSE After",
                "Pmpp R2 Before",
                "Pmpp R2 After",
                "Isc R2 Before",
                "Isc R2 After",
            ],
            _build_villegas_improvement_table(villegas_weather, villegas_transfer),
        )
    )

    lines.extend(
        [
            "The rerun demonstrates that the corrected Villegas pipeline is substantially stronger than the pre-fix state, especially for `Pmpp` and `Isc`. "
            "The transfer + weather branch became the best electrical model after the fix.",
            "",
            "## Final Model Results",
            "",
            "### DeepSolarEye",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Model", "MAE", "RMSE", "R2", "Spearman", "Severity Accuracy", "Severity Macro F1"],
            _build_deepsolareye_table(deepsolareye_baseline, deepsolareye_multitask),
        )
    )

    lines.extend(
        [
            "DeepSolarEye is the strongest branch in the final system. The multitask variant improved continuous regression slightly over the baseline and added usable severity outputs, but its severity F1 still leaves room for class-balancing improvements.",
            "",
            "### Villegas",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Model", "Pmpp RMSE", "Pmpp R2", "Isc RMSE", "Isc R2", "FF RMSE", "FF R2"],
            _build_villegas_table(villegas_image_only, villegas_weather, villegas_transfer),
        )
    )

    lines.extend(
        [
            "Final interpretation for Villegas:",
            "",
            "- `Image + weather` clearly improves `Pmpp` and `Isc` over the corrected image-only baseline.",
            "- `DeepSolarEye transfer + weather` is the best final electrical branch, especially for `Pmpp` and `Isc`.",
            "- `FF` remains unstable and should be treated as experimental rather than headline evidence.",
            "",
            "### TRSAI",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Metric", "Value"],
            [[key, value] for key, value in trsai_metrics.items()],
        )
    )

    lines.extend(
        [
            "TRSAI currently behaves as a one-class hotspot prototype because all parsed labels resolve to the positive class. "
            "Therefore, the perfect test metrics are not evidence of a balanced hotspot classifier; they only show that the thermal branch runs correctly end-to-end on the available parsed labels.",
            "",
            "## Final Checkpoints Selected",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Branch", "Checkpoint"],
            [
                ["DeepSolarEye baseline", "outputs/models/deepsolareye/best_powerloss.pt"],
                ["DeepSolarEye multitask", "outputs/models/deepsolareye_multitask/best_powerloss_multitask.pt"],
                ["Villegas best branch", "outputs/models/villegas_transfer/best_electrical_transfer.pt"],
                ["TRSAI thermal prototype", "outputs/models/trsai/best_hotspot.pt"],
            ],
        )
    )

    lines.extend(
        [
            "## Runtime and Deployment Notes",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Hardware / Software", "Value"],
            [
                ["GPU", hardware.get("gpu_name")],
                ["GPU memory", f"{hardware.get('gpu_memory_mib', 'N/A')} MiB"],
                ["Driver", hardware.get("driver_version")],
                ["Python", hardware.get("python_version")],
                ["PyTorch", hardware.get("torch_version")],
                ["Device used for final runs", hardware.get("device")],
            ],
        )
    )

    lines.extend(["### Training Time", ""])
    lines.extend(
        _markdown_table(
            ["Run", "Measured Time", "Note"],
            [
                ["DeepSolarEye baseline", _fmt_seconds(training_seconds.get("deepsolareye_baseline_approx")), "Approximate artifact window"],
                ["DeepSolarEye multitask", _fmt_seconds(training_seconds.get("deepsolareye_multitask_approx")), "Approximate artifact window"],
                ["Villegas image-only (post-fix)", _fmt_seconds(training_seconds.get("villegas_image_only_post_fix")), "Direct timing"],
                ["Villegas image + weather (post-fix)", _fmt_seconds(training_seconds.get("villegas_weather_post_fix")), "Direct timing"],
                ["Villegas transfer + weather (post-fix)", _fmt_seconds(training_seconds.get("villegas_transfer_post_fix")), "Direct timing"],
                ["TRSAI", _fmt_seconds(training_seconds.get("trsai_direct")), "Direct timing"],
                ["Initial full GPU experiment cycle", _fmt_seconds(training_seconds.get("full_gpu_cycle_initial_approx")), "Approximate end-to-end window before the Villegas parser fix"],
            ],
        )
    )

    lines.extend(["### Evaluation Time", ""])
    lines.extend(
        _markdown_table(
            ["Run", "Measured Time"],
            [
                ["DeepSolarEye multitask test pass", _fmt_seconds(evaluation_seconds.get("deepsolareye_multitask_test"))],
                ["Villegas image-only test pass (post-fix)", _fmt_seconds(evaluation_seconds.get("villegas_image_only_post_fix_test"))],
                ["Villegas image + weather test pass (post-fix)", _fmt_seconds(evaluation_seconds.get("villegas_weather_post_fix_test"))],
                ["Villegas transfer + weather test pass (post-fix)", _fmt_seconds(evaluation_seconds.get("villegas_transfer_post_fix_test"))],
                ["TRSAI thermal test pass", _fmt_seconds(evaluation_seconds.get("trsai_test"))],
            ],
        )
    )

    lines.extend(["### Single-Image GPU Latency", ""])
    latency_rows = []
    for key, value in latency.items():
        if key == "device" or not isinstance(value, dict):
            continue
        latency_rows.append([key, value.get("mean_ms"), value.get("median_ms"), value.get("min_ms"), value.get("max_ms"), value.get("repeats")])
    lines.extend(_markdown_table(["Branch", "Mean ms", "Median ms", "Min ms", "Max ms", "Repeats"], latency_rows))

    lines.extend(
        [
            "The proposal target was to keep processing comfortably below 5 seconds per image on practical hardware. "
            "On the workstation GPU, the final branch models are far below that threshold, in the low single-digit millisecond range for single-image inference.",
            "",
            "## Fusion and Demo Outputs",
            "",
            "- Fusion was regenerated using the best available electrical branch (`Villegas transfer + weather`).",
            "- The electrical degradation score was corrected so that lower `Pmpp`, `Isc`, and `FF` now increase risk, which aligns the decision rules with PV performance physics.",
            "- Demo bundles were regenerated from the updated fusion configuration.",
            "",
        ]
    )

    lines.extend(
        _markdown_table(
            ["Fusion Priority", "Count"],
            [[priority, count] for priority, count in priority_counts.items()],
        )
    )

    lines.extend(
        [
            "Key generated artifacts:",
            "",
            "- `outputs/reports/final_report.md`",
            "- `outputs/reports/fusion_report.md`",
            "- `outputs/reports/demo_showcase.md`",
            "- `outputs/predictions/fusion_predictions.csv`",
            "- `outputs/predictions/demo_fusion_cases.csv`",
            "",
            "## Honest Limitations",
            "",
            "- No dedicated object detection benchmark or segmentation branch was completed, even though the proposal discussed them as desirable directions.",
            "- TRSAI thermal evaluation is limited by one-class parsed labels.",
            "- Villegas `FF` prediction is still unstable and should not be overstated.",
            "- The fusion layer is a realistic prototype, but not a rigorously aligned benchmark because the three public datasets do not share panel-level sample IDs.",
            "- No real SCADA integration, edge deployment, or small controlled lab capture validation was completed in the final project state.",
            "",
            "## Teacher-Facing Conclusion",
            "",
            (
                "The project successfully delivered the core internship objective as a modular AI-based PV monitoring prototype built on public datasets. "
                "The strongest evidence comes from the DeepSolarEye branch for RGB power-loss estimation and the corrected Villegas transfer + weather branch for electrical-impact estimation. "
                "The thermal branch is implemented as a prototype, and the fusion layer produces maintenance-oriented outputs suitable for demonstration."
            ),
            "",
            (
                "The main missing items relative to the original proposal are detection / segmentation benchmarking, richer thermal labels, aligned cross-dataset fusion evaluation, "
                "and real deployment integration. These gaps are now clearly documented, so the final report can be submitted honestly: the project is complete as a strong software prototype, "
                "with several well-defined extensions left for future work rather than hidden as if they were finished."
            ),
            "",
        ]
    )

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report summary written to {output_path}")


if __name__ == "__main__":
    main()
