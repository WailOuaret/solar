# Final Report

## Executive Summary

This report documents the final state of the PV monitoring internship project after reviewing the proposal (`proposal_last_version.docx`), rebuilding the data pipeline on the real datasets, training the branch models on GPU, correcting a Villegas metadata unit issue, rerunning the electrical experiments, and regenerating the fusion and demo artifacts.

The strongest final results are the DeepSolarEye RGB branch and the Villegas transfer + weather electrical branch. The thermal TRSAI branch runs end-to-end but remains a prototype because the available parsed labels currently collapse to a single positive class.

## What Was Implemented

- Modular branch architecture: DeepSolarEye RGB power-loss branch, Villegas RGB-electrical branch, TRSAI thermal branch, and a fused maintenance-decision layer.
- Frozen raw datasets under `data/raw/` with unified metadata, leakage-aware splits, and audit outputs.
- Full training and evaluation scripts for RGB regression, multitask severity prediction, electrical regression, transfer learning, thermal hotspot scoring, fusion, latency measurement, and demo generation.
- Teacher-facing artifacts including prediction CSVs, training curves, demo bundles, a fusion report, and this final report.

## Proposal Alignment Review

| Proposal Area | Status | Comment |
| --- | --- | --- |
| Public-dataset-only workflow | Implemented | The project uses frozen datasets under data/raw and does not attempt drone or robot communication. |
| RGB-based anomaly analysis | Implemented | DeepSolarEye and Villegas branches cover RGB power-loss estimation and RGB-to-electrical estimation. |
| Severity scoring and maintenance recommendation | Implemented | The pipeline exports severity labels, fused risk scores, and operator-facing recommendation text. |
| Thermal hotspot branch | Partially implemented | TRSAI runs end-to-end, but the current labels collapse to a single positive class, so evaluation is only a prototype. |
| Detection / localization | Partially implemented | The system supports severity and demo overlays, but no dedicated YOLO-style detection model or bounding-box benchmark was completed. |
| Segmentation stretch goal | Not implemented | No U-Net or Mask R-CNN style segmentation branch was added in the final project state. |
| SCADA / dashboard integration | Not implemented | The report includes deployment notes, but no live dashboard or SCADA connector was built. |
| Validation and deployment timing | Implemented | Held-out evaluation, GPU latency measurement, and branch-wise runtime reporting were completed. |
| Controlled laboratory captures | Not implemented | The final system relies on public datasets only, which is consistent with the proposal scope reduction. |

## Workflow Executed

```powershell
python scripts/build_metadata.py --config configs/data.yaml
python scripts/create_splits.py --config configs/data.yaml
python scripts/run_data_audit.py --config configs/data.yaml --max-hash-files-per-dataset 4000
python scripts/train_rgb_powerloss.py --config configs/train_rgb_full.yaml
python scripts/train_rgb_powerloss.py --config configs/train_rgb_multitask_full.yaml
python scripts/train_rgb_electrical.py --config configs/train_regression_image_only_full.yaml
python scripts/train_rgb_electrical.py --config configs/train_regression_full.yaml
python scripts/train_rgb_electrical.py --config configs/train_regression_transfer_full.yaml
python scripts/train_thermal_hotspot.py --config configs/train_thermal_full.yaml
python scripts/evaluate_rgb.py --config configs/train_rgb_full.yaml --split test
python scripts/evaluate_rgb.py --config configs/train_rgb_multitask_full.yaml --split test
python scripts/evaluate_regression.py --config configs/train_regression_image_only_full.yaml --split test
python scripts/evaluate_regression.py --config configs/train_regression_full.yaml --split test
python scripts/evaluate_regression.py --config configs/train_regression_transfer_full.yaml --split test
python scripts/evaluate_thermal.py --config configs/train_thermal_full.yaml --split test
python scripts/run_fusion.py --config configs/fusion.yaml
python scripts/generate_demo_outputs.py --metadata data/processed/unified_metadata/metadata_master.csv --fusion-config configs/fusion.yaml
python scripts/measure_latency.py --rgb-config configs/train_rgb_multitask_full.yaml --regression-config configs/train_regression_transfer_full.yaml --thermal-config configs/train_thermal_full.yaml --repeats 20
```

## Dataset Summary

| Dataset | Total Rows | Train | Val | Test | Width | Height |
| --- | --- | --- | --- | --- | --- | --- |
| deepsolareye | 21,490 | 13,602 | 2,516 | 5,372 | 192 | 192 |
| villegas | 5,211 | 3,600 | 649 | 962 | 810 | 672 |
| trsai | 1,260 | 1,048 | 140 | 72 | 640 | 640 |

### Effective Rows Used For Training / Evaluation

| Dataset | Usable Train | Usable Val | Usable Test |
| --- | --- | --- | --- |
| deepsolareye | 13,584 | 2,516 | 5,372 |
| villegas | 3,597 | 648 | 962 |
| trsai | 1,048 | 140 | 72 |

### Audit Highlights

- DeepSolarEye missing `power_loss_pct` rows: 18
- DeepSolarEye session count: 16
- DeepSolarEye near-duplicate temporal pairs: 511
- DeepSolarEye extremely close near-duplicates (hash <= 5): 506
- Villegas missing `ff` rows after parsing: 4
- Villegas weather coverage: irradiance=1.00, temperature=1.00, azimuth=1.00, zenith=1.00, albedo=1.00
- TRSAI hotspot balance: class 1=1260

### Exact Duplicate Scan

| Dataset | Files Hashed | Duplicate Groups | Duplicate Files |
| --- | --- | --- | --- |
| deepsolareye | 4,000 | 0 | 0 |
| trsai | 1,260 | 56 | 112 |
| villegas | 4,000 | 0 | 0 |

## Data Preparation Timing

The first full data rebuild was measured before the Villegas parser correction. After identifying mixed electrical units in Villegas, the parser was fixed and the full rebuild was rerun.

| Step | Initial Full Rebuild | Post-Fix Rebuild |
| --- | --- | --- |
| build_metadata.py | 51.3 s (0.85 min) | 47.6 s (0.79 min) |
| create_splits.py | 2.6 s (0.04 min) | 3.1 s (0.05 min) |
| run_data_audit.py --max-hash-files-per-dataset 4000 | 16.1 s (0.27 min) | 15.1 s (0.25 min) |
| Total | 70.0 s (1.17 min) | 65.9 s (1.10 min) |

## Improvement Performed After The Initial Full Run

The main corrective improvement was applied to the Villegas metadata parser in `scripts/build_metadata.py`.

- Problem found: `Pmpp` and `Isc` values were mixed across different unit scales (`mW` vs `W`, `mA` vs `A`), and a few `FF` entries were invalid outliers.
- Fix applied: harmonize large `Pmpp` values by dividing by `1000` when needed, harmonize large `Isc` values by dividing by `1000` when needed, and null out implausible `FF` values above `2`.
- Result: the electrical targets became physically consistent enough to rerun Villegas experiments meaningfully.

### Villegas Before / After Unit Harmonization

| Run | Pmpp RMSE Before | Pmpp RMSE After | Isc RMSE Before | Isc RMSE After | Pmpp R2 Before | Pmpp R2 After | Isc R2 Before | Isc R2 After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image + weather | 26,690.29 | 17.41 | 2,237.13 | 0.9270 | -0.7057 | 0.2905 | -0.4128 | 0.7346 |
| Transfer + weather | 26,689.98 | 13.67 | 2,236.65 | 0.7974 | -0.7057 | 0.5625 | -0.4122 | 0.8036 |

The rerun demonstrates that the corrected Villegas pipeline is substantially stronger than the pre-fix state, especially for `Pmpp` and `Isc`. The transfer + weather branch became the best electrical model after the fix.

## Final Model Results

### DeepSolarEye

| Model | MAE | RMSE | R2 | Spearman | Severity Accuracy | Severity Macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline regression | 11.26 | 15.90 | 0.5006 | 0.6372 | N/A | N/A |
| Multitask regression + severity | 11.03 | 15.57 | 0.5212 | 0.6219 | 0.5086 | 0.4056 |

DeepSolarEye is the strongest branch in the final system. The multitask variant improved continuous regression slightly over the baseline and added usable severity outputs, but its severity F1 still leaves room for class-balancing improvements.

### Villegas

| Model | Pmpp RMSE | Pmpp R2 | Isc RMSE | Isc R2 | FF RMSE | FF R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Image only | 27.05 | -0.7121 | 1.9478 | -0.1720 | 0.0502 | 0.3585 |
| Image + weather | 17.41 | 0.2905 | 0.9270 | 0.7346 | 0.0983 | -1.4571 |
| DeepSolarEye transfer + weather | 13.67 | 0.5625 | 0.7974 | 0.8036 | 0.0903 | -1.0754 |

Final interpretation for Villegas:

- `Image + weather` clearly improves `Pmpp` and `Isc` over the corrected image-only baseline.
- `DeepSolarEye transfer + weather` is the best final electrical branch, especially for `Pmpp` and `Isc`.
- `FF` remains unstable and should be treated as experimental rather than headline evidence.

### TRSAI

| Metric | Value |
| --- | --- |
| accuracy | 1 |
| f1_macro | 1 |
| precision_macro | 1 |
| recall_macro | 1 |

TRSAI currently behaves as a one-class hotspot prototype because all parsed labels resolve to the positive class. Therefore, the perfect test metrics are not evidence of a balanced hotspot classifier; they only show that the thermal branch runs correctly end-to-end on the available parsed labels.

## Final Checkpoints Selected

| Branch | Checkpoint |
| --- | --- |
| DeepSolarEye baseline | outputs/models/deepsolareye/best_powerloss.pt |
| DeepSolarEye multitask | outputs/models/deepsolareye_multitask/best_powerloss_multitask.pt |
| Villegas best branch | outputs/models/villegas_transfer/best_electrical_transfer.pt |
| TRSAI thermal prototype | outputs/models/trsai/best_hotspot.pt |

## Runtime and Deployment Notes

| Hardware / Software | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8188 MiB |
| Driver | 591.44 |
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu126 |
| Device used for final runs | cuda |

### Training Time

| Run | Measured Time | Note |
| --- | --- | --- |
| DeepSolarEye baseline | 720.0 s (12.00 min) | Approximate artifact window |
| DeepSolarEye multitask | 630.0 s (10.50 min) | Approximate artifact window |
| Villegas image-only (post-fix) | 420.3 s (7.01 min) | Direct timing |
| Villegas image + weather (post-fix) | 426.5 s (7.11 min) | Direct timing |
| Villegas transfer + weather (post-fix) | 426.3 s (7.10 min) | Direct timing |
| TRSAI | 65.7 s (1.09 min) | Direct timing |
| Initial full GPU experiment cycle | 4500.0 s (75.00 min) | Approximate end-to-end window before the Villegas parser fix |

### Evaluation Time

| Run | Measured Time |
| --- | --- |
| DeepSolarEye multitask test pass | 27.0 s (0.45 min) |
| Villegas image-only test pass (post-fix) | 20.0 s (0.33 min) |
| Villegas image + weather test pass (post-fix) | 20.0 s (0.33 min) |
| Villegas transfer + weather test pass (post-fix) | 20.2 s (0.34 min) |
| TRSAI thermal test pass | 9.7 s (0.16 min) |

### Single-Image GPU Latency

| Branch | Mean ms | Median ms | Min ms | Max ms | Repeats |
| --- | --- | --- | --- | --- | --- |
| deepsolareye_rgb_powerloss_multitask | 3.7949 | 3.2884 | 2.3763 | 6.5141 | 20 |
| villegas_rgb_electrical_transfer | 2.4714 | 2.3884 | 2.0543 | 3.0939 | 20 |
| trsai_thermal_hotspot | 3.0796 | 2.3800 | 2.0822 | 11.44 | 20 |

The proposal target was to keep processing comfortably below 5 seconds per image on practical hardware. On the workstation GPU, the final branch models are far below that threshold, in the low single-digit millisecond range for single-image inference.

## Fusion and Demo Outputs

- Fusion was regenerated using the best available electrical branch (`Villegas transfer + weather`).
- The electrical degradation score was corrected so that lower `Pmpp`, `Isc`, and `FF` now increase risk, which aligns the decision rules with PV performance physics.
- Demo bundles were regenerated from the updated fusion configuration.

| Fusion Priority | Count |
| --- | --- |
| high_priority | 72 |
| monitor | 3,081 |
| schedule_check | 3,253 |

Key generated artifacts:

- `outputs/reports/final_report.md`
- `outputs/reports/fusion_report.md`
- `outputs/reports/demo_showcase.md`
- `outputs/predictions/fusion_predictions.csv`
- `outputs/predictions/demo_fusion_cases.csv`

## Honest Limitations

- No dedicated object detection benchmark or segmentation branch was completed, even though the proposal discussed them as desirable directions.
- TRSAI thermal evaluation is limited by one-class parsed labels.
- Villegas `FF` prediction is still unstable and should not be overstated.
- The fusion layer is a realistic prototype, but not a rigorously aligned benchmark because the three public datasets do not share panel-level sample IDs.
- No real SCADA integration, edge deployment, or small controlled lab capture validation was completed in the final project state.

## Teacher-Facing Conclusion

The project successfully delivered the core internship objective as a modular AI-based PV monitoring prototype built on public datasets. The strongest evidence comes from the DeepSolarEye branch for RGB power-loss estimation and the corrected Villegas transfer + weather branch for electrical-impact estimation. The thermal branch is implemented as a prototype, and the fusion layer produces maintenance-oriented outputs suitable for demonstration.

The main missing items relative to the original proposal are detection / segmentation benchmarking, richer thermal labels, aligned cross-dataset fusion evaluation, and real deployment integration. These gaps are now clearly documented, so the final report can be submitted honestly: the project is complete as a strong software prototype, with several well-defined extensions left for future work rather than hidden as if they were finished.
