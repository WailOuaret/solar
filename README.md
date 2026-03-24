# Solar PV Monitoring System

Modular photovoltaic monitoring system for:

- RGB soiling and power-loss estimation
- RGB shading and electrical-impact regression
- Thermal hotspot detection
- Fusion into maintenance-oriented decisions

## Scope

This repository follows the project plan defined in the proposal:

- `DeepSolarEye` for RGB soiling and power-loss learning
- `Villegas et al. 2022` for RGB plus electrical behavior regression
- `TRSAI 2024` for thermal hotspot detection

The datasets are not mixed into one flat label space. Each dataset drives a separate branch, and the final system fuses branch outputs into actionable maintenance recommendations.

## Repository Layout

```text
solar/
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
├── src/
└── tests/
```

## Project Branches

### 1. RGB Power-Loss Branch

- Primary dataset: `DeepSolarEye`
- Inputs: RGB image, optional irradiance and timestamp features
- Outputs: `power_loss_pct`, severity class, optional localization cues

### 2. RGB Electrical Branch

- Primary dataset: `Villegas`
- Inputs: RGB image, optional weather features
- Outputs: `pmpp`, `isc`, `ff`, electrical degradation proxy

### 3. Thermal Hotspot Branch

- Primary dataset: `TRSAI`
- Inputs: thermal image
- Outputs: hotspot probability, hotspot severity

### 4. Fusion Branch

- Inputs: branch scores and branch predictions
- Outputs: final severity and maintenance recommendation

## Quick Start

### 1. Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare datasets

```bash
python scripts/download_data.py --config configs/data.yaml
```

This script creates the expected dataset structure and records dataset manifests. Some sources may still require manual download because of DOI, publisher, or Mendeley restrictions.

### 3. Build metadata

```bash
python scripts/build_metadata.py --config configs/data.yaml
python scripts/create_splits.py --config configs/data.yaml
```

### 4. Train baseline branches

```bash
python scripts/train_rgb_powerloss.py --config configs/train_rgb.yaml
python scripts/train_rgb_electrical.py --config configs/train_regression.yaml
python scripts/train_thermal_hotspot.py --config configs/train_thermal.yaml
```

### 5. Run fusion

```bash
python scripts/run_fusion.py --config configs/fusion.yaml
```

## Evaluation

### DeepSolarEye branch

- MAE
- RMSE
- R2
- Spearman correlation
- severity-bin F1

### Villegas branch

- RMSE on `pmpp`
- RMSE on `isc`
- MAE or RMSE on `ff`
- calibration plots

### TRSAI branch

- precision
- recall
- mAP if detection is added
- IoU or Dice if segmentation is added

### Full system

- final severity accuracy
- maintenance-priority quality
- inference time per image

## Exact Build Order

1. Freeze raw datasets
2. Build unified metadata
3. Create leakage-safe splits
4. Train DeepSolarEye baseline
5. Train Villegas baseline
6. Train TRSAI baseline
7. Build fusion logic
8. Evaluate and report

## Notes

- Raw datasets should remain immutable under `data/raw/`.
- Splits should avoid temporal or session leakage.
- This codebase is written to support the proposal structure even before all datasets are present locally.

