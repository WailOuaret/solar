# PV Monitoring Project Checklist

Last updated: 2026-03-24
Workspace: `c:\Users\wailo\Desktop\solar`

## Current Status

- The repository scaffold is complete.
- The three raw datasets are frozen under `data/raw/`.
- Metadata, leakage-safe splits, and audit artifacts have been generated on the real data.
- Full-data GPU experiments have been completed for DeepSolarEye, Villegas, and TRSAI.
- The Villegas parser was corrected for mixed electrical target units and the electrical experiments were rerun on the fixed metadata.
- Demo-ready prediction bundles and a manual fusion showcase have been generated.
- A fully aligned cross-dataset fusion benchmark is still pending because the datasets do not share common sample IDs.

## Done Already (Planning / Scope)

- [x] Project idea defined: modular PV monitoring system
- [x] Main datasets identified
- [x] DeepSolarEye chosen for RGB soiling and power-loss learning
- [x] Villegas 2022 chosen for RGB + electrical modeling
- [x] TRSAI 2024 chosen for thermal hotspot detection
- [x] Decision made to use the datasets as separate branches, not one mixed classifier
- [x] High-level architecture defined
- [x] RGB power-loss branch defined
- [x] RGB electrical-impact branch defined
- [x] Thermal hotspot branch defined
- [x] Fusion and maintenance-decision branch defined
- [x] Repo structure planned
- [x] Training phases planned
- [x] Evaluation metrics planned
- [x] Week-by-week roadmap planned
- [x] Novelty/contribution statements drafted

## Not Done Yet In This Repo (Implementation)

### Phase 1: Repo Bootstrap

- [x] Initialize the project repository
- [x] Create `README.md`
- [x] Create `requirements.txt`
- [x] Create `.gitignore`
- [x] Create `pyproject.toml`
- [x] Create the folder structure from the proposal

### Phase 2: Configs

- [x] Create `configs/data.yaml`
- [x] Create RGB training config
- [x] Create electrical regression config
- [x] Create thermal training config
- [x] Create fusion config
- [x] Create inference config

### Phase 3: Dataset Acquisition

- [x] Download DeepSolarEye
- [x] Download Villegas dataset
- [x] Download TRSAI dataset
- [x] Freeze raw datasets under `data/raw/`
- [x] Record dataset licenses
- [x] Record dataset versions
- [x] Record dataset checksums
- [x] Write dataset cards for all datasets

### Phase 4: Metadata and Splits

- [x] Create unified schema
- [x] Implement DeepSolarEye metadata parser
- [x] Implement Villegas metadata parser
- [x] Implement TRSAI metadata parser
- [x] Build one master metadata file
- [x] Design leakage-safe split logic
- [x] Create DeepSolarEye split file
- [x] Create Villegas split file
- [x] Create TRSAI split file
- [x] Add leakage checks

### Phase 5: Data Layer

- [x] Create `src/data/schema.py`
- [x] Create `src/data/datasets.py`
- [x] Create RGB transforms
- [x] Create thermal transforms
- [x] Create data loaders

### Phase 6: EDA and Audit

- [x] Create `notebooks/01_data_audit.ipynb`
- [x] Audit sample counts
- [x] Audit image sizes
- [x] Audit missing metadata
- [x] Audit class imbalance
- [x] Audit duplicates / near-duplicates
- [x] Audit temporal leakage risk
- [x] Save audit figures

### Phase 7: Models

- [x] Create shared RGB backbone
- [x] Create power-loss head
- [x] Create electrical head
- [x] Create thermal hotspot head
- [x] Create fusion head

### Phase 8: Training Infrastructure

- [x] Create reusable training engine
- [x] Create loss functions
- [x] Create metrics module
- [x] Create callbacks / checkpointing

### Phase 9: First Baselines

- [x] Create DeepSolarEye training script
- [x] Create Villegas training script
- [x] Create TRSAI training script
- [x] Train DeepSolarEye RGB baseline
- [x] Evaluate power-loss regression
- [x] Add severity classification
- [x] Train Villegas image-only regression
- [x] Train Villegas image + weather regression
- [x] Compare image-only vs image + weather
- [x] Train TRSAI thermal baseline
- [x] Evaluate hotspot branch

### Phase 10: Transfer and Fusion

- [x] Pretrain RGB branch on DeepSolarEye
- [x] Fine-tune on Villegas
- [x] Compare transfer vs scratch
- [x] Build severity mapping
- [x] Build decision rules
- [x] Build fusion pipeline

### Phase 11: Inference and Reporting

- [x] Create end-to-end inference pipeline
- [x] Support RGB input
- [x] Support thermal input
- [x] Output severity score
- [x] Output maintenance recommendation
- [x] Generate report artifacts

### Phase 12: Analysis and Finalization

- [x] Create error-analysis notebook
- [x] Measure latency / inference time
- [x] Document limitations
- [x] Write final report
- [x] Prepare demo-ready outputs
- [x] Add tests

## Honest Progress Summary

- Planning: done
- Code implementation: scaffold, branch scripts, and demo/report utilities completed
- Data acquisition: done
- Metadata and splits: done on real data
- Audit: done on real data
- Training: full-data GPU runs completed for all three branches
- Evaluation: final branch metrics, post-fix Villegas reruns, latency, fusion outputs, and smoke-test inference runs completed
- Final system: branch checkpoints exist, manual fusion showcase is generated, aligned fusion benchmarking still pending

## Recommended Next 5 Actions

1. If TRSAI labels can be enriched with negatives, rerun the thermal branch as a real binary hotspot task.
2. Build an aligned fusion evaluation set with shared inspection cases across modalities.
3. Improve Villegas target quality and scaling before claiming strong electrical-impact performance.
4. Add localization or weak-visualization outputs to improve interpretability in the final demo.
5. Re-export the final report after any new checkpoint or label changes.

## Update Rule

Use this file as the master checklist.

- Mark planning items as done only once they are clearly decided.
- Mark implementation items as done only when the file/script/notebook exists and runs.
- Do not mark dataset steps as done until the raw data is actually present in the repo structure.
- Do not mark training steps as done until a checkpoint and metrics are saved.
