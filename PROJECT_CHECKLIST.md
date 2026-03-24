# PV Monitoring Project Checklist

Last updated: 2026-03-23
Workspace: `c:\Users\wailo\Desktop\solar`

## Current Status

- The proposal/planning work is mostly defined.
- The repository scaffold has been created.
- Core code for metadata, datasets, models, training, inference, scripts, notebooks, and tests is now present.
- Real dataset download, model training, and experiment results are still pending.

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

- [ ] Download DeepSolarEye
- [ ] Download Villegas dataset
- [ ] Download TRSAI dataset
- [ ] Freeze raw datasets under `data/raw/`
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
- [ ] Audit sample counts
- [ ] Audit image sizes
- [ ] Audit missing metadata
- [ ] Audit class imbalance
- [ ] Audit duplicates / near-duplicates
- [ ] Audit temporal leakage risk
- [ ] Save audit figures

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
- [ ] Train DeepSolarEye RGB baseline
- [ ] Evaluate power-loss regression
- [ ] Add severity classification
- [ ] Train Villegas image-only regression
- [ ] Train Villegas image + weather regression
- [ ] Compare image-only vs image + weather
- [ ] Train TRSAI thermal baseline
- [ ] Evaluate hotspot branch

### Phase 10: Transfer and Fusion

- [ ] Pretrain RGB branch on DeepSolarEye
- [ ] Fine-tune on Villegas
- [ ] Compare transfer vs scratch
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
- [ ] Measure latency / inference time
- [ ] Document limitations
- [x] Write final report
- [ ] Prepare demo-ready outputs
- [x] Add tests

## Honest Progress Summary

- Planning: done at a high level
- Code implementation: scaffold and first full pass completed
- Data acquisition: download workflow prepared, raw data still missing
- Training: not started in this repo
- Evaluation: not started in this repo
- Final system: architecture and inference logic implemented, but no trained checkpoints yet

## Recommended Next 5 Actions

1. Create the repo structure and the base files: `README.md`, `requirements.txt`, `.gitignore`, `configs/`.
2. Download the three datasets and freeze them in `data/raw/`.
3. Implement the unified schema and metadata builder.
4. Create leakage-safe splits before training anything.
5. Train the first DeepSolarEye baseline only after the audit notebook is complete.

## Update Rule

Use this file as the master checklist.

- Mark planning items as done only once they are clearly decided.
- Mark implementation items as done only when the file/script/notebook exists and runs.
- Do not mark dataset steps as done until the raw data is actually present in the repo structure.
- Do not mark training steps as done until a checkpoint and metrics are saved.
