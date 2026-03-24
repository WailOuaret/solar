# Final Report

## Dataset Summary

- Total metadata rows: 27961
- deepsolareye: 21490 samples
- trsai: 1260 samples
- villegas: 5211 samples

## Split Summary

- deepsolareye: test: 1391, train: 16462, val: 3637
- trsai: test: 72, train: 1048, val: 140
- villegas: test: 631, train: 3681, val: 899

## DeepSolarEye Results

### Regression Baseline
- mae: 17.5901
- rmse: 23.6549
- r2: 0.2326
- spearman: 0.6339

### Multitask Regression + Severity
- mae: 18.5128
- rmse: 24.0990
- r2: 0.2035
- spearman: 0.4478
- accuracy: 0.9062
- f1_macro: 0.2377
- precision_macro: 0.2266
- recall_macro: 0.2500

## Villegas Results

### Image-Only Baseline
- pmpp_mae: 15690.6505
- pmpp_rmse: 20794.1976
- pmpp_r2: -0.1703
- pmpp_spearman: 0.4973
- isc_mae: 733.5198
- isc_rmse: 1409.3096
- isc_r2: 0.4090
- isc_spearman: 0.5174
- ff_mae: 2.1745
- ff_rmse: 3.6625
- ff_r2: -3639.1043
- ff_spearman: -0.4445

### Image + Weather
- pmpp_mae: 14129.9790
- pmpp_rmse: 18077.2153
- pmpp_r2: 0.1156
- pmpp_spearman: 0.2207
- isc_mae: 1334.0368
- isc_rmse: 1716.2681
- isc_r2: 0.1235
- isc_spearman: 0.2165
- ff_mae: 2.4124
- ff_rmse: 4.4342
- ff_r2: -5334.6892
- ff_spearman: 0.4806

### DeepSolarEye Transfer + Weather
- pmpp_mae: 13900.0272
- pmpp_rmse: 17543.8020
- pmpp_r2: 0.1670
- pmpp_spearman: 0.2301
- isc_mae: 1319.6074
- isc_rmse: 1685.9702
- isc_r2: 0.1542
- isc_spearman: 0.2214
- ff_mae: 0.9137
- ff_rmse: 1.7054
- ff_r2: -788.2316
- ff_spearman: 0.5115

## TRSAI Results

- accuracy: 1.0000
- f1_macro: 1.0000
- precision_macro: 1.0000
- recall_macro: 1.0000

## Limitations

- TRSAI metadata currently resolves to a single positive class, so its thermal results are one-class hotspot scoring only.
- DeepSolarEye contains many near-duplicate temporal neighbors; leakage-safe splits were required and random image splits would be misleading.
- The reported model runs use reduced sample caps for CPU feasibility and should be treated as prototype baselines, not final full-dataset numbers.
- Cross-branch fusion is implemented in code, but a fully aligned fused evaluation is still pending because the three datasets do not share common sample IDs.

## Deployment Notes

- Raw datasets are frozen under data/raw and parsed into unified metadata without modifying the original files.
- Fast audit options are available in scripts/run_data_audit.py to cap hashing work on large rebuilds.
- Transfer learning from DeepSolarEye to Villegas is now supported through configs/train_regression_transfer.yaml.

## Latency Summary

- device: cpu
- deepsolareye_rgb_powerloss: mean 22.90 ms, median 23.38 ms
- villegas_rgb_electrical_transfer: mean 24.67 ms, median 24.88 ms
- trsai_thermal_hotspot: mean 25.32 ms, median 25.26 ms