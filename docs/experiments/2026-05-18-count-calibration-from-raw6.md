# Count Calibration from Raw6 CNN Predictions

Date: 2026-05-18

## Goal

Test whether a real-time-safe post-hoc count calibrator can reduce the raw CNN over-count bias without retraining the CNN or changing rep boundaries.

## Method

- Script: `scripts/new_c_pipeline/count_calibration_from_raw_results.py`
- Command: `python -u scripts\new_c_pipeline\count_calibration_from_raw_results.py --input artifacts\cnn_variant_comparison\raw6_cnn_comprehensive_9fold_gpu_h64e20.json --output artifacts\cnn_variant_comparison\count_calibration_from_raw6_loso.json`
- Source predictions: `artifacts/cnn_variant_comparison/raw6_cnn_comprehensive_9fold_gpu_h64e20.json`
- Protocol: subject-wise LOSO over existing held-out raw6 CNN stream predictions
- Allowed inference-time features: `action`, `pred_count`
- Important caveat: calibrated count does not change rep boundaries, so this is a final displayed-count correction, not a Rep F1 replacement.

## Results

| Method | Exact | Within-1 | Count MAE | RMSE | Bias | Over | Under |
|--------|------:|---------:|----------:|-----:|-----:|-----:|------:|
| Raw identity | 0.518 | 0.673 | 1.505 | 2.782 | +1.405 | 0.459 | 0.023 |
| Global bias | 0.118 | 0.605 | 1.782 | 2.544 | +0.182 | 0.305 | 0.577 |
| Action bias | 0.282 | 0.582 | 1.655 | 2.484 | +0.091 | 0.305 | 0.414 |
| Action linear | 0.573 | **0.859** | **0.759** | **1.664** | -0.005 | 0.200 | 0.227 |
| Action pred-count median lookup | 0.645 | 0.841 | 0.918 | 2.102 | +0.355 | 0.200 | 0.155 |
| Nested action selector, MAE | 0.623 | 0.845 | 0.859 | 1.978 | +0.068 | **0.191** | 0.186 |
| Nested action selector, Exact | **0.668** | 0.836 | 1.073 | 2.616 | +0.564 | 0.245 | 0.086 |

## Duration Feature Probe

Command: `python -u scripts\new_c_pipeline\count_calibration_from_raw_results.py --include-duration --input artifacts\cnn_variant_comparison\raw6_cnn_comprehensive_9fold_gpu_h64e20.json --output artifacts\cnn_variant_comparison\count_calibration_from_raw6_loso_duration.json`

| Method | Exact | Within-1 | Count MAE | RMSE | Bias | Over | Under |
|--------|------:|---------:|----------:|-----:|-----:|-----:|------:|
| Action linear | 0.573 | 0.859 | **0.759** | **1.664** | -0.005 | 0.200 | 0.227 |
| Action linear + total duration | 0.577 | 0.859 | 0.800 | 1.748 | +0.064 | 0.223 | 0.200 |
| Nested selector, Exact + duration candidates | **0.677** | 0.841 | 1.059 | 2.612 | +0.550 | 0.236 | 0.086 |

Adding total stream duration does not improve the MAE-oriented calibrator. This rules out a simple duration-only correction as the next path; richer prediction-derived features are needed, especially predicted rep duration distribution and short-fragment counts.

## Interpretation

- Count calibration has real signal even with only `action + raw predicted count`.
- `action_linear` is the best MAE-oriented option: Count MAE drops from 1.505 to 0.759 and count bias is essentially removed.
- `nested_action_select_exact` is the best exact-count option: Exact Count improves from 0.518 to 0.668, but it leaves larger heavy-tail errors.
- Simple global/action mean bias correction over-corrects and creates under-counting, so the count relationship is not just a constant offset.

## Decision

Keep `action_linear` as the best immediate real-time-safe final-count correction candidate. It still does not reach the Count MAE < 0.6 target, and total stream duration does not help. The next experiment should add prediction-derived features such as active duration, predicted rep duration statistics, short-rep count, phase confidence, and active fragmentation, or move to a boundary/count-density head that directly supervises count events.
