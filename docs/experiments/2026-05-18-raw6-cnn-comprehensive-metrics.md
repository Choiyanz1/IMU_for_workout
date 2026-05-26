# Raw6 1D Causal CNN Comprehensive Metrics

Date: 2026-05-18

## Goal

Evaluate the current strongest sequence model with a fuller set of metrics in one run, instead of mixing metrics from older scripts.

## Setup

- Script: `scripts/new_c_pipeline/raw6_cnn_comprehensive_9fold.py`
- Command: `python -u scripts\new_c_pipeline\raw6_cnn_comprehensive_9fold.py --epochs 20 --hidden 64 --output artifacts\cnn_variant_comparison\raw6_cnn_comprehensive_9fold_gpu_h64e20.json`
- Model: Global 2-class 1D Causal CNN
- Input: raw 6-axis IMU, `[batch, 6, 300]`
- Channels: `ax, ay, az, gx, gy, gz`
- Decoder: MA25 + Viterbi penalty 0.3
- Evaluation: 9-fold LOSO, light-weight sessions excluded
- Device: CUDA GPU

## Overall Results

| Metric | Value |
|--------|------:|
| Streams | 220 |
| Rep Precision | 0.8144 |
| Rep Recall | 0.9115 |
| Rep F1 | 0.8602 |
| Exact Count Acc | 0.5182 |
| Within-1 Count Acc | 0.6727 |
| Count MAE | 1.5045 |
| Median Count AE | 0.0000 |
| Count RMSE | 2.7822 |
| Count Bias (pred-gt) | +1.4045 |
| Mean Pred Count | 13.1727 |
| Mean GT Count | 11.7682 |
| Over-count Rate | 0.4591 |
| Under-count Rate | 0.0227 |
| Phase Macro F1 | 0.7589 |
| Phase Accuracy | 0.7636 |
| Transition MAE | 310.0 ms |
| Concentric Segment IoU-F1@0.50 | 0.7295 |
| Eccentric Segment IoU-F1@0.50 | 0.7111 |
| Avg Phase Segment IoU-F1@0.50 | 0.7203 |
| C/E Ratio MAE | 0.6704 |

## Per-Action Highlights

| Action | Rep F1 | Exact | Count MAE | Phase IoU-F1 Avg | C/E MAE |
|--------|------:|------:|----------:|------------------:|--------:|
| db_biceps_curl | 0.9937 | 0.8929 | 0.1429 | 0.9414 | 0.2054 |
| db_squat | 0.9201 | 0.4815 | 1.0000 | 0.7887 | 0.5138 |
| db_triceps_curl | 0.8515 | 0.6296 | 0.4815 | 0.6432 | 0.4968 |
| db_weighted_crunch | 0.8426 | 0.5556 | 1.1852 | 0.6494 | 0.5726 |
| db_rdl | 0.8366 | 0.2143 | 2.9286 | 0.6685 | 1.4011 |
| one_arm_db_row | 0.8270 | 0.3929 | 1.4643 | 0.6674 | 0.6265 |
| db_bench_press | 0.8187 | 0.5357 | 2.2143 | 0.7286 | 0.7747 |
| db_shoulder_press | 0.8012 | 0.4444 | 2.5926 | 0.6704 | 0.7603 |

## Interpretation

The model is strong as a temporal phase model: phase macro F1, transition MAE, and phase IoU-F1 are much better than the RF-style window baseline. The main remaining deployment risk is count bias: recall is high but the model over-counts in 45.9% of streams.

RDL and shoulder press remain the count-stability bottlenecks. Biceps curl is effectively solved under this protocol.
