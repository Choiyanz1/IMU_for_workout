# Per-Action Duration Merge Oracle

Date: 2026-05-18

## Goal

Check whether per-action duration merge tuning can reduce Count MAE below 0.6.

## Method

- Script: `scripts/new_c_pipeline/per_action_duration_merge_oracle_9fold.py`
- Command: `python -u scripts\new_c_pipeline\per_action_duration_merge_oracle_9fold.py --epochs 20 --hidden 64 --percentiles "5,10,15,20,25,30" --max-gap-samples 50 --output artifacts\cnn_variant_comparison\per_action_duration_merge_oracle_9fold_gpu_h64e20.json`
- Model: raw 6-axis global 2-class 1D Causal CNN
- Decoder: MA25 + Viterbi p=0.3, followed by optional duration merge
- Options per action: `none`, p5, p10, p15, p20, p25, p30

Important: this is an oracle upper bound because the best option per action is selected from held-out results.

## Results

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|
| Raw none | 0.8560 | 0.518 | 0.673 | 1.61 | +1.46 | 0.450 | 0.032 |
| Global p5 | 0.8663 | 0.505 | 0.750 | 1.19 | -0.82 | 0.100 | 0.395 |
| Per-action oracle | 0.8781 | 0.568 | 0.755 | 1.03 | -0.41 | 0.168 | 0.264 |

## Selected Options

| Action | Option | Action MAE | Exact | Rep F1 |
|--------|--------|-----------:|------:|------:|
| db_bench_press | p15 | 1.286 | 0.536 | 0.817 |
| db_biceps_curl | none | 0.179 | 0.893 | 0.992 |
| db_rdl | p10 | 1.607 | 0.429 | 0.836 |
| db_shoulder_press | p5 | 1.148 | 0.481 | 0.887 |
| db_squat | none | 0.963 | 0.556 | 0.919 |
| db_triceps_curl | none | 0.593 | 0.630 | 0.850 |
| db_weighted_crunch | p5 | 1.111 | 0.519 | 0.895 |
| one_arm_db_row | p5 | 1.321 | 0.500 | 0.825 |

## Decision

Per-action duration tuning improves over the raw decoder but cannot plausibly reach Count MAE below 0.6. The next promising direction is stream-level count calibration or a count-constrained decoder.
