# Selective Duration-Aware Rep Merge Decoder

Date: 2026-05-18

## Goal

Reduce over-segmentation without the under-counting caused by global duration merge.

## Method

- Script: `scripts/new_c_pipeline/selective_duration_merge_decoder_9fold.py`
- Command: `python -u scripts\new_c_pipeline\selective_duration_merge_decoder_9fold.py --epochs 20 --hidden 64 --percentiles "5,10" --max-gap-samples 50 --output artifacts\cnn_variant_comparison\selective_duration_merge_decoder_9fold_gpu_h64e20.json`
- Model: raw 6-axis global 2-class 1D Causal CNN
- Base decoder: MA25 + Viterbi penalty 0.3
- Merge rule: apply duration merge only to selected over-count-prone action sets.

## Action Sets

| Name | Actions |
|------|---------|
| top3 | `db_rdl`, `db_shoulder_press`, `db_bench_press` |
| top4 | top3 + `one_arm_db_row` |
| top5 | top4 + `db_weighted_crunch` |
| over50 | `db_rdl`, `db_shoulder_press`, `db_squat`, `one_arm_db_row` |
| compound6 | `db_bench_press`, `db_rdl`, `db_shoulder_press`, `db_squat`, `db_weighted_crunch`, `one_arm_db_row` |

## Results

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate | C/E MAE |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|--------:|
| Raw MA25+Viterbi | 0.8582 | 0.5091 | 0.6727 | 1.4955 | +1.38 | 0.4682 | 0.0227 | 0.6544 |
| top3 p5 | 0.8702 | 0.5727 | 0.7409 | 1.0182 | +0.28 | 0.3091 | 0.1182 | 0.6234 |
| top4 p5 | 0.8700 | 0.5909 | 0.7409 | 0.9955 | -0.07 | 0.2273 | 0.1818 | 0.6077 |
| top5 p5 | 0.8737 | 0.5909 | 0.7545 | 0.9727 | -0.36 | 0.1682 | 0.2409 | 0.6010 |
| over50 p5 | 0.8614 | 0.5864 | 0.7364 | 1.2273 | +0.02 | 0.2182 | 0.1955 | 0.5977 |
| compound6 p5 | 0.8672 | 0.6000 | 0.7545 | 1.0500 | -0.64 | 0.1182 | 0.2818 | 0.5860 |

## Decision

Selective duration merge passes the strict gate. The best balanced candidate is `top5_p5`, which improves Rep F1, Exact Count, Within-1 Count, Count MAE, over-rate, and C/E MAE versus raw MA25+Viterbi.

Use `compound6_p5` only if exact count and over-rate are prioritized over under-count risk.
