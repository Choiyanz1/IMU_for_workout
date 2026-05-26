# Duration-Aware Rep Merge Decoder Probe

Date: 2026-05-18

## Goal

Reduce over-segmentation without retraining the CNN by merging predicted reps that are shorter than train-fold per-action duration priors.

## Method

- Script: `scripts/new_c_pipeline/duration_merge_decoder_9fold.py`
- Command: `python -u scripts\new_c_pipeline\duration_merge_decoder_9fold.py --epochs 20 --hidden 64 --percentiles "5,10,15,20,25" --max-gap-samples 50 --output artifacts\cnn_variant_comparison\duration_merge_decoder_9fold_gpu_h64e20.json`
- Model: raw 6-axis global 2-class 1D Causal CNN
- Base decoder: MA25 + Viterbi penalty 0.3
- Merge rule: if a predicted rep duration is shorter than the train-fold per-action GT duration percentile, merge it with the next/previous nearby rep.
- Evaluation: 9-fold LOSO, light-weight sessions excluded

## Results

| Decoder | Rep F1 | Exact | Within-1 | Count MAE | Bias | Over Rate | Under Rate | C/E MAE |
|---------|------:|------:|---------:|----------:|-----:|----------:|-----------:|--------:|
| Raw MA25+Viterbi | 0.8607 | 0.5045 | 0.6773 | 1.5091 | +1.40 | 0.4682 | 0.0273 | 0.7003 |
| Merge p5 | 0.8659 | 0.4909 | 0.7500 | 1.1455 | -0.83 | 0.1045 | 0.4045 | 0.6426 |
| Merge p10 | 0.8475 | 0.4591 | 0.7273 | 1.3409 | -1.09 | 0.0909 | 0.4500 | 0.6422 |
| Merge p15 | 0.8278 | 0.4500 | 0.6591 | 1.5409 | -1.34 | 0.0682 | 0.4864 | 0.6473 |
| Merge p20 | 0.8100 | 0.3682 | 0.5818 | 1.7818 | -1.60 | 0.0682 | 0.5636 | 0.6537 |
| Merge p25 | 0.7939 | 0.3545 | 0.5045 | 1.9955 | -1.88 | 0.0409 | 0.6045 | 0.6652 |

## Interpretation

Duration-aware merge confirms the over-segmentation hypothesis. It sharply reduces over-counting and improves Count MAE at p5, but global application over-corrects into under-counting and slightly lowers exact count accuracy.

## Decision

Do not replace the baseline decoder with global duration merge. The next useful variant is selective duration merge applied only to over-count-prone actions or streams flagged as likely over-segmented.
