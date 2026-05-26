# C/E Duration-Constrained Decoder

Date: 2026-05-18

## Goal

Test a decoder-only alternative that directly targets C/E phase fragmentation without adding model heads or post-hoc count calibration.

## Method

- Script: `scripts/new_c_pipeline/ce_duration_constrained_decoder_9fold.py`
- Command: `python -u scripts\new_c_pipeline\ce_duration_constrained_decoder_9fold.py --epochs 20 --hidden 64 --percentiles "1,5,10,15" --output artifacts\cnn_variant_comparison\ce_duration_constrained_decoder_9fold_gpu_h64e20.json`
- Model: raw 6-axis global 2-class 1D Causal CNN
- Base decoder: MA25 + Viterbi penalty 0.3
- Experiment decoder: suppress predicted C/E fragments shorter than the train-fold per-action, per-phase GT duration percentile.
- Reference decoder in same run: `top5_p5` selective duration merge.

## Results

| Decoder | Rep F1 | Exact | Count MAE | C/E MAE | Over | Under |
|---------|------:|------:|----------:|--------:|-----:|------:|
| Raw MA25+Viterbi | 0.8549 | 0.518 | 1.573 | 0.675 | 0.459 | 0.023 |
| top5_p5 selective merge | **0.8766** | **0.568** | **0.955** | **0.636** | 0.186 | 0.245 |
| C/E duration p1 | 0.8373 | 0.477 | 1.332 | 0.701 | **0.086** | 0.436 |
| C/E duration p5 | 0.7776 | 0.336 | 1.918 | 0.828 | 0.077 | 0.586 |
| C/E duration p10 | 0.7348 | 0.218 | 2.377 | 0.927 | 0.082 | 0.700 |
| C/E duration p15 | 0.7097 | 0.182 | 2.595 | 0.985 | 0.082 | 0.736 |

## Interpretation

- Simple C/E fragment suppression reduces over-counting, but it over-corrects into severe under-counting.
- C/E MAE gets worse for every tested percentile, so this method fails the C/E-aware acceptance gate.
- The failure suggests that many short predicted C/E fragments are not safely removable at the phase-run level; removing them destroys legitimate reps for subjects/actions with shorter or irregular phases.
- The current `top5_p5` rep-level merge remains the better decoder baseline because it improves count metrics while preserving C/E labels more than direct phase-run suppression.

## Decision

Reject simple C/E duration-constrained phase suppression. Do not promote it to the main pipeline. The next step should not be more aggressive duration thresholds; if we need to improve both Count MAE and C/E MAE, use a more informed boundary/event model or a confidence-aware rep-level decoder that does not erase C/E phase structure.
