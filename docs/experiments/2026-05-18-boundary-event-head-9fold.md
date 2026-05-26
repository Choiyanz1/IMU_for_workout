# Boundary/Event Head 9-Fold Probe

Date: 2026-05-18

## Goal

Test whether a lightweight boundary/event head can improve rep grouping and C/E timing while preserving real-time feasibility.

## Method

- Script: `scripts/new_c_pipeline/boundary_event_head_9fold.py`
- Command: `python -u scripts\new_c_pipeline\boundary_event_head_9fold.py --epochs 20 --hidden 64 --boundary-weights "0.1,0.2,0.4" --output artifacts\cnn_variant_comparison\boundary_event_head_9fold_gpu_h64e20.json`
- Input: raw 6-axis IMU, `[batch, 6, 300]`
- Baseline model: phase-only raw6 1D Causal CNN
- Boundary model: same encoder plus a `1x1 Conv` boundary head trained on C/E transition labels expanded by +/-10 samples
- Boundary loss: BCE with `pos_weight=10`, weighted by `0.5`
- Decoders:
  - `raw`: phase-only CNN + MA25 + Viterbi p=0.3
  - `top5_p5`: raw baseline plus selective rep duration merge reference
  - `boundary_phase`: boundary-aware CNN phase head + MA25 + Viterbi p=0.3
  - `boundary_b*`: boundary-aware CNN phase head + MA25 + boundary-bonus Viterbi

## Results

| Decoder | Rep F1 | Exact | Count MAE | C/E MAE | Over | Under |
|---------|------:|------:|----------:|--------:|-----:|------:|
| Raw MA25+Viterbi | 0.8555 | 0.518 | 1.573 | 0.678 | 0.436 | 0.045 |
| top5_p5 selective merge | **0.8702** | **0.577** | **0.977** | **0.604** | 0.173 | 0.250 |
| Boundary phase only | 0.8464 | 0.555 | 1.391 | 0.715 | 0.418 | 0.027 |
| Boundary bonus 0.1 | 0.8437 | 0.536 | 1.450 | 0.722 | 0.436 | 0.027 |
| Boundary bonus 0.2 | 0.8424 | 0.523 | 1.500 | 0.738 | 0.450 | 0.027 |
| Boundary bonus 0.4 | 0.7824 | 0.282 | 2.323 | 0.801 | 0.332 | 0.386 |

## Interpretation

- The boundary head does not pass the C/E-aware gate.
- `boundary_phase` improves Exact Count versus raw in this run, but it hurts Rep F1, Count MAE, and C/E MAE.
- Boundary-bonus Viterbi worsens as the transition bonus increases, suggesting the learned boundary probabilities are not reliable enough for hard transition encouragement.
- `top5_p5` remains the best practical decoder baseline despite being a post-processing method.

## Decision

Reject this boundary/event head configuration. Do not replace `top5_p5`. If boundary supervision is revisited, it should use a different target formulation, such as explicit rep start/transition/end event targets or confidence-aware rep-level proposal scoring, rather than a broad C/E transition band plus transition-bonus Viterbi.
