# Phase Latency Ablation

## Goal
- Diagnose why strict realtime replay performed poorly.
- Separate four possible causes: global active gate, trailing CNN probability quality, offline Viterbi/post-processing, and stateful online rep parsing.
- Test whether accepting about 1s latency can recover useful performance.

## Script
- `scripts/evaluate_phase_latency_ablation.py`

## Variants
- `offline_predict_fast`: global active detector, offline active segments, existing overlapping-window phase inference, MA25, full-sequence Viterbi, offline `parse_reps`.
- `causal_raw_stateful`: trailing-window CNN probabilities, no full-sequence post-processing, stateful online parser.
- `causal_past_ma25_parse`: trailing-window CNN probabilities, causal/past MA25, offline `parse_reps`. This is not event-online, but it tests whether causal smoothing is enough if rep finalization can be delayed.
- `causal_full_viterbi_parse`: trailing-window CNN probabilities, MA25, full-sequence Viterbi, offline `parse_reps`. This is a diagnostic upper bound, not a valid 1s-latency method.
- `fixed_lag_viterbi_50/100`: trailing-window CNN probabilities, MA25, Viterbi labels finalized after only `50` or `100` future samples. These are legal bounded-latency realtime variants.
- `lookahead_center_ma_parse`: trailing-window CNN probabilities with centered smoothing using `100` future samples. This approximates a simple 1s lookahead smoother, but does not include fixed-lag Viterbi.

## Command
```sh
python scripts/evaluate_phase_latency_ablation.py \
  --epochs 5 \
  --hidden 64 \
  --phase-step-samples 10 \
  --lookahead-samples 100 \
  --fixed-lags-samples 50,100 \
  --output artifacts/action_recognition/phase_latency_ablation/summary_e5_h64_step10_fixed_lag.json
```

## 9-Fold Result
| Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|
| offline predict_fast | `0.804` | `0.441` | `2.00` | `0.644` | `1.106` |
| causal raw stateful | `0.697` | `0.255` | `3.40` | `0.594` | `1.204` |
| causal past MA25 + parse | `0.791` | `0.350` | `2.02` | `0.613` | `0.744` |
| causal full Viterbi + parse | `0.805` | `0.386` | `1.78` | `0.628` | `0.770` |
| fixed-lag Viterbi 0.5s | `0.810` | `0.386` | `1.78` | `0.629` | `0.856` |
| fixed-lag Viterbi 1.0s | `0.810` | `0.386` | `1.78` | `0.637` | `0.787` |
| 1s centered MA + parse | `0.756` | `0.255` | `2.48` | `0.586` | `1.095` |

## Step-1 Fold Check
One fold was rerun with `--phase-step-samples 1` to check update-rate sensitivity.

| Fold | Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---|---:|---:|---:|---:|---:|
| `_tsenyu_temp`, e5/h64 | offline predict_fast | `0.879` | `0.480` | `1.36` | `0.769` | `0.897` |
| `_tsenyu_temp`, e5/h64 | causal raw stateful | `0.574` | `0.400` | `2.08` | `0.469` | `2.608` |
| `_tsenyu_temp`, e5/h64 | causal past MA25 + parse | `0.803` | `0.320` | `2.60` | `0.613` | `0.818` |
| `_tsenyu_temp`, e5/h64 | causal full Viterbi + parse | `0.880` | `0.520` | `1.00` | `0.685` | `0.602` |
| `_tsenyu_temp`, e5/h64 | 1s centered MA + parse | `0.859` | `0.240` | `1.16` | `0.666` | `0.581` |

## Interpretation
- The phase CNN is not completely unusable in causal trailing-window mode. When trailing probabilities receive stronger post-processing, performance approaches offline replay.
- The strict stateful parser is the weakest link: `causal_raw_stateful` drops to Rep F1 `0.692` and Count MAE `3.33`, while the same trailing probabilities with causal MA25 and offline parser reach Rep F1 `0.779` and Count MAE `2.13`.
- Full-sequence Viterbi recovers nearly all offline Count MAE (`1.78` vs `2.00`) and Rep F1 (`0.805` vs `0.804`).
- Fixed-lag Viterbi achieves the same bounded-latency recovery without using the whole future sequence: both 0.5s and 1.0s lag reach Rep F1 `0.810` and Count MAE `1.78`.
- The fixed-lag result is currently the closest valid realtime reproduction of offline performance. The main remaining deficit is Exact Count (`0.386` vs offline `0.441`) and some C/E/phase alignment, not count MAE or Rep F1.
- Simple 1s centered moving average is not enough by itself in the 9-fold run. It over-smooths some subjects and hurts Count MAE (`2.53`) versus causal past MA25 (`2.13`).
- On one fold with per-sample updates, 1s centered smoothing performed well on Count MAE (`1.16`) and C/E MAE (`0.581`), so latency can help, but the strategy must be tuned per the full LOSO protocol.

## Causal vs 1s-Delay Decision
- If output must be immediate, the decoder must stay causal and needs a better online state machine.
- If about 1s delay is acceptable, the system may use bounded lookahead. That does not permit using the whole future sequence, but it does allow fixed-lag smoothing, fixed-lag Viterbi, and delayed rep finalization.
- Because the current phase model is a causal CNN, future samples do not help the target sample inside the CNN itself. The useful place to spend 1s latency is post-processing: active hysteresis, fixed-lag phase filtering, transition confirmation, and rep finalization.
- A truly non-causal model can be used only if it is explicitly bounded to the chosen latency window, e.g. a CNN/TCN that sees at most 1s future context and is trained/evaluated under that same constraint.

## Next Step
- Promote fixed-lag Viterbi to the realtime decoder path.
- Add soft top5 merge after fixed-lag Viterbi labels, then rerun the full automatic pipeline.
- Improve Exact Count with confirmed-transition rep finalization and duration-aware merge calibration.
