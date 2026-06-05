# Realtime Soft Top5 Replay

## Goal
- Test the current best automatic direction under stricter realtime constraints.
- Remove offline active-segment extraction and full-sequence rep parsing from the evaluation path.

## Script
- `scripts/evaluate_realtime_soft_top5_pipeline.py`

## Streaming Design
- Global active detector: trailing IMU windows only.
- Action RF: trailing 200-sample windows, posterior accumulated online.
- Phase CNN: trailing 300-sample windows only.
- Phase update: default every `10` samples (`100ms` at 100 Hz); smoke test also tried every sample.
- Rep parser: stateful C/E run parser, emits reps only after a concentric run followed by an eccentric run is observed.
- Soft merge: delayed one-rep merge using the current online action posterior. It can hold a short rep until the next rep arrives, but it does not use future samples beyond that delayed finalization.
- Bounded-latency variant: raw trailing-window phase posterior is smoothed with causal `MA25`, then decoded with fixed-lag Viterbi. With `--fixed-lag-samples 100`, each finalized label uses at most `1.0s` future context.
- Implementation note: windows are batched for speed during replay, but each prediction window contains only samples available up to that replay time.

## Command
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --phase-step-samples 10 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_e5_h64_step10.json
```

## 9-Fold Result
This is a 5-epoch sanity run, not the 20-epoch deployment-quality run.

| Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|
| raw online | `0.455` | `0.086` | `5.22` | `0.196` | `1.902` |
| soft online | `0.454` | `0.091` | `5.36` | `0.196` | `1.879` |

## Corrected Fixed-Lag Result
The first fixed-lag integration mistakenly fed stateful-smoothed phase probabilities into Viterbi. The corrected run matches the phase-latency ablation path: trailing-window phase posterior, causal `MA25`, then fixed-lag Viterbi.

Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8,1.0,1.2 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_fixed_lag100_soft_corrected_e5_h64.json
```

| Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|
| raw online | `0.452` | `0.073` | `5.20` | `0.195` | `1.751` |
| soft online | `0.453` | `0.086` | `5.35` | `0.195` | `1.767` |
| fixed-lag raw | `0.791` | `0.359` | `1.71` | `0.631` | `0.745` |
| fixed-lag soft x0.8 | `0.825` | `0.427` | `1.34` | `0.631` | `0.700` |
| fixed-lag soft x1.0 | `0.809` | `0.395` | `1.55` | `0.631` | `0.681` |
| fixed-lag soft x1.2 | `0.782` | `0.350` | `1.95` | `0.631` | `0.661` |

Best 5-epoch automatic bounded-latency variant: `fixed-lag soft x0.8`. It improves over fixed-lag raw on Rep F1, Exact Count, Count MAE, and C/E MAE while keeping the same phase labels.

## 20-Epoch Follow-Up
The same corrected bounded-latency pipeline was rerun with the 20-epoch phase setting.

Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 20 \
  --hidden 64 \
  --action-n-estimators 50 \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8,1.0,1.2 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_fixed_lag100_soft_corrected_e20_h64.json
```

| Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---:|---:|---:|---:|---:|
| raw online | `0.437` | `0.082` | `5.10` | `0.197` | `1.878` |
| soft online | `0.435` | `0.082` | `5.24` | `0.197` | `1.892` |
| fixed-lag raw | `0.809` | `0.309` | `1.87` | `0.662` | `0.779` |
| fixed-lag soft x0.8 | `0.846` | `0.400` | `1.43` | `0.662` | `0.735` |
| fixed-lag soft x1.0 | `0.824` | `0.382` | `1.61` | `0.662` | `0.708` |
| fixed-lag soft x1.2 | `0.795` | `0.355` | `1.93` | `0.662` | `0.697` |

The 20-epoch model improves phase quality and Rep F1 compared with the 5-epoch run, but Count MAE for `x0.8` is slightly worse (`1.43` vs `1.34`). `x0.8` remains the best count-stable soft merge setting. `x1.2` gives the best C/E MAE but over-merges enough to hurt count.

## Full-Session Rest Gate Check
The evaluator was extended to test full-session failure modes:
- `rest-only`: held-out `big_rest` and `rest_after_set*` fragments should emit zero reps.
- `set + rest tail`: each set is followed by up to `20s` of its matching `rest_after_set*`; predicted reps overlapping the appended rest are counted as rest-tail false positives.
- The fixed-lag parser now uses an active mask, so inactive samples become no-phase boundaries instead of being parsed as C/E labels.

Command:
```sh
python scripts/evaluate_realtime_soft_top5_pipeline.py \
  --epochs 5 \
  --hidden 64 \
  --action-n-estimators 50 \
  --active-threshold 0.7 \
  --phase-step-samples 10 \
  --fixed-lag-samples 100 \
  --merge-threshold-scales 0.8 \
  --rest-tail-seconds 20 \
  --output artifacts/action_recognition/realtime_soft_top5/summary_full_realtime_rest_checks_e5_h64_active07.json
```

| Setup | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE | Rest False Reps | Rest-Tail Overlap Reps |
|---|---:|---:|---:|---:|---:|---:|---:|
| active-masked fixed-lag soft x0.8, active threshold 0.7 | `0.651` | `0.205` | `2.90` | `0.489` | `1.423` | `346 / 274 rest streams` | `131 / 181 appended streams` |

Additional one-fold smoke tests showed the tradeoff:
- Raising active threshold to `0.7` can suppress some rest-only false positives in a small fold, but it undercounts heavily.
- Adding action-active gate `>=0.75` reduced appended-rest overlap in the smoke fold but collapsed set performance (`fixed-lag soft x0.8` Rep F1 `0.310`, Count MAE `7.04`).
- Short active-burst filtering was not sufficient; some rest fragments are classified as sustained active motion.

## Step-1 Smoke Check
To test whether the low result was mainly caused by coarse 100ms phase updates, one held-out fold was rerun with `--phase-step-samples 1`.

| Fold | Variant | Rep IoU-F1@50 | Exact Count | Count MAE | Phase IoU-F1@50 | C/E MAE |
|---|---|---:|---:|---:|---:|---:|
| `_tsenyu_temp`, e1/h16 | raw online step1 | `0.561` | `0.080` | `2.28` | `0.372` | `1.212` |
| `_tsenyu_temp`, e1/h16 | soft online step1 | `0.569` | `0.080` | `2.28` | `0.372` | `1.190` |

The step-1 smoke improves count error on that fold, but phase IoU remains low. The strict online gap is therefore not only an update-rate issue.

## Interpretation
- The current offline/global-active pipeline is not yet ready for strict realtime use.
- Soft top5 helps in offline/global-active replay, but under strict online phase decoding the phase/rep signal is too weak for the merge policy to help.
- Fixed-lag Viterbi changes the conclusion for bounded-latency realtime: with `1.0s` label delay, the automatic pipeline recovers most of the offline/global-active Rep F1 and soft top5 becomes useful again.
- Full-session rest checks change the deployment conclusion: the current active gate is not safe enough for complete automatic realtime use.
- The offline pipeline relies on operations that are not equivalent to strict online inference:
  - active probabilities are computed over full start windows and converted to full active segments;
  - phase probabilities are averaged over overlapping windows across an active segment;
  - moving average and Viterbi decoding are applied over complete sequences;
  - rep parsing is run over the full sequence.
- The online script removes those advantages and exposes the current realtime bottleneck: online phase decoding and active gating.

## Decision
- Do not claim 0-lag fully automatic realtime readiness.
- Treat `fixed-lag Viterbi + soft top5 x0.8` as the best bounded-latency decoder for active/cropped sets, not yet a complete full-session deployment pipeline.
- Complete realtime use is blocked by active gating/rest suppression, not by the phase decoder alone.
- Next engineering target should be a streaming active state machine trained and calibrated on full sessions/rest tails, then rerun this full-session gate before exporting the corrected decoder into the live replay script.
