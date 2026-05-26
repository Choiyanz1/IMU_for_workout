# 2026-05-15 - Direct Event RF Probe

## Goal

Test a lightweight direct-rep alternative that does not first predict
`concentric` / `eccentric` phases.

## Method

- Script: `scripts/evaluate_direct_event_rf.py`
- Model: causal trailing-window RF
- Targets:
  - `other`
  - `start`
  - `transition`
  - `end`
- Decoder: greedy ordered event pairing
  - start -> transition -> end

## Probe 1

- Held-out subject: `yoru`
- Actions: `db_bench_press`, `db_triceps_curl`
- Settings:
  - `window_size = 50`
  - `train_stride = 10`
  - `event_radius = 2`
  - `smoothing_window = 9`
  - `event_threshold = 0.30`
  - `n_estimators = 20`
  - `max_depth = 12`

Output:

- `artifacts/baseline_comparison/direct_event_rf_yoru_probe_v2`

Result:

- overall: `rep_f1 = 0.0000`
- no reps were emitted on either action

## Probe 2

- Held-out subject: `yoru`
- Action: `db_triceps_curl`
- Relaxed settings:
  - `event_radius = 6`
  - `smoothing_window = 15`
  - `event_threshold = 0.05`
  - `n_estimators = 30`
  - `max_depth = 14`

Output:

- `artifacts/baseline_comparison/direct_event_rf_yoru_probe_v3_triceps`

Result:

- precision: `0.9474`
- recall: `0.5143`
- rep F1: `0.6667`
- exact-count streams: `0 / 3`
- all three streams remained under-segmented

## Comparison

Current phase-first RF baseline on held-out `yoru`:

- `db_bench_press`: precision `0.7838`, recall `0.8788`, rep F1 `0.8286`
- `db_triceps_curl`: precision `1.0000`, recall `1.0000`, rep F1 `1.0000`

## Reading

- A minimal direct-event RF is not good enough yet.
- Even after relaxing the sparse-event labeling and threshold, it still misses
  too many reps.
- This suggests the direct-boundary direction is still plausible in principle,
  but a naive RF event detector is weaker than the current phase-first RF on
  this dataset.
