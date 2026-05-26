# 2026-05-15 - Nested Per-Action RF Benchmark Smoke Validation

## Goal

Validate the new nested cross-subject benchmark pipeline before launching the
full modality and window sweep.

## Script

- `scripts/benchmark_per_action_rf_refiner.py`

## What The Script Does

- Loads set-level streams from the configured subjects and actions.
- Uses train subjects only to compute per-action rep duration statistics.
- Generates per-action trailing-window candidates from train-fold median rep
  duration.
- Generates per-action edge-window candidates from train-fold median rep
  duration.
- Derives per-action min/max rep-duration priors from train-fold quantiles.
- Runs inner subject-wise model selection on train subjects only.
- Re-trains the selected setting on the full outer train fold.
- Evaluates on the held-out subject.
- Writes JSON, CSV, HTML, and replay SVG artifacts.

## Smoke Command

```bash
python scripts/benchmark_per_action_rf_refiner.py \
  --config configs/micro_macro_recognition_8act_test_yushuan.yaml \
  --output artifacts/baseline_comparison/smoke_nested_rf_tiny \
  --outer-subjects yushuan \
  --include-actions db_bench_press \
  --modalities acc \
  --max-inner-subjects 1 \
  --trailing-multipliers 0.25 \
  --edge-multipliers 0.10 \
  --n-estimators 10 \
  --max-depth 10 \
  --max-samples 0.5 \
  --target-matched-reps 100 \
  --max-refiner-train-streams 10
```

## Smoke Scope

- Outer held-out subject: `yushuan`
- Action subset: `db_bench_press`
- Modality subset: `acc`
- Inner validation subjects: first train subject only
- Reduced RF / refiner budget to keep runtime small

## Output

- `artifacts/baseline_comparison/smoke_nested_rf_tiny`

Files confirmed:

- `results.json`
- `stream_metrics.csv`
- `index.html`
- `summary.html`
- `best_config_per_action.csv`
- `selection_summary.csv`
- `subject_action_coverage.csv`
- `yushuan/duration_stats.json`
- `yushuan/duration_report.md`
- `yushuan/stream_replays/.../*.svg`

## Result Snapshot

Tuned nested output:

- `rep_f1 = 0.6667`
- `precision = 0.9412`
- `recall = 0.5161`
- `micro_f1@50 = 0.3968`

Baseline output:

- `rep_f1 = 0.6486`
- `precision = 0.5581`
- `recall = 0.7742`
- `micro_f1@50 = 0.3723`

Selected config in this smoke:

- action: `db_bench_press`
- modality: `acc`
- trailing window: `80`
- edge window: `32`
- duration prior: `232-472` samples

## Interpretation

- This run is only a tooling smoke test.
- It should not be treated as a headline result because:
  - it uses one action
  - it uses one modality subset
  - it uses one outer subject
  - it truncates inner validation and RF / refiner capacity for speed
- What it does prove:
  - the nested protocol is executable
  - duration-statistics generation works
  - train-only inner selection works
  - outer tuned-vs-baseline comparison works
  - expected inspection artifacts are produced

## Next Step

Run the full nested benchmark across multiple held-out subjects, all target
actions, and the complete 7-subset modality search.
