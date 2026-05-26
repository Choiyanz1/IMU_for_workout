# 2026-05-15 - Optimized Multi-Subject Nested RF Smoke Benchmark

## Goal

Verify that the accelerated nested benchmark can complete a multi-subject
held-out run after the caching, vectorization, and process-level sharding
changes.

## Command Shape

This smoke used:

- outer held-out subjects: `yushuan`, `kevin`
- actions: `db_bench_press`, `db_rdl`
- modalities: `acc`, `acc+gyro`
- outer-fold process parallelism: `2`
- reduced RF and refiner budgets for runtime sanity
- aggressive refiner train-stream / matched-rep truncation

## Output

- `artifacts/baseline_comparison/smoke_nested_rf_multisubject_v3`

## Overall Result

Tuned nested output:

- `precision = 0.9754`
- `recall = 0.5640`
- `rep_f1 = 0.7147`
- `micro_f1@50 = 0.4777`

Baseline output:

- `precision = 0.7348`
- `recall = 0.8009`
- `rep_f1 = 0.7664`
- `micro_f1@50 = 0.4935`

## Fold-Level Snapshot

Held-out `yushuan`:

- tuned: `rep_f1 = 0.6122`, `micro_f1@50 = 0.3812`
- baseline: `rep_f1 = 0.6165`, `micro_f1@50 = 0.3250`

Held-out `kevin`:

- tuned: `rep_f1 = 0.7574`, `micro_f1@50 = 0.5303`
- baseline: `rep_f1 = 0.8312`, `micro_f1@50 = 0.5853`

## Selected Configs In This Smoke

- `yushuan / db_bench_press`:
  - modality: `acc`
  - trailing window: `160`
  - edge window: `32`
- `yushuan / db_rdl`:
  - modality: `acc+gyro`
  - trailing window: `90`
  - edge window: `36`
- `kevin / db_bench_press`:
  - modality: `acc+gyro`
  - trailing window: `85`
  - edge window: `34`
- `kevin / db_rdl`:
  - modality: `acc+gyro`
  - trailing window: `90`
  - edge window: `36`

## Interpretation

- This run succeeded as an engineering smoke test.
- It confirms that the optimized benchmark now supports:
  - multi-subject held-out execution
  - train-only inner selection
  - cached evaluation and refiner features
  - process-level outer-fold parallelism
- It should not be treated as a final research result because the runtime-saving
  settings are aggressive enough to distort the quality tradeoff.

Most important behavioral pattern:

- tuned nested settings became very precision-heavy and consistently
  under-segmented
- baseline remained much more recall-heavy and often over-segmented

That means the current smoke is useful for validating infrastructure, but not yet
for answering the final scientific questions about the best per-action modality
and window policy.

## Recommendation

For the next run, keep the optimized infrastructure but relax the most
aggressive truncation knobs before drawing conclusions:

- increase RF tree count
- increase refiner tree count
- increase `target_matched_reps`
- increase `max_refiner_train_streams`
- increase or disable per-stream / per-subject matched-rep caps
