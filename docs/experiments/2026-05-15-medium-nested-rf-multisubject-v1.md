# 2026-05-15 - Medium-Fidelity Nested RF Multi-Subject Benchmark

## Goal

Run a more credible follow-up after the aggressive smoke benchmark by relaxing
the strongest runtime-saving truncation knobs while keeping the optimized cached
and parallelized infrastructure.

## Output

- `artifacts/baseline_comparison/medium_nested_rf_multisubject_v1`

## Scope

- Outer held-out subjects:
  - `yushuan`
  - `kevin`
- Actions:
  - `db_bench_press`
  - `db_rdl`
- Modalities searched:
  - `acc`
  - `acc+gyro`
- Inner held-out subjects per outer fold:
  - `2`
- Trailing-window multipliers:
  - `0.25`
  - `0.5`
- Edge-window multipliers:
  - `0.10`
  - `0.15`

## Runtime Policy

This is still not the final full sweep. It keeps reduced training cost, but is
less aggressive than the earlier smoke:

- RF trees: `20`
- RF depth: `12`
- refiner trees: `80`
- refiner depth: `14`
- target matched reps: `200`
- refiner train streams: `20`

## Overall Result

Tuned nested output:

- `precision = 0.9769`
- `recall = 0.6019`
- `rep_f1 = 0.7449`
- `micro_f1@50 = 0.5485`

Baseline output:

- `precision = 0.7348`
- `recall = 0.8009`
- `rep_f1 = 0.7664`
- `micro_f1@50 = 0.5132`

## Comparison Against Earlier Aggressive Smoke

Earlier aggressive smoke:

- tuned `rep_f1 = 0.7147`
- tuned `micro_f1@50 = 0.4777`

This run:

- tuned `rep_f1 = 0.7449`
- tuned `micro_f1@50 = 0.5485`

Reading:

- Relaxing the truncation knobs materially improved tuned quality.
- The tuned nested pipeline still remains more precision-heavy and more
  under-segmented than the baseline.
- But it now shows a meaningful IoU-style gain over the baseline on this
  benchmark slice.

## Fold-Level Result

Held-out `yushuan`:

- tuned: `rep_f1 = 0.7037`, `micro_f1@50 = 0.4644`
- baseline: `rep_f1 = 0.6565`, `micro_f1@50 = 0.2831`

Held-out `kevin`:

- tuned: `rep_f1 = 0.7639`, `micro_f1@50 = 0.5943`
- baseline: `rep_f1 = 0.8129`, `micro_f1@50 = 0.6387`

## Selected Configs

- `yushuan / db_bench_press`:
  - modality: `acc+gyro`
  - trailing window: `80`
  - edge window: `32`
- `yushuan / db_rdl`:
  - modality: `acc`
  - trailing window: `90`
  - edge window: `36`
- `kevin / db_bench_press`:
  - modality: `acc`
  - trailing window: `160`
  - edge window: `34`
- `kevin / db_rdl`:
  - modality: `acc+gyro`
  - trailing window: `90`
  - edge window: `36`

## Stability Reading

- This run still does not show stable action-specific winners.
- `db_bench_press` changed best modality/window across the two held-out folds.
- `db_rdl` also changed best modality across folds.
- Therefore this run is still too small to claim deployable per-action settings.

## Conclusion

- The optimized benchmark is now capable of a more credible medium-fidelity run.
- The current best interpretation is:
  - nested tuning can improve boundary-quality style metrics on some folds
  - fixed baseline can still win on rep F1 and recall on other folds
  - more outer subjects and more actions are required before deciding whether the
    per-action tuned policy is truly superior overall

## Recommended Next Step

Run a broader benchmark with the same optimized infrastructure but expanded
coverage:

- more outer held-out subjects
- more actions
- same cache + outer-fold parallelism
- similar or slightly stronger RF / refiner budgets
