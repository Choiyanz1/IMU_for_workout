# 2026-05-09 Stage Ablation And Handoff

## Summary

- Added configurable macro-stage count to `DSMSTCN`.
- Compared 4-stage, 3-stage, and 2-stage causal DS-MS-TCN on held-out subject `kevin`.
- Best current result is 3-stage, but it is still below practical rep-counting quality.

## Implemented Code Changes

- `models/ds_ms_tcn.py`
  - Added `DSMSTCNConfig.num_macro_stages`.
  - Replaced hardcoded stage 3 and stage 4 modules with a `ModuleList` of refinement stages.
  - Added `final_macro_logits()` helper.
  - Made loss aggregation dynamic across available macro stages.
- `train/micro_macro_recognition.py`
  - Added `MicroMacroConfig.num_macro_stages`.
  - Updated full-sequence prediction path to use the final macro stage dynamically.
- `configs/micro_macro_recognition.yaml`
  - Current working config is set to 3-stage for the best-so-far board-style run.
- Added experiment configs:
  - `configs/micro_macro_recognition_stage3_beta0.yaml`
  - `configs/micro_macro_recognition_stage3_40ep.yaml`

## Completed Runs

### 4-stage baseline

- Run dir: `artifacts/micro_macro_recognition/stage34_4stage_v3/tcn`
- Command:
  - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_4stage_v3`

### 3-stage

- Run dir: `artifacts/micro_macro_recognition/stage34_3stage/tcn`
- Command:
  - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage`

### 2-stage

- Run dir: `artifacts/micro_macro_recognition/stage34_2stage_v3/tcn`
- Command:
  - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_2stage_v3`

## Results

| Metric | 4-stage | 3-stage | 2-stage |
|---|---:|---:|---:|
| Rep F1 | 0.4923 | 0.5970 | 0.5725 |
| Rep Precision | 0.5000 | 0.5882 | 0.5620 |
| Rep Recall | 0.4848 | 0.6061 | 0.5833 |
| Rep Action Accuracy | 0.5938 | 0.7000 | 0.7792 |
| Macro IoU-F1@50 | 0.0295 | 0.2123 | 0.1472 |

## Key Diagnosis

- Stage count is not the only problem.
- 3-stage is the best current compromise, but quality is still too low for a usable counter.
- Pairing diagnostics show many phase-order failures:
  - `missing_eccentric_after_concentric`: 65
  - `unexpected_phase_before_concentric`: 59
  - `phase_gap_too_large`: 6
- `db_rdl` is the clearest failure mode:
  - `kevin/db_rdl/set0` rep F1: `0.2857`
  - `kevin/db_rdl/set1` rep F1: `0.4528`
  - Rep-level action confusion maps all matched `db_rdl` reps to `db_weighted_crunch`.
- This is unlikely to be caused only by sample imbalance because train-subject sample counts for `db_rdl` and `db_weighted_crunch` are similar.

## Handoff For Next Agent

1. Run TMSE ablation:
   - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_beta0.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_beta0`
2. Run longer training:
   - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_40ep.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_40ep`
3. Compare both against `artifacts/micro_macro_recognition/stage34_3stage/tcn/metrics/summary.json`.
4. Focus diagnosis on `db_rdl` vs `db_weighted_crunch` confusion and phase-order failures, not only on aggregate F1.

## Open Question

- If `beta=0` and longer training both fail to recover `db_rdl`, the next likely issue is the micro-label design or subject-specific motion mismatch rather than the number of refinement stages.
