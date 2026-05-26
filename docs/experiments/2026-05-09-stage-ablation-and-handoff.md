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

## Follow-up Runs Completed

### TMSE ablation (`beta=0`)

- Run dir: `artifacts/micro_macro_recognition/stage34_3stage_beta0/tcn`
- Command:
  - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_beta0.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_beta0`
- Result summary:
  - Rep F1: `0.3390`
  - Precision: `0.3846`
  - Recall: `0.3030`
  - Rep action accuracy: `0.9000`
  - Macro IoU-F1@50: `0.1439`
  - Micro IoU-F1@50: `0.1409`

Diagnosis:

- This run is much worse than the baseline 3-stage run.
- Pairing got worse, especially `unexpected_phase_before_concentric=143`.
- Conclusion: removing TMSE does **not** fix the core issue and in fact damages sequence quality.

### Longer training (`40 epochs`)

- Run dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
- Command:
  - `python -m train.micro_macro_recognition --config configs/micro_macro_recognition_stage3_40ep.yaml --micro-source tcn --no-timestamp --run-stamp stage34_3stage_40ep`
- Result summary:
  - Rep F1: `0.5735`
  - Precision: `0.5571`
  - Recall: `0.5909`
  - Rep action accuracy: `0.6795`
  - Macro IoU-F1@50: `0.1894`
  - Micro IoU-F1@50: `0.3160`

Diagnosis:

- This recovers much of the gap from the failed `beta=0` run.
- It slightly improves micro IoU-F1@50 and some timing metrics.
- But it still does not beat the baseline 3-stage run on rep F1 / recall / macro IoU-F1@50.
- Conclusion: undertraining is not the main explanation.

## RDL-Specific Comparison

Baseline from earlier notes:

- `kevin/db_rdl/set0` F1: `0.2857`
- `kevin/db_rdl/set1` F1: `0.4528`
- All 16 matched `db_rdl` reps were mapped to `db_weighted_crunch`.

New runs:

- `beta=0`
  - `set0` F1: `0.0714`
  - `set1` F1: `0.1224`
  - confusion row: `db_rdl=3`, `db_weighted_crunch=0`, `one_arm_db_row=1`
- `40ep`
  - `set0` F1: `0.8000`
  - `set1` F1: `0.3077`
  - confusion row: `db_rdl=6`, `db_weighted_crunch=7`, `uncertain=5`

Interpretation:

- `beta=0` fails badly on both RDL streams.
- `40ep` partially fixes the “everything becomes crunch” failure, especially on `set0`, but `set1` is still weak and confusion remains severe.

## Additional Inspection: Raw Data Separability

The new script `scripts/analyze_rdl_vs_crunch.py` was run on rep-level CSVs.

Key observations:

- Non-Kevin train data is not strongly imbalanced for these two actions.
- Kevin `db_rdl` reps are longer and more concentric-heavy than Kevin `db_weighted_crunch` reps.
- A simple rep-signature classifier trained on non-Kevin subjects achieved **100% accuracy** on Kevin for `db_rdl` vs `db_weighted_crunch`.

This suggests the persistent model confusion is likely caused by stream-level sequence labeling / pairing / macro assignment issues rather than simple raw-feature overlap.

## Updated Next Direction

The next work should prioritize:

1. Inspect micro-label design for `db_rdl` and `db_weighted_crunch`, especially whether the current concentric/eccentric assumptions fit both actions equally well.
2. Inspect pairing logic on irregular streams, especially `unexpected_phase_before_concentric` and `missing_eccentric_after_concentric` failure cases.
3. Inspect stream-level macro aggregation on `kevin/db_rdl/set1`, since coarse rep-level features still appear separable.

## Extra Findings After Follow-up Analysis

- `whole_session` data is not actually entering the current micro/macro training runs in this local dataset snapshot because `_load_streams(..., ["whole"])` returns zero streams.
- Ground-truth phase ordering is overwhelmingly `concentric -> eccentric`, including Kevin's `db_rdl` reps, so the fixed pairing direction is not the primary global failure.

## New Promising Adjustment

A causal moving-average smoothing over `micro_probs` gives a strong lift on held-out Kevin for the 40-epoch 3-stage checkpoint without retraining.

Measured with `scripts/grid_micro_smoothing.py` on `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`:

- Window `1`: `micro_sample_macro_f1 = 0.4259`, `micro_f1_at_50 = 0.3209`
- Window `15`: `micro_sample_macro_f1 = 0.4317`, `micro_f1_at_50 = 0.4719`

This suggests the model's frame-level phase probabilities are noisy, but the underlying phase signal is still present.

Code support added:

- `micro_smoothing_window` in `MicroMacroConfig` for evaluation-time causal smoothing.
- 3-stage checkpoint loading fix in the streaming/postprocess loaders so stage count is restored correctly from checkpoint config.

## Full Re-evaluation Of 40ep Checkpoint With Smoothing

The new script `scripts/reevaluate_micro_macro_run.py` was used to re-evaluate the 40-epoch checkpoint with `micro_smoothing_window=15`.

- Source run: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn`
- Re-eval output: `artifacts/micro_macro_recognition/stage34_3stage_40ep/tcn/reeval_smooth15`

Overall result:

- Rep F1: `0.7789`
- Precision: `0.7255`
- Recall: `0.8409`
- start MAE: `433.1 ms`
- micro sample macro F1: `0.4293`
- micro IoU-F1@50: `0.4582`

Compared with the raw 40-epoch checkpoint:

- raw 40ep rep F1: `0.5735`
- smoothed 40ep rep F1: `0.7789`
- raw 40ep micro IoU-F1@50: `0.3160`
- smoothed 40ep micro IoU-F1@50: `0.4582`

The improvement is large enough to change the diagnosis:

- The current model already contains useful phase information.
- The main issue is that its framewise micro probabilities are too jittery for direct hard-label decoding.
- Causal smoothing is therefore a high-value runtime/postprocess component, not just a cosmetic tweak.

RDL-specific impact after smoothing:

- `kevin/db_rdl/set0` F1: `0.8800`
- `kevin/db_rdl/set1` F1: `0.6538`

Pairing diagnostics after smoothing:

- `missing_eccentric_after_concentric=10`
- `unexpected_phase_before_concentric=7`

So smoothing does not just improve samplewise metrics; it also strongly reduces the pairing failures that were blocking rep quality.

## Current Best Next Experiment

The strongest next training run is now:

- `configs/micro_macro_recognition_stage3_40ep_alpha2_smooth15.yaml`

Rationale:

1. `beta=0` failed, so keep TMSE.
2. `40ep` helped but was not enough by itself.
3. Smoothing works extremely well, suggesting the next lever is to make micro predictions less noisy before smoothing by increasing the micro-task weight.

## Follow-up Experiment Result: `alpha=2.0 + smooth15`

This experiment was run with:

- config: `configs/micro_macro_recognition_stage3_40ep_alpha2_smooth15.yaml`
- run dir: `artifacts/micro_macro_recognition/stage34_3stage_40ep_alpha2_smooth15/tcn`

Result summary:

- Rep F1: `0.6341`
- Precision: `0.5871`
- Recall: `0.6894`
- Rep action accuracy: `0.8901`
- micro sample macro F1: `0.4309`
- micro IoU-F1@50: `0.4156`
- macro sample macro F1: `0.2247`

Interpretation:

- This is better than the raw 40-epoch checkpoint.
- But it is still clearly worse than simply taking the raw 40-epoch checkpoint and applying `micro_smoothing_window=15` during re-evaluation.

Important tradeoff observed:

- `db_rdl` action confusion improved strongly (`db_rdl` no longer collapses into `db_weighted_crunch`).
- But `db_rdl` rep F1 became much worse than the `smooth15` reevaluation-only path:
  - `set0 = 0.4375`
  - `set1 = 0.2069`

So a higher `alpha` appears to improve action identity but damage the rep segmentation / pairing behavior.

## Updated Best Practical Setting

At this point, the best practical configuration is:

1. Use the `stage34_3stage_40ep` checkpoint.
2. Apply `micro_smoothing_window=15` during evaluation/runtime.

That path currently gives the strongest combination of:

- rep F1,
- recall,
- lower pairing errors,
- better `db_rdl` rep detection.

## Updated Next Direction

The next experiment should target **noise reduction without over-pushing micro CE**. The most likely next levers are:

1. try intermediate `alpha` values like `1.25` or `1.5` rather than `2.0`;
2. keep `micro_smoothing_window` enabled;
3. consider class-weighting or focal-style weighting on the micro task, especially to stabilize `other` without hurting rep boundaries.
