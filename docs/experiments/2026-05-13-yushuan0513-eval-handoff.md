# 2026-05-13 - Yushuan 0513 Evaluation Handoff

## Scope

- Evaluated new data under `datasets/raw_data/yushuan/yushuan0513workout`.
- Rep detector used current best checkpoint for rep quality:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn`
- Action identity was compared using rep-level classifier methods on top of the streamed rep outputs.

## Important Context

- `rep0` contamination cleanup was checked first with `--dry-run` and the new `yushuan` data did not show the earlier non-CE issue:
  - `trimmed=0 deleted=0 unchanged=248`
- The current DS-MS-TCN checkpoint only knows macro classes:
  - `other`, `db_bench_press`, `db_rdl`, `one_arm_db_row`
- Therefore, on 7/8-action new-subject evaluation, macro labels are out-of-domain for several actions.
- This is why the current confidence-based hybrid rule underperformed on `yushuan` even though it was best on held-out `kevin` 3-action data.

## Files Added/Changed For This Evaluation

- Added eval-only configs:
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_eval_yushuan.yaml`
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_eval_yushuan.yaml`
- Fixed empty-output handling in:
  - `scripts/evaluate_rep_complete_action_classifier.py`

The classifier comparison script now handles:

- empty `online_rep_detections.csv`
- streams with zero predicted rep segments

## Streaming Eval Outputs

- Per-set streaming outputs:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn/streaming_eval_yushuan0513_rawmacro`
- Probe run used during validation:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn/streaming_eval_yushuan_probe`

## Comparison Outputs

- 8-action comparison:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn/rep_action_compare_yushuan0513_8act.json`
- 7-action comparison excluding `db_weighted_crunch`:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn/rep_action_compare_yushuan0513_7act_no_crunch.json`

## Headline Results

### Rep Detection

8 actions:

- `tp=129 fp=144 fn=151`
- Precision: `0.4725`
- Recall: `0.4607`
- F1: `0.4665`

7 actions excluding `db_weighted_crunch`:

- `tp=121 fp=135 fn=123`
- Precision: `0.4727`
- Recall: `0.4959`
- F1: `0.4840`

### Rep-Level Action Identity On IoU-Matched Reps

8 actions, `129` matched reps:

- `online_macro_aggregation`: `20.93%`
- `rep_complete_classifier`: `56.59%`
- `rep_complete_hierarchical`: `49.61%`
- `hybrid_routing`: `46.51%`
- `confidence_hybrid`: `23.26%`

7 actions excluding `db_weighted_crunch`, `121` matched reps:

- `online_macro_aggregation`: `22.31%`
- `rep_complete_classifier`: `52.07%`
- `rep_complete_hierarchical`: `52.07%`
- `hybrid_routing`: `46.28%`
- `confidence_hybrid`: `18.18%`

## Per-Action Readout (7 Actions, No Crunch)

- `db_bench_press`: macro `0.0`, classifier `0.8889`
- `db_biceps_curl`: macro `0.0`, classifier `1.0`
- `db_rdl`: macro `1.0`, classifier `0.4444`
- `db_shoulder_press`: macro `0.0`, classifier `0.0`
- `db_squat`: no matched reps
- `db_triceps_curl`: macro `0.0`, classifier `1.0`
- `one_arm_db_row`: macro `0.75`, classifier `0.0`

Interpretation:

- Macro and classifier do have complementary strengths.
- The current confidence-threshold hybrid is the wrong fusion rule for this dataset.
- A class-aware router or meta-classifier is the most natural next step.

## Rep Count / Fragmentation Notes

All 8 actions:

- predicted reps: `273`
- true reps: `280`
- exact-count streams: `4 / 25`
- over-segmented streams: `12 / 25`
- under-segmented streams: `9 / 25`
- streams with `tp=0`: `6 / 25`

7 actions excluding crunch:

- predicted reps: `256`
- true reps: `244`
- exact-count streams: `4 / 22`
- over-segmented streams: `11 / 22`
- under-segmented streams: `7 / 22`
- streams with `tp=0`: `4 / 22`

Notable per-action rep behavior:

- `db_bench_press`: strong over-segmentation (`62` predicted vs `31` true)
- `db_shoulder_press`: over-segmentation (`52` vs `33`)
- `one_arm_db_row`: moderate over-segmentation (`45` vs `36`)
- `db_rdl`: under-segmentation (`20` vs `36`)
- `db_squat`: severe under-segmentation (`3` vs `36`)
- `db_weighted_crunch`: under-segmentation (`17` vs `36`)

## Recommended Next Step For The Next Agent

1. Keep current rep detector checkpoint for now if the goal is continuity with the best rep-F1 run.
2. Do not use the current `confidence_hybrid` rule on this 7/8-action `yushuan` evaluation.
3. Try a class-aware fusion strategy:
   - trust macro for `db_rdl` and `one_arm_db_row`
   - trust classifier for `db_bench_press`, `db_biceps_curl`, `db_triceps_curl`, `db_weighted_crunch`
   - inspect `db_shoulder_press` and `db_squat` separately
4. Investigate rep fragmentation on `db_squat`, `db_rdl`, and `db_bench_press` first.
