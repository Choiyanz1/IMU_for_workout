# 2026-05-13 - 3-Action All-Train Streaming Audit

## Scope

- Retrained the current best 3-action DS-MS-TCN setup on **all available 3-action data**.
- Replayed **every** eligible 3-action set directory and aggregated:
  - rep count accuracy,
  - over/under segmentation,
  - zero-hit streams,
  - rep-level action identity.
- This is an **in-sample train-all audit**, not a headline subject-held-out benchmark.

## Model Combination Used

- Config:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_alltrain.yaml`
- Environment:
  - `conda run -n imu_for_workout`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn`

Current stack for this run:

1. DS-MS-TCN, 6 layers, 64 filters, causal, 20 s slices
2. dual-head Stage 1
   - phase head: `other / concentric / eccentric`
   - auxiliary semantic head: action-aware phase labels
3. `semantic_alpha = 0.5`
4. causal micro smoothing window `15`
5. `viterbi` micro decoder
   - `switch=1.5`
   - `invalid=10.0`
   - `min_run=16`

## Important Replay Alignment Fix

Before this audit, `evaluation/streaming_micro_macro.py --method fast` was not faithfully reflecting the current best offline configuration because it:

- used `macro2_logits` directly instead of the checkpoint's final macro stage
- ignored the configured `micro_decoder=viterbi`

This audit reran replay **after** fixing both behaviors.

## Data Coverage

- Subjects included:
  - `haoyu`
  - `kevin`
  - `thomas`
  - `yanz`
  - `yoru`
  - `yushuan`
  - `ziho`
- Action set:
  - `db_bench_press`
  - `db_rdl`
  - `one_arm_db_row`
- Evaluated streams:
  - `101`

## Results

### Offline Summary From The Training Run

This is the run's own aggregate summary on the same train-all protocol:

| Metric | Value |
|---|---:|
| Precision | 0.6915 |
| Recall | 0.3397 |
| Rep F1 | 0.4556 |
| rep_action_accuracy | 0.9975 |
| micro_f1_at_50 | 0.2067 |

### Full Streaming Replay Aggregate

| Metric | Value |
|---|---:|
| Predicted reps | 646 |
| True reps | 1202 |
| TP | 390 |
| FP | 256 |
| FN | 812 |
| Precision | 0.6037 |
| Recall | 0.3245 |
| Rep F1 | 0.4221 |
| Exact-count streams | 6 / 101 |
| Over-segmented streams | 25 / 101 |
| Under-segmented streams | 70 / 101 |
| Zero-TP streams | 52 / 101 |

### Per-Action Count / Fragmentation Readout

| Action | Streams | Pred reps | True reps | Precision | Recall | F1 | Exact-count | Over | Under | Zero-TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `db_bench_press` | 35 | 514 | 411 | 0.6304 | 0.7883 | 0.7005 | 5 | 23 | 7 | 1 |
| `db_rdl` | 33 | 0 | 396 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 33 | 33 |
| `one_arm_db_row` | 33 | 132 | 395 | 0.5000 | 0.1671 | 0.2505 | 1 | 2 | 30 | 18 |

### Rep-Level Action Identity On IoU-Matched Reps

| Method | Matched reps | Accuracy |
|---|---:|---:|
| Online macro aggregation | 390 | 0.9974 |
| Rep-complete classifier | 390 | 0.9872 |
| Rep-complete hierarchical | 390 | 0.9872 |
| Hybrid routing | 390 | 0.9872 |
| Confidence hybrid | 390 | 1.0000 |

Interpretation:

- Once a rep is successfully formed, action identity is almost solved in this 3-action setting.
- The hard problem is rep formation itself.

## Unexpected Results And Root-Cause Reading

### 1. `db_rdl` collapsed completely

Representative file:

- `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn/streaming_eval_all/kevin/kevin/db_rdl/set0/streaming_summary.json`

Observed behavior:

- `pred_micro_counts = {"eccentric": 6241}`
- GT has both phases:
  - `concentric = 4153`
  - `eccentric = 2088`
- Macro/action identity is still correct:
  - `macro_sample_accuracy = 1.0`
  - all sample-level macro labels are `db_rdl`
- But no reps are emitted because the stream never presents a valid decoded `concentric -> eccentric` pattern.

Pairing diagnostic:

- `unexpected_phase_before_concentric` over the whole stream.

Reading:

- This is a **phase-head failure**, not an action-classification failure.
- The model knows the stream is `db_rdl`, but its shared 3-class phase head maps the active motion almost entirely to one phase.

### 2. `one_arm_db_row` is heavily under-segmented

Representative file:

- `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn/streaming_eval_all/kevin/kevin0509workout/one_arm_db_row/set1/streaming_summary.json`

Observed behavior:

- predicted micro counts:
  - `eccentric = 3211`
  - `concentric = 40`
- GT:
  - `eccentric = 1701`
  - `concentric = 1550`
- final display action is still correct:
  - `one_arm_db_row`
- but only `2` reps are emitted for `12` true reps.

Reading:

- This is the same underlying problem as RDL, but less extreme.
- The action path is good enough to keep the displayed class stable, yet the phase path still cannot recover enough concentric runs to build reps.

### 3. `db_bench_press` is the strong class, but often over-segmented

Representative file:

- `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn/streaming_eval_all/haoyu/haoyu0512workout/db_bench_press/set0/streaming_summary.json`

Observed behavior:

- `23` predicted reps vs `12` true
- action display is still correct and stable
- no pairing-diagnostic rows were recorded

Reading:

- Bench does not show the RDL/row-style total phase collapse.
- Instead, the decoded phase sequence alternates too often, so one true rep is sometimes split into multiple legal `concentric -> eccentric` pairs.
- That is a fragmentation / over-segmentation problem, not an identity problem.

## Overall Interpretation

This run shows a clearer diagnosis than the earlier held-out notes:

1. The current architecture can learn action identity very well in 3-action in-sample conditions.
2. The dominant remaining failure is still the **shared 3-class phase head**.
3. The failure is strongly action-dependent:
   - `db_bench_press`: over-segmentation
   - `db_rdl`: all-eccentric collapse
   - `one_arm_db_row`: eccentric-dominant under-segmentation

So the most important next model question is not "which action classifier should we use?" but:

- how to make the phase decoder preserve action-specific phase structure without losing causal rep pairing stability.

## Takeaway

- For 3 actions, the current best dual-head + smoothing + viterbi stack is **not** uniformly good across action families even when evaluated in-sample.
- `db_rdl` is still the clearest blocking failure mode.
- `rep_action_accuracy` numbers can look excellent while actual rep counting is still poor, because matched-rep action identity and rep formation are different bottlenecks.
