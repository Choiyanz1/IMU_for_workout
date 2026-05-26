# 2026-05-13 - 7/8-Action All-Train Streaming Audit

## Scope

- Retrained and replay-audited the current best practical DS-MS-TCN stack on the larger 7-action and 8-action settings.
- Both runs used:
  - dual-head Stage 1
  - `semantic_alpha = 0.5`
  - causal smoothing window `15`
  - `viterbi` micro decoder
- Both are **train-all in-sample audits**, not subject-held-out headline numbers.

## Runs

### 7 actions (excluding `db_weighted_crunch`)

- Config:
  - `configs/micro_macro_recognition_7act_no_crunch_dualhead_viterbi_alltrain.yaml`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_7act_alltrain_dualhead_viterbi_imuenv/tcn`

### 8 actions

- Config:
  - `configs/micro_macro_recognition_8act_dualhead_viterbi_alltrain.yaml`
- Run dir:
  - `artifacts/micro_macro_recognition/20260513_8act_alltrain_dualhead_viterbi_imuenv/tcn`

## Summary Table

| Setting | Streams | Replay Precision | Replay Recall | Replay F1 | Exact-count | Over | Under | Zero-TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 actions | 199 | 0.3310 | 0.2823 | 0.3047 | 5 | 83 | 111 | 92 |
| 8 actions | 231 | 0.2580 | 0.1870 | 0.2168 | 3 | 72 | 156 | 112 |

## 7-Action Results

Offline summary from the run:

| Metric | Value |
|---|---:|
| Precision | 0.4163 |
| Recall | 0.2847 |
| Rep F1 | 0.3381 |
| rep_action_accuracy | 0.9255 |

Replay per-action view:

| Action | Precision | Recall | F1 | Main pattern |
|---|---:|---:|---:|---|
| `db_bench_press` | 0.4320 | 0.2628 | 0.3268 | under-segmentation on many streams |
| `db_biceps_curl` | 0.0000 | 0.0000 | 0.0000 | almost total collapse |
| `db_rdl` | 0.4889 | 0.0556 | 0.0998 | severe under-detection |
| `db_shoulder_press` | 0.2976 | 0.5455 | 0.3851 | strong over-segmentation |
| `db_squat` | 0.0000 | 0.0000 | 0.0000 | almost total collapse |
| `db_triceps_curl` | 0.2509 | 0.5087 | 0.3360 | strong over-segmentation |
| `one_arm_db_row` | 0.3840 | 0.6076 | 0.4706 | best replay F1, but over-count heavy |

Matched-rep action identity:

| Method | Accuracy |
|---|---:|
| Online macro aggregation | 0.9279 |
| Rep-complete classifier | 0.8679 |
| Confidence hybrid | 0.9670 |

Important notes:

- `db_rdl` still often collapses to all `eccentric` while sample-level macro identity remains correct.
- `db_biceps_curl` can even stabilize to the wrong display action in replay, e.g. `db_triceps_curl`.
- The larger action set changed the failure mode of `db_bench_press`: it became much less dominant than in the 3-action audit.

## 8-Action Results

Offline summary from the run:

| Metric | Value |
|---|---:|
| Precision | 0.4274 |
| Recall | 0.1867 |
| Rep F1 | 0.2599 |
| rep_action_accuracy | 0.8711 |

Replay per-action view:

| Action | Precision | Recall | F1 | Main pattern |
|---|---:|---:|---:|---|
| `db_bench_press` | 0.2601 | 0.4088 | 0.3179 | over-segmentation |
| `db_biceps_curl` | 0.6212 | 0.1419 | 0.2310 | strong under-detection |
| `db_rdl` | 0.1014 | 0.0177 | 0.0301 | near-total collapse |
| `db_shoulder_press` | 0.2581 | 0.4909 | 0.3383 | over-segmentation |
| `db_squat` | 0.0000 | 0.0000 | 0.0000 | total collapse |
| `db_triceps_curl` | 0.1600 | 0.0139 | 0.0256 | near-total collapse |
| `db_weighted_crunch` | 0.2667 | 0.0623 | 0.1011 | under-detection |
| `one_arm_db_row` | 0.2376 | 0.3392 | 0.2795 | over-segmentation + misses |

Matched-rep action identity:

| Method | Accuracy |
|---|---:|
| Online macro aggregation | 0.8986 |
| Rep-complete classifier | 0.8850 |
| Confidence hybrid | 0.9220 |

Important notes:

- `db_weighted_crunch` can keep the correct displayed action while still badly under-counting reps.
- `db_triceps_curl` showed a representative stream with all predicted active frames collapsing to `concentric` only.
- `db_squat` remained completely unrecoverable in this setting.

## Cross-Setting Interpretation

The main pattern is now very consistent:

1. Expanding the action space hurts replay rep-counting sharply.
2. Matched-rep action identity remains much better than rep formation.
3. The project's central bottleneck is still the shared phase head, not the rep-complete action classifier.

This means that adding more action classes has not solved the rep-counting problem; it has mostly exposed more action-specific phase-collapse modes.
