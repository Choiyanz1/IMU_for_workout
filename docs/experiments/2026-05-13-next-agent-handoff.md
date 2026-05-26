# 2026-05-13 Next-Agent Handoff

## User Priorities

The user's evaluation priorities are now explicitly:

1. **Rep boundary quality first**
   - Most important: `start_mae_ms`, `end_mae_ms`, `transition_mae_ms`
   - Then: `rep precision / recall / f1`
2. **Rep count second**
   - `n_pred` vs `n_true`
   - `count_diff`
   - `exact-count streams`
   - `over-segmented / under-segmented / zero-TP streams`
3. **Action classification third**
   - Only meaningful after reps are successfully cut

The user explicitly stated:

- if rep cutting is correct, count should usually follow
- therefore boundary quality is the primary optimization target

## Most Important Current Findings

### 1. The main bottleneck is already present in whole-set inference

Whole-set offline results are only slightly better than streaming replay, so the core failure is **not** mainly the online decoder.

Current all-data in-sample whole-set summaries:

| Setting | Precision | Recall | F1 | start MAE ms | end MAE ms | transition MAE ms |
|---|---:|---:|---:|---:|---:|---:|
| 3 actions | 0.6915 | 0.3397 | 0.4556 | 690.4 | 549.4 | 780.2 |
| 7 actions | 0.4163 | 0.2847 | 0.3381 | 681.9 | 769.8 | 3156.0 |
| 8 actions | 0.4274 | 0.1867 | 0.2599 | 757.1 | 726.8 | 4562.6 |

Current all-data streaming replay summaries:

| Setting | Precision | Recall | F1 |
|---|---:|---:|---:|
| 3 actions | 0.6037 | 0.3245 | 0.4221 |
| 7 actions | 0.3310 | 0.2823 | 0.3047 |
| 8 actions | 0.2580 | 0.1870 | 0.2168 |

Interpretation:

- streaming replay is worse, but not by enough to change the diagnosis
- the phase / rep boundary model is already weak before online replay constraints are added

### 2. The shared 3-class phase head collapses on many actions

Phase-collapse analysis was added via:

- `scripts/analyze_phase_collapse.py`

Strong patterns found:

- **3 actions**
  - `db_rdl`: `33/33` streams are `all_eccentric`
  - `one_arm_db_row`: `13/33` `all_eccentric`, `14/33` `eccentric_dominant`
  - `db_bench_press`: not mainly collapse; more fragmentation / over-segmentation
- **7 actions**
  - `db_biceps_curl`: `23/26` `all_eccentric`
  - `db_rdl`: `29/33` `all_eccentric`
  - `db_squat`: `21/24` `all_eccentric`
  - `db_bench_press`: often `all_concentric` or `concentric_dominant`
- **8 actions**
  - `db_rdl`: `28/33` `all_eccentric`
  - `db_squat`: `21/24` `all_eccentric`
  - `db_triceps_curl`: `10/24` `all_concentric`, `13/24` `concentric_dominant`
  - `db_weighted_crunch`: `27/32` `eccentric_dominant`

Interpretation:

- this is not just noisy segmentation
- many action families collapse into a single active phase, so no valid rep pairing is possible

### 3. Matched-rep action identity is not the main problem

Even when rep cutting is poor, matched-rep action classification often remains relatively strong.

Examples:

- 3-action all-data replay:
  - `confidence_hybrid = 1.0000`
- 7-action all-data replay:
  - `confidence_hybrid = 0.9670`
- 8-action all-data replay:
  - `confidence_hybrid = 0.9220`

Interpretation:

- if a rep is already formed, identity is often fine
- the real failure is rep formation itself

### 4. Simpler phase-only baselines are strong enough to beat the current DS-MS-TCN rerun

Held-out Kevin baseline comparison:

- output dir:
  - `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin`

Headline comparison:

| Model | StartMAE | EndMAE | TransMAE | Rep F1 | Precision | Recall | ExactCt | Over | Under | micro_f1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 466.97 | 457.60 | **461.70** | 0.8235 | 0.7538 | **0.9074** | 4 | 13 | 0 | 0.4942 |
| BiLSTM | 489.27 | **440.28** | 605.18 | **0.8463** | **0.8647** | 0.8287 | **7** | 4 | 6 | 0.5450 |
| Phase-only causal TCN | **400.14** | 496.92 | 599.58 | 0.7706 | 0.7636 | 0.7778 | 2 | 9 | 6 | 0.4918 |
| Phase-only non-causal TCN | 481.49 | 445.63 | 573.11 | 0.8419 | 0.8112 | 0.8750 | **7** | 8 | 2 | **0.5973** |
| DS-MS-TCN (ours, single rerun) | 640.08 | 613.78 | 1160.50 | 0.5322 | 0.5106 | 0.5556 | 1 | 11 | 5 | 0.2405 |

Interpretation:

- a **phase-only causal TCN** is already a serious candidate for the main rep-cutting branch
- non-causal TCN / BiLSTM are strong offline upper bounds
- the current multi-task DS-MS-TCN is likely misaligned with the user's main target

### 5. A small inference-only semantic-to-phase blend did not fix collapse

Added minimal probe support:

- `semantic_phase_fusion_weight`
- probe config:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_alltrain_semphase025.yaml`

Probe on representative failures:

- `kevin/kevin/db_rdl/set0`
- `kevin/kevin0509workout/one_arm_db_row/set1`

Result:

- `db_rdl/set0` stayed all `eccentric`, `n_pred = 0`
- `one_arm_db_row/set1` stayed strongly `eccentric`-biased, `n_pred = 2`, `tp = 0`

Interpretation:

- light inference-time semantic fusion is not enough
- the next useful change likely needs to happen at **training time**, not only in post-processing

## Important Artifact Paths

### All-data audits

- 3 actions:
  - `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn`
- 7 actions:
  - `artifacts/micro_macro_recognition/20260513_7act_alltrain_dualhead_viterbi_imuenv/tcn`
- 8 actions:
  - `artifacts/micro_macro_recognition/20260513_8act_alltrain_dualhead_viterbi_imuenv/tcn`

### Baseline comparison

- `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin/comparison_results.json`
- `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin/comparison_results.md`

### Main docs updated already

- `docs/dev-log.md`
- `docs/specs/system.md`
- `docs/specs/model.md`
- `docs/specs/metrics.md`

## Best Current Reading

The current project state is:

1. **Rep boundary quality is the main problem**
2. **Rep count is the second-order downstream problem**
3. **Action identity is not the main bottleneck**
4. **Simpler phase-only models are currently more promising for rep cutting than the present multi-task DS-MS-TCN rerun**

## Recommended Next Steps

For the next agent, the most justified next branch is:

1. Promote **phase-only causal TCN** to a first-class training/evaluation pipeline
2. Compare it against:
   - non-causal TCN
   - BiLSTM
   - RF
   using the user's priority metrics first
3. If still needed, explore a **training-time** phase-representation change instead of more inference-time fusion

Good experimental questions:

1. Can a phase-only causal TCN become the main deployment rep-cutting branch?
2. Can a non-causal TCN define a better offline upper bound than BiLSTM for this dataset?
3. Can training-time action-aware phase supervision improve boundaries without collapsing to single-phase outputs?
