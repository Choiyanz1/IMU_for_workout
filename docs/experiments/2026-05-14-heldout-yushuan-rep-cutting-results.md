# 2026-05-14 - Held-Out Yushuan Rep-Cutting Results

## Scope

This note records the current most credible rep-cutting results under a clean
subject-wise split using:

- `test_subject = yushuan`
- train subjects = all remaining subjects
- no `__all__` train/test overlap

These results should be treated as the current main reference, rather than the
earlier `train_all_in_sample` upper-bound runs.

## 1. Held-Out 8-Action Baseline Comparison

Config:

- `configs/micro_macro_recognition_8act_test_yushuan.yaml`

Output:

- `artifacts/baseline_comparison/20260514_8act_test_yushuan_fullbaseline`

Headline metrics:

| Model | Start MAE | End MAE | Transition MAE | Rep F1 | Precision | Recall | micro_f1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 511.56 | 693.72 | 636.30 | 0.7726 | 0.7410 | 0.8071 | **0.6348** |
| BiLSTM | **403.64** | 530.14 | 848.99 | **0.7809** | **0.8829** | 0.7000 | 0.5021 |
| 1D CNN | 676.17 | 518.12 | 725.17 | 0.6634 | 0.6041 | 0.7357 | 0.4670 |
| Phase-only causal TCN | 611.05 | 571.44 | 704.09 | 0.6926 | 0.6571 | 0.7321 | 0.4466 |
| Phase-only non-causal TCN | 479.57 | **500.22** | **563.52** | 0.7661 | 0.7290 | 0.8071 | 0.5670 |
| DS-MS-TCN | 769.17 | 838.28 | 1472.94 | 0.3086 | 0.3641 | 0.2679 | 0.0918 |

Reading:

- Best overall rep F1 in held-out mode: `BiLSTM`
- Best held-out IoU-style boundary score: `Random Forest`
- Best non-causal TCN reference: `Phase-only non-causal TCN`
- Best causal TCN reference: `Phase-only causal TCN`
- DS-MS-TCN remains badly misaligned with pure rep cutting

## 2. Held-Out Strict Causal RF

Output:

- `artifacts/baseline_comparison/20260514_causal_rf_8act_test_yushuan`

Best tested smoothing window:

- `1`

Metrics:

- `rep_f1 = 0.7407`
- `precision = 0.7006`
- `recall = 0.7857`
- `start_mae_ms = 432.79`
- `end_mae_ms = 548.80`
- `transition_mae_ms = 443.75`
- `micro_f1@50 = 0.5894`

Reading:

- Causal RF remains competitive under a clean held-out split.
- It is currently the strongest deployment-style baseline tested so far.

## 3. Held-Out Shared RF Boundary Refiner

Output:

- `artifacts/baseline_comparison/20260514_rf_boundary_refiner_8act_test_yushuan`

Metrics:

- `rep_f1 = 0.7576`
- `precision = 0.7166`
- `recall = 0.8036`
- `start_mae_ms = 423.06`
- `end_mae_ms = 527.04`
- `transition_mae_ms = 428.91`
- `micro_f1@50 = 0.6402`

Reading:

- This improves over strict causal RF on both rep F1 and IoU.
- The gain is real under subject-wise split, so it is not a leakage artifact.

## 4. Held-Out Per-Action RF Boundary Refiner

Output:

- `artifacts/baseline_comparison/20260514_per_action_rf_boundary_refiner_yushuan`

Overall metrics:

- `rep_f1 = 0.7820`
- `precision = 0.7321`
- `recall = 0.8393`
- `start_mae_ms = 427.82`
- `end_mae_ms = 690.78`
- `transition_mae_ms = 432.35`
- `micro_f1@50 = 0.6583`

Comparison vs shared RF boundary refiner:

- shared refiner: `rep_f1 = 0.7576`, `micro_f1@50 = 0.6402`
- per-action refiner: `rep_f1 = 0.7820`, `micro_f1@50 = 0.6583`

Per-action summary:

| Action | Rep F1 | micro_f1@50 |
|---|---:|---:|
| `db_bench_press` | 0.5526 | 0.3037 |
| `db_biceps_curl` | 1.0000 | 1.0000 |
| `db_rdl` | 0.6269 | 0.3215 |
| `db_shoulder_press` | 0.4667 | 0.3650 |
| `db_squat` | 0.8684 | 0.7631 |
| `db_triceps_curl` | 0.9589 | 0.9542 |
| `db_weighted_crunch` | 0.9041 | 0.7539 |
| `one_arm_db_row` | 0.9459 | 0.9231 |

Reading:

- If action identity is already known, per-action modeling is better than a
  fully shared RF boundary refiner.
- Gains are concentrated in easier or medium-difficulty actions.
- The main remaining bottlenecks are:
  - `db_bench_press`
  - `db_rdl`
  - `db_shoulder_press`

## 5. What Is Credible vs Not Credible

Credible for generalization:

- everything in this file

Not credible as headline generalization:

- earlier `test_subject = __all__` runs such as:
  - `artifacts/baseline_comparison/20260514_8act_alltrain_fullbaseline`
  - `artifacts/baseline_comparison/20260514_causal_rf_8act_alltrain_sweep_small`

Those remain useful as stress tests / upper bounds, but not as held-out claims.

## 6. Where To Inspect Stream-Level Results

Most useful inspection entrypoints:

- shared RF refiner:
  - `artifacts/baseline_comparison/20260514_rf_boundary_refiner_8act_test_yushuan/index.html`
- per-action RF refiner:
  - `artifacts/baseline_comparison/20260514_per_action_rf_boundary_refiner_yushuan/index.html`

These HTML files link to per-stream SVGs showing rep boundaries against the raw
IMU magnitude traces.

## 7. Per-Action Known-Action Modality Comparison

We compared per-action causal RF + boundary refiner under the same held-out
`yushuan` split with different sensor subsets.

Outputs:

- `acc-only`:
  - `artifacts/baseline_comparison/20260514_per_action_rf_refiner_yushuan_acc`
- `gyro-only`:
  - `artifacts/baseline_comparison/20260514_per_action_rf_refiner_yushuan_gyro`
- `acc+gyro`:
  - `artifacts/baseline_comparison/20260514_per_action_rf_boundary_refiner_yushuan`
- `acc+gyro+mag`:
  - `artifacts/baseline_comparison/20260514_per_action_rf_refiner_yushuan_acc_gyro_mag_part1`
  - `artifacts/baseline_comparison/20260514_per_action_rf_refiner_yushuan_acc_gyro_mag_part2`

Overall comparison:

| Modality | Rep F1 | Precision | Recall | micro_f1@50 |
|---|---:|---:|---:|---:|
| `acc-only` | **0.8647** | **0.8512** | **0.8786** | 0.6169 |
| `gyro-only` | 0.6909 | 0.6000 | 0.8143 | 0.5893 |
| `acc+gyro` | 0.7820 | 0.7321 | 0.8393 | **0.6583** |
| `acc+gyro+mag` | 0.7218 | 0.7118 | 0.7321 | 0.5689 |

Reading:

- Best overall rep detection/counting quality: `acc-only`
- Best overall IoU-style boundary quality: `acc+gyro`
- Adding magnetometer globally did **not** improve overall held-out performance.

Best modality by action (`micro_f1@50`):

| Action | Best Modality | Best micro_f1@50 |
|---|---|---:|
| `db_bench_press` | `acc` | 0.3777 |
| `db_biceps_curl` | `acc` | 1.0000 |
| `db_rdl` | `gyro` | 0.3870 |
| `db_shoulder_press` | `acc` | 0.5306 |
| `db_squat` | `acc+gyro+mag` | 0.8470 |
| `db_triceps_curl` | `acc+gyro+mag` | 0.9657 |
| `db_weighted_crunch` | `acc+gyro` | 0.7539 |
| `one_arm_db_row` | `acc+gyro` | 0.9231 |

Interpretation:

- The user's observation is supported by held-out experiments:
  - some actions are cleaner in accelerometer space
  - some actions benefit from gyroscope information
  - magnetometer is not uniformly helpful, but can help specific actions
- This strongly supports **action-specific sensor selection** over a single fixed
  global sensor subset.
