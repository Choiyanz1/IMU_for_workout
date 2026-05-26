# 2026-05-13 - Phase-Only Baseline Comparison

## Goal

Check whether simpler phase-focused models can outperform the current multi-task DS-MS-TCN on the user's highest-priority target:

1. rep boundary quality
2. rep count

The comparison is on whole-set inference with the same held-out subject (`kevin`) and the same rep-decoding/evaluation logic.

## Config

- Config used:
  - `configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_testkevin.yaml`
- Output dir:
  - `artifacts/baseline_comparison/20260513_phase_only_baselines_testkevin`

Action set:

- `db_bench_press`
- `db_rdl`
- `one_arm_db_row`

## Models Compared

1. Sliding-window Random Forest
2. BiLSTM
3. Simple 1D CNN
4. Phase-only causal TCN
5. Phase-only non-causal TCN
6. DS-MS-TCN (dual-head + viterbi + smoothing)

Important scope note:

- The new TCN baselines are **phase-only** models.
- They do not try to learn the macro/action task jointly.
- This is intentional, because the question here is whether simplifying the task helps rep cutting.

## Results

| Model | Start MAE | End MAE | Transition MAE | Rep F1 | Precision | Recall | Exact-count | Over | Under | micro_f1@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 466.97 | 457.60 | **461.70** | 0.8235 | 0.7538 | **0.9074** | 4 | 13 | 0 | 0.4942 |
| BiLSTM | 489.27 | **440.28** | 605.18 | **0.8463** | **0.8647** | 0.8287 | **7** | 4 | 6 | 0.5450 |
| 1D CNN | 467.68 | 628.61 | 758.92 | 0.5721 | 0.5413 | 0.6065 | 3 | 10 | 4 | 0.2974 |
| Phase-only causal TCN | **400.14** | 496.92 | 599.58 | 0.7706 | 0.7636 | 0.7778 | 2 | 9 | 6 | 0.4918 |
| Phase-only non-causal TCN | 481.49 | 445.63 | 573.11 | 0.8419 | 0.8112 | 0.8750 | **7** | 8 | 2 | **0.5973** |
| DS-MS-TCN (ours, single rerun) | 640.08 | 613.78 | 1160.50 | 0.5322 | 0.5106 | 0.5556 | 1 | 11 | 5 | 0.2405 |

## What This Means

### 1. Simpler models can absolutely be better

At least in this fair rerun, the answer is yes.

- `Phase-only causal TCN` beat the DS-MS-TCN rerun by a large margin on:
  - start MAE
  - Rep F1
  - `micro_f1@50`
- `BiLSTM` and `Phase-only non-causal TCN` were even better as offline whole-set models.

### 2. A phase-only model is a serious candidate for the main rep-cutting branch

This is the most important practical conclusion.

- The user priority is rep boundary quality first, count second.
- A phase-only causal TCN is already strong enough to beat the current multi-task rerun on exactly those targets.

### 3. Multi-task complexity may be hurting rep cutting

The DS-MS-TCN is trying to do both:

- phase segmentation
- action modeling

The simpler baselines suggest that this extra objective can pull the representation away from clean rep boundaries.

### 4. Non-causal models define the offline ceiling

- `BiLSTM` and `Phase-only non-causal TCN` show what happens when the whole set can be used with future context.
- Their advantage indicates that the task is learnable offline; the main challenge is preserving boundary quality in a causal formulation.

## Recommended Reading

The most useful current model interpretation is now:

1. For deployment-style rep cutting, try a **phase-only causal TCN** as the next serious branch.
2. For offline upper-bound analysis, keep the **non-causal TCN / BiLSTM** as reference models.
3. Keep rep-level action identity as a second-stage problem rather than forcing it into the same backbone too early.
