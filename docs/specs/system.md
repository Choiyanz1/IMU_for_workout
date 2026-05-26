# System Specification

## Goal

Build an IMU-based workout recognition system for resistance training that can run offline for experiments and eventually run online on a development board.

## Input Data

- Required IMU columns: `sensor_ts`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`.
- Required metadata columns for training/evaluation: `action_type`, `subject_id`.
- Optional supervision columns: `phase`, `rep`, `set`.
- Current target actions: `db_bench_press`, `db_rdl`, `db_weighted_crunch`, `one_arm_db_row`.

## Evaluation Rules

- Use subject-wise held-out testing for headline metrics.
- Current held-out subject for the reviewed micro/macro model: `kevin`.
- Treat **rep boundary quality** as the primary evaluation target.
- Report rep-level precision, recall, F1, start MAE, end MAE, transition MAE.
- Treat **rep count** as a primary metric as well, immediately after boundary quality.
  - Always include `n_pred`, `n_true`, and stream-level count diagnostics such as exact-count / over-segmented / under-segmented streams when available.
- Report sample-level micro and macro metrics as secondary diagnostics.
- For board readiness, also run streaming-style replay with a causal checkpoint.

各指標的統一解釋請見：

- `docs/specs/metrics.md`
- `docs/specs/reporting_plan.md`：報告 / 論文流程與所需輸出材料
- `docs/specs/reporting_plan_assets.html`：圖表與表格樣板

## Deployment Requirements

- The model must be causal for online streaming.
- The deploy bundle must include model weights or exported runtime artifact, z-score normalization stats, class labels, and the exact inference window/buffer settings.
- The current `deploy/export_onnx.py` path exports the `InertialStudent` model, not the DS-MS-TCN micro/macro checkpoint.
- DS-MS-TCN deployment needs a dedicated export/runtime path before it can be treated as board-ready.
- The current RF + boundary-refiner rep-cutting branch also lacks a dedicated
  deploy/runtime artifact path for Luckfox Pico Zero. There is no validated
  ONNX / RKNN / board-side implementation for the sklearn RF + refiner stack in
  this repo yet.

## Current Status As Of 2026-05-13

- Best current **subject-held-out** 3-action checkpoint remains:
  - `artifacts/micro_macro_recognition/20260512_192822/tcn`
  - held-out `kevin` rep metrics: Precision `0.7112`, Recall `0.7639`, F1 `0.7366`
- Latest **all-data in-sample** 3-action audit:
  - `artifacts/micro_macro_recognition/20260513_3act_alltrain_dualhead_viterbi_imuenv/tcn`
  - full replay aggregate across `101` streams: Precision `0.6037`, Recall `0.3245`, F1 `0.4221`
- Latest **all-data in-sample** larger-action audits:
  - 7 actions: `artifacts/micro_macro_recognition/20260513_7act_alltrain_dualhead_viterbi_imuenv/tcn`
    - replay aggregate: Precision `0.3310`, Recall `0.2823`, F1 `0.3047`
  - 8 actions: `artifacts/micro_macro_recognition/20260513_8act_alltrain_dualhead_viterbi_imuenv/tcn`
    - replay aggregate: Precision `0.2580`, Recall `0.1870`, F1 `0.2168`
- Important interpretation:
  - headline quality must still come from subject-wise held-out evaluation
  - the all-data audits are useful for debugging action-family failure modes, not for claiming generalization
- Deployment status is still **not ready** for DS-MS-TCN rep counting because:
  - replay performance is highly action-dependent
  - `db_rdl` still collapses at the phase-decoding level in the latest 3-action audit
  - larger 7/8-action audits further degrade rep counting even though matched-rep action identity stays relatively strong
  - DS-MS-TCN still lacks a dedicated board export/runtime path
- Current RF rep-cutting status:
  - a guarded held-out `yoru` run reached Precision `0.8648`, Recall `0.8869`,
    Rep F1 `0.8757`, and exact-count `19 / 25`
  - this is strong enough to treat RF as the current offline rep-cutting
    reference
  - but it is still **not board-ready** yet because latency / memory have not
    been validated in a deployable Luckfox runtime and the current benchmark is
    still Python/sklearn-based
