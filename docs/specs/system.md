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
- Report rep-level precision, recall, F1, start MAE, end MAE, transition MAE.
- Report sample-level micro and macro metrics as secondary diagnostics.
- For board readiness, also run streaming-style replay with a causal checkpoint.

## Deployment Requirements

- The model must be causal for online streaming.
- The deploy bundle must include model weights or exported runtime artifact, z-score normalization stats, class labels, and the exact inference window/buffer settings.
- The current `deploy/export_onnx.py` path exports the `InertialStudent` model, not the DS-MS-TCN micro/macro checkpoint.
- DS-MS-TCN deployment needs a dedicated export/runtime path before it can be treated as board-ready.

## Current Status As Of 2026-05-08

- Complete evaluated checkpoint: `artifacts/micro_macro_recognition/20260508_143504/tcn`.
- This checkpoint is causal and can run streaming-style replay.
- It is not ready as the development-board rep-counting model because held-out rep recall and F1 are below an acceptable level.
