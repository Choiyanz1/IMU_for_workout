# Development Log

## 2026-05-09 - Stage 3/4 Ablation Experiment

### What Changed

- Modified `DSMSTCN` to support configurable `num_macro_stages` via `DSMSTCNConfig.num_macro_stages`.
  - `num_macro_stages=4`: full pipeline (Stage 1 micro + Stage 2 macro + Stages 3-4 refinement)
  - `num_macro_stages=2`: ablation (Stage 1 micro + Stage 2 macro only, no refinement stages)
- Refinement stages are now built via `nn.ModuleList` instead of hardcoded stage3_macro/stage4_macro.
- Added `final_macro_logits()` method to select the correct final output key regardless of num_macro_stages.
- `ds_ms_tcn_loss()` now dynamically discovers macro keys via `startswith("macro") and endswith("_logits")`.
- `OnlineDSMSTCNPredictor` uses `final_macro_logits()` for correct final macro selection.
- `MicroMacroConfig` now includes `num_macro_stages` field (default 4).
- Config `configs/micro_macro_recognition.yaml` includes `num_macro_stages` setting.

### Hypothesis

Stages 3 and 4 in DS-MS-TCN hurt performance because:
1. Error amplification: Stage 2 mistakes propagate and worsen through refinement.
2. Over-smoothing: TMSE loss pushes predictions toward uniform continuity, destroying action boundaries.
3. Information bottleneck: Stages 3-4 only see macro softmax from previous stage, losing raw IMU signal.
4. Training instability: Conflicting gradients between micro CE (needs sharp predictions) and TMSE (needs smooth predictions).

### Commands Run

- 4-stage (baseline): `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_4stage_v3`
- 2-stage (ablation): `python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn --no-timestamp --run-stamp stage34_2stage_v3`

### Shared Config

- sample_rate: 100 Hz
- slice_seconds: 20.0
- num_layers: 6, num_filters: 64
- causal: true
- epochs: 20
- alpha: 1.0, beta: 0.15, tmse_threshold: 4.0
- test_subject: kevin

### Result Summary (held-out subject: kevin, 20 epochs)

| Metric | 4-stage | 3-stage | 2-stage | Best |
|---|---:|---:|---:|---|
| **Rep F1** | 0.4923 | 0.5970 | 0.5725 | **3-stage** |
| **Rep Precision** | 0.5000 | 0.5882 | 0.5620 | **3-stage** |
| **Rep Recall** | 0.4848 | **0.6061** | 0.5833 | 3-stage |
| **Rep Action Accuracy** | 0.5938 | 0.7000 | **0.7792** | 2-stage |
| start_mae_ms | 586.2 | 600.1 | **522.6** | 2-stage |
| end_mae_ms | 600.5 | **554.2** | 670.2 | 3-stage |
| transition_mae_ms | 605.9 | **504.4** | 514.2 | 3-stage |
| macro_f1_at_50 | 0.0295 | **0.2123** | 0.1472 | 3-stage |
| micro_f1_at_50 | 0.2297 | 0.2986 | 0.2978 | **3-stage** |

3-stage is best on the most metrics. 2-stage is best on rep action accuracy and start_mae. 4-stage is last on all metrics.

### Artifact Paths

- 4-stage: `artifacts/micro_macro_recognition/stage34_4stage_v3/tcn`
- 3-stage: `artifacts/micro_macro_recognition/stage34_3stage/tcn`
- 2-stage: `artifacts/micro_macro_recognition/stage34_2stage_v3/tcn`

### Current Diagnosis

- Best current setting is 3-stage with held-out `kevin` rep F1 `0.5970`, but this is still not deployable.
- Failure is not only over-segmentation. There are two clear blockers:
  - Micro phase pairing errors remain frequent: `missing_eccentric_after_concentric=65`, `unexpected_phase_before_concentric=59`, `phase_gap_too_large=6`.
  - Macro action confusion is severe on `db_rdl`: all 16 matched `db_rdl` reps were classified as `db_weighted_crunch` in `rep_action_confusion_tcn.csv`.
- Per-stream metrics show `kevin/db_rdl/set0` and `kevin/db_rdl/set1` are the worst streams, with macro sample accuracy `0.2220` and `0.0555`.
- Training-data volume does not explain the RDL failure by itself: `db_rdl` and `db_weighted_crunch` have similar sample counts in the train subjects.

### Next Queued Experiments

- Run `configs/micro_macro_recognition_stage3_beta0.yaml` to isolate whether TMSE smoothing is hurting 3-stage quality.
- Run `configs/micro_macro_recognition_stage3_40ep.yaml` to check whether the 20-epoch 3-stage run is undertrained.
- If both still fail on `db_rdl`, inspect feature/label mismatch rather than stage count alone.

## 2026-05-08 - Current Model Review For Development Board

### What Changed

- Added project documentation files for long-term rules, system specification, model specification, and experiment tracking.
- Ran comparison over existing evaluation summaries with the `imu` conda environment.
- Ran a streaming-style replay on the complete current DS-MS-TCN checkpoint.

### Commands Run

- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.compare_runs --root artifacts --output artifacts/run_comparison_20260508.csv`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_fast --method fast --max-samples 3000 --device cpu`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step300 --method step --max-samples 300 --device cpu --progress-interval 100`

### Result Summary

- Complete DS-MS-TCN checkpoint reviewed: `artifacts/micro_macro_recognition/20260508_143504/tcn`.
- Newer candidate folders `20260508_145027/tcn` and `board_100hz_l6_20260508/tcn` had no model checkpoint or summary metrics.
- Held-out `kevin` summary for the complete checkpoint:
  - precision: 0.4545.
  - recall: 0.1136.
  - rep F1: 0.1818.
  - rep action accuracy: 0.2667.
  - micro sample accuracy: 0.5125.
  - macro sample accuracy: 0.2424.
  - macro sample macro F1: 0.0919.
- Post-process grid on the same checkpoint improved rep F1 to 0.3235, but recall remained only 0.25.
- Streaming fast replay on `kevin/db_weighted_crunch/set0` processed 3000 samples in 0.228 s on CPU and produced:
  - micro accuracy: 0.6627.
  - macro accuracy: 0.9790.
- Streaming step replay on 300 samples took 2.338 s on CPU, about 128 samples/s, with a 4089-sample buffer.

### Decision

- Do not ship this checkpoint as the development-board rep-counting model.
- It may be used only as a debugging/demo checkpoint for causal replay, not as a reliable workout counter.

### Next Work

- Finish a clean training run for the current board-style config: 100 Hz, 20 s slices, 6 layers.
- Add DS-MS-TCN export/runtime support if DS-MS-TCN is the intended board model.
- Improve recall before deployment by addressing phase-label order errors and post-processing thresholds.

## 2026-05-08 - Streaming Step Replay Improvements

### What Changed

- Updated `OnlineDSMSTCNPredictor` to use a fixed-length `deque` and `torch.inference_mode()` for step-by-step streaming inference.
- Updated `evaluation.streaming_micro_macro` so `streaming_summary.json` includes throughput, real-time factor, sample accuracies, and predicted/ground-truth label counts.
- Re-tested with actual `--method step` replay, not fast full-sequence inference.

### Commands Run

- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m py_compile models/ds_ms_tcn.py evaluation/streaming_micro_macro.py`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_after --method step --max-samples 3000 --device cpu --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b512 --method step --max-samples 3000 --device cpu --buffer-size 512 --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b256 --method step --max-samples 3000 --device cpu --buffer-size 256 --progress-interval 500`
- `PATH="/bin:/usr/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH" /opt/homebrew/Caskroom/miniconda/base/bin/conda run -n imu python -m evaluation.streaming_micro_macro --run-dir artifacts/micro_macro_recognition/20260508_143504/tcn --csv datasets/raw_data/kevin/db_weighted_crunch/set0 --output-dir artifacts/micro_macro_recognition/20260508_143504/tcn/streaming_eval/kevin_db_weighted_crunch_set0_step3000_b128 --method step --max-samples 3000 --device cpu --buffer-size 128 --progress-interval 500`

### Result Summary

| Runtime buffer | Elapsed s | Samples/s | Real-time factor | Micro acc | Macro acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4089 | 76.882 | 39.02 | 0.352 | 0.6627 | 0.9790 |
| 512 | 37.779 | 79.41 | 0.715 | 0.6630 | 0.9790 |
| 256 | 23.485 | 127.74 | 1.151 | 0.6417 | 0.9790 |
| 128 | 21.401 | 140.18 | 1.263 | 0.6553 | 0.9790 |

### Decision

- For this checkpoint and replay, `--buffer-size 128` or `--buffer-size 256` is the practical real-time setting on CPU.
- This improves streaming runtime measurement and runtime configuration, but it does not solve the global held-out rep-counting quality issue.
