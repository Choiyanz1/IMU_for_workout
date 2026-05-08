# XTinyHAR Resistance Student (Multi-File)

This project trains and evaluates IMU models for resistance-training recognition:

- action classification
- phase segmentation
- repetition segmentation with single/multi-template SDTW
- optional teacher-to-student distillation for lightweight deployment

## Expected Input Schema

Required CSV columns:

- `sensor_ts`
- `ax`, `ay`, `az`, `gx`, `gy`, `gz`
- `action_type`
- `subject_id`

Optional but recommended columns:

- `mx`, `my`, `mz` for magnetometer-aware repetition segmentation
- `phase`, `rep`, `set` for phase and whole-session segmentation evaluation

Recommended raw-data layout:

```text
datasets/raw_data/{subject_id}/
  {subject_id}_whole_session_*.csv
  {action_type}/
    set0/rep*.csv
    set1/rep*.csv
    rest_after_set0/rest*.csv
  big_rest/session0/rest*.csv
```

Use `data.include_actions` in `config.yaml` to control which action folders are included in training/evaluation.

## Project Layout

- `datasets/custom_resistance_dataset.py`: CSV loading, schema checks, sequence assembly
- `preprocessing/window_pipeline.py`: resampling, z-score, subject-wise split, sliding windows
- `models/inertial_student.py`: XTinyHAR-style student model
- `train/student.py`: student model training entrypoint
- `train/action_classification.py`: action classification training with tabular models and rich feature engineering
- `train/phase_segmentation.py`: per-rep eccentric/concentric phase segmentation training
- `train/distillation.py`: knowledge distillation (teacher → student transformer)
- `train/hybrid_rep_segmentation.py`: hybrid SDTW + AutoGluon classifier rep segmentation
- `preprocessing/sdtw_rep_segmentation.py`: SDTW-based repetition segmentation utilities
- `preprocessing/hybrid_rep_features.py`: per-candidate features for the hybrid classifier
- `evaluation/reporting.py`: shared run manifests, summaries, and Markdown reports
- `evaluation/compare_runs.py`: collect run summaries into one comparison table
- `evaluation/model_suite.py`: run selected comparable models into one suite
  folder and write shared comparison CSV/Markdown reports
- `evaluation/rep_segmentation.py`: rep segmentation evaluation and SVG plot generation
- `evaluation/streaming_micro_macro.py`: offline replay of causal online predictions
- `configs/`: task-scoped config copies for experiments
- `scripts/plot_phase_segments.py`: phase segmentation plot helper
- `scripts/plot_rep_phase_prediction.py`: one-rep phase prediction plot helper
- `deploy/export_onnx.py`: checkpoint to ONNX export
- `deploy/luckfox_infer.py`: ONNX runtime sliding-window inference helper

Generated outputs are kept under `artifacts/` and are ignored by git.

## Standard Run Outputs

Training and evaluation entrypoints keep their original artifacts, but now also
write a common comparison layer whenever possible:

- `report.md`: human-readable run report with key metrics and artifact links.
- `metrics/summary.json`: standardized summary with `overall` and
  `primary_metrics`.
- `metadata/run_manifest.json`: task, model name, config path, split details,
  and run metadata.
- `metadata/config_snapshot.yaml`: config copied at run time.

This keeps model-specific details intact while making different models easier
to compare.

Create a cross-run comparison table:

```bash
python -m evaluation.compare_runs --root artifacts --output artifacts/run_comparison.csv
```

The resulting CSV and Markdown report normalize common fields such as
`accuracy`, `macro_f1`, `precision`, `recall`, `f1`, and `iou_f1_50` across
action, phase, rep, and micro/macro runs.

Run comparable rep/action models from one command:

```bash
python -m evaluation.model_suite --models ds_ms_tcn --mode sets
```

Useful model choices:

- `ds_ms_tcn`: runs `ds_ms_tcn_tcn` and `ds_ms_tcn_dtw`.
- `sdtw`: runs the plain SDTW rep segmentation baseline.
- `hybrid`: runs SDTW candidates plus the AutoGluon classifier/refiner.
- `all`: runs DS-MS-TCN TCN, DS-MS-TCN DTW, SDTW, and hybrid.

Outputs are grouped under `artifacts/model_suites/<run_id>/`, with each model's
own artifacts kept separate and shared `comparison.csv` and `comparison.md`
files at the suite root.

## Python Version

- Python `3.10+`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml`:

- `data.data_dir` and `data.csv_glob`
- training/device settings
- window/model parameters

For cleaner experiments, copy/edit one of:

- `configs/action_classification.yaml`
- `configs/phase_segmentation.yaml`
- `configs/rep_segmentation.yaml`

## Train

```bash
python -m train.student --config config.yaml
```

## Action Classification Training (Tabular)

This runs a comprehensive model search using tabular models on the **same windowing + subject-wise split** as the student model.

### Feature Modes

Configure `autogluon.feature_mode` in `config.yaml` (using AutoGluon as the underlying framework):

- **`stats`** — per-channel summary statistics (mean, std, min, max, median). Fast, 30 features for 6-axis IMU.
- **`flatten`** — raw window flattened to a vector. Higher dim but preserves temporal order.
- **`rich`** *(recommended)* — stats + RMS + IQR + zero-crossing rate + lag-1 autocorrelation + skewness + kurtosis + FFT features (energy, entropy, top-K frequencies) + inter-axis Pearson correlation + accel/gyro magnitude features. ~150+ features for 6-axis IMU.

### Bagging & Stacking

- `num_bag_folds: 8` — enables 8-fold bagging for better generalization
- `num_stack_levels: 1` — enables 1-level stacking (ensemble of ensembles)

### Usage

Install dependency:

```bash
pip install -r requirements.txt
```

Dry-run (build features only, inspect feature dimensions):

```bash
python -m train.action_classification --config configs/action_classification.yaml --dry-run
```

Train and evaluate:

```bash
python -m train.action_classification --config configs/action_classification.yaml
```

### Action Classification Artifacts (`./artifacts/action_classification/`)

- `models/` — trained models (timestamped subfolders)
- `leaderboard.csv` — ranked model performance on test set
- `confusion_matrix.csv` — per-class confusion matrix
- `classification_report.json` / `.txt` — precision, recall, F1 per class
- `feature_importance.csv` — permutation-based feature importance
- `train_soft_labels.csv` — teacher probability outputs (for distillation)
- `test_soft_labels.csv`
- `autogluon_summary.json`
- `report.md`, `metrics/summary.json`, `metadata/run_manifest.json` —
  standardized comparison outputs

## Knowledge Distillation (Tabular Teacher → Student)

After running action classification training, distill the teacher's knowledge into the lightweight student transformer:

```bash
python -m train.distillation --config configs/action_classification.yaml
```

Configure in `config.yaml` under `distill:`:

- `temperature: 3.0` — softens teacher probabilities (higher = softer)
- `alpha: 0.7` — weight of soft-label KL loss vs hard-label CE loss
- `epochs: 50` — distillation training epochs

Artifacts are written to `./artifacts_student/distilled/`:

- `student_distilled.pt` — distilled student checkpoint
- `distill_summary.json` — metrics and configuration

## Phase Segmentation

Phase segmentation is trained and evaluated as a **single-rep** problem.
The intended runtime order is: detect/crop reps first, then run this model
inside each cropped rep to cut eccentric vs concentric phases.

```bash
python -m train.phase_segmentation --config configs/phase_segmentation.yaml
```

Artifacts are written to timestamped folders under `./artifacts/phase_segmentation/`:

- `models/`
- `leaderboard.csv`
- `confusion_matrix.csv`
- `classification_report.json` / `.txt`
- `feature_importance.csv`
- `phase_summary.json`
- `rep_transition_metrics.csv`
- `rep_transition_summary.json`
- `dataset_shapes.json`

## Repetition Segmentation

Run both set-level and whole-session evaluation in one timestamped folder:

```bash
python -m evaluation.rep_segmentation --config configs/rep_segmentation.yaml
```

Run only one mode when needed:

```bash
python -m evaluation.rep_segmentation --config configs/rep_segmentation.yaml --mode sets
python -m evaluation.rep_segmentation --config configs/rep_segmentation.yaml --mode whole
```

Artifacts are written to timestamped folders under `./artifacts/rep_segmentation/`:

```text
YYYYMMDD_HHMMSS/
  README.md
  summary.json
  sets/
    metrics/summary.json
    metrics/stream_metrics.csv
    detections/detections.csv
    templates/templates.csv
    plots/{action}/{subject}/*.svg
    metadata/config_snapshot.yaml
    metadata/run_manifest.json
  whole/
    metrics/summary.json
    metrics/stream_metrics.csv
    detections/detections.csv
    templates/templates.csv
    plots/{action}/{subject}/*.svg
    metadata/config_snapshot.yaml
    metadata/run_manifest.json
```

The SDTW segmenter is configured under `segmentation.sdtw` in `config.yaml`.
Current defaults use middle-repetition templates, up to three templates per action/subject group, accelerometer + gyroscope + magnetometer channels, and stricter thresholds to reduce rest-period false positives.

## Hybrid Rep Segmentation (SDTW + AutoGluon)

Three-stage pipeline trained per fold with Leave-One-Subject-Out:

1. **Candidate generation (SDTW)** — middle-repetition templates produce a
   superset of candidate windows per stream (controlled by `cost_threshold_scale`).
2. **Binary classifier (AutoGluon)** — drops false positives among the
   candidates. The fold's calibrated `decision_threshold` is preferred over a
   fixed cutoff (see `use_calibrated_threshold`).
3. **Boundary refiner (AutoGluon regression)** — two regressors predict
   `delta_start_samples` / `delta_end_samples` for each kept candidate using
   features computed over local IMU windows around the predicted edges, so the
   SDTW boundaries can be nudged closer to ground truth before NMS.

```bash
python -m train.hybrid_rep_segmentation --config configs/hybrid_rep_segmentation.yaml
python -m train.hybrid_rep_segmentation --config configs/hybrid_rep_segmentation.yaml --mode whole
```

Key knobs (in `configs/hybrid_rep_segmentation.yaml` under `hybrid:`):

- `label_iou` — IoU threshold to label a candidate as TP during classifier training.
- `cost_threshold_scale` — multiplies the SDTW cost threshold during candidate
  generation. Values > 1.0 admit more candidates (more FPs to learn from).
- `prob_threshold` — hard floor on classifier probability (used together with
  `use_calibrated_threshold` so AutoGluon's per-fold optimum is preferred).
- `subject_stratified_validation` — hold out one *training* subject as the
  AutoGluon tuning set so threshold calibration is LOSO-aware.
- `nms_iou` — IoU used for prob-ranked NMS on kept (refined) candidates.
- `enable_boundary_refiner` — toggles stage 3 on / off.
- `edge_window_samples` — half-width of the local IMU window around each edge
  used as refiner features; also caps the refiner's effective shift range.
- `refiner_train_iou` — only candidates with best-IoU above this are used as
  regression rows (so deltas are well-defined).
- `refiner_max_shift_samples` — safety cap; predicted shifts beyond this fall
  back to the original SDTW boundary.
- AutoGluon presets / model filters mirror the action-classification config.
  CatBoost / LightGBM / XGBoost are included by default.

Artifacts share the same `./artifacts/rep_segmentation/<timestamp>/` layout
as the plain SDTW evaluator, with a `hybrid/` sub-folder so the two methods
can be compared side-by-side at any point in time:

```text
artifacts/rep_segmentation/
  20260429_183922/                # plain SDTW eval (evaluation/rep_segmentation.py)
    sets/, whole/, summary.json
  20260429_201500/                # hybrid run
    hybrid/
      sets/
        metrics/{summary.json, stream_metrics.csv}
        detections/detections.csv          # kept reps (classifier_prob, refiner_delta_*_samples)
        candidates/labeled_candidates.csv  # all training candidates with TP/FP labels and refiner regression targets
        models/{action}/{test_subject}/    # AutoGluon classifier + _refiner_start / _refiner_end (deleted after eval by default)
        templates/templates.csv
        plots/{action}/{subject}/*.svg
        metadata/{run_manifest.json, config_snapshot.yaml}
      whole/...
      summary.json
```

## DS-MS-TCN Micro/Macro Rep Segmentation

This pipeline implements a DS-MS-TCN-style sequence-to-sequence recognizer for
rep segmentation and action recognition. Stage 1 predicts fixed micro labels
(`other`, `concentric`, `eccentric`) and reps are extracted only from the fixed
order `concentric -> eccentric`. Stages 2-4 predict and refine macro labels
(`other + action_type`), so the same model can also provide per-rep action
labels.

By default, `configs/micro_macro_recognition.yaml` runs both Stage 1 sources
under the same timestamped output folder:

```bash
python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml
```

Run only the learned Stage 1 TCN version:

```bash
python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn
```

Run only the DTW Stage 1 comparison version:

```bash
python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source dtw
```

You can also switch the same setting in `configs/micro_macro_recognition.yaml`:

```yaml
micro_macro:
  micro_source: both  # both | tcn | dtw
```

The CLI flag is only an override for quick comparisons; the fixed phase order
`concentric -> eccentric` is intentionally not configurable.

Resource settings are automatic by default:

```yaml
train:
  device: auto       # CUDA -> MPS -> CPU
  num_workers: auto
  pin_memory: auto   # enabled only for CUDA
  amp: true          # mixed precision only on CUDA
```

Set `device: cpu`, `device: cuda`, or `device: mps` if you want to force a
specific backend. The run prints the resolved device, AMP state, DataLoader
workers, and pinned-memory setting before training starts.

The split is subject-wise: streams are assigned by their subject folder, and the
run aborts if any split subject appears in both train and test. Set the held-out
person with:

```yaml
train:
  test_subject: kevin
```

The paper uses non-causal temporal convolutions, which can look at future
samples. This pipeline defaults to causal convolutions for real-time use:

```yaml
micro_macro:
  causal: true
```

Set `causal: false` to run the paper-style symmetric/non-causal TCN for offline
comparison.

Training slices are sized from the actual training-stream sample rate, so
`slice_seconds: 40.0` remains 40 seconds even when the raw data is not exactly
the fallback `window.sample_rate_hz`.

After training a causal TCN run, generate a streaming-style online replay with:

```bash
python -m evaluation.streaming_micro_macro \
  --run-dir artifacts/micro_macro_recognition/<timestamp>/tcn \
  --csv datasets/raw_data/<subject>/<action>/<set>/your.csv
```

For set-level replay, pass the set directory directly. The script will merge
`rep*.csv` in natural order and write `merged_set_input.csv` beside the replay
artifacts:

```bash
python -m evaluation.streaming_micro_macro \
  --run-dir artifacts/micro_macro_recognition/<timestamp>/tcn \
  --csv datasets/raw_data/<subject>/<action>/set0 \
  --max-samples -1
```

This writes a sample-by-sample CSV, a static SVG overview, and an interactive
HTML replay. The static replay is useful for inspection after inference is
done.

To simulate a board receiving samples in real time, use live mode. This runs
`OnlineDSMSTCNPredictor.update()` for each incoming sample and updates a browser
dashboard while the replay is running:

```bash
python -m evaluation.streaming_micro_macro \
  --run-dir artifacts/micro_macro_recognition/<timestamp>/tcn \
  --csv datasets/raw_data/<subject>/<action>/set0 \
  --max-samples -1 \
  --live \
  --replay-speed 1.0 \
  --live-window-seconds 15 \
  --live-history-seconds 60
```

Open the printed `http://127.0.0.1:<port>/live_dashboard.html` URL while the
command is running. Use `--replay-speed 2.0` or `5.0` for faster-than-real-time
inspection, or `--keep-server-open` if you want the local dashboard server to
stay open after replay finishes.

For board-style streams without reliable `sensor_ts`, pass the known sensor rate
explicitly, for example `--sample-rate 50`. The live dashboard keeps only the
recent `--live-history-seconds` samples in `live_state.json` so long realtime
tests do not keep rewriting an ever-growing JSON file; `streaming_predictions.csv`
still stores the full replay.

For speed, replay defaults to `--method fast`, which runs the full sequence once
with the causal TCN. For a causal checkpoint, each output still only depends on
past samples, so this is online-equivalent for visualization. Use
`--method step` only when you specifically want to benchmark sample-by-sample
loop overhead.

The DTW Stage 1 comparison is CPU-bound, so long whole-session streams can be
slow. The config uses a coarse search by default:

```yaml
micro_macro:
  dtw:
    detection_stride: 8
    duration_stride: 8
    dtw_downsample_factor: 2
    max_windows_per_label: 5000
```

Lower these values for a denser but slower DTW search.

Artifacts are written under `./artifacts/micro_macro_recognition/`:

- `report.md`: one-page experiment report with key metrics, split details,
  config highlights, and artifact links.
- `models/ds_ms_tcn.pt`: trained four-stage model.
- `detections/rep_detections_{tcn,dtw}.csv`: predicted reps with transition,
  micro confidence, and predicted action label.
- `detections/pairing_diagnostics_{tcn,dtw}.csv`: invalid or unpaired micro
  runs such as missing eccentric segments.
- `metrics/stream_metrics_{tcn,dtw}.csv`: rep-level segmentation metrics.
  This also includes paper-style temporal segmentation metrics for both micro
  and macro streams: sample accuracy, sample macro-F1, edit score, and
  segment IoU F1@10/25/50.
- `metrics/rep_action_confusion_{tcn,dtw}.csv`: per-rep action confusion matrix.
- `plots/rep/{tcn,dtw}/*.svg`: waveform overlays for rep segmentation
  debugging.
- `plots/action/{tcn,dtw}/*.svg`: separate color-coded GT/pred action plots
  with predicted rep action labels.

Streaming replay artifacts are written by `evaluation.streaming_micro_macro`
under `<run-dir>/streaming_eval/<stream_id>/`:

- `streaming_predictions.csv`: sample-by-sample online labels and confidences.
- `streaming_replay.svg`: color-block overview with synchronized IMU waveforms.
- `streaming_replay.html`: interactive replay with a moving time cursor.
- `live_dashboard.html` / `live_state.json`: live browser dashboard files when
  `--live` is used.
- `streaming_summary.json`: input/output paths and buffer information.

## Student Model Artifacts (`io.output_dir`)

- `student_best.pt`
- `label_map.json`
- `zscore_stats.json`
- `train_summary.json`
- `effective_config.json`

## Export ONNX

```bash
python -m deploy.export_onnx --config config.yaml
```

Optional overrides:

```bash
python -m deploy.export_onnx --config config.yaml --checkpoint ./artifacts_student/student_best.pt --output ./artifacts_student/student_model.onnx --opset 17
```

## Luckfox-Style ONNX Inference

```bash
python -m deploy.luckfox_infer \
  --onnx ./artifacts_student/student_model.onnx \
  --stats ./artifacts_student/zscore_stats.json \
  --label-map ./artifacts_student/label_map.json \
  --csv ./your_stream_or_record.csv
```

This script uses a rolling window buffer and prints one predicted action label per ready window.
