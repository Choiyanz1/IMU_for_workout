# Full Auto Realtime Bundle

## Goal
- Package the current best engineering version of the automatic workout pipeline for later Luckfox Pico Zero deployment work.
- Include automatic action recognition, active/rest gating, bounded-latency phase segmentation, soft duration merge, and rep counting.

## Bundle
Current artifact:

```sh
artifacts/deploy/full_auto_realtime_current/
```

Files:

| File | Purpose |
|---|---|
| `phase_model.pt` | raw6 causal CNN phase model checkpoint |
| `phase_model.onnx` | ONNX phase model for CPU/ONNXRuntime and RKNN conversion input |
| `active_gate_rf.joblib` | periodic active/rest Random Forest gate |
| `active_gate_scaler.joblib` | scaler for active gate features |
| `active_gate_rf.json` | pure-NumPy RF tree export for active gate |
| `active_gate_scaler.json` | pure-NumPy scaler export for active gate |
| `action_active_rf.joblib` | RF active head for action branch |
| `action_rf.joblib` | RF action classifier over 8 workout actions |
| `action_active_rf.json` | pure-NumPy RF tree export for action-active head |
| `action_rf.json` | pure-NumPy RF tree export for action classifier |
| `normalization.json` | phase/action normalization stats and IMU column order |
| `pipeline_config.json` | active/action/phase/soft-top5/event settings |
| `metadata.json` | training/export metadata and caveats |

## Export Command
```sh
python scripts/export_full_auto_realtime_bundle.py \
  --epochs 5 \
  --hidden 64 \
  --output artifacts/deploy/full_auto_realtime_current
```

This trains on all valid set streams plus available rest streams. It is for deployment engineering, not held-out reporting.

## Runtime Command
Torch CPU replay:

```sh
python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input path/to/imu.csv \
  --runtime torch \
  --device cpu
```

ONNX replay:

```sh
python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input path/to/imu.csv \
  --runtime onnx
```

Luckfox-friendly replay path using JSON RF trees instead of sklearn/joblib:

```sh
python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input path/to/imu.csv \
  --runtime onnx \
  --rf-runtime json
```

JSONL event mode for live-style output:

```sh
python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input path/to/imu.csv \
  --runtime onnx \
  --rf-runtime json \
  --emit-mode jsonl-events \
  --emit-stride-samples 50
```

Each detected rep is emitted as one JSON line:

```json
{"type":"rep","count":1,"samples_seen":800,"top_action":"db_biceps_curl","start_idx":0,"end_idx":397}
```

The final line is a summary with the final count and batch-equivalent count.

More board-like stateful mode:

```sh
python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input path/to/imu.csv \
  --runtime onnx \
  --rf-runtime json \
  --emit-mode stateful-jsonl
```

`stateful-jsonl` updates active/action RFs at their configured strides, runs the phase CNN at the phase step, finalizes labels after the fixed lag, and feeds only newly finalized labels into a streaming parser. It is closer to a board loop than `jsonl-events`, which recomputes over the accumulated buffer.

Raw `zig_bt_client --stdout` style stdin:

```sh
zig_bt_client --stdout --no-file | python scripts/run_full_auto_realtime_bundle.py \
  --artifact artifacts/deploy/full_auto_realtime_current \
  --input - \
  --runtime onnx \
  --rf-runtime json \
  --emit-mode jsonl-events
```

Input formats:
- saved CSV with `ax,ay,az,gx,gy,gz` columns;
- raw client rows: `serial,type,ts,host_ts,ppg_a..j,ax,ay,az,gx,gy,gz,mx,my,mz`.

## Pipeline Settings
The current bundle uses:
- active gate: periodic RF, `200` sample window, `50` stride, enter/exit threshold `0.7`;
- action branch: RF dual head, `200` sample window, `100` stride;
- phase model: raw6 CNN, `300` sample trailing window;
- phase decode: causal `MA25`, fixed-lag Viterbi with `100` sample delay;
- rep parser: active-masked C/E parser;
- duration merge: soft posterior-gated top5, threshold scale `0.8`;
- event confirmation: min `2` reps, event gap `1000` samples.

## Smoke Checks
Smoke export:

```sh
python scripts/export_full_auto_realtime_bundle.py \
  --epochs 1 \
  --hidden 16 \
  --skip-onnx \
  --output artifacts/deploy/full_auto_realtime_smoke
```

Smoke replay on `_tsenyu_temp/tsenyu0515workout/db_biceps_curl/set0` exported to `sample_replay.csv` passed:
- Torch runtime: predicted action `db_biceps_curl`, count `12`.
- ONNX runtime on current bundle: predicted action `db_biceps_curl`, count `12`.
- ONNX + JSON RF runtime: predicted action `db_biceps_curl`, count `12`; no pandas/sklearn/joblib inference dependency.
- JSONL event mode with ONNX + JSON RF emitted `12` rep events and a final summary with `count=12`, matching final replay.
- Stateful JSONL mode with ONNX + JSON RF also emitted `12` rep events on the same sample. Boundaries differ slightly from final replay because causal active entry cannot backfill labels before the gate first enters active: batch active samples `6525`, stateful active samples `6476`, first rep start sample `0 -> 49`.

## Luckfox Notes
- The bundle now includes JSON RF tree exports. Use `--rf-runtime json` on the board to avoid sklearn/joblib for active/action inference.
- The remaining heavyweight runtime dependency is the phase CNN runtime: ONNXRuntime or RKNN runtime. For production Luckfox use, convert `phase_model.onnx` to `phase_model.rknn` and run with `--runtime rknn --rf-runtime json`.
- `phase_model.onnx` is available for RKNN conversion, but `.rknn` was not generated on this Windows machine because RKNN-Toolkit2 is not installed here.
- The runner supports two event modes. `jsonl-events` recomputes over the accumulated buffer every `--emit-stride-samples` samples and releases reps after a fixed safety delay, which is easiest to compare with batch replay. `stateful-jsonl` updates RF/CNN/parser state incrementally and is the better starting point for Luckfox integration, but its causal active gate does not backfill pre-entry labels, so boundaries can differ from batch replay.

## Portable Package

Prepared a minimal manual-transfer package at:

- Folder: `artifacts/deploy/full_auto_realtime_portable/`
- Zip: `artifacts/deploy/full_auto_realtime_portable.zip`

This package intentionally includes only the ONNX + JSON RF inference path: runner, `phase_model.onnx`, JSON RF trees/scaler, normalization/config/metadata, smoke sample, README, requirements, and manifest. It excludes datasets, PyTorch `.pt`, sklearn `.joblib`, training scripts, and evaluation artifacts.

Portable validation from inside the package passed:

```sh
python -m py_compile run_full_auto_realtime_bundle.py
python run_full_auto_realtime_bundle.py --artifact artifact --input samples/sample_replay.csv --runtime onnx --rf-runtime json
python run_full_auto_realtime_bundle.py --artifact artifact --input samples/sample_replay.csv --runtime onnx --rf-runtime json --emit-mode stateful-jsonl
```

Expected smoke result remains `db_biceps_curl`, count `12`.

## Caveat
Full-session quality is still limited by active/rest gating. The current bundle is the best packaged engineering baseline, not a final deployment-ready gate. In particular, low-amplitude biceps-style motion can be under-covered by strict active thresholds; tune `pipeline_config.json` active thresholds/bridge settings only after replay validation.
