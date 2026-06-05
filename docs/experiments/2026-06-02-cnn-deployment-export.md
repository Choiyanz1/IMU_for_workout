# CNN Deployment Export

## What Changed
- Added `scripts/export_raw6_cnn_deploy.py` to train the raw6 phase-only 1D causal CNN on all valid set streams and export a deployment artifact.
- Added `scripts/stream_raw6_cnn_top5_p5.py` to replay saved workout CSV rows or consume `zig_bt_client --stdout` raw IMU rows with the CNN and `top5_p5` decoder.

## Deployment Artifact
- Default path: `artifacts/deploy/raw6_cnn_top5_p5_current/`
- Files: `model.pt`, `model.onnx`, `normalization.json`, `decoder_config.json`, `metadata.json`, optional `model_rknn.onnx`, and optional `model.rknn`.
- Input: raw 6-axis IMU columns `ax, ay, az, gx, gy, gz` at 100 Hz.
- Model window: `[batch, 6, 300]`.

## Commands
```sh
python scripts/export_raw6_cnn_deploy.py --epochs 20 --hidden 64
python scripts/export_raw6_cnn_rknn_onnx.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current
python scripts/convert_raw6_cnn_to_rknn.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --target rv1103 --dtype fp
python scripts/stream_raw6_cnn_top5_p5.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --input Raw_data.csv --action db_rdl --runtime torch
zig_bt_client --stdout --no-file | python scripts/stream_raw6_cnn_top5_p5.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --input - --action db_rdl --runtime onnx
zig_bt_client --stdout --no-file | python scripts/stream_raw6_cnn_top5_p5.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --input - --action db_rdl --runtime rknn
```

## RKNN Notes
- Luckfox Pico Zero maps to the `RV1103_Luckfox_Pico` SDK board, so the default RKNN conversion target is `rv1103`.
- Rockchip's RKNN flow is: convert ONNX to RKNN on Ubuntu/WSL/Linux with `RKNN-Toolkit2`, then run the `.rknn` on the board with RKNN Runtime or `rknn-toolkit-lite2`.
- Native Windows usually cannot run the converter because Rockchip distributes Linux Python wheels for `RKNN-Toolkit2`.
- Use `--dtype fp` first. INT8 (`--dtype i8`) needs a calibration dataset and should be validated against ONNX/PyTorch outputs before deployment.

## Caveat
- This artifact is trained on all valid set streams for deployment. It must not be used as a held-out metric source.
- The current live path assumes the input interval is an active workout set and requires action context via `--action`; full-session rest-aware active gating remains a separate deployment gate.
