# RKNN Conversion for Luckfox Pico Zero

## Findings
- Rockchip's RKNN stack converts models on a PC with `RKNN-Toolkit2`, then runs `.rknn` models on the board with RKNN Runtime or `RKNN-Toolkit-Lite2`.
- Current Rockchip `RKNN-Toolkit2` supports `RV1103/RV1106` and ONNX opset 12-19.
- Luckfox Pico SDK lists `RV1103_Luckfox_Pico` as board option `[0]`; for Pico Zero deployment, use RKNN conversion target `rv1103` first.
- RKNN Model Zoo Linux demo build uses target `rv1106` as a grouped label for `rv1103/rv1106`, but conversion examples accept both `rv1103` and `rv1106`.

## Added Files
- `scripts/export_raw6_cnn_rknn_onnx.py`: exports `model_rknn.onnx` using constant causal padding for RKNN compatibility.
- `scripts/convert_raw6_cnn_to_rknn.py`: converts `model_rknn.onnx` to `model.rknn` with `RKNN-Toolkit2`.
- `scripts/stream_raw6_cnn_top5_p5.py`: now supports `--runtime rknn` for board-side Python runtime if `rknnlite` is available.

## Host Conversion Commands
Run on Ubuntu/WSL/Linux with `RKNN-Toolkit2` installed:

```sh
python scripts/export_raw6_cnn_rknn_onnx.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current
python scripts/convert_raw6_cnn_to_rknn.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --target rv1103 --dtype fp
```

If `rv1103` fails because of toolkit/runtime version mismatch, retry:

```sh
python scripts/convert_raw6_cnn_to_rknn.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --target rv1106 --dtype fp
```

## Board Runtime Command
After copying the artifact folder to Luckfox, run:

```sh
zig_bt_client --stdout --no-file | python scripts/stream_raw6_cnn_top5_p5.py --artifact artifacts/deploy/raw6_cnn_top5_p5_current --input - --action db_rdl --runtime rknn
```

## Notes
- Use FP conversion first. INT8 requires calibration data and should be validated against ONNX/PyTorch outputs before deployment.
- The exported RKNN-friendly ONNX uses constant zero causal padding instead of reflect padding. Since inputs are z-scored, zero padding corresponds to the training-set mean.
- Native Windows conversion was not possible in this session because `rknn.api` is not installed and Rockchip conversion wheels are normally Linux-targeted.
