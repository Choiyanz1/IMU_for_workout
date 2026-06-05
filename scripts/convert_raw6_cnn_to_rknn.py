"""Convert the raw6 CNN ONNX model to RKNN format.

Run this on Ubuntu/WSL/Linux with RKNN-Toolkit2 installed. Windows Python wheels
are not provided by Rockchip, so this script usually cannot run in native Win32.
"""
from __future__ import annotations

import argparse
from pathlib import Path


VALID_TARGETS = (
    "rv1103",
    "rv1106",
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
    "rv1126b",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw6 CNN ONNX to RKNN.")
    parser.add_argument("--artifact", default="artifacts/deploy/raw6_cnn_top5_p5_current")
    parser.add_argument("--onnx", default=None, help="Default: <artifact>/model_rknn.onnx")
    parser.add_argument("--output", default=None, help="Default: <artifact>/model.rknn")
    parser.add_argument("--target", default="rv1103", choices=VALID_TARGETS)
    parser.add_argument("--dtype", default="fp", choices=["fp", "i8"], help="fp=no quantization; i8=INT8 quantization")
    parser.add_argument("--dataset", default=None, help="Calibration dataset txt for --dtype i8")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RKNN-Toolkit2 is not installed. Install it in Ubuntu/WSL from "
            "https://github.com/airockchip/rknn-toolkit2/releases, then rerun this script."
        ) from exc

    artifact_dir = Path(args.artifact)
    onnx_path = Path(args.onnx) if args.onnx else artifact_dir / "model_rknn.onnx"
    output_path = Path(args.output) if args.output else artifact_dir / "model.rknn"
    if args.dtype == "i8" and not args.dataset:
        raise SystemExit("--dataset is required for --dtype i8 quantization")

    rknn = RKNN(verbose=args.verbose)
    print("--> Config RKNN")
    ret = rknn.config(target_platform=args.target)
    if ret != 0:
        raise SystemExit(f"RKNN config failed: {ret}")

    print("--> Load ONNX")
    ret = rknn.load_onnx(model=str(onnx_path), inputs=["imu"], input_size_list=[[1, 6, 300]])
    if ret != 0:
        raise SystemExit(f"RKNN load_onnx failed: {ret}")

    do_quant = args.dtype == "i8"
    print(f"--> Build RKNN dtype={args.dtype}")
    ret = rknn.build(do_quantization=do_quant, dataset=args.dataset)
    if ret != 0:
        raise SystemExit(f"RKNN build failed: {ret}")

    print("--> Export RKNN")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise SystemExit(f"RKNN export failed: {ret}")
    rknn.release()
    print(f"[OK] exported RKNN: {output_path}")


if __name__ == "__main__":
    main()
