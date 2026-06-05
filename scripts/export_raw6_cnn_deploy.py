"""Train and export the raw6 causal CNN deployment artifact.

This exports the current research deployment target: raw 6-axis IMU input,
the phase-only 1D causal CNN, and the top5_p5 selective duration-merge decoder
configuration. The artifact is trained on all valid set streams, so it is for
deployment/replay, not for reporting held-out quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.new_c_pipeline.duration_merge_decoder_9fold import build_duration_priors  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    stream_action,
    train_raw6_model,
)
from scripts.new_c_pipeline.selective_duration_merge_decoder_9fold import ACTION_SETS  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, set_seed, should_exclude  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


def _json_dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_onnx(model: torch.nn.Module, path: Path, input_channels: int, slice_len: int) -> bool:
    dummy = torch.zeros(1, input_channels, slice_len, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["imu"],
            output_names=["phase_logits"],
            opset_version=17,
        )
    except Exception as exc:  # pragma: no cover - depends on local ONNX stack
        print(f"[WARN] ONNX export failed: {exc}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployable raw6 CNN + top5_p5 artifact.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/deploy/raw6_cnn_top5_p5_current")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-onnx", action="store_true", help="Only write the PyTorch checkpoint.")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in streams if not should_exclude(sid)]
    imu_columns = list(raw_cfg.get("feature", {}).get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    sample_rate_hz = int(raw_cfg.get("window", {}).get("sample_rate_hz", 100))

    print(f"[EXPORT] valid streams={len(streams)}")
    print(f"[EXPORT] actions={sorted({stream_action(sid) for sid, _ in streams})}")
    print(f"[EXPORT] training raw6 CNN hidden={args.hidden}, epochs={args.epochs}, device={device}")
    model, mean, std, n_segments = train_raw6_model(streams, imu_columns, args.hidden, args.epochs, device)
    duration_priors = build_duration_priors(streams, [5])

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "CausalCNN_PhaseOnly",
        "input_channels": len(imu_columns),
        "hidden": int(args.hidden),
        "num_classes": 2,
        "dropout": 0.2,
    }
    torch.save(checkpoint, out_dir / "model.pt")

    normalization = {
        "imu_columns": imu_columns,
        "mean": np.asarray(mean, dtype=float).tolist(),
        "std": np.asarray(std, dtype=float).tolist(),
    }
    decoder_config = {
        "phase_labels": {"0": "eccentric", "1": "concentric"},
        "slice_len": 300,
        "overlap_stride": 150,
        "smoothing_window": 25,
        "viterbi_penalty": 0.3,
        "min_phase_samples": 3,
        "max_phase_gap_samples": 3,
        "selective_merge_name": "top5_p5",
        "selective_merge_actions": ACTION_SETS["top5"],
        "selective_merge_percentile": 5,
        "selective_merge_max_gap_samples": 50,
        "duration_priors": duration_priors,
        "active_policy": "deployment_replay_treats_input_set_as_active_by_default",
    }
    metadata = {
        "artifact_name": "raw6_cnn_top5_p5_current",
        "model_family": "raw6 phase-only 1D causal CNN",
        "source_config": str(args.config),
        "input_shape": "[batch, 6, 300]",
        "sample_rate_hz": sample_rate_hz,
        "epochs": int(args.epochs),
        "hidden": int(args.hidden),
        "seed": int(args.seed),
        "train_streams": len(streams),
        "train_active_segments": int(n_segments),
        "excluded_sessions": EXCLUDED_SESSIONS,
        "files": {
            "pytorch_checkpoint": "model.pt",
            "normalization": "normalization.json",
            "decoder_config": "decoder_config.json",
            "onnx": "model.onnx" if not args.skip_onnx else None,
        },
        "note": "Trained on all valid set streams for deployment. Do not use this artifact for headline held-out metrics.",
    }

    _json_dump(out_dir / "normalization.json", normalization)
    _json_dump(out_dir / "decoder_config.json", decoder_config)
    _json_dump(out_dir / "metadata.json", metadata)

    if not args.skip_onnx:
        onnx_ok = _write_onnx(model, out_dir / "model.onnx", len(imu_columns), 300)
        metadata["files"]["onnx"] = "model.onnx" if onnx_ok else None
        _json_dump(out_dir / "metadata.json", metadata)

    print(f"[OK] exported CNN deployment artifact to {out_dir}")


if __name__ == "__main__":
    main()
