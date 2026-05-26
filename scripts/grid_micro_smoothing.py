from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.streaming_micro_macro import _resolve_run_dir
from scripts.grid_micro_macro_postprocess import _load_model_for_grid
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    labels_to_runs,
    micro_labels_from_phase,
    sample_classification_metrics,
    segment_iou_f1,
)
from preprocessing.window_pipeline import ZScoreStats, apply_zscore
from train.micro_macro_recognition import _available_actions, _load_config, _load_set_sequences


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def _causal_average(probs: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return probs.copy()
    out = np.zeros_like(probs)
    csum = np.cumsum(probs, axis=0)
    for i in range(len(probs)):
        start = max(0, i - window + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search causal smoothing for micro sample-wise and IoU metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--subject", default="kevin")
    parser.add_argument("--window-sizes", default="1,3,5,7,9,11,15")
    parser.add_argument("--min-phase-samples", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-streams", type=int, default=-1)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(Path(args.run_dir))
    cfg = _load_config(run_dir / "metadata" / "config_snapshot.yaml")
    device_setting = str(args.device)
    if device_setting == "auto":
        device_setting = str((cfg.get("train", {}) or {}).get("device", "auto"))
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)

    causal_override = "true" if bool((cfg.get("micro_macro", {}) or {}).get("causal", True)) else "false"
    model, macro_classes, micro_classes, _semantic_micro_classes, imu_columns = _load_model_for_grid(run_dir, device, causal_override)
    stats = ZScoreStats.load(run_dir / "metadata" / "zscore_stats.json")

    data_cfg = cfg.get("data", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])
    actions = _available_actions(data_dir, data_cfg.get("include_actions"))

    streams = []
    for action in actions:
        streams.extend(_load_set_sequences(data_dir, str(args.subject), action, exclude_patterns))
    if args.max_streams > 0:
        streams = streams[: int(args.max_streams)]

    cache = []
    for stream_id, df_raw in streams:
        df = apply_zscore(df_raw, imu_columns, stats)
        x = torch.from_numpy(df[imu_columns].to_numpy(dtype=np.float32))[None, :, :].to(device)
        with torch.no_grad():
            out = model(x)
        probs = out["micro_probs"].detach().cpu().numpy()[0]
        truth_labels = micro_labels_from_phase(df["phase"].to_numpy())
        truth_runs = labels_to_runs(
            truth_labels,
            positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
            min_length=int(args.min_phase_samples),
        )
        cache.append((stream_id, probs, truth_labels, truth_runs))

    results = []
    for window in _parse_ints(args.window_sizes):
        sample_acc = []
        sample_f1 = []
        iou10 = []
        iou25 = []
        iou50 = []
        for stream_id, probs, truth_labels, truth_runs in cache:
            smoothed = _causal_average(probs, window)
            pred_labels = np.asarray([micro_classes[int(i)] for i in np.argmax(smoothed, axis=1)], dtype=object)
            pred_runs = labels_to_runs(
                pred_labels,
                positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
                probabilities=smoothed,
                min_length=int(args.min_phase_samples),
            )
            sample_metrics = sample_classification_metrics(truth_labels, pred_labels, MICRO_LABELS)
            seg_metrics = segment_iou_f1(truth_runs, pred_runs)
            sample_acc.append(sample_metrics["accuracy"])
            sample_f1.append(sample_metrics["macro_f1"])
            iou10.append(seg_metrics["f1_at_10"])
            iou25.append(seg_metrics["f1_at_25"])
            iou50.append(seg_metrics["f1_at_50"])
        row = {
            "window": window,
            "micro_sample_accuracy": float(np.mean(sample_acc)),
            "micro_sample_macro_f1": float(np.mean(sample_f1)),
            "micro_f1_at_10": float(np.mean(iou10)),
            "micro_f1_at_25": float(np.mean(iou25)),
            "micro_f1_at_50": float(np.mean(iou50)),
        }
        results.append(row)

    results.sort(key=lambda r: (r["micro_f1_at_50"], r["micro_sample_macro_f1"], r["micro_f1_at_25"]), reverse=True)
    print(json.dumps({"top": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
