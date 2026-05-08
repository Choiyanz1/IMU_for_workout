from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from evaluation.streaming_micro_macro import _load_model, _resolve_run_dir
from models.ds_ms_tcn import DSMSTCN, DSMSTCNConfig
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    labels_to_runs,
    pair_concentric_eccentric_reps,
    rep_metrics,
    truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz
from preprocessing.window_pipeline import ZScoreStats, apply_zscore
from train.micro_macro_recognition import _available_actions, _load_set_sequences


def _resolve_device(device_setting: str) -> torch.device:
    requested = str(device_setting).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return torch.device("cpu")
    return torch.device(requested)


def _parse_ints(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(x) for x in str(value).split(",") if str(x).strip()]


def _filter_reps(reps, sample_rate_hz: float, min_duration_seconds: float, min_confidence: float):
    min_samples = max(0, int(round(float(min_duration_seconds) * float(sample_rate_hz))))
    out = []
    for rep in reps:
        if min_samples > 0 and int(rep.end_idx) - int(rep.start_idx) < min_samples:
            continue
        if float(min_confidence) > 0 and float(rep.micro_confidence) < float(min_confidence):
            continue
        out.append(rep)
    return out


def _load_model_for_grid(run_dir: Path, device: torch.device, causal_override: str):
    if causal_override == "auto":
        return _load_model(run_dir, device)
    ckpt_path = run_dir / "models" / "ds_ms_tcn.pt"
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    macro_classes = [str(x) for x in payload["macro_classes"]]
    micro_classes = [str(x) for x in payload.get("micro_classes", MICRO_LABELS)]
    imu_columns = [str(x) for x in payload["imu_columns"]]
    cfg_raw = dict(payload.get("config", {}) or {})
    model = DSMSTCN(
        DSMSTCNConfig(
            input_channels=len(imu_columns),
            micro_classes=len(micro_classes),
            macro_classes=len(macro_classes),
            num_filters=int(cfg_raw.get("num_filters", 64)),
            num_layers=int(cfg_raw.get("num_layers", 9)),
            kernel_size=int(cfg_raw.get("kernel_size", 3)),
            dropout=float(cfg_raw.get("dropout", 0.2)),
            causal=causal_override == "true",
        )
    )
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, macro_classes, micro_classes, imu_columns


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {key: float(sum(float(row[key]) for row in rows)) for key in ("n_pred", "n_true", "tp", "fp", "fn")}
    tp, fp, fn = out["tp"], out["fp"], out["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out.update({"precision": float(precision), "recall": float(recall), "f1": float(f1)})
    return out


def run(args: argparse.Namespace) -> None:
    run_dir = _resolve_run_dir(Path(args.run_dir))
    config_path = Path(args.config) if args.config else run_dir / "metadata" / "config_snapshot.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = _resolve_device(str(args.device or (raw.get("train", {}) or {}).get("device", "auto")))
    model, macro_classes, micro_classes, imu_columns = _load_model_for_grid(run_dir, device, str(args.causal))
    stats = ZScoreStats.load(Path(args.stats) if args.stats else run_dir / "metadata" / "zscore_stats.json")

    data_cfg = raw.get("data", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    include_actions = data_cfg.get("include_actions")
    actions = _available_actions(data_dir, include_actions)
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])

    streams = []
    for action in actions:
        streams.extend(_load_set_sequences(data_dir, str(args.subject), action, exclude_patterns))
    if args.max_streams > 0:
        streams = streams[: int(args.max_streams)]
    if not streams:
        raise RuntimeError(f"No set streams found for subject={args.subject}")

    cache = []
    for idx, (stream_id, df_raw) in enumerate(streams, start=1):
        df = apply_zscore(df_raw, imu_columns, stats)
        sample_rate = infer_sample_rate_hz(df)
        x = torch.from_numpy(df[imu_columns].to_numpy(dtype=np.float32))[None, :, :].to(device)
        with torch.no_grad():
            out = model(x)
        micro_probs = out["micro_probs"].detach().cpu().numpy()[0]
        pred_labels = [micro_classes[int(i)] for i in np.argmax(micro_probs, axis=1)]
        truth_reps = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=int(args.truth_min_phase_samples),
        )
        cache.append(
            {
                "stream_id": stream_id,
                "sample_rate": sample_rate,
                "micro_probs": micro_probs,
                "pred_labels": pred_labels,
                "truth_reps": truth_reps,
            }
        )
        print(f"[INFO] predicted {idx}/{len(streams)} stream={stream_id} samples={len(df)}", flush=True)

    results = []
    for min_phase in _parse_ints(args.min_phase_samples):
        for max_gap in _parse_ints(args.max_phase_gap_samples):
            for min_duration in _parse_floats(args.min_rep_duration_seconds):
                for min_conf in _parse_floats(args.min_rep_confidence):
                    metric_rows = []
                    for item in cache:
                        pred_runs = labels_to_runs(
                            item["pred_labels"],
                            positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
                            probabilities=item["micro_probs"],
                            min_length=min_phase,
                        )
                        pred_reps, _ = pair_concentric_eccentric_reps(
                            pred_runs,
                            micro_source="tcn",
                            max_gap_samples=max_gap,
                        )
                        pred_reps = _filter_reps(pred_reps, item["sample_rate"], min_duration, min_conf)
                        metric_rows.append(rep_metrics(pred_reps, item["truth_reps"], item["sample_rate"]))
                    row = _aggregate(metric_rows)
                    row.update(
                        {
                            "min_phase_samples": int(min_phase),
                            "max_phase_gap_samples": int(max_gap),
                            "min_rep_duration_seconds": float(min_duration),
                            "min_rep_confidence": float(min_conf),
                        }
                    )
                    results.append(row)

    results.sort(key=lambda r: (r["f1"], r["precision"], r["recall"]), reverse=True)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "postprocess_grid"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / "tcn_postprocess_grid.csv", index=False)
    (output_dir / "best_tcn_postprocess.json").write_text(
        json.dumps({"best": results[0], "top10": results[:10]}, indent=2),
        encoding="utf-8",
    )
    print("[OK] top postprocess settings:", flush=True)
    for row in results[: int(args.print_top)]:
        print(json.dumps(row, separators=(",", ":")), flush=True)
    print(f"[OK] wrote {output_dir / 'tcn_postprocess_grid.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search DS-MS-TCN rep postprocessing without retraining.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subject", default="thomas_2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--causal", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--max-streams", type=int, default=-1)
    parser.add_argument("--truth-min-phase-samples", type=int, default=3)
    parser.add_argument("--min-phase-samples", default="3,5,8,10,12,15,20,25,30")
    parser.add_argument("--max-phase-gap-samples", default="0,3,5,8,10,15,20,30,40")
    parser.add_argument("--min-rep-duration-seconds", default="0.8,1.0,1.2,1.3,1.4,1.5")
    parser.add_argument("--min-rep-confidence", default="0.0,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--print-top", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
