"""Streaming-style evaluation for a trained DS-MS-TCN run.

This module does not train. It loads a saved TCN run, feeds one sample at a
time through a rolling-buffer causal predictor, and writes CSV/SVG/HTML records
for inspecting online predictions.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from models.ds_ms_tcn import DSMSTCN, DSMSTCNConfig, OnlineDSMSTCNPredictor
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    OTHER_LABEL,
    labels_to_runs,
    macro_labels_from_action,
    micro_labels_from_phase,
    write_streaming_prediction_svg,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz
from preprocessing.window_pipeline import ZScoreStats, apply_zscore


def _load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _load_model(run_dir: Path, device: torch.device) -> tuple[DSMSTCN, list[str], list[str], list[str]]:
    ckpt_path = run_dir / "models" / "ds_ms_tcn.pt"
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
            causal=bool(cfg_raw.get("causal", True)),
        )
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    if not model.cfg.causal:
        raise ValueError("Streaming evaluation requires a causal checkpoint.")
    return model, macro_classes, micro_classes, imu_columns


def _write_replay_html(
    path: Path,
    stream_id: str,
    rows: pd.DataFrame,
    sample_rate_hz: float,
    svg_rel: str,
) -> None:
    data = rows[
        [
            "sample_idx",
            "online_micro_label",
            "online_micro_confidence",
            "online_macro_label",
            "online_macro_confidence",
            "gt_micro_label",
            "gt_macro_label",
        ]
    ].to_dict(orient="records")
    payload = json.dumps(data)
    title = html.escape(stream_id)
    svg_src = html.escape(svg_rel)
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Streaming replay - {title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }}
    button {{ padding: 6px 12px; border: 1px solid #cbd5e1; background: #fff; border-radius: 6px; cursor: pointer; }}
    input[type="range"] {{ width: 520px; }}
    .readout {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 8px; margin: 12px 0 18px; }}
    .cell {{ border: 1px solid #e2e8f0; padding: 8px 10px; border-radius: 6px; background: #f8fafc; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
    .frame {{ position: relative; display: inline-block; }}
    .cursor {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #ef4444; pointer-events: none; transform: translateX(110px); }}
    img {{ width: 1280px; max-width: 100%; height: auto; border: 1px solid #e2e8f0; }}
  </style>
</head>
<body>
  <h1>Streaming replay: {title}</h1>
  <div class="toolbar">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="0" max="{max(0, len(data) - 1)}" value="0">
    <span id="time">0.00s</span>
  </div>
  <div class="readout">
    <div class="cell"><div class="label">Online micro</div><div id="omicro" class="value"></div></div>
    <div class="cell"><div class="label">Online action</div><div id="omacro" class="value"></div></div>
    <div class="cell"><div class="label">GT micro</div><div id="gmicro" class="value"></div></div>
    <div class="cell"><div class="label">GT action</div><div id="gmacro" class="value"></div></div>
  </div>
  <div class="frame">
    <img id="plot" src="{svg_src}" alt="streaming prediction plot">
    <div id="cursor" class="cursor"></div>
  </div>
  <script>
    const data = {payload};
    const sampleRate = {float(sample_rate_hz):.12f};
    const slider = document.getElementById('slider');
    const cursor = document.getElementById('cursor');
    let timer = null;
    function render(i) {{
      i = Math.max(0, Math.min(data.length - 1, Number(i)));
      slider.value = i;
      const row = data[i];
      document.getElementById('time').textContent = (row.sample_idx / sampleRate).toFixed(2) + 's';
      document.getElementById('omicro').textContent = row.online_micro_label + ' ' + Number(row.online_micro_confidence).toFixed(2);
      document.getElementById('omacro').textContent = row.online_macro_label + ' ' + Number(row.online_macro_confidence).toFixed(2);
      document.getElementById('gmicro').textContent = row.gt_micro_label;
      document.getElementById('gmacro').textContent = row.gt_macro_label;
      const x0 = 110;
      const plotW = 1080;
      const x = x0 + (row.sample_idx / Math.max(1, data.length - 1)) * plotW;
      cursor.style.transform = `translateX(${{x}}px)`;
    }}
    slider.addEventListener('input', () => render(slider.value));
    document.getElementById('play').addEventListener('click', () => {{
      if (timer) return;
      timer = setInterval(() => {{
        const next = Number(slider.value) + Math.max(1, Math.round(sampleRate / 20));
        if (next >= data.length - 1) {{
          render(data.length - 1);
          clearInterval(timer);
          timer = null;
        }} else {{
          render(next);
        }}
      }}, 50);
    }});
    document.getElementById('pause').addEventListener('click', () => {{
      clearInterval(timer);
      timer = null;
    }});
    render(0);
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    config_path = Path(args.config) if args.config else run_dir / "metadata" / "config_snapshot.yaml"
    raw = _load_config(config_path)
    device = _resolve_device(str(args.device or (raw.get("train", {}) or {}).get("device", "auto")))
    model, macro_classes, _micro_classes, imu_columns = _load_model(run_dir, device)

    df = pd.read_csv(args.csv)
    stats_path = Path(args.stats) if args.stats else run_dir / "metadata" / "zscore_stats.json"
    stats = ZScoreStats.load(stats_path)
    df = apply_zscore(df, imu_columns, stats)
    max_samples = int(args.max_samples)
    if max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)
    stream_id = args.stream_id or Path(args.csv).stem
    sample_rate = infer_sample_rate_hz(df)

    predictor = OnlineDSMSTCNPredictor(model, imu_columns, device, buffer_size=args.buffer_size)
    micro_probs, macro_probs = [], []
    for sample in df[imu_columns].to_numpy(dtype=np.float32):
        out = predictor.update(sample)
        micro_probs.append(out["micro_probs"].numpy())
        macro_probs.append(out["macro4_probs"].numpy())
    micro_probs_np = np.stack(micro_probs, axis=0)
    macro_probs_np = np.stack(macro_probs, axis=0)
    online_micro_labels = [MICRO_LABELS[int(i)] for i in np.argmax(micro_probs_np, axis=1)]
    online_macro_labels = [macro_classes[int(i)] for i in np.argmax(macro_probs_np, axis=1)]
    gt_micro_labels = micro_labels_from_phase(df["phase"].to_numpy()) if "phase" in df.columns else np.asarray([OTHER_LABEL] * len(df), dtype=object)
    gt_macro_labels = (
        macro_labels_from_action(df["action_type"].astype(str).to_numpy(), gt_micro_labels)
        if "action_type" in df.columns
        else np.asarray([OTHER_LABEL] * len(df), dtype=object)
    )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "streaming_eval" / stream_id
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "sample_idx": np.arange(len(df), dtype=int),
            "online_micro_label": online_micro_labels,
            "online_micro_confidence": np.max(micro_probs_np, axis=1),
            "online_macro_label": online_macro_labels,
            "online_macro_confidence": np.max(macro_probs_np, axis=1),
            "gt_micro_label": gt_micro_labels,
            "gt_macro_label": gt_macro_labels,
        }
    )
    csv_path = output_dir / "streaming_predictions.csv"
    rows.to_csv(csv_path, index=False)

    gt_micro_runs = labels_to_runs(gt_micro_labels, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=1)
    online_micro_runs = labels_to_runs(online_micro_labels, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=1)
    positive_macro = [c for c in macro_classes if c != OTHER_LABEL]
    gt_macro_runs = labels_to_runs(gt_macro_labels, positive_labels=positive_macro, min_length=1)
    online_macro_runs = labels_to_runs(online_macro_labels, positive_labels=positive_macro, min_length=1)
    svg_path = output_dir / "streaming_replay.svg"
    write_streaming_prediction_svg(
        svg_path,
        stream_id,
        df,
        gt_micro_runs,
        online_micro_runs,
        gt_macro_runs,
        online_macro_runs,
        sample_rate,
        predictor.buffer_size,
    )
    html_path = output_dir / "streaming_replay.html"
    _write_replay_html(html_path, stream_id, rows, sample_rate, svg_path.name)
    summary = {
        "run_dir": run_dir.as_posix(),
        "csv": Path(args.csv).as_posix(),
        "stream_id": stream_id,
        "samples": int(len(df)),
        "sample_rate_hz": float(sample_rate),
        "buffer_size": int(predictor.buffer_size),
        "buffer_seconds": float(predictor.buffer_size / sample_rate) if sample_rate > 0 else None,
        "streaming_predictions_csv": csv_path.as_posix(),
        "streaming_replay_svg": svg_path.as_posix(),
        "streaming_replay_html": html_path.as_posix(),
    }
    (output_dir / "streaming_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming DS-MS-TCN evaluation and replay")
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained tcn run directory")
    parser.add_argument("--csv", type=Path, required=True, help="Raw CSV to replay")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--buffer-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
