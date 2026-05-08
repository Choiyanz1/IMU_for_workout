"""Streaming-style evaluation for a trained DS-MS-TCN run.

This module does not train. It loads a saved TCN run, feeds one sample at a
time through a rolling-buffer causal predictor, and writes CSV/SVG/HTML records
for inspecting online predictions.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
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


def _natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_run_dir(path: Path) -> Path:
    """Accept either the concrete TCN run directory or its timestamp parent."""
    path = path.expanduser()
    ckpt_path = path / "models" / "ds_ms_tcn.pt"
    if ckpt_path.exists():
        return path
    tcn_ckpt_path = path / "tcn" / "models" / "ds_ms_tcn.pt"
    if tcn_ckpt_path.exists():
        return path / "tcn"
    raise FileNotFoundError(
        "Could not find DS-MS-TCN checkpoint. Expected either "
        f"{ckpt_path.as_posix()} or {tcn_ckpt_path.as_posix()}. "
        "Use a completed causal TCN run, e.g. "
        "artifacts/micro_macro_recognition/<timestamp>/tcn."
    )


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
    if "causal" not in cfg_raw:
        raise ValueError(
            "Checkpoint does not record config.causal, so it may be a legacy non-causal model. "
            "Streaming inference requires a checkpoint trained with micro_macro.causal: true."
        )
    model = DSMSTCN(
        DSMSTCNConfig(
            input_channels=len(imu_columns),
            micro_classes=len(micro_classes),
            macro_classes=len(macro_classes),
            num_filters=int(cfg_raw.get("num_filters", 64)),
            num_layers=int(cfg_raw.get("num_layers", 9)),
            kernel_size=int(cfg_raw.get("kernel_size", 3)),
            dropout=float(cfg_raw.get("dropout", 0.2)),
            causal=bool(cfg_raw["causal"]),
        )
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    if not model.cfg.causal:
        raise ValueError("Streaming evaluation requires a causal checkpoint.")
    return model, macro_classes, micro_classes, imu_columns


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _motion_magnitude(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    available = [c for c in columns if c in df.columns]
    if not available:
        return np.zeros(len(df), dtype=float)
    return np.linalg.norm(df[available].to_numpy(dtype=float), axis=1)


def _prediction_summary(rows: pd.DataFrame, elapsed_seconds: float, sample_rate_hz: float) -> Dict[str, object]:
    samples = int(len(rows))
    throughput = float(samples / elapsed_seconds) if elapsed_seconds > 0 else None
    summary: Dict[str, object] = {
        "throughput_samples_per_second": throughput,
        "real_time_factor": float(throughput / sample_rate_hz) if throughput is not None and sample_rate_hz > 0 else None,
    }
    if rows.empty:
        return summary
    if {"online_micro_label", "gt_micro_label"}.issubset(rows.columns):
        summary["micro_sample_accuracy"] = float((rows["online_micro_label"].astype(str) == rows["gt_micro_label"].astype(str)).mean())
        summary["pred_micro_counts"] = {str(k): int(v) for k, v in rows["online_micro_label"].value_counts().items()}
        summary["gt_micro_counts"] = {str(k): int(v) for k, v in rows["gt_micro_label"].value_counts().items()}
    if {"online_macro_label", "gt_macro_label"}.issubset(rows.columns):
        summary["macro_sample_accuracy"] = float((rows["online_macro_label"].astype(str) == rows["gt_macro_label"].astype(str)).mean())
        summary["pred_macro_counts"] = {str(k): int(v) for k, v in rows["online_macro_label"].value_counts().items()}
        summary["gt_macro_counts"] = {str(k): int(v) for k, v in rows["gt_macro_label"].value_counts().items()}
    return summary


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
    .frame {{ position: relative; display: inline-block; line-height: 0; }}
    .cursor {{ position: absolute; top: 1px; bottom: 1px; width: 2px; background: #ef4444; pointer-events: none; transform: translateX(0); }}
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
    const svgWidth = 1280;
    const plotX0 = 110;
    const plotW = 1080;
    const slider = document.getElementById('slider');
    const cursor = document.getElementById('cursor');
    const img = document.getElementById('plot');
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
      const svgX = plotX0 + (row.sample_idx / Math.max(1, data.length - 1)) * plotW;
      const x = (svgX / svgWidth) * img.clientWidth;
      cursor.style.height = img.clientHeight + 'px';
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
    window.addEventListener('resize', () => render(slider.value));
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _label_color(label: str) -> str:
    palette = {
        "other": "#94a3b8",
        "concentric": "#16a34a",
        "eccentric": "#f97316",
        "db_bench_press": "#2563eb",
        "db_rdl": "#7c3aed",
        "db_weighted_crunch": "#dc2626",
        "one_arm_db_row": "#0891b2",
        "db_squat": "#ca8a04",
        "db_biceps_curl": "#db2777",
        "db_shoulder_press": "#0f766e",
        "db_triceps_curl": "#9333ea",
    }
    return palette.get(str(label), "#64748b")


def _write_json_atomic(path: Path, payload: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _trim_live_state(state: Dict[str, object], max_samples: int) -> None:
    if max_samples <= 0:
        return
    sample_idx = state.get("sample_idx")
    if not isinstance(sample_idx, list) or len(sample_idx) <= max_samples:
        return
    drop = len(sample_idx) - max_samples
    for key in (
        "sample_idx",
        "acc_mag",
        "gyro_mag",
        "online_micro_label",
        "online_micro_confidence",
        "online_macro_label",
        "online_macro_confidence",
        "gt_micro_label",
        "gt_macro_label",
    ):
        value = state.get(key)
        if isinstance(value, list):
            del value[:drop]


def _write_live_dashboard_html(path: Path, stream_id: str, sample_rate_hz: float, window_seconds: float) -> None:
    title = html.escape(stream_id)
    label_colors = json.dumps({k: _label_color(k) for k in [
        "other",
        "concentric",
        "eccentric",
        "db_bench_press",
        "db_rdl",
        "db_weighted_crunch",
        "one_arm_db_row",
        "db_squat",
        "db_biceps_curl",
        "db_shoulder_press",
        "db_triceps_curl",
    ]})
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live streaming replay - {title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #111827; background: #ffffff; }}
    .top {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .pill {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; background: #f8fafc; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 16px; font-weight: 700; margin-left: 4px; }}
    canvas {{ width: 1280px; max-width: 100%; height: 680px; border: 1px solid #e2e8f0; display: block; }}
  </style>
</head>
<body>
  <h1>Live streaming replay: {title}</h1>
  <div class="top">
    <div class="pill"><span class="label">Status</span><span id="status" class="value">loading</span></div>
    <div class="pill"><span class="label">Time</span><span id="time" class="value">0.00s</span></div>
    <div class="pill"><span class="label">Online micro</span><span id="micro" class="value">-</span></div>
    <div class="pill"><span class="label">Online action</span><span id="macro" class="value">-</span></div>
    <div class="pill"><span class="label">GT</span><span id="gt" class="value">-</span></div>
  </div>
  <canvas id="plot" width="1280" height="680"></canvas>
  <script>
    const sampleRate = {float(sample_rate_hz):.12f};
    const defaultWindowSamples = Math.max(10, Math.round({float(window_seconds):.6f} * sampleRate));
    const colors = {label_colors};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    function color(label) {{ return colors[label] || '#64748b'; }}
    function norm(values, idxs) {{
      let min = Infinity, max = -Infinity;
      for (const i of idxs) {{ const v = values[i]; if (Number.isFinite(v)) {{ min = Math.min(min, v); max = Math.max(max, v); }} }}
      if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return {{min: -1, max: 1}};
      const pad = (max - min) * 0.12;
      return {{min: min - pad, max: max + pad}};
    }}
    function drawSeries(values, idxs, x0, y0, w, h, stroke) {{
      const r = norm(values, idxs);
      ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.beginPath();
      idxs.forEach((idx, j) => {{
        const x = x0 + (j / Math.max(1, idxs.length - 1)) * w;
        const y = y0 + h - ((values[idx] - r.min) / Math.max(1e-9, r.max - r.min)) * h;
        if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}
    function drawBands(labels, idxs, x0, y, w, h) {{
      if (!idxs.length) return;
      let startJ = 0, cur = labels[idxs[0]];
      for (let j = 1; j <= idxs.length; j++) {{
        const lab = j < idxs.length ? labels[idxs[j]] : null;
        if (lab !== cur) {{
          const x = x0 + (startJ / Math.max(1, idxs.length - 1)) * w;
          const ww = Math.max(1, ((j - startJ) / Math.max(1, idxs.length - 1)) * w);
          ctx.fillStyle = color(cur); ctx.globalAlpha = cur === 'other' ? 0.18 : 0.72;
          ctx.fillRect(x, y, ww, h); ctx.globalAlpha = 1;
          startJ = j; cur = lab;
        }}
      }}
    }}
    function text(x, y, value, size = 13, weight = '500') {{
      ctx.font = `${{weight}} ${{size}}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
      ctx.fillStyle = '#111827'; ctx.fillText(value, x, y);
    }}
    function draw(state) {{
      const n = state.sample_idx.length;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!n) {{ text(40, 60, 'Waiting for samples...', 18, '700'); return; }}
      const end = n - 1;
      const start = Math.max(0, end - (state.window_samples || defaultWindowSamples) + 1);
      const idxs = [];
      for (let i = start; i <= end; i++) idxs.push(i);
      const x0 = 110, w = 1080;
      text(40, 52, `Live window ${{(state.sample_idx[start] / sampleRate).toFixed(1)}}s - ${{(state.sample_idx[end] / sampleRate).toFixed(1)}}s`, 18, '700');
      text(40, 104, 'GT micro', 14, '700'); ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(x0, 84, w, 24); drawBands(state.gt_micro_label, idxs, x0, 84, w, 24);
      text(40, 142, 'Pred micro', 14, '700'); ctx.strokeRect(x0, 122, w, 24); drawBands(state.online_micro_label, idxs, x0, 122, w, 24);
      text(40, 194, 'GT action', 14, '700'); ctx.strokeRect(x0, 174, w, 28); drawBands(state.gt_macro_label, idxs, x0, 174, w, 28);
      text(40, 236, 'Pred action', 14, '700'); ctx.strokeRect(x0, 216, w, 28); drawBands(state.online_macro_label, idxs, x0, 216, w, 28);
      text(40, 312, 'acc_mag', 14, '700'); ctx.strokeRect(x0, 285, w, 125); drawSeries(state.acc_mag, idxs, x0, 285, w, 125, '#2563eb');
      text(40, 477, 'gyro_mag', 14, '700'); ctx.strokeRect(x0, 450, w, 125); drawSeries(state.gyro_mag, idxs, x0, 450, w, 125, '#7c3aed');
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0 + w, 76); ctx.lineTo(x0 + w, 585); ctx.stroke();
      const row = end;
      document.getElementById('status').textContent = state.done ? 'done' : 'running';
      document.getElementById('time').textContent = (state.sample_idx[row] / sampleRate).toFixed(2) + 's';
      document.getElementById('micro').textContent = state.online_micro_label[row] + ' ' + Number(state.online_micro_confidence[row]).toFixed(2);
      document.getElementById('macro').textContent = state.online_macro_label[row] + ' ' + Number(state.online_macro_confidence[row]).toFixed(2);
      document.getElementById('gt').textContent = state.gt_micro_label[row] + ' / ' + state.gt_macro_label[row];
    }}
    async function poll() {{
      try {{
        const res = await fetch('live_state.json?ts=' + Date.now(), {{cache: 'no-store'}});
        if (res.ok) draw(await res.json());
      }} catch (err) {{
        document.getElementById('status').textContent = 'waiting';
      }}
      setTimeout(poll, 150);
    }}
    poll();
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def _start_http_server(output_dir: Path, port: int) -> tuple[ThreadingHTTPServer, int]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _infer_metadata_from_set_dir(set_dir: Path) -> tuple[str | None, str | None]:
    action_dir = set_dir.parent
    subject_dir = action_dir.parent
    action = action_dir.name if action_dir.name else None
    subject = subject_dir.name if subject_dir.name else None
    return subject, action


def _make_sensor_time_monotonic(frames: list[pd.DataFrame], time_column: str = "sensor_ts") -> list[pd.DataFrame]:
    if not frames or time_column not in frames[0].columns:
        return frames
    out: list[pd.DataFrame] = []
    next_start: float | None = None
    fallback_dt = 0.01
    for frame in frames:
        df = frame.copy()
        ts = pd.to_numeric(df[time_column], errors="coerce")
        valid_diffs = ts.diff().dropna()
        valid_diffs = valid_diffs[valid_diffs > 0]
        dt = float(valid_diffs.median()) if not valid_diffs.empty else fallback_dt
        if not np.isfinite(dt) or dt <= 0:
            dt = fallback_dt
        if next_start is not None and not ts.empty:
            start = float(ts.iloc[0])
            if not np.isfinite(start):
                start = 0.0
            # Separate rep CSVs often restart timestamps. Shift each later rep
            # forward only when needed so the merged set is a valid stream.
            if start <= next_start:
                df[time_column] = ts + (next_start - start)
                ts = pd.to_numeric(df[time_column], errors="coerce")
        if not ts.empty:
            end = float(ts.iloc[-1])
            next_start = end + dt if np.isfinite(end) else next_start
            fallback_dt = dt
        out.append(df)
    return out


def _load_replay_input(path: Path) -> tuple[pd.DataFrame, str, str, list[str]]:
    if path.is_file():
        return pd.read_csv(path), "csv", path.stem, [path.as_posix()]
    if not path.is_dir():
        raise FileNotFoundError(f"Replay input does not exist: {path}")

    csv_paths = sorted(path.glob("rep*.csv"), key=_natural_key)
    if not csv_paths:
        csv_paths = sorted(path.glob("*.csv"), key=_natural_key)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in set directory: {path}")

    subject, action = _infer_metadata_from_set_dir(path)
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = df.copy()
        if subject is not None and "subject_id" not in df.columns:
            df["subject_id"] = subject
        if action is not None and "action_type" not in df.columns:
            df["action_type"] = action
        df["_source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise ValueError(f"All CSV files in set directory are empty: {path}")
    frames = _make_sensor_time_monotonic(frames)
    return pd.concat(frames, ignore_index=True), "set_dir", path.name, [p.as_posix() for p in csv_paths]


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    run_dir = _resolve_run_dir(Path(args.run_dir))
    config_path = Path(args.config) if args.config else run_dir / "metadata" / "config_snapshot.yaml"
    print(f"[INFO] loading config={config_path}", flush=True)
    raw = _load_config(config_path)
    device = _resolve_device(str(args.device or (raw.get("train", {}) or {}).get("device", "auto")))
    print(f"[INFO] loading model run_dir={run_dir} device={device}", flush=True)
    model, macro_classes, _micro_classes, imu_columns = _load_model(run_dir, device)

    input_path = Path(args.csv)
    print(f"[INFO] loading replay input={input_path}", flush=True)
    df, input_kind, default_stream_id, source_files = _load_replay_input(input_path)
    print(f"[INFO] input_kind={input_kind} source_files={len(source_files)} raw_samples={len(df)}", flush=True)
    _require_columns(df, imu_columns, "Replay input")
    if "action_type" in df.columns:
        input_actions = sorted(set(df["action_type"].astype(str)) - {OTHER_LABEL})
        missing_actions = [action for action in input_actions if action not in macro_classes]
        if missing_actions:
            print(
                "[WARN] replay input contains action labels not present in this checkpoint: "
                f"{missing_actions}. Known macro classes={macro_classes}",
                flush=True,
            )
    stats_path = Path(args.stats) if args.stats else run_dir / "metadata" / "zscore_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Z-score stats file not found: {stats_path}")
    stats = ZScoreStats.load(stats_path)
    df = apply_zscore(df, imu_columns, stats)
    max_samples = int(args.max_samples)
    if max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)
    stream_id = args.stream_id or default_stream_id
    fallback_sample_rate = float((raw.get("window", {}) or {}).get("sample_rate_hz", 50.0))
    sample_rate = float(args.sample_rate) if args.sample_rate is not None else infer_sample_rate_hz(df, default=fallback_sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate_hz={sample_rate}")
    print(f"[INFO] stream_id={stream_id} samples={len(df)} sample_rate_hz={sample_rate:.2f}", flush=True)

    predictor = OnlineDSMSTCNPredictor(model, imu_columns, device, buffer_size=args.buffer_size)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "streaming_eval" / stream_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_kind == "set_dir":
        df.to_csv(output_dir / "merged_set_input.csv", index=False)
        print(f"[INFO] wrote merged set input={output_dir / 'merged_set_input.csv'}", flush=True)

    gt_micro_labels = micro_labels_from_phase(df["phase"].to_numpy()) if "phase" in df.columns else np.asarray([OTHER_LABEL] * len(df), dtype=object)
    gt_macro_labels = (
        macro_labels_from_action(df["action_type"].astype(str).to_numpy(), gt_micro_labels)
        if "action_type" in df.columns
        else np.asarray([OTHER_LABEL] * len(df), dtype=object)
    )

    if args.live:
        live_html_path = output_dir / "live_dashboard.html"
        live_state_path = output_dir / "live_state.json"
        window_samples = max(10, int(float(args.live_window_seconds) * sample_rate))
        _write_live_dashboard_html(live_html_path, stream_id, sample_rate, float(args.live_window_seconds))
        server = None
        if int(args.live_port) >= 0:
            try:
                server, port = _start_http_server(output_dir, int(args.live_port))
            except OSError:
                if int(args.live_port) == 0:
                    raise
                print(f"[WARN] port {args.live_port} is busy; falling back to a free port", flush=True)
                server, port = _start_http_server(output_dir, 0)
            print(f"[OK] live dashboard=http://127.0.0.1:{port}/live_dashboard.html", flush=True)
        else:
            print(
                "[WARN] --live-port < 0 writes files only; most browsers block file:// polling. "
                "Use the default HTTP server for the live dashboard.",
                flush=True,
            )
            print(f"[OK] live dashboard file={live_html_path}", flush=True)

        samples = df[list(imu_columns)].to_numpy(dtype=np.float32)
        acc_mag = _motion_magnitude(df, ("ax", "ay", "az"))
        gyro_mag = _motion_magnitude(df, ("gx", "gy", "gz"))
        state: Dict[str, object] = {
            "stream_id": stream_id,
            "sample_rate_hz": float(sample_rate),
            "window_samples": int(window_samples),
            "done": False,
            "sample_idx": [],
            "acc_mag": [],
            "gyro_mag": [],
            "online_micro_label": [],
            "online_micro_confidence": [],
            "online_macro_label": [],
            "online_macro_confidence": [],
            "gt_micro_label": [],
            "gt_macro_label": [],
        }
        live_history_samples = int(round(float(args.live_history_seconds) * sample_rate)) if float(args.live_history_seconds) > 0 else 0
        if live_history_samples > 0:
            live_history_samples = max(window_samples, live_history_samples)
        _write_json_atomic(live_state_path, state)
        rows = []
        update_interval = max(1, int(args.live_update_interval))
        replay_speed = float(args.replay_speed)
        live_start = time.perf_counter()
        print(
            f"[INFO] live step inference samples={len(samples)} replay_speed={replay_speed}x "
            f"update_interval={update_interval}",
            flush=True,
        )
        try:
            for idx, sample in enumerate(samples):
                out = predictor.update(sample)
                micro_probs = out["micro_probs"].numpy()
                macro_probs = out["macro4_probs"].numpy()
                micro_idx = int(np.argmax(micro_probs))
                macro_idx = int(np.argmax(macro_probs))
                row = {
                    "sample_idx": int(idx),
                    "online_micro_label": MICRO_LABELS[micro_idx],
                    "online_micro_confidence": float(micro_probs[micro_idx]),
                    "online_macro_label": macro_classes[macro_idx],
                    "online_macro_confidence": float(macro_probs[macro_idx]),
                    "gt_micro_label": str(gt_micro_labels[idx]),
                    "gt_macro_label": str(gt_macro_labels[idx]),
                }
                rows.append(row)
                state["sample_idx"].append(int(idx))
                state["acc_mag"].append(float(acc_mag[idx]))
                state["gyro_mag"].append(float(gyro_mag[idx]))
                state["online_micro_label"].append(row["online_micro_label"])
                state["online_micro_confidence"].append(row["online_micro_confidence"])
                state["online_macro_label"].append(row["online_macro_label"])
                state["online_macro_confidence"].append(row["online_macro_confidence"])
                state["gt_micro_label"].append(row["gt_micro_label"])
                state["gt_macro_label"].append(row["gt_macro_label"])
                _trim_live_state(state, live_history_samples)
                if idx == 0 or idx == len(samples) - 1 or (idx + 1) % update_interval == 0:
                    _write_json_atomic(live_state_path, state)
                    elapsed = time.perf_counter() - t0
                    print(f"[INFO] live inference {idx + 1}/{len(samples)} elapsed={elapsed:.1f}s", flush=True)
                if replay_speed > 0:
                    target_elapsed = idx / max(1e-9, sample_rate * replay_speed)
                    wait_s = target_elapsed - (time.perf_counter() - live_start)
                    if wait_s > 0:
                        time.sleep(wait_s)
            state["done"] = True
            _write_json_atomic(live_state_path, state)
        finally:
            if server is not None and args.shutdown_server_on_done:
                server.shutdown()

        rows_df = pd.DataFrame(rows)
        csv_path = output_dir / "streaming_predictions.csv"
        rows_df.to_csv(csv_path, index=False)
        elapsed_seconds = float(time.perf_counter() - t0)
        summary = {
            "run_dir": run_dir.as_posix(),
            "input_path": input_path.as_posix(),
            "input_kind": input_kind,
            "source_files": source_files,
            "stream_id": stream_id,
            "samples": int(len(df)),
            "sample_rate_hz": float(sample_rate),
            "buffer_size": int(predictor.buffer_size),
            "buffer_seconds": float(predictor.buffer_size / sample_rate) if sample_rate > 0 else None,
            "inference_method": "live_step",
            "elapsed_seconds": elapsed_seconds,
            "streaming_predictions_csv": csv_path.as_posix(),
            "live_dashboard_html": live_html_path.as_posix(),
            "live_state_json": live_state_path.as_posix(),
            "live_history_seconds": float(args.live_history_seconds),
        }
        summary.update(_prediction_summary(rows_df, elapsed_seconds, sample_rate))
        (output_dir / "streaming_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] wrote live dashboard={live_html_path}", flush=True)
        print(json.dumps(summary, indent=2))
        if server is not None and args.keep_server_open:
            print("[INFO] live server is still open. Press Ctrl-C to stop.", flush=True)
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                server.shutdown()
        return

    method = str(args.method).lower()
    if method == "fast":
        print(
            "[INFO] inference method=fast causal_full_sequence "
            "(online-equivalent for causal checkpoints)",
            flush=True,
        )
        x = torch.from_numpy(df[list(imu_columns)].to_numpy(dtype=np.float32))[None, :, :].to(device)
        with torch.inference_mode():
            out = model(x)
        micro_probs_np = out["micro_probs"].detach().cpu().numpy()[0]
        # Use Stage 2 output (more accurate than Stage 4 for this checkpoint)
        macro_probs_np = torch.softmax(out["macro2_logits"], dim=-1).detach().cpu().numpy()[0]
    elif method == "step":
        print(
            f"[INFO] inference method=step buffer_size={predictor.buffer_size} "
            f"progress_interval={args.progress_interval}",
            flush=True,
        )
        micro_probs, macro_probs = [], []
        samples = df[imu_columns].to_numpy(dtype=np.float32)
        progress_interval = max(1, int(args.progress_interval))
        for idx, sample in enumerate(samples, start=1):
            out = predictor.update(sample)
            micro_probs.append(out["micro_probs"].numpy())
            macro_probs.append(out["macro4_probs"].numpy())
            if idx == 1 or idx == len(samples) or idx % progress_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"[INFO] step inference {idx}/{len(samples)} elapsed={elapsed:.1f}s", flush=True)
        micro_probs_np = np.stack(micro_probs, axis=0)
        macro_probs_np = np.stack(macro_probs, axis=0)
    else:
        raise ValueError(f"Unknown inference method: {args.method}")
    online_micro_labels = [MICRO_LABELS[int(i)] for i in np.argmax(micro_probs_np, axis=1)]
    online_macro_labels = [macro_classes[int(i)] for i in np.argmax(macro_probs_np, axis=1)]
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
    elapsed_seconds = float(time.perf_counter() - t0)
    summary = {
        "run_dir": run_dir.as_posix(),
        "input_path": input_path.as_posix(),
        "input_kind": input_kind,
        "source_files": source_files,
        "stream_id": stream_id,
        "samples": int(len(df)),
        "sample_rate_hz": float(sample_rate),
        "buffer_size": int(predictor.buffer_size),
        "buffer_seconds": float(predictor.buffer_size / sample_rate) if sample_rate > 0 else None,
        "inference_method": method,
        "elapsed_seconds": elapsed_seconds,
        "streaming_predictions_csv": csv_path.as_posix(),
        "streaming_replay_svg": svg_path.as_posix(),
        "streaming_replay_html": html_path.as_posix(),
    }
    summary.update(_prediction_summary(rows, elapsed_seconds, sample_rate))
    (output_dir / "streaming_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote replay html={html_path}", flush=True)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming DS-MS-TCN evaluation and replay")
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained tcn run directory")
    parser.add_argument("--csv", type=Path, required=True, help="Raw CSV or set directory to replay")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Override inferred sample rate in Hz; useful for board streams without reliable sensor_ts",
    )
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=["fast", "step"],
        default="fast",
        help="fast runs the full causal sequence once; step replays one sample at a time",
    )
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run true sample-by-sample inference and update a browser dashboard while data arrives",
    )
    parser.add_argument("--live-port", type=int, default=8765, help="HTTP port for --live; use 0 for any free port")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="CSV replay speed multiplier for --live; 1.0 means real time")
    parser.add_argument("--live-window-seconds", type=float, default=15.0)
    parser.add_argument(
        "--live-history-seconds",
        type=float,
        default=60.0,
        help="Seconds of recent samples kept in live_state.json; <=0 keeps the full replay",
    )
    parser.add_argument("--live-update-interval", type=int, default=5)
    parser.add_argument(
        "--keep-server-open",
        action="store_true",
        help="Keep the live HTTP server open after replay finishes until Ctrl-C",
    )
    parser.add_argument(
        "--shutdown-server-on-done",
        action="store_true",
        help="Explicitly shut down the live HTTP server when replay completes",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
