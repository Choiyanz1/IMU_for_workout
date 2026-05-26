from __future__ import annotations

import argparse
import fnmatch
import html
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.window_pipeline import apply_zscore, compute_train_stats, resample_sequence


IMU_DEFAULT = ["ax", "ay", "az", "gx", "gy", "gz"]
PHASE_COLORS = {
    "eccentric": "#4C9BE8",
    "concentric": "#E8854C",
    "inter_set_rest": "#AAAAAA",
    "none": "#DDDDDD",
}


@dataclass
class FileCheck:
    title: str
    relative_path: str
    sample_count: int
    inferred_hz_raw: float
    target_hz: int
    sensor_ts_monotonic: bool
    sensor_ts_duplicates: int
    missing_counts: dict[str, int]
    phase_counts: dict[str, int]
    action_counts: dict[str, int]
    raw_plot: str
    resampled_plot: str
    zscore_plot: str


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _timestamp_dir(base_dir: Path) -> Path:
    return base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_numeric_or_datetime(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=np.float64)
    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        raise ValueError("time column could not be parsed as numeric or datetime")
    return (dt.astype("int64") / 1e9).to_numpy(dtype=np.float64)


def _infer_hz_from_series(series: pd.Series) -> float:
    values = _ensure_numeric_or_datetime(series)
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median = float(np.median(diffs))
    rates = []
    for divisor in (1.0, 1e3, 1e6):
        rate = divisor / median
        if 1.0 <= rate <= 2000.0:
            rates.append(rate)
    return float(rates[0] if rates else 0.0)


def _rle(values: Sequence[str]) -> list[tuple[str, int, int]]:
    if not values:
        return []
    runs: list[tuple[str, int, int]] = []
    current = str(values[0])
    start = 0
    for i in range(1, len(values)):
        value = str(values[i])
        if value != current:
            runs.append((current, start, i))
            current = value
            start = i
    runs.append((current, start, len(values)))
    return runs


def _plot_signal_panels(df: pd.DataFrame, imu_cols: Sequence[str], title: str, sample_rate_hz: float, save_path: Path) -> None:
    accel_cols = [c for c in imu_cols if c.startswith("a")]
    gyro_cols = [c for c in imu_cols if c.startswith("g")]
    n = len(df)
    hz = sample_rate_hz if sample_rate_hz > 0 else 100.0
    t = np.arange(n, dtype=np.float64) / hz
    phases = df["phase"].astype(str).tolist() if "phase" in df.columns else []
    runs = _rle(phases)

    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    for ax, cols, ylabel in ((ax_a, accel_cols, "Accel"), (ax_g, gyro_cols, "Gyro")):
        if runs:
            for phase, start, end in runs:
                color = PHASE_COLORS.get(phase, "#EEEEEE")
                ax.axvspan(t[start], t[max(start, end - 1)], alpha=0.14, color=color, linewidth=0)
        for col in cols:
            if col in df.columns:
                ax.plot(t, df[col].to_numpy(dtype=np.float64), linewidth=0.8, label=col)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8, ncol=max(1, len(cols)))
    ax_g.set_xlabel("Time (s)")
    ax_a.set_title(title, fontsize=12, fontweight="bold")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _matches_any(path: Path, base_dir: Path, patterns: Sequence[str]) -> bool:
    try:
        parts = path.relative_to(base_dir).parts
    except ValueError:
        parts = path.parts
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in patterns)


def _discover_training_frames(
    data_dir: Path,
    include_actions: Sequence[str],
    exclude_patterns: Sequence[str],
    imu_cols: Sequence[str],
    subject_column: str,
    test_subject: str | None,
) -> list[pd.DataFrame]:
    include_set = set(include_actions)
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(data_dir.rglob("*.csv")):
        if _matches_any(csv_path, data_dir, exclude_patterns):
            continue
        rel = csv_path.relative_to(data_dir)
        if len(rel.parts) < 2:
            continue
        action = rel.parts[1]
        if include_set and action not in include_set:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if not all(col in df.columns for col in imu_cols):
            continue
        if subject_column not in df.columns:
            subject = rel.parts[0]
            df = df.copy()
            df[subject_column] = subject
        subject_value = str(df.iloc[0][subject_column]) if len(df) else rel.parts[0]
        if test_subject is not None and subject_value == test_subject:
            continue
        frames.append(df)
    return frames


def _select_files(data_dir: Path, csvs: Sequence[str], subject: str | None, action: str | None, limit: int) -> list[Path]:
    if csvs:
        return [Path(p) if Path(p).is_absolute() else (data_dir / p) for p in csvs]
    pattern = data_dir
    if subject:
        pattern = pattern / subject
    if action:
        pattern = pattern / action
    files = sorted(pattern.rglob("*.csv"))
    filtered = [p for p in files if "rest" not in p.as_posix().lower() and "whole_session" not in p.name.lower()]
    return filtered[:limit] if limit > 0 else filtered


def _build_check(
    csv_path: Path,
    data_dir: Path,
    imu_cols: Sequence[str],
    time_col: str,
    target_hz: int,
    zscore_stats,
    plots_dir: Path,
) -> FileCheck:
    df = pd.read_csv(csv_path)
    raw_hz = _infer_hz_from_series(df[time_col]) if time_col in df.columns else 0.0
    monotonic = bool(df[time_col].is_monotonic_increasing) if time_col in df.columns else False
    dupes = int(df[time_col].duplicated().sum()) if time_col in df.columns else 0
    missing = {col: int(df[col].isna().sum()) for col in df.columns if int(df[col].isna().sum()) > 0}
    phase_counts = df["phase"].astype(str).value_counts().to_dict() if "phase" in df.columns else {}
    action_counts = df["action_type"].astype(str).value_counts().to_dict() if "action_type" in df.columns else {}

    rel = csv_path.relative_to(data_dir).as_posix() if csv_path.is_relative_to(data_dir) else csv_path.as_posix()
    safe_name = rel.replace("/", "__").replace(":", "_")

    raw_plot = plots_dir / f"{safe_name}__raw.png"
    resampled_plot = plots_dir / f"{safe_name}__resampled.png"
    zscore_plot = plots_dir / f"{safe_name}__zscore.png"

    _plot_signal_panels(df, imu_cols, f"Raw: {rel}", raw_hz, raw_plot)
    resampled_df = resample_sequence(df, imu_cols, time_col, target_hz) if time_col in df.columns else df.copy()
    _plot_signal_panels(resampled_df, imu_cols, f"Resampled: {rel}", float(target_hz), resampled_plot)
    z_df = apply_zscore(resampled_df, imu_cols, zscore_stats)
    _plot_signal_panels(z_df, imu_cols, f"Z-score: {rel}", float(target_hz), zscore_plot)

    return FileCheck(
        title=csv_path.stem,
        relative_path=rel,
        sample_count=int(len(df)),
        inferred_hz_raw=float(raw_hz),
        target_hz=int(target_hz),
        sensor_ts_monotonic=monotonic,
        sensor_ts_duplicates=dupes,
        missing_counts=missing,
        phase_counts={str(k): int(v) for k, v in phase_counts.items()},
        action_counts={str(k): int(v) for k, v in action_counts.items()},
        raw_plot=raw_plot.name,
        resampled_plot=resampled_plot.name,
        zscore_plot=zscore_plot.name,
    )


def _render_counts_table(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"<p><strong>{html.escape(title)}:</strong> none</p>"
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{v}</td></tr>"
        for k, v in counts.items()
    )
    return (
        f"<details open><summary><strong>{html.escape(title)}</strong></summary>"
        f"<table><tr><th>Label</th><th>Count</th></tr>{rows}</table></details>"
    )


def _render_file_card(check: FileCheck) -> str:
    missing_json = html.escape(json.dumps(check.missing_counts, ensure_ascii=False))
    return f"""
<section class="card">
  <h2>{html.escape(check.relative_path)}</h2>
  <div class="meta-grid">
    <div><strong>samples</strong><br>{check.sample_count}</div>
    <div><strong>raw Hz</strong><br>{check.inferred_hz_raw:.2f}</div>
    <div><strong>target Hz</strong><br>{check.target_hz}</div>
    <div><strong>sensor_ts monotonic</strong><br>{check.sensor_ts_monotonic}</div>
    <div><strong>sensor_ts duplicates</strong><br>{check.sensor_ts_duplicates}</div>
    <div><strong>missing counts</strong><br><code>{missing_json}</code></div>
  </div>
  {_render_counts_table('phase counts', check.phase_counts)}
  {_render_counts_table('action counts', check.action_counts)}
  <div class="plots">
    <figure><img src="plots/{html.escape(check.raw_plot)}" alt="raw plot"><figcaption>Raw</figcaption></figure>
    <figure><img src="plots/{html.escape(check.resampled_plot)}" alt="resampled plot"><figcaption>Resampled</figcaption></figure>
    <figure><img src="plots/{html.escape(check.zscore_plot)}" alt="zscore plot"><figcaption>Z-score</figcaption></figure>
  </div>
</section>
"""


def _write_index(out_dir: Path, checks: Sequence[FileCheck], config_path: Path) -> None:
    cards = "\n".join(_render_file_card(c) for c in checks)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Data Inspection Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
    h1 {{ margin-bottom: 8px; }}
    .subtitle {{ color: #555; margin-bottom: 24px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 12px 0 16px; }}
    .plots {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; border: 1px solid #ccc; border-radius: 8px; background: #fff; }}
    figcaption {{ margin-top: 6px; font-size: 13px; color: #555; }}
    table {{ border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>Data Inspection Dashboard</h1>
  <p class="subtitle">Config: <code>{html.escape(config_path.as_posix())}</code></p>
  <p>Use this page to visually inspect raw source CSVs, resampled signals, and z-score normalized signals side by side.</p>
  {cards}
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a plot-based dashboard for raw and processed data checks.")
    parser.add_argument("--config", type=Path, default=Path("configs/micro_macro_recognition_stage3_40ep.yaml"))
    parser.add_argument("--csv", action="append", default=[], help="Relative-to-data-dir or absolute CSV path. Can be passed multiple times.")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--action", type=str, default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data_inspection"))
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    cfg = _read_yaml(args.config)
    data_cfg = cfg.get("data", {}) or {}
    feature_cfg = cfg.get("feature", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    window_cfg = cfg.get("window", {}) or {}

    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    include_actions = list(data_cfg.get("include_actions", []) or [])
    exclude_patterns = list(data_cfg.get("exclude_patterns", []) or [])
    imu_cols = list(feature_cfg.get("imu_columns", IMU_DEFAULT))
    time_col = str(feature_cfg.get("time_column", "sensor_ts"))
    subject_col = str(feature_cfg.get("subject_column", "subject_id"))
    test_subject = train_cfg.get("test_subject")
    target_hz = int(window_cfg.get("sample_rate_hz", 100))

    out_dir = args.out_dir if args.no_timestamp else _timestamp_dir(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_frames = _discover_training_frames(data_dir, include_actions, exclude_patterns, imu_cols, subject_col, test_subject)
    if not train_frames:
        raise RuntimeError("No training frames discovered for z-score stats.")
    stats = compute_train_stats(train_frames, imu_cols)

    selected = _select_files(data_dir, args.csv, args.subject, args.action, args.limit)
    if not selected:
        raise RuntimeError("No CSV files selected for inspection.")

    checks = [
        _build_check(csv_path, data_dir, imu_cols, time_col, target_hz, stats, plots_dir)
        for csv_path in selected
    ]
    _write_index(out_dir, checks, args.config)
    print(f"[OK] Wrote dashboard to {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
