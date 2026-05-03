"""Evaluate single-exemplar SDTW rep segmentation.

This script focuses on active repetition boundaries only:
phase in {eccentric, concentric}.  It intentionally does not evaluate the
eccentric/concentric transition yet; that comes after rep boundaries are stable.

Usage:
    python -m evaluation.rep_segmentation --config config.yaml
    python -m evaluation.rep_segmentation --config config.yaml --mode sets
    python -m evaluation.rep_segmentation --config config.yaml --mode whole
"""
from __future__ import annotations

import argparse
import fnmatch
import html
import json
import re
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - fallback for lightweight runtimes
    yaml = None

from preprocessing.sdtw_rep_segmentation import (
    SDTWConfig,
    SegmentDetection,
    active_segments_from_phase,
    detect_reps_sdtw_templates,
    fit_sdtw_templates,
    infer_sample_rate_hz,
    summarize_detection_metrics,
)


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _load_config(config_path: Path) -> Dict:
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)

    root: Dict = {}
    stack: List[Tuple[int, Dict]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _natural_sort_key(path: Path) -> List[int | str]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(p) if p.isdigit() else p for p in parts]


def _matches_any_path_part(path: Path, base_dir: Path, patterns: Sequence[str]) -> bool:
    try:
        parts = path.relative_to(base_dir).parts
    except ValueError:
        parts = path.parts
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in patterns)


def _subject_dirs(data_dir: Path) -> List[Path]:
    return [p for p in sorted(data_dir.iterdir()) if p.is_dir()]


def _load_rep_csvs(
    data_dir: Path,
    subject: str,
    action: str,
    exclude_patterns: Sequence[str],
    sdtw_cfg: SDTWConfig,
) -> List[pd.DataFrame]:
    action_dir = data_dir / subject / action
    if not action_dir.exists():
        return []
    frames: List[pd.DataFrame] = []
    for set_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("set")):
        if _matches_any_path_part(set_dir, data_dir, exclude_patterns):
            continue
        csv_paths = [
            p for p in sorted(set_dir.glob("*.csv"), key=_natural_sort_key)
            if not _matches_any_path_part(p, data_dir, exclude_patterns)
            and "whole_session" not in p.name
        ]
        n_reps = len(csv_paths)
        if n_reps == 0:
            continue

        middle_fraction = min(max(float(sdtw_cfg.template_middle_fraction), 0.0), 1.0)
        side_fraction = (1.0 - middle_fraction) / 2.0
        start_rank = int(np.floor(n_reps * side_fraction))
        end_rank = int(np.ceil(n_reps * (1.0 - side_fraction)))
        edge = int(sdtw_cfg.template_min_edge_reps)
        if n_reps >= edge * 2 + 1:
            start_rank = max(start_rank, edge)
            end_rank = min(end_rank, n_reps - edge)
        if start_rank >= end_rank:
            center = n_reps // 2
            start_rank = center
            end_rank = center + 1

        for rep_rank, csv_path in enumerate(csv_paths):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "phase" not in df.columns:
                continue
            rep_position = rep_rank / max(1, n_reps - 1)
            is_template_candidate = start_rank <= rep_rank < end_rank
            df = df.copy()
            df["_source_path"] = str(csv_path.relative_to(data_dir))
            df["_subject_id"] = subject
            df["_action_type"] = action
            df["_set_name"] = set_dir.name
            df["_rep_rank_in_set"] = rep_rank
            df["_rep_count_in_set"] = n_reps
            df["_rep_position"] = rep_position
            df["_template_candidate"] = is_template_candidate
            df.attrs["source_path"] = str(csv_path.relative_to(data_dir))
            df.attrs["source_id"] = f"{subject}/{action}/{set_dir.name}/{csv_path.stem}"
            df.attrs["rep_position"] = rep_position
            df.attrs["template_candidate"] = is_template_candidate
            frames.append(df)
    return frames


def _load_set_streams(
    data_dir: Path,
    subject: str,
    action: str,
    exclude_patterns: Sequence[str],
) -> List[Tuple[str, pd.DataFrame]]:
    action_dir = data_dir / subject / action
    if not action_dir.exists():
        return []
    streams: List[Tuple[str, pd.DataFrame]] = []
    for set_dir in sorted(action_dir.iterdir()):
        if not set_dir.is_dir() or not set_dir.name.startswith("set"):
            continue
        if _matches_any_path_part(set_dir, data_dir, exclude_patterns):
            continue
        csvs = sorted(set_dir.glob("*.csv"), key=_natural_sort_key)
        frames: List[pd.DataFrame] = []
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "phase" not in df.columns:
                continue
            df = df.copy()
            df["_source_file"] = csv_path.name
            frames.append(df)
        if frames:
            streams.append((f"{subject}/{action}/{set_dir.name}", pd.concat(frames, ignore_index=True)))
    return streams


def _load_whole_streams(
    data_dir: Path,
    subject: str,
    action: str,
) -> List[Tuple[str, pd.DataFrame]]:
    whole_files = sorted((data_dir / subject).glob("*whole_session*.csv"))
    streams: List[Tuple[str, pd.DataFrame]] = []
    for whole_file in whole_files:
        try:
            df = pd.read_csv(whole_file)
        except Exception:
            continue
        required = {"action_type", "phase"}
        if not required.issubset(df.columns):
            continue

        # Known-action simulation: evaluate each contiguous block where the
        # session says the current action is this action.
        mask = df["action_type"].astype(str).eq(action).to_numpy()
        start = None
        block_idx = 0
        for idx, keep in enumerate(mask):
            if keep and start is None:
                start = idx
            elif not keep and start is not None:
                chunk = df.iloc[start:idx].copy().reset_index(drop=True)
                if len(chunk) > 0:
                    streams.append((f"{subject}/{whole_file.stem}/{action}/block{block_idx}", chunk))
                    block_idx += 1
                start = None
        if start is not None:
            chunk = df.iloc[start:].copy().reset_index(drop=True)
            streams.append((f"{subject}/{whole_file.stem}/{action}/block{block_idx}", chunk))
    return streams


def _timestamped_dir(base_dir: Path, use_timestamp: bool, label: str) -> Path:
    if not use_timestamp:
        return base_dir
    return base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"


def _timestamped_run_dir(base_dir: Path, use_timestamp: bool) -> Path:
    if not use_timestamp:
        return base_dir
    return base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "stream"


def _prepare_output_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "metrics": out_dir / "metrics",
        "detections": out_dir / "detections",
        "templates": out_dir / "templates",
        "plots": out_dir / "plots",
        "metadata": out_dir / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _aggregate_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    total_tp = sum(row["tp"] for row in rows)
    total_fp = sum(row["fp"] for row in rows)
    total_fn = sum(row["fn"] for row in rows)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    out = {
        "streams": float(len(rows)),
        "n_pred": float(sum(row["n_pred"] for row in rows)),
        "n_true": float(sum(row["n_true"] for row in rows)),
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    for key in ("mean_iou", "start_mae_ms", "end_mae_ms", "duration_mae_ms"):
        values = [row[key] for row in rows if np.isfinite(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def _detection_rows(stream_id: str, detections: Sequence[SegmentDetection]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, det in enumerate(detections):
        rows.append(
            {
                "stream_id": stream_id,
                "det_idx": idx,
                "action_type": det.action_type,
                "start_idx": det.start_idx,
                "end_idx": det.end_idx,
                "cost": det.cost,
                "normalized_cost": det.normalized_cost,
                "feature": det.feature,
                "template_id": det.template_id,
                "exemplar_source": det.exemplar_source,
            }
        )
    return rows


def _magnitude(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    available = [col for col in cols if col in df.columns]
    if not available:
        return np.zeros(len(df), dtype=np.float64)
    values = df.loc[:, available].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(values ** 2, axis=1))


def _polyline_points(values: np.ndarray, x0: float, y0: float, width: float, height: float, max_points: int = 1600) -> str:
    if len(values) == 0:
        return ""
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[idx]
        xs = np.linspace(x0, x0 + width, len(values))
    else:
        xs = x0 + np.arange(len(values)) / max(1, len(values) - 1) * width

    lo = float(np.nanpercentile(values, 2))
    hi = float(np.nanpercentile(values, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        lo = float(np.nanmin(values)) if len(values) else 0.0
        hi = float(np.nanmax(values)) if len(values) else 1.0
    if hi - lo < 1e-8:
        hi = lo + 1.0
    clipped = np.clip(values, lo, hi)
    ys = y0 + height - ((clipped - lo) / (hi - lo) * height)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _segment_rects(
    segments: Sequence[Tuple[int, int]],
    n_samples: int,
    x0: float,
    width: float,
    y0: float,
    height: float,
    fill: str,
    stroke: str,
    opacity: float,
) -> str:
    rects: List[str] = []
    for start, end in segments:
        x = x0 + start / max(1, n_samples) * width
        w = max(1.0, (end - start) / max(1, n_samples) * width)
        rects.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{height:.1f}" '
            f'fill="{fill}" fill-opacity="{opacity:.3f}" stroke="{stroke}" stroke-width="1"/>'
        )
    return "\n".join(rects)


def _lane_rects(
    segments: Sequence[Tuple[int, int]],
    n_samples: int,
    x0: float,
    width: float,
    y0: float,
    height: float,
    fill: str,
    stroke: str,
) -> str:
    rects: List[str] = []
    for start, end in segments:
        x = x0 + start / max(1, n_samples) * width
        w = max(1.5, (end - start) / max(1, n_samples) * width)
        rects.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{height:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="2"/>'
        )
    return "\n".join(rects)


def _boundary_lines(
    segments: Sequence[Tuple[int, int]],
    n_samples: int,
    x0: float,
    width: float,
    y0: float,
    height: float,
    color: str,
    dash: str,
    opacity: float = 0.9,
) -> str:
    lines: List[str] = []
    for start, end in segments:
        for idx in (start, end):
            x = x0 + idx / max(1, n_samples) * width
            lines.append(
                f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y0 + height:.1f}" '
                f'stroke="{color}" stroke-width="1.4" stroke-opacity="{opacity:.2f}" stroke-dasharray="{dash}"/>'
            )
    return "\n".join(lines)


def _write_segmentation_svg(
    path: Path,
    stream_id: str,
    stream_df: pd.DataFrame,
    truth: Sequence[Tuple[int, int]],
    detections: Sequence[SegmentDetection],
    metrics: Dict[str, float],
    sample_rate_hz: float,
) -> None:
    width = 1280
    height = 680
    margin_left = 70
    plot_width = 1140
    panel_h = 190
    lane_y = 112
    gt_lane_y = lane_y + 24
    pred_lane_y = lane_y + 52
    lane_h = 16
    acc_y = 210
    gyro_y = 450
    n = len(stream_df)
    pred_segments = [(det.start_idx, det.end_idx) for det in detections]

    acc_mag = _magnitude(stream_df, ["ax", "ay", "az"])
    gyro_mag = _magnitude(stream_df, ["gx", "gy", "gz"])
    acc_points = _polyline_points(acc_mag, margin_left, acc_y, plot_width, panel_h)
    gyro_points = _polyline_points(gyro_mag, margin_left, gyro_y, plot_width, panel_h)

    gt_lane = _lane_rects(truth, n, margin_left, plot_width, gt_lane_y, lane_h, "#2da44e", "#166534")
    pred_lane = _lane_rects(pred_segments, n, margin_left, plot_width, pred_lane_y, lane_h, "#fb8500", "#c2410c")
    gt_acc_lines = _boundary_lines(truth, n, margin_left, plot_width, acc_y, panel_h, "#15803d", "5 4", 0.75)
    pred_acc_lines = _boundary_lines(pred_segments, n, margin_left, plot_width, acc_y, panel_h, "#ea580c", "none", 0.65)
    gt_gyro_lines = _boundary_lines(truth, n, margin_left, plot_width, gyro_y, panel_h, "#15803d", "5 4", 0.75)
    pred_gyro_lines = _boundary_lines(pred_segments, n, margin_left, plot_width, gyro_y, panel_h, "#ea580c", "none", 0.65)

    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0
    title = html.escape(stream_id)
    subtitle = (
        f"F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"GT={int(metrics['n_true'])}  Pred={int(metrics['n_pred'])}  Duration={duration_s:.1f}s"
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1; }}
    .label {{ font-size: 14px; font-weight: 600; }}
    .small {{ font-size: 12px; fill: #475569; }}
  </style>
  <text x="40" y="36" font-size="20" font-weight="700">{title}</text>
  <text x="40" y="62" class="small">{html.escape(subtitle)}</text>
  <rect x="40" y="78" width="16" height="10" fill="#2da44e" stroke="#166534"/>
  <text x="62" y="88" class="small">GT rep lane / dashed boundaries</text>
  <rect x="260" y="78" width="16" height="10" fill="#fb8500" stroke="#c2410c"/>
  <text x="282" y="88" class="small">SDTW rep lane / solid boundaries</text>

  <text x="40" y="{gt_lane_y + 12}" class="label">GT</text>
  <rect x="{margin_left}" y="{gt_lane_y}" width="{plot_width}" height="{lane_h}" fill="#ecfdf5" stroke="#bbf7d0"/>
  {gt_lane}
  <text x="40" y="{pred_lane_y + 12}" class="label">Pred</text>
  <rect x="{margin_left}" y="{pred_lane_y}" width="{plot_width}" height="{lane_h}" fill="#fff7ed" stroke="#fed7aa"/>
  {pred_lane}

  <text x="40" y="{acc_y + 18}" class="label">acc_mag</text>
  <rect x="{margin_left}" y="{acc_y}" width="{plot_width}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1"/>
  {gt_acc_lines}
  {pred_acc_lines}
  <polyline points="{acc_points}" fill="none" stroke="#2563eb" stroke-width="1.4"/>
  <line x1="{margin_left}" y1="{acc_y + panel_h}" x2="{margin_left + plot_width}" y2="{acc_y + panel_h}" class="axis"/>

  <text x="40" y="{gyro_y + 18}" class="label">gyro_mag</text>
  <rect x="{margin_left}" y="{gyro_y}" width="{plot_width}" height="{panel_h}" fill="#f8fafc" stroke="#cbd5e1"/>
  {gt_gyro_lines}
  {pred_gyro_lines}
  <polyline points="{gyro_points}" fill="none" stroke="#7c3aed" stroke-width="1.4"/>
  <line x1="{margin_left}" y1="{gyro_y + panel_h}" x2="{margin_left + plot_width}" y2="{gyro_y + panel_h}" class="axis"/>
  <text x="{margin_left}" y="{height - 24}" class="small">0.0s</text>
  <text x="{margin_left + plot_width - 80}" y="{height - 24}" class="small">{duration_s:.1f}s</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _truth_segments_for_stream(stream_df: pd.DataFrame) -> List[Tuple[int, int]]:
    if "_source_file" not in stream_df.columns:
        if {"set", "rep", "phase"}.issubset(stream_df.columns):
            phases = stream_df["phase"].astype(str).to_numpy()
            sets = stream_df["set"].to_numpy()
            reps = stream_df["rep"].to_numpy()
            segments: List[Tuple[int, int]] = []
            start = None
            cur_key = None
            for idx, phase in enumerate(phases):
                is_active = phase in {"eccentric", "concentric"}
                key = (sets[idx], reps[idx])
                if is_active and (start is None or key != cur_key):
                    if start is not None:
                        segments.append((start, idx))
                    start = idx
                    cur_key = key
                elif (not is_active or key != cur_key) and start is not None:
                    segments.append((start, idx))
                    start = None
                    cur_key = None
            if start is not None:
                segments.append((start, len(stream_df)))
            return segments
        return active_segments_from_phase(stream_df)

    segments: List[Tuple[int, int]] = []
    offset = 0
    for _, group in stream_df.groupby("_source_file", sort=False):
        local = active_segments_from_phase(group)
        segments.extend((start + offset, end + offset) for start, end in local)
        offset += len(group)
    return segments


def evaluate(
    config_path: Path,
    mode: str,
    out_dir: Path,
    use_timestamp: bool,
    iou_threshold: float,
    make_plots: bool,
    max_plots: int,
) -> None:
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("feature", {})
    seg_cfg = cfg.get("segmentation", {})
    io_cfg = cfg.get("io", {})

    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    if out_dir == Path("artifacts/rep_segmentation"):
        out_dir = Path(io_cfg.get("rep_segmentation_output_dir", out_dir))
    include_actions = data_cfg.get("include_actions") or []
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])

    raw_sdtw_cfg = dict(seg_cfg.get("sdtw", {}))
    motion_columns = raw_sdtw_cfg.pop(
        "motion_columns",
        feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]),
    )
    sdtw_cfg = SDTWConfig(**raw_sdtw_cfg)
    imu_columns = tuple(
        motion_columns
    )
    out_dir = _timestamped_dir(out_dir, use_timestamp, mode)
    dirs = _prepare_output_dirs(out_dir)

    subjects = [p.name for p in _subject_dirs(data_dir)]
    if not include_actions:
        action_names = sorted({p.name for subject in _subject_dirs(data_dir) for p in subject.iterdir() if p.is_dir()})
        include_actions = [a for a in action_names if "rest" not in a]

    metrics_rows: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []
    templates_summary: List[Dict[str, object]] = []
    plot_count = 0

    print(f"[INFO] mode={mode}, subjects={subjects}, actions={include_actions}")
    print(f"[INFO] output={out_dir}")

    for test_subject in subjects:
        train_subjects = [subject for subject in subjects if subject != test_subject]
        for action in include_actions:
            train_reps: List[pd.DataFrame] = []
            for subject in train_subjects:
                train_reps.extend(_load_rep_csvs(data_dir, subject, action, exclude_patterns, sdtw_cfg))
            if len(train_reps) < 3:
                print(f"[WARN] Skip {action} for test={test_subject}: only {len(train_reps)} train reps")
                continue

            try:
                templates = fit_sdtw_templates(action, train_reps, imu_columns, sdtw_cfg)
            except Exception as exc:
                print(f"[WARN] Template failed for {action} test={test_subject}: {exc}")
                continue

            for template in templates:
                templates_summary.append(
                    {
                        "test_subject": test_subject,
                        "action_type": action,
                        "train_subjects": train_subjects,
                        "template_id": template.template_id,
                        "dtw_feature": template.dtw_feature,
                        "ranked_features": template.ranked_features[:5],
                        "feature_names": template.feature_names,
                        "active_duration": template.active_duration,
                        "min_duration": template.min_duration,
                        "max_duration": template.max_duration,
                        "cost_threshold": template.cost_threshold,
                        "n_calibration_costs": len(template.calibration_costs),
                        "exemplar_source": template.exemplar_source,
                        "exemplar_rep_position": template.exemplar_rep_position,
                    }
                )

            if mode == "sets":
                streams = _load_set_streams(data_dir, test_subject, action, exclude_patterns)
            elif mode == "whole":
                streams = _load_whole_streams(data_dir, test_subject, action)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            for stream_id, stream_df in streams:
                truth = _truth_segments_for_stream(stream_df)
                if not truth:
                    continue
                detections = detect_reps_sdtw_templates(stream_df, templates, imu_columns, sdtw_cfg)
                sample_rate = infer_sample_rate_hz(stream_df)
                metrics = summarize_detection_metrics(detections, truth, sample_rate, iou_threshold)
                metrics_rows.append(
                    {
                        "test_subject": test_subject,
                        "action_type": action,
                        "stream_id": stream_id,
                        "sample_rate_hz": sample_rate,
                        **metrics,
                    }
                )
                detection_rows.extend(_detection_rows(stream_id, detections))
                if make_plots and (max_plots <= 0 or plot_count < max_plots):
                    plot_name = _safe_name(stream_id) + ".svg"
                    plot_path = dirs["plots"] / action / test_subject / plot_name
                    _write_segmentation_svg(
                        plot_path,
                        stream_id,
                        stream_df,
                        truth,
                        detections,
                        metrics,
                        sample_rate,
                    )
                    plot_count += 1

    metrics_df = pd.DataFrame(metrics_rows)
    detections_df = pd.DataFrame(detection_rows)
    templates_df = pd.DataFrame(templates_summary)

    metrics_df.to_csv(dirs["metrics"] / "stream_metrics.csv", index=False)
    detections_df.to_csv(dirs["detections"] / "detections.csv", index=False)
    templates_df.to_csv(dirs["templates"] / "templates.csv", index=False)

    summary: Dict[str, object] = {
        "mode": mode,
        "iou_threshold": iou_threshold,
        "config": asdict(sdtw_cfg),
        "overall": _aggregate_metrics(metrics_rows),
        "by_action": {},
        "by_subject": {},
        "outputs": {
            "metrics": str(dirs["metrics"]),
            "detections": str(dirs["detections"]),
            "templates": str(dirs["templates"]),
            "plots": str(dirs["plots"]) if make_plots else None,
        },
    }
    if not metrics_df.empty:
        for action, group in metrics_df.groupby("action_type"):
            summary["by_action"][str(action)] = _aggregate_metrics(group.to_dict("records"))
        for subject, group in metrics_df.groupby("test_subject"):
            summary["by_subject"][str(subject)] = _aggregate_metrics(group.to_dict("records"))

    (dirs["metrics"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dirs["metadata"] / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "config_path": str(config_path),
                "iou_threshold": iou_threshold,
                "make_plots": make_plots,
                "max_plots": max_plots,
                "plot_count": plot_count,
                "subjects": subjects,
                "actions": include_actions,
                "motion_columns": list(imu_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        shutil.copy2(config_path, dirs["metadata"] / "config_snapshot.yaml")
    except Exception:
        pass
    (dirs["root"] / "README.md").write_text(
        "\n".join(
            [
                f"# SDTW Rep Segmentation ({mode})",
                "",
                "- `metrics/summary.json`: overall, by-action, and by-subject scores.",
                "- `metrics/stream_metrics.csv`: one row per evaluated stream.",
                "- `detections/detections.csv`: predicted rep start/end indices.",
                "- `templates/templates.csv`: exemplar-derived feature and threshold settings.",
                "- `plots/`: SVG overlays with ground truth active reps and SDTW predictions.",
                "- `metadata/config_snapshot.yaml`: config used for this run.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SDTW REP SEGMENTATION SUMMARY")
    print("=" * 72)
    print(json.dumps(summary["overall"], indent=2))
    if not metrics_df.empty:
        cols = ["action_type", "n_true", "n_pred", "precision", "recall", "f1", "start_mae_ms", "end_mae_ms"]
        by_action = metrics_df.groupby("action_type")[cols[1:]].mean(numeric_only=True).reset_index()
        print("\n[BY ACTION]")
        print(by_action[cols].to_string(index=False))
    print(f"\n[OK] Wrote organized outputs to {out_dir}")
    if make_plots:
        print(f"[OK] Wrote {plot_count} SVG plots under {dirs['plots']}")


def evaluate_both(
    config_path: Path,
    out_dir: Path,
    use_timestamp: bool,
    iou_threshold: float,
    make_plots: bool,
    max_plots: int,
) -> None:
    cfg = _load_config(config_path)
    io_cfg = cfg.get("io", {})
    if out_dir == Path("artifacts/rep_segmentation"):
        out_dir = Path(io_cfg.get("rep_segmentation_output_dir", out_dir))

    run_dir = _timestamped_run_dir(out_dir, use_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] combined output={run_dir}")
    evaluate(
        config_path=config_path,
        mode="sets",
        out_dir=run_dir / "sets",
        use_timestamp=False,
        iou_threshold=iou_threshold,
        make_plots=make_plots,
        max_plots=max_plots,
    )
    evaluate(
        config_path=config_path,
        mode="whole",
        out_dir=run_dir / "whole",
        use_timestamp=False,
        iou_threshold=iou_threshold,
        make_plots=make_plots,
        max_plots=max_plots,
    )

    combined: Dict[str, object] = {
        "mode": "both",
        "iou_threshold": iou_threshold,
        "outputs": {
            "sets": str(run_dir / "sets"),
            "whole": str(run_dir / "whole"),
        },
        "sets": {},
        "whole": {},
    }
    for mode in ("sets", "whole"):
        summary_path = run_dir / mode / "metrics" / "summary.json"
        if summary_path.exists():
            try:
                combined[mode] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                combined[mode] = {"error": f"Could not read {summary_path}"}

    (run_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "\n".join(
            [
                "# SDTW Rep Segmentation Run",
                "",
                "This folder contains both evaluation modes for the same parameter run.",
                "",
                "- `sets/`: evaluation on concatenated per-set rep CSVs.",
                "- `whole/`: evaluation on whole-session action blocks.",
                "- `summary.json`: combined pointer summary for both modes.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] Wrote combined sets+whole outputs to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SDTW rep segmentation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=["sets", "whole", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/rep_segmentation"))
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--no-plots", action="store_true", help="Disable SVG segmentation plots")
    parser.add_argument("--max-plots", type=int, default=0, help="Maximum plots to write; 0 = no limit")
    parser.add_argument("--no-timestamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "both":
        evaluate_both(
            config_path=args.config,
            out_dir=args.out_dir,
            use_timestamp=not args.no_timestamp,
            iou_threshold=args.iou_threshold,
            make_plots=not args.no_plots,
            max_plots=args.max_plots,
        )
    else:
        evaluate(
            config_path=args.config,
            mode=args.mode,
            out_dir=args.out_dir,
            use_timestamp=not args.no_timestamp,
            iou_threshold=args.iou_threshold,
            make_plots=not args.no_plots,
            max_plots=args.max_plots,
        )


if __name__ == "__main__":
    main()
