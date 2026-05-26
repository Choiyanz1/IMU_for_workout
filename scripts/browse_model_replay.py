"""Interactive model replay browser.

Navigate: model -> subject -> session -> action -> set, then view GT vs predictions.

Supports:
  - TCN models with saved streaming_predictions.csv
  - Per-Action / Causal RF models (trained on-the-fly)
  - Other baselines with saved fold results

Usage:
    python scripts/browse_model_replay.py
    python scripts/browse_model_replay.py --data-dir datasets/raw_data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")  # Interactive GUI backend; switch to "Agg" for headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.window_pipeline import ZScoreStats, apply_zscore, compute_train_stats
from preprocessing.micro_macro_segments import rep_metrics

import compare_baselines as cb
import importlib.util


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load boundary refiner module for _coarse_predict_reps, _refine_reps, etc.
base = _load_mod(ROOT / "scripts" / "train_rf_boundary_refiner.py", "train_rf_boundary_refiner_mod")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _natural_sort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _subdirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted([d for d in parent.iterdir() if d.is_dir()], key=_natural_sort_key)


def _csv_files(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted(parent.glob("*.csv"), key=_natural_sort_key)


def _choose(prompt: str, options: List[str], allow_back: bool = True) -> int | None:
    print()
    print(f"--- {prompt} ---")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    if allow_back:
        print("  [b] <- back")
    print("  [q] quit")
    while True:
        raw = input(">>> ").strip().lower()
        if raw == "q":
            sys.exit(0)
        if raw == "b" and allow_back:
            return None
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Invalid input, try again.")


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models() -> List[Dict]:
    """Discover available models with replay capability."""
    models = []

    # 1. TCN / DL models with streaming_eval_all
    tcn_root = ROOT / "artifacts" / "micro_macro_recognition"
    if tcn_root.is_dir():
        for exp_dir in sorted(tcn_root.iterdir(), key=_natural_sort_key):
            if not exp_dir.is_dir():
                continue
            tcn_sub = exp_dir / "tcn"
            if not tcn_sub.is_dir():
                continue
            streaming_dir = tcn_sub / "streaming_eval_all"
            if streaming_dir.is_dir():
                models.append({
                    "name": f"[TCN] {exp_dir.name}",
                    "type": "tcn",
                    "path": str(tcn_sub),
                    "exp_dir": str(exp_dir),
                })

    # 2. RF / Baseline models from baseline_comparison
    bc_root = ROOT / "artifacts" / "baseline_comparison"
    if bc_root.is_dir():
        for model_dir in sorted(bc_root.iterdir(), key=_natural_sort_key):
            if not model_dir.is_dir():
                continue
            # Check if this looks like a per-action RF model
            actions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "grand_summary.json"]
            has_action_folds = False
            for action in actions:
                action_dir = model_dir / action
                if any(f.name.startswith("fold_") for f in action_dir.iterdir() if f.is_file()):
                    has_action_folds = True
                    break
            if has_action_folds:
                # Determine if per-action or global
                summary_file = model_dir / "grand_summary.json"
                protocol = "per_action_rf"
                if summary_file.is_file():
                    try:
                        summary = json.loads(summary_file.read_text())
                        protocol = summary.get("evaluation_protocol", "per_action_rf")
                    except Exception:
                        pass
                models.append({
                    "name": f"[RF] {model_dir.name}",
                    "type": "rf",
                    "path": str(model_dir),
                    "protocol": protocol,
                })

    return models


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _compute_train_fold_stats(data_root: Path, heldout_subject: str, imu_columns: List[str]) -> ZScoreStats | None:
    sequences: List[pd.DataFrame] = []
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        if rel.parts and rel.parts[0] == heldout_subject:
            continue
        try:
            df = pd.read_csv(csv_path, usecols=imu_columns)
        except Exception:
            continue
        if not df.empty:
            sequences.append(df)
    if not sequences:
        return None
    return compute_train_stats(sequences, imu_columns)


def _load_stream(data_root: Path, subject: str, session: str, action: str, set_name: str) -> List[Tuple[str, pd.DataFrame]]:
    """Load all rep CSVs for a given set."""
    set_dir = data_root / subject / session / action / set_name
    if not set_dir.is_dir():
        return []
    streams = []
    for csv_path in sorted(set_dir.glob("*.csv"), key=_natural_sort_key):
        try:
            df = pd.read_csv(csv_path)
            rel = csv_path.relative_to(data_root)
            stream_id = "/".join(rel.parts)
            streams.append((stream_id, df))
        except Exception:
            pass
    return streams


# ---------------------------------------------------------------------------
# Prediction loading / generation
# ---------------------------------------------------------------------------

def load_tcn_predictions(streaming_pred_path: Path) -> Optional[pd.DataFrame]:
    """Load TCN streaming predictions."""
    if not streaming_pred_path.is_file():
        return None
    try:
        return pd.read_csv(streaming_pred_path)
    except Exception:
        return None


def _fit_boundary_refiner(train_streams, train_prob_cache, mm_cfg, imu_columns, edge_window=20, max_shift=20):
    """Train a simple boundary refiner from matched coarse reps and GT reps."""
    from sklearn.ensemble import ExtraTreesRegressor
    
    # Collect matched examples
    matched_examples = []
    for stream_id, df in train_streams:
        probs = train_prob_cache.get(stream_id)
        if probs is None:
            continue
        
        # Coarse prediction
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        truth = cb.truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        
        # Match coarse to truth by IoU
        matches = cb.match_segments(
            [(r.start_idx, r.end_idx) for r in coarse_reps],
            [(r.start_idx, r.end_idx) for r in truth],
            iou_threshold=0.3,
        )
        
        for pred_idx, true_idx, _ in matches:
            p = coarse_reps[pred_idx]
            t = truth[true_idx]
            matched_examples.append({
                "df": df,
                "probs": probs,
                "pred_rep": p,
                "start_shift": int(np.clip(t.start_idx - p.start_idx, -max_shift, max_shift)),
                "transition_shift": int(np.clip(t.transition_idx - p.transition_idx, -max_shift, max_shift)),
                "end_shift": int(np.clip(t.end_idx - p.end_idx, -max_shift, max_shift)),
            })
    
    if len(matched_examples) < 5:
        print(f"  [Refiner] Only {len(matched_examples)} matched reps, skipping refiner.")
        return None
    
    print(f"  [Refiner] Training on {len(matched_examples)} matched reps...")
    
    # Build features
    start_rows, trans_rows, end_rows = [], [], []
    y_start, y_trans, y_end = [], [], []
    
    for ex in matched_examples:
        start_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "start", edge_window, imu_columns))
        trans_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "transition", edge_window, imu_columns))
        end_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "end", edge_window, imu_columns))
        y_start.append(float(ex["start_shift"]))
        y_trans.append(float(ex["transition_shift"]))
        y_end.append(float(ex["end_shift"]))
    
    x_start, feature_keys = base._rows_to_matrix(start_rows)
    x_trans, _ = base._rows_to_matrix(trans_rows)
    x_end, _ = base._rows_to_matrix(end_rows)
    
    # Fit regressors
    def _fit_reg(x, y):
        model = ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42)
        model.fit(x, y)
        return model
    
    return {
        "start": _fit_reg(x_start, np.asarray(y_start, dtype=np.float32)),
        "transition": _fit_reg(x_trans, np.asarray(y_trans, dtype=np.float32)),
        "end": _fit_reg(x_end, np.asarray(y_end, dtype=np.float32)),
        "feature_keys": feature_keys,
        "edge_window": edge_window,
        "max_shift": max_shift,
    }


def run_rf_predictions(model_info: Dict, action: str, test_subject: str,
                       data_root: Path, imu_columns: List[str],
                       mm_cfg,
                       window_size: int = 100, stride: int = 10,
                       n_estimators: int = 100, max_depth: int = 15,
                       max_samples: float = 0.7, smoothing_window: int = 15,
                       edge_window: int = 20, max_shift: int = 20) -> List[Tuple[str, pd.DataFrame, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """Train per-action RF + boundary refiner and predict on test subject streams.
    
    Returns list of (stream_id, df, probs, gt_rep_segments, pred_rep_segments).
    """
    # Load all streams for this action
    all_streams = []
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        parts = rel.parts
        if len(parts) >= 4 and parts[2] == action:
            try:
                df = pd.read_csv(csv_path)
                stream_id = "/".join(parts)
                all_streams.append((stream_id, df))
            except Exception:
                pass

    train_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] != test_subject]
    test_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] == test_subject]

    if not test_streams:
        print(f"  [WARN] No test streams found for {test_subject}/{action}")
        return []

    # Z-score normalize
    stats = _compute_train_fold_stats(data_root, test_subject, imu_columns)
    if stats is None:
        print("  [WARN] Could not compute z-score stats")
        return []

    train_z = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_z = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

    # Compute duration prior from training data GT reps
    train_durations = []
    for _, df in train_streams:
        gt_reps_train = cb.truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        for r in gt_reps_train:
            train_durations.append(int(r.end_idx) - int(r.start_idx))
    
    if train_durations:
        min_dur = int(np.quantile(train_durations, 0.05)) if len(train_durations) >= 10 else min(train_durations)
        max_dur = int(np.quantile(train_durations, 0.95)) if len(train_durations) >= 10 else max(train_durations)
        min_rep_duration_samples = max(1, int(min_dur * 0.5))
        max_rep_duration_samples = max(min_rep_duration_samples + 1, int(max_dur * 2.0))
        print(f"  [DurationPrior] train reps={len(train_durations)}, min={min_rep_duration_samples}, max={max_rep_duration_samples} samples")
    else:
        sample_rate = cb.infer_sample_rate_hz(train_streams[0][1]) if train_streams else 100.0
        min_rep_duration_samples = int(mm_cfg.min_rep_duration_seconds * sample_rate) if mm_cfg.min_rep_duration_seconds > 0 else 0
        max_rep_duration_samples = int(10.0 * sample_rate)
        print(f"  [DurationPrior] fallback min={min_rep_duration_samples}, max={max_rep_duration_samples} samples")

    # Train RF
    print(f"  [RF] Training on {len(train_z)} streams...")
    crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")
    clf = crf.train_causal_rf(
        train_z, imu_columns,
        window_size=window_size, stride=stride,
        n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples,
    )

    # Predict probabilities for all streams
    train_prob_cache = {}
    for sid, df in train_z:
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
        if smoothing_window > 1:
            smoothed = np.zeros_like(probs)
            csum = np.cumsum(probs, axis=0)
            for i in range(len(probs)):
                start = max(0, i - smoothing_window + 1)
                total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                smoothed[i] = total / float(i - start + 1)
            probs = smoothed
        train_prob_cache[sid] = probs

    # Train boundary refiner
    refiner = _fit_boundary_refiner(train_z, train_prob_cache, mm_cfg, imu_columns, edge_window=edge_window, max_shift=max_shift)

    # Predict all test streams and extract reps
    results = []
    for sid, df in test_z:
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
        
        # Smooth
        if smoothing_window > 1:
            smoothed = np.zeros_like(probs)
            csum = np.cumsum(probs, axis=0)
            for i in range(len(probs)):
                start = max(0, i - smoothing_window + 1)
                total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                smoothed[i] = total / float(i - start + 1)
            probs = smoothed
        
        # Extract GT reps
        gt_reps = cb.truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        gt_segments = [(int(r.start_idx), int(r.end_idx)) for r in gt_reps]
        
        # Coarse prediction
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        coarse_reps = _apply_duration_prior(coarse_reps, min_rep_duration_samples, max_rep_duration_samples)
        
        # Refine boundaries
        if refiner is not None:
            pred_reps = base._refine_reps(df, probs, coarse_reps, refiner, imu_columns, edge_window=edge_window, max_shift=max_shift)
            print(f"    [Refiner] Applied to {len(coarse_reps)} coarse reps")
        else:
            pred_reps = list(coarse_reps)
        
        # Final duration filter
        pred_reps = _apply_duration_prior(pred_reps, min_rep_duration_samples, max_rep_duration_samples)
        print(f"    Stream: {len(gt_segments)} GT reps, {len(pred_reps)} pred reps (coarse={len(coarse_reps)}, after refiner+duration)")
        pred_segments = [(int(r.start_idx), int(r.end_idx)) for r in pred_reps]
        
        # Compute per-stream metrics
        sample_rate = cb.infer_sample_rate_hz(df)
        metrics = rep_metrics(pred_reps, gt_reps, sample_rate_hz=sample_rate)
        
        results.append((sid, df, probs, gt_segments, pred_segments, metrics))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "concentric": "#3b82f6",
    "eccentric": "#ef4444",
    "none": "#d1d5db",
    "other": "#d1d5db",
}

PRED_COLORS = {
    "concentric": "#60a5fa",  # lighter blue
    "eccentric": "#fca5a5",  # lighter red
    "none": "#e5e7eb",
    "other": "#e5e7eb",
}


def _add_phase_spans(ax: plt.Axes, t: np.ndarray, phases: np.ndarray, alpha: float = 0.24) -> None:
    i = 0
    while i < len(phases):
        j = i + 1
        while j < len(phases) and phases[j] == phases[i]:
            j += 1
        phase = str(phases[i]).lower()
        color = PHASE_COLORS.get(phase, "#e5e7eb")
        ax.axvspan(t[i], t[min(j - 1, len(t) - 1)], alpha=alpha, color=color, linewidth=0)
        i = j


def _add_pred_spans(ax: plt.Axes, t: np.ndarray, preds: np.ndarray, alpha: float = 0.15) -> None:
    """Add prediction spans as top overlay."""
    i = 0
    while i < len(preds):
        j = i + 1
        while j < len(preds) and preds[j] == preds[i]:
            j += 1
        pred = str(preds[i]).lower()
        color = PRED_COLORS.get(pred, "#e5e7eb")
        # Only draw if different from "none"/"other"
        if pred not in ("none", "other", ""):
            ax.axvspan(t[i], t[min(j - 1, len(t) - 1)], alpha=alpha, color=color, linewidth=0)
        i = j


def _concatenate_reps_with_segments(
    streams: List[Tuple[str, pd.DataFrame]],
    gt_segments_per_stream: List[List[Tuple[int, int]]],
    pred_segments_per_stream: List[List[Tuple[int, int]]],
    gap_s: float = 0.15,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Concatenate all reps in a set with gaps.
    
    Args:
        streams: list of (stream_id, df)
        gt_segments_per_stream: GT rep segments for each stream (concentric+eccentric pairs)
        pred_segments_per_stream: Pred rep segments for each stream
    
    Returns:
        concat_df, t, phases, gt_rep_segments, pred_rep_segments (all adjusted for concatenation)
    """
    if not streams:
        return pd.DataFrame(), np.array([]), np.array([]), [], []
    
    frames = []
    phases_list = []
    gt_rep_segments = []
    pred_rep_segments = []
    offset_s = 0.0
    sample_rate = 100.0  # Hz
    cumulative_idx = 0
    
    for (stream_id, df), gt_segs, pred_segs in zip(streams, gt_segments_per_stream, pred_segments_per_stream):
        if df.empty:
            continue
        
        n = len(df)
        
        # Adjust GT segments for concatenation offset
        for start, end in gt_segs:
            gt_rep_segments.append((start + cumulative_idx, end + cumulative_idx))
        
        # Adjust Pred segments for concatenation offset
        for start, end in pred_segs:
            pred_rep_segments.append((start + cumulative_idx, end + cumulative_idx))
        
        cumulative_idx += n
        
        # Copy and adjust time
        df_copy = df.copy()
        if "sensor_ts" in df_copy.columns:
            t_local = (df_copy["sensor_ts"] - df_copy["sensor_ts"].iloc[0]).to_numpy(dtype=float) / 1e6
        else:
            t_local = np.arange(n) / sample_rate
        
        df_copy["_t"] = t_local + offset_s
        frames.append(df_copy)
        
        rep_phases = df_copy["phase"].astype(str).to_numpy() if "phase" in df_copy.columns else np.full(n, "none")
        phases_list.append(rep_phases)
        
        # Add gap
        duration = float(t_local[-1]) if len(t_local) > 0 else 0.0
        offset_s += duration + gap_s
    
    concat_df = pd.concat(frames, ignore_index=True)
    t = concat_df["_t"].to_numpy()
    phases = np.concatenate(phases_list)
    
    return concat_df, t, phases, gt_rep_segments, pred_rep_segments


def _apply_duration_prior(reps, min_samples: int, max_samples: int):
    """Filter reps by duration (in samples)."""
    out = []
    for rep in reps:
        duration = int(rep.end_idx) - int(rep.start_idx)
        if int(min_samples) > 0 and duration < int(min_samples):
            continue
        if int(max_samples) > 0 and duration > int(max_samples):
            continue
        out.append(rep)
    return out


def _magnitude(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Compute magnitude of columns."""
    arr = np.stack([df[c].to_numpy(dtype=np.float32) for c in cols if c in df.columns], axis=1)
    return np.sqrt(np.sum(arr ** 2, axis=1))


def _extract_rep_segments(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Extract rep segments (concentric + eccentric) from phase labels."""
    if "phase" not in df.columns:
        return []
    phases = df["phase"].astype(str).to_numpy()
    segments = []
    start = None
    for i, phase in enumerate(phases):
        is_active = phase in {"concentric", "eccentric"}
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(phases)))
    return segments


def _rle_decode_runs(runs: List[Tuple[int, int]], n: int) -> np.ndarray:
    """Convert run segments to per-sample boolean mask."""
    mask = np.zeros(n, dtype=bool)
    for start, end in runs:
        mask[start:end] = True
    return mask


def _plot_set_replay_clean(
    set_title: str,
    concat_df: pd.DataFrame,
    gt_rep_segments: List[Tuple[int, int]],
    pred_rep_segments: List[Tuple[int, int]],
    metrics: Optional[Dict[str, float]] = None,
    sample_rate: float = 100.0,
) -> None:
    """Plot clean lane-style replay. Each block = one rep (one CSV file).
    
    Layout:
      [Title + metrics]
      [GT rep lane]     -- green blocks showing each GT rep (CSV range)
      [Pred rep lane]   -- orange blocks showing each predicted rep
      [acc_mag trace]   -- with GT dashed / Pred solid boundary lines
      [gyro_mag trace]  -- with GT dashed / Pred solid boundary lines
    """
    n = len(concat_df)
    duration_s = n / sample_rate if sample_rate > 0 else 0.0
    
    # Compute magnitudes
    acc_mag = _magnitude(concat_df, ["ax", "ay", "az"])
    gyro_mag = _magnitude(concat_df, ["gx", "gy", "gz"])
    t = np.arange(n) / sample_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [0.6, 0.6, 2.5, 2.5]})
    fig.suptitle(set_title, fontsize=12, fontweight="bold")
    
    # Title / metrics
    subtitle = ""
    if metrics:
        subtitle = (
            f"F1={metrics.get('f1', 0):.3f}  P={metrics.get('precision', 0):.3f}  "
            f"R={metrics.get('recall', 0):.3f}  GT={int(metrics.get('n_true', 0))}  "
            f"Pred={int(metrics.get('n_pred', 0))}  Duration={duration_s:.1f}s"
        )
        # Add rep count accuracy if available
        if 'exact_count_ratio' in metrics:
            subtitle += (
                f"  |  ExactCt={metrics['exact_count_streams']}/{metrics.get('n_streams', '?')}"
                f" ({metrics['exact_count_ratio']:.0%})"
                f"  Over={metrics.get('over_segmented_streams', 0)}"
                f"  Under={metrics.get('under_segmented_streams', 0)}"
                f"  MADiff={metrics.get('mean_abs_count_diff', 0):.1f}"
            )
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color="#475569")
    
    # --- GT Rep Lane ---
    ax_gt = axes[0]
    ax_gt.set_xlim(0, duration_s)
    ax_gt.set_ylim(0, 1)
    ax_gt.set_ylabel("GT", fontsize=10, fontweight="bold", color="#166534")
    ax_gt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_visible(False)
    # Background
    ax_gt.axvspan(0, duration_s, ymin=0.1, ymax=0.9, color="#ecfdf5", alpha=1.0)
    # GT rep blocks (each CSV = one rep)
    for start, end in gt_rep_segments:
        x_start = start / sample_rate
        x_end = end / sample_rate
        ax_gt.axvspan(x_start, x_end, ymin=0.1, ymax=0.9, color="#2da44e", alpha=0.9)
    # Rep count text
    ax_gt.text(0.01, 0.5, f"{len(gt_rep_segments)} reps", transform=ax_gt.transAxes,
               fontsize=9, color="#166534", va="center")
    
    # --- Pred Rep Lane ---
    ax_pred = axes[1]
    ax_pred.set_xlim(0, duration_s)
    ax_pred.set_ylim(0, 1)
    ax_pred.set_ylabel("Pred", fontsize=10, fontweight="bold", color="#c2410c")
    ax_pred.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_visible(False)
    # Background
    ax_pred.axvspan(0, duration_s, ymin=0.1, ymax=0.9, color="#fff7ed", alpha=1.0)
    # Pred rep blocks
    for start, end in pred_rep_segments:
        x_start = start / sample_rate
        x_end = end / sample_rate
        ax_pred.axvspan(x_start, x_end, ymin=0.1, ymax=0.9, color="#fb8500", alpha=0.9)
    # Rep count text
    ax_pred.text(0.01, 0.5, f"{len(pred_rep_segments)} reps", transform=ax_pred.transAxes,
                fontsize=9, color="#c2410c", va="center")
    
    # --- Acc Mag Trace ---
    ax_acc = axes[2]
    ax_acc.plot(t, acc_mag, linewidth=1.2, color="#2563eb", label="acc_mag")
    ax_acc.set_ylabel("acc_mag", fontsize=10, fontweight="bold", color="#1d4ed8")
    ax_acc.tick_params(labelsize=8)
    # GT boundaries (dashed green) = each CSV boundary
    for start, end in gt_rep_segments:
        for idx in (start, end):
            if 0 <= idx < n:
                ax_acc.axvline(t[idx], color="#15803d", linestyle="--", linewidth=1.2, alpha=0.7)
    # Pred boundaries (solid orange)
    for start, end in pred_rep_segments:
        for idx in (start, end):
            if 0 <= idx < n:
                ax_acc.axvline(t[idx], color="#ea580c", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_acc.set_xlim(0, duration_s)
    
    # --- Gyro Mag Trace ---
    ax_gyro = axes[3]
    ax_gyro.plot(t, gyro_mag, linewidth=1.2, color="#7c3aed", label="gyro_mag")
    ax_gyro.set_ylabel("gyro_mag", fontsize=10, fontweight="bold", color="#6d28d9")
    ax_gyro.set_xlabel("Time (s)", fontsize=9)
    ax_gyro.tick_params(labelsize=8)
    # GT boundaries (dashed green)
    for start, end in gt_rep_segments:
        for idx in (start, end):
            if 0 <= idx < n:
                ax_gyro.axvline(t[idx], color="#15803d", linestyle="--", linewidth=1.2, alpha=0.7)
    # Pred boundaries (solid orange)
    for start, end in pred_rep_segments:
        for idx in (start, end):
            if 0 <= idx < n:
                ax_gyro.axvline(t[idx], color="#ea580c", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_gyro.set_xlim(0, duration_s)
    
    # Legend
    legend_elements = [
        Patch(facecolor="#2da44e", alpha=0.9, label="GT rep"),
        Patch(facecolor="#fb8500", alpha=0.9, label="Pred rep"),
        plt.Line2D([0], [0], color="#15803d", linestyle="--", linewidth=1.2, label="GT boundary (CSV)"),
        plt.Line2D([0], [0], color="#ea580c", linestyle="-", linewidth=1.0, label="Pred boundary"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.show(block=False)
    plt.pause(0.1)
    input("  Press Enter to close plot...")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------------

def _navigate_data_tree(data_root: Path) -> Optional[Tuple[str, str, str, str, List[Tuple[str, pd.DataFrame]]]]:
    """Navigate subject -> session -> action -> set, return streams."""
    subjects = _subdirs(data_root)
    if not subjects:
        print("No subjects found in data directory.")
        return None

    idx = _choose("Select SUBJECT", [s.name for s in subjects], allow_back=False)
    if idx is None:
        return None
    subject = subjects[idx].name

    sessions = _subdirs(data_root / subject)
    if not sessions:
        print("  No sessions found.")
        return None
    idx = _choose(f"Select SESSION ({subject})", [s.name for s in sessions])
    if idx is None:
        return None
    session = sessions[idx].name

    actions = _subdirs(data_root / subject / session)
    if not actions:
        print("  No actions found.")
        return None
    idx = _choose(f"Select ACTION ({subject}/{session})", [s.name for s in actions])
    if idx is None:
        return None
    action = actions[idx].name

    sets = [d for d in _subdirs(data_root / subject / session / action) if d.name.startswith("set")]
    if not sets:
        print("  No sets found.")
        return None
    display = [f"{s.name} ({len(list(s.glob('*.csv')))} reps)" for s in sets]
    idx = _choose(f"Select SET ({subject}/{session}/{action})", display)
    if idx is None:
        return None
    set_name = sets[idx].name

    streams = _load_stream(data_root, subject, session, action, set_name)
    if not streams:
        print("  No rep CSVs found.")
        return None

    return subject, session, action, set_name, streams


def _navigate_tcn_tree(streaming_root: Path) -> Optional[Tuple[str, str, str, str, Path]]:
    """Navigate TCN streaming_eval tree to find predictions CSV."""
    subjects = _subdirs(streaming_root)
    if not subjects:
        print("No subjects found in streaming eval.")
        return None

    idx = _choose("Select SUBJECT", [s.name for s in subjects], allow_back=False)
    if idx is None:
        return None
    subject = subjects[idx]

    sessions = _subdirs(subject)
    if not sessions:
        print("  No sessions found.")
        return None
    idx = _choose(f"Select SESSION ({subject.name})", [s.name for s in sessions])
    if idx is None:
        return None
    session = sessions[idx]

    actions = _subdirs(session)
    if not actions:
        print("  No actions found.")
        return None
    idx = _choose(f"Select ACTION ({subject.name}/{session.name})", [s.name for s in actions])
    if idx is None:
        return None
    action = actions[idx]

    sets = _subdirs(action)
    if not sets:
        print("  No sets found.")
        return None
    idx = _choose(f"Select SET ({subject.name}/{session.name}/{action.name})", [s.name for s in sets])
    if idx is None:
        return None
    set_dir = sets[idx]

    pred_file = set_dir / "streaming_predictions.csv"
    if not pred_file.is_file():
        print(f"  No streaming_predictions.csv found in {set_dir}")
        return None

    return subject.name, session.name, action.name, set_dir.name, pred_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse and replay model predictions")
    parser.add_argument("--data-dir", default="datasets/raw_data", help="Raw data root")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        print(f"Data directory not found: {data_root}")
        sys.exit(1)

    # Load config for imu_columns
    config_path = Path(args.config)
    imu_columns = ["ax", "ay", "az", "gx", "gy", "gz"]
    if config_path.is_file():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            feature_cfg = raw.get("feature", {}) or {}
            imu_columns = list(feature_cfg.get("imu_columns", imu_columns))
        except Exception:
            pass

    # Discover models
    print("Scanning for available models...")
    models = discover_models()
    if not models:
        print("No models found with replay capability.")
        print("Expected:")
        print("  - TCN: artifacts/micro_macro_recognition/<exp>/tcn/streaming_eval_all/")
        print("  - RF:  artifacts/baseline_comparison/<model>/")
        sys.exit(1)

    model_names = [m["name"] for m in models]
    idx = _choose("Select MODEL", model_names, allow_back=False)
    if idx is None:
        sys.exit(0)
    model = models[idx]
    print(f"\nSelected: {model['name']}")

    # Handle by model type
    if model["type"] == "tcn":
        streaming_root = Path(model["path"]) / "streaming_eval_all"
        result = _navigate_tcn_tree(streaming_root)
        if result is None:
            sys.exit(0)
        subject, session, action, set_name, pred_file = result

        # Find corresponding raw data
        raw_set_dir = data_root / subject / session / action / set_name
        streams = _load_stream(data_root, subject, session, action, set_name)
        if not streams:
            print("  No raw data found for this stream.")
            sys.exit(0)

        predictions = load_tcn_predictions(pred_file)
        if predictions is None:
            print("  Failed to load predictions.")
            sys.exit(0)

        # Concatenate all reps for set-level view
        # For TCN, extract GT reps from each CSV's phase labels
        gt_segments_per_stream = []
        for stream_id, df in streams:
            gt_reps = cb.truth_reps_from_labels(
                df["phase"].to_numpy(),
                actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
                min_phase_samples=mm_cfg.min_phase_samples,
            )
            gt_segments_per_stream.append([(int(r.start_idx), int(r.end_idx)) for r in gt_reps])
        
        # For TCN, extract Pred reps from predictions using same pipeline
        pred_segments_per_stream = []
        pred_offset = 0
        for stream_id, df in streams:
            n = len(df)
            pred_end = min(pred_offset + n, len(predictions))
            if pred_offset < len(predictions):
                pred_slice = predictions.iloc[pred_offset:pred_end]
                if "online_micro_label" in pred_slice.columns:
                    pred_labels = pred_slice["online_micro_label"].astype(str).to_numpy()
                    # Build micro_probs from confidence (simplified)
                    n_samples = len(pred_labels)
                    micro_probs = np.zeros((n_samples, len(cb.MICRO_LABELS)), dtype=np.float32)
                    for i, label in enumerate(pred_labels):
                        if label in cb.MICRO_LABELS:
                            micro_probs[i, cb.MICRO_LABELS.index(label)] = 1.0
                    
                    pred_micro_runs = cb.labels_to_runs(
                        pred_labels,
                        positive_labels=(cb.CONCENTRIC_LABEL, cb.ECCENTRIC_LABEL),
                        probabilities=micro_probs,
                        min_length=mm_cfg.min_phase_samples,
                    )
                    pred_reps, _ = cb.pair_concentric_eccentric_reps(
                        pred_micro_runs, micro_source="tcn", max_gap_samples=mm_cfg.max_phase_gap_samples,
                    )
                    pred_reps = cb._filter_predicted_reps(
                        pred_reps, sample_rate_hz=cb.infer_sample_rate_hz(df),
                        min_duration_seconds=mm_cfg.min_rep_duration_seconds,
                        min_confidence=mm_cfg.min_rep_confidence,
                    )
                    pred_segments_per_stream.append([(int(r.start_idx), int(r.end_idx)) for r in pred_reps])
                else:
                    pred_segments_per_stream.append([])
            else:
                pred_segments_per_stream.append([])
            pred_offset += n
        
        concat_df, _, _, gt_rep_segments, pred_rep_segments = _concatenate_reps_with_segments(
            streams, gt_segments_per_stream, pred_segments_per_stream
        )
        if len(concat_df) == 0:
            print("  No data to plot.")
            sys.exit(0)

        title = f"[TCN] {model['name']} | {subject}/{session}/{action}/{set_name}"
        _plot_set_replay_clean(title, concat_df, gt_rep_segments, pred_rep_segments, sample_rate=100.0)

    elif model["type"] == "rf":
        # Navigate data tree
        result = _navigate_data_tree(data_root)
        if result is None:
            sys.exit(0)
        subject, session, action, set_name, streams = result

        # Load fold config to get parameters
        model_dir = Path(model["path"])
        action_dir = model_dir / action
        fold_files = sorted(action_dir.glob("fold_*.json"))

        # Default params
        window_size, stride = 100, 10
        n_estimators, max_depth = 100, 15
        max_samples = 0.7
        smoothing_window = 15

        if fold_files:
            try:
                fold_data = json.loads(fold_files[0].read_text())
                cfg = fold_data.get("config", {})
                window_size = cfg.get("window_size", window_size)
                stride = cfg.get("train_stride", stride)
                n_estimators = cfg.get("n_estimators", n_estimators)
                max_depth = cfg.get("max_depth", max_depth)
                max_samples = cfg.get("max_samples", max_samples)
                smoothing_window = cfg.get("smoothing_window", smoothing_window)
            except Exception:
                pass

        # Load config for mm_cfg (needed for evaluate_all_streams)
        config_path = Path(args.config)
        mm_cfg = None
        if config_path.is_file():
            try:
                raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                mm_raw = raw.get("micro_macro", {}) or {}
                mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
            except Exception:
                pass
        if mm_cfg is None:
            mm_cfg = cb.MicroMacroConfig()

        print(f"\n  Training RF for action={action}, test_subject={subject}")
        print(f"  Params: window={window_size}, stride={stride}, trees={n_estimators}, depth={max_depth}")

        predictions = run_rf_predictions(
            model, action, subject, data_root, imu_columns, mm_cfg,
            window_size=window_size, stride=stride,
            n_estimators=n_estimators, max_depth=max_depth,
            max_samples=max_samples, smoothing_window=smoothing_window,
        )

        # Run predictions with rep extraction
        print(f"  Concatenating {len(streams)} reps into set-level view...")
        
        pred_map = {Path(sid).name: (probs, gt_segs, pred_segs, metrics) 
                    for sid, _, probs, gt_segs, pred_segs, metrics in predictions}
        gt_segments_per_stream = []
        pred_segments_per_stream = []
        valid_streams = []
        all_stream_metrics = []
        
        for stream_id, df in streams:
            rep_name = Path(stream_id).name
            if rep_name in pred_map:
                probs, gt_segs, pred_segs, metrics = pred_map[rep_name]
                gt_segments_per_stream.append(gt_segs)
                pred_segments_per_stream.append(pred_segs)
                valid_streams.append((stream_id, df))
                all_stream_metrics.append(metrics)
            else:
                print(f"  [WARN] Skipping {rep_name} (no prediction)")
        
        if not valid_streams:
            print("  [ERROR] No predictions could be matched to any rep.")
            sys.exit(0)
        
        # Aggregate metrics across all streams
        if all_stream_metrics:
            total_pred = sum(m['n_pred'] for m in all_stream_metrics)
            total_true = sum(m['n_true'] for m in all_stream_metrics)
            total_tp = sum(m['tp'] for m in all_stream_metrics)
            total_fp = total_pred - total_tp
            total_fn = total_true - total_tp
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Rep count metrics
            exact_count = sum(1 for m in all_stream_metrics if m['n_pred'] == m['n_true'])
            over_count = sum(1 for m in all_stream_metrics if m['n_pred'] > m['n_true'])
            under_count = sum(1 for m in all_stream_metrics if m['n_pred'] < m['n_true'])
            count_diffs = [abs(m['n_pred'] - m['n_true']) for m in all_stream_metrics]
            
            # Boundary MAE (average across matched streams)
            start_mae_vals = [m['start_mae_ms'] for m in all_stream_metrics if m['tp'] > 0 and np.isfinite(m['start_mae_ms'])]
            end_mae_vals = [m['end_mae_ms'] for m in all_stream_metrics if m['tp'] > 0 and np.isfinite(m['end_mae_ms'])]
            
            agg_metrics = {
                'n_pred': total_pred,
                'n_true': total_true,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'start_mae_ms': float(np.mean(start_mae_vals)) if start_mae_vals else float('nan'),
                'end_mae_ms': float(np.mean(end_mae_vals)) if end_mae_vals else float('nan'),
                'exact_count_streams': exact_count,
                'over_segmented_streams': over_count,
                'under_segmented_streams': under_count,
                'exact_count_ratio': exact_count / len(all_stream_metrics),
                'mean_abs_count_diff': float(np.mean(count_diffs)) if count_diffs else 0.0,
                'n_streams': len(all_stream_metrics),
            }
            
            print(f"\n  === Rep Count Metrics ===")
            print(f"    Streams: {agg_metrics['n_streams']}")
            print(f"    Total GT reps: {total_true:.0f}, Pred reps: {total_pred:.0f}")
            print(f"    Exact count: {exact_count}/{len(all_stream_metrics)} ({agg_metrics['exact_count_ratio']:.1%})")
            print(f"    Over-segmented: {over_count}, Under-segmented: {under_count}")
            print(f"    Mean abs count diff: {agg_metrics['mean_abs_count_diff']:.2f}")
            print(f"    Rep F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            if start_mae_vals:
                print(f"    Start MAE: {agg_metrics['start_mae_ms']:.1f}ms, End MAE: {agg_metrics['end_mae_ms']:.1f}ms")
        else:
            agg_metrics = None
        
        # Concatenate reps with proper segment alignment
        concat_df, _, _, gt_rep_segments, pred_rep_segments = _concatenate_reps_with_segments(
            valid_streams, gt_segments_per_stream, pred_segments_per_stream
        )
        
        title = f"[RF] {model['name']} | {subject}/{session}/{action}/{set_name}"
        _plot_set_replay_clean(title, concat_df, gt_rep_segments, pred_rep_segments, metrics=agg_metrics, sample_rate=100.0)


if __name__ == "__main__":
    main()
