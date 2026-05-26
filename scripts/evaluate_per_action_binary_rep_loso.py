"""Per-Action LOSO evaluation of Binary Rep Detection (no phase classification).

Instead of classifying concentric/eccentric, this directly classifies
inside_rep vs outside_rep, then extracts rep boundaries from continuous
inside_rep segments.

Usage:
    python scripts/evaluate_per_action_binary_rep_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/per_action_binary_rep \
        --subjects haoyu,hsianshun,kevin
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")

from preprocessing.micro_macro_segments import (
    RepDetection, SegmentRun, labels_to_runs, match_segments, rep_metrics,
    truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz


BINARY_LABELS = ("outside_rep", "inside_rep")
OUTSIDE_REP = "outside_rep"
INSIDE_REP = "inside_rep"


def _phase_to_binary_label(phase_label: str, mode: str = "inside_rep") -> str:
    """Convert phase label to binary rep label."""
    if mode == "inside_rep":
        if phase_label in ("concentric", "eccentric"):
            return INSIDE_REP
        return OUTSIDE_REP
    elif mode == "concentric":
        if phase_label == "concentric":
            return "concentric"
        return "not_concentric"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _prepare_binary_labels(df: pd.DataFrame, mode: str = "inside_rep") -> np.ndarray:
    """Convert phase column to binary labels.
    
    mode="inside_rep": 1=concentric/eccentric, 0=other
    mode="concentric": 1=concentric, 0=eccentric/other
    """
    phases = df["phase"].to_numpy()
    if mode == "inside_rep":
        labels = np.array([
            1 if str(p) in ("concentric", "eccentric") else 0
            for p in phases
        ], dtype=np.int64)
    elif mode == "concentric":
        labels = np.array([
            1 if str(p) == "concentric" else 0
            for p in phases
        ], dtype=np.int64)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return labels


def _build_trailing_feature_matrix(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Copy from evaluate_causal_rf.py"""
    ends = np.arange(1, len(x) + 1, int(max(1, stride)), dtype=np.int64)
    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float32), ends
    window_size = int(max(1, window_size))
    prefix = np.repeat(x[:1], max(0, window_size - 1), axis=0)
    padded = np.concatenate([prefix, x], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    selected = windows[np.maximum(0, ends - 1)]
    return cb._extract_window_features_batch(selected), ends


def train_binary_causal_rf(
    train_streams,
    imu_columns: Sequence[str],
    window_size: int = 100,
    stride: int = 10,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    mode: str = "inside_rep",
) -> object:
    """Train binary RF."""
    X_all, y_all = [], []
    for stream_idx, (_, df) in enumerate(train_streams, start=1):
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        binary_labels = _prepare_binary_labels(df, mode=mode)
        X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(X_batch):
            X_all.append(X_batch)
            y_all.append(binary_labels[np.maximum(0, ends - 1)])
        if stream_idx % 25 == 0 or stream_idx == len(train_streams):
            print(f"  [BinaryRF] prepared {stream_idx}/{len(train_streams)} train streams", flush=True)
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    print(
        f"  [BinaryRF] Training on {len(X_all)} trailing windows "
        f"({window_size} samples, stride {stride}, trees={n_estimators}, max_depth={max_depth})"
    )
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples) if max_samples and max_samples < 1.0 else max_samples,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight="balanced",  # Handle class imbalance
    )
    clf.fit(X_all, y_all)
    clf.verbose = 0
    return clf


def predict_binary_causal_rf(
    clf,
    df,
    imu_columns: Sequence[str],
    window_size: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Predict inside_rep probability for each sample."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    # Binary probability: P(inside_rep) for each sample
    probs = np.zeros((n, 2), dtype=np.float32)
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), int(stride))
    raw_batch = clf.predict_proba(X_batch) if len(X_batch) else np.zeros((0, len(class_map)), dtype=np.float32)
    if len(raw_batch):
        full_batch = np.zeros((len(raw_batch), 2), dtype=np.float32)
        for cls_idx, mi in class_map.items():
            full_batch[:, cls_idx] = raw_batch[:, mi]
        probs[np.maximum(0, ends - 1)] = full_batch
    return probs


def _extract_reps_from_binary_probs_v2(
    df: pd.DataFrame,
    probs: np.ndarray,
    sample_rate: float,
    threshold: float = 0.3,
    min_set_duration_seconds: float = 3.0,
) -> List[RepDetection]:
    """Two-stage rep detection:
    
    Stage 1: Binary RF detects active set segments (inside_rep = set in progress)
    Stage 2: Peak Detection on acc_mag within each active segment to find rep boundaries
    
    Key insight: inside_rep is a continuous block containing all reps in a set.
    We find this block, then use signal processing to split it into individual reps.
    """
    from scipy.signal import find_peaks
    
    n = len(probs)
    inside_prob = probs[:, 1]  # P(inside_rep)
    
    # Stage 1: Find active set segments with low threshold + class_weight balanced
    binary_preds = (inside_prob >= threshold).astype(int)
    pred_labels = np.where(binary_preds == 1, INSIDE_REP, OUTSIDE_REP)
    
    runs = labels_to_runs(
        pred_labels,
        positive_labels=(INSIDE_REP,),
        probabilities=probs,
        min_length=1,
    )
    
    # Merge nearby runs (gaps up to 0.5s are likely noise between reps)
    max_merge_gap = int(0.5 * sample_rate)
    if len(runs) <= 1:
        merged = runs
    else:
        merged = []
        current = SegmentRun(runs[0].label, runs[0].start_idx, runs[0].end_idx, runs[0].confidence)
        for run in runs[1:]:
            gap = run.start_idx - current.end_idx
            if gap <= max_merge_gap:
                current.end_idx = run.end_idx
                current.confidence = max(current.confidence, run.confidence)
            else:
                merged.append(current)
                current = SegmentRun(run.label, run.start_idx, run.end_idx, run.confidence)
        merged.append(current)
    
    # Filter: a valid set segment must be reasonably long (at least a few reps)
    min_set_samples = int(min_set_duration_seconds * sample_rate)
    active_segments = [run for run in merged if run.end_idx - run.start_idx >= min_set_samples]
    
    if not active_segments:
        return []
    
    # Stage 2: Within each active segment, use Peak Detection on acc_mag to find reps
    reps = []
    acc_mag = np.sqrt(np.sum(df[["ax", "ay", "az"]].to_numpy() ** 2, axis=1))
    
    for segment in active_segments:
        seg_start, seg_end = segment.start_idx, segment.end_idx
        seg_mag = acc_mag[seg_start:seg_end]
        
        if len(seg_mag) < 10:
            continue
        
        # Smooth acc_mag slightly
        from scipy.ndimage import uniform_filter1d
        smooth_mag = uniform_filter1d(seg_mag, size=5)
        
        # Estimate rep interval from segment duration
        # Typical rep duration: 2-4s at 100Hz = 200-400 samples
        est_rep_duration = int(2.5 * sample_rate)  # ~250 samples
        min_distance = int(1.5 * sample_rate)  # at least 1.5s apart
        
        # Find peaks in smoothed acc_mag
        peaks, _ = find_peaks(
            smooth_mag,
            distance=min_distance,
            prominence=np.std(smooth_mag) * 0.3,
        )
        
        # Convert peaks to rep boundaries
        # Each peak is the center of a rep; rep spans roughly ±rep_duration/2 around peak
        half_rep = est_rep_duration // 2
        
        for peak in peaks:
            rep_start = max(0, peak - half_rep)
            rep_end = min(len(seg_mag), peak + half_rep)
            
            # Convert to absolute indices
            abs_start = seg_start + rep_start
            abs_end = seg_start + rep_end
            abs_transition = seg_start + peak
            
            reps.append(RepDetection(
                start_idx=abs_start,
                transition_idx=abs_transition,
                end_idx=abs_end,
                micro_source="binary_rf_peak",
                micro_confidence=segment.confidence,
            ))
    
    return reps


def _extract_reps_from_binary_probs_concentric(
    df: pd.DataFrame,
    probs: np.ndarray,
    sample_rate: float,
    threshold: float = 0.5,
    max_eccentric_duration_seconds: float = 3.0,
) -> List[RepDetection]:
    """Extract reps using SAME logic as phase-based RF.
    
    Binary model predicts: concentric(1) vs not_concentric(0)
    where not_concentric = eccentric + other.
    
    Rep extraction: find concentric runs, pair with following not_concentric run.
    If not_concentric run is too long (> max_eccentric_duration), it's probably
    "other" not "eccentric", so skip.
    """
    n = len(probs)
    concentric_prob = probs[:, 1]  # P(concentric)
    binary_preds = (concentric_prob >= threshold).astype(int)
    pred_labels = np.where(binary_preds == 1, "concentric", "not_concentric")
    
    # Find all runs (both concentric and not_concentric)
    runs = labels_to_runs(
        pred_labels,
        positive_labels=("concentric", "not_concentric"),
        probabilities=probs,
        min_length=1,
    )
    
    if not runs:
        return []
    
    max_eccentric_samples = int(max_eccentric_duration_seconds * sample_rate)
    reps = []
    
    for i, run in enumerate(runs):
        if run.label != "concentric":
            continue
        
        # Find the next run (should be not_concentric)
        if i + 1 >= len(runs):
            continue
        next_run = runs[i + 1]
        
        if next_run.label != "not_concentric":
            continue
        
        # Filter: if not_concentric is too long, it's probably "other" not "eccentric"
        next_duration = next_run.end_idx - next_run.start_idx
        if next_duration > max_eccentric_samples:
            continue
        
        # Valid rep: concentric run + following not_concentric run
        reps.append(RepDetection(
            start_idx=run.start_idx,
            transition_idx=next_run.start_idx,  # boundary = transition point
            end_idx=next_run.end_idx,
            micro_source="binary_concentric_rf",
            micro_confidence=run.confidence,
        ))
    
    return reps


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def _extract_subject_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def evaluate_stream_binary(
    df: pd.DataFrame,
    probs: np.ndarray,
    sample_rate: float,
    mode: str = "inside_rep",
) -> Dict:
    """Evaluate binary rep detection on a single stream."""
    if mode == "inside_rep":
        pred_reps = _extract_reps_from_binary_probs_v2(
            df, probs, sample_rate,
        )
    elif mode == "concentric":
        pred_reps = _extract_reps_from_binary_probs_concentric(
            df, probs, sample_rate,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    truth_reps = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
    )
    metrics = rep_metrics(pred_reps, truth_reps, sample_rate_hz=sample_rate)

    # Sample-level accuracy
    gt_binary = _prepare_binary_labels(df, mode=mode)
    if mode == "inside_rep":
        pred_binary = (probs[:, 1] >= 0.3).astype(int)
    else:
        pred_binary = (probs[:, 1] >= 0.5).astype(int)
    sample_acc = np.mean(gt_binary == pred_binary)

    return {
        **metrics,
        "binary_sample_accuracy": float(sample_acc),
        "stream_id": "unknown",
        "count_diff": metrics.get("n_pred", 0) - metrics.get("n_true", 0),
    }


def run_action_loso(
    action: str,
    all_streams: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    subjects: List[str],
    imu_columns: Sequence[str],
    window_size: int = 100,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
    mode: str = "inside_rep",
) -> Dict:
    """Run LOSO for a single action with binary rep detection."""
    mode_label = "Binary Rep Detection" if mode == "inside_rep" else "Binary Concentric Detector"
    print(f"\n{'='*60}")
    print(f"ACTION: {action} | {mode_label} (mode={mode})")
    print(f"{'='*60}")

    action_streams = [(sid, df) for sid, df in all_streams if _extract_action_from_stream_id(sid) == action]
    print(f"[INFO] Total {action} streams: {len(action_streams)}")

    if not action_streams:
        return {"action": action, "error": "No streams found"}

    action_subjects = sorted({_extract_subject_from_stream_id(sid) for sid, _ in action_streams})
    print(f"[INFO] Subjects with {action}: {action_subjects}")

    all_results = []
    for test_subject in subjects:
        if test_subject not in action_subjects:
            print(f"[Skip] {test_subject} has no {action} data")
            continue

        fold_file = output_dir / action / f"fold_{test_subject}.json"
        if fold_file.exists():
            print(f"[Resume] Loading {fold_file}")
            with open(fold_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        train_streams = [(sid, df) for sid, df in action_streams
                        if _extract_subject_from_stream_id(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in action_streams
                       if _extract_subject_from_stream_id(sid) == test_subject]

        if not test_streams:
            print(f"[Skip] No test streams for {test_subject}/{action}")
            continue

        print(f"\n[Fold] action={action} test={test_subject} train={len(train_streams)} test={len(test_streams)}")

        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

        # Train binary causal RF
        t0 = time.time()
        clf = train_binary_causal_rf(
            train_z, imu_columns,
            window_size=int(window_size), stride=int(train_stride),
            n_estimators=int(n_estimators), max_depth=int(max_depth), max_samples=float(max_samples),
            mode=mode,
        )
        train_time = time.time() - t0

        # Predict
        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
            probs = predict_binary_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 10 == 0 or stream_idx == len(test_z):
                print(f"  [BinaryRF] predicted {stream_idx}/{len(test_z)} test streams", flush=True)

        # Apply smoothing
        smooth_w = smoothing_window
        smoothed_streams = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smooth_w) > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smooth_w) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    count = i - start + 1
                    smoothed[i] = total / float(count)
                cur = smoothed
            smoothed_streams.append((stream_id, df, cur))

        # Evaluate each stream
        rows = []
        for stream_id, df, probs in smoothed_streams:
            sample_rate = infer_sample_rate_hz(df)
            row = evaluate_stream_binary(df, probs, sample_rate, mode=mode)
            row["stream_id"] = stream_id
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"  [BinaryRF] evaluated {len(rows)}/{len(smoothed_streams)}", flush=True)

        # Aggregate
        total_tp = sum(r["tp"] for r in rows)
        total_fp = sum(r["fp"] for r in rows)
        total_fn = sum(r["fn"] for r in rows)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0

        model_name = "Binary Rep Detection (Per-Action)" if mode == "inside_rep" else "Binary Concentric Detector (Per-Action)"
        results = {
            "model_name": model_name,
            "evaluation_protocol": "loso_per_action",
            "test_subject": test_subject,
            "action": action,
            "train_time_s": train_time,
            "smoothing_window": smooth_w,
            "config": {
                "window_size": int(window_size), "train_stride": int(train_stride),
                "smoothing_window": int(smooth_w), "n_estimators": int(n_estimators),
                "max_depth": int(max_depth), "max_samples": float(max_samples),
            },
            "stream_count": len(rows),
            "n_pred": sum(r["n_pred"] for r in rows),
            "n_true": sum(r["n_true"] for r in rows),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": p, "recall": r, "rep_f1": f1,
            "binary_sample_accuracy": float(np.mean([r["binary_sample_accuracy"] for r in rows])),
            "stream_rows": rows,
        }

        fold_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved: {fold_file} | Rep F1={f1:.4f}, Sample Acc={results['binary_sample_accuracy']:.4f}")
        all_results.append(results)

    if not all_results:
        return {"action": action, "error": "No results"}

    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    overall = {
        "n_folds": len(all_results), "subjects": subjects,
        "streams": sum(r.get("stream_count", 0) for r in all_results),
        "n_true": sum(r["n_true"] for r in all_results), "n_pred": sum(r["n_pred"] for r in all_results),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": p, "recall": r, "rep_f1": f1,
        "binary_sample_accuracy": float(np.mean([r["binary_sample_accuracy"] for r in all_results])),
    }

    summary = {"action": action, "overall": overall, "fold_results": all_results}

    with open(output_dir / action / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[Action {action} Summary] Rep F1={f1:.4f}, Precision={p:.4f}, Recall={r:.4f}")
    return summary


def run_all_actions(
    config_path: Path,
    output_dir: Path,
    subjects: List[str] | None = None,
    actions: List[str] | None = None,
    window_size: int = 100,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
    mode: str = "inside_rep",
) -> Dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))

    print("[INFO] Loading all streams...")
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)

    if subjects is None:
        subjects = sorted(set(all_subjects))
    if actions is None:
        actions = available_actions

    print(f"[INFO] Subjects: {subjects}")
    print(f"[INFO] Actions: {actions}")
    print(f"[INFO] Total streams: {len(all_streams)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    action_results = {}
    for action in actions:
        action_summary = run_action_loso(
            action=action,
            all_streams=all_streams,
            output_dir=output_dir,
            subjects=subjects,
            imu_columns=imu_columns,
            window_size=window_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            train_stride=train_stride,
            smoothing_window=smoothing_window,
            mode=mode,
        )
        action_results[action] = action_summary

        completed = sum(1 for a in actions if a in action_results)
        f1s = [r["overall"]["rep_f1"] for r in action_results.values() if "overall" in r and "rep_f1" in r["overall"]]
        if f1s:
            print(f"\n[Progress] {completed}/{len(actions)} actions completed")
            print(f"  Mean Rep F1 across actions: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    all_folds = []
    for action, summary in action_results.items():
        if "fold_results" in summary:
            all_folds.extend(summary["fold_results"])

    if all_folds:
        total_tp = sum(r["tp"] for r in all_folds)
        total_fp = sum(r["fp"] for r in all_folds)
        total_fn = sum(r["fn"] for r in all_folds)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0

        model_label = "Binary Rep Detection (Per-Action LOSO)" if mode == "inside_rep" else "Binary Concentric Detector (Per-Action LOSO)"
        grand_summary = {
            "model": model_label,
            "overall": {
                "n_folds": len(all_folds), "actions": actions, "subjects": subjects,
                "streams": sum(r.get("stream_count", 0) for r in all_folds),
                "n_true": sum(r["n_true"] for r in all_folds), "n_pred": sum(r["n_pred"] for r in all_folds),
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": p, "recall": r, "rep_f1": f1,
                "binary_sample_accuracy": float(np.mean([r["binary_sample_accuracy"] for r in all_folds])),
            },
            "by_action": {action: res.get("overall", {}) for action, res in action_results.items()},
        }

        with open(output_dir / "grand_summary.json", "w", encoding="utf-8") as f:
            json.dump(grand_summary, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("GRAND SUMMARY (All Actions)")
        print(f"{'='*60}")
        print(json.dumps(grand_summary["overall"], indent=2))
        print(f"\n[OK] Results saved to {output_dir}")
        return grand_summary

    return {}


def main():
    parser = argparse.ArgumentParser(description="Per-Action LOSO with Binary Rep Detection")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/per_action_binary_rep")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--binary-mode", choices=["inside_rep", "concentric"], default="inside_rep",
                        help="Binary mode: inside_rep=detect active set, concentric=detect concentric phase")
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    actions = [a.strip() for a in args.actions.split(",") if a.strip()] if args.actions else None

    run_all_actions(
        Path(args.config), Path(args.output),
        subjects=subjects, actions=actions,
        window_size=args.window_size,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        max_samples=args.max_samples, train_stride=args.train_stride,
        smoothing_window=args.smoothing_window,
        mode=args.binary_mode,
    )


if __name__ == "__main__":
    main()
