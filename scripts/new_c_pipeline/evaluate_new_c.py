"""
New C Pipeline: Full Evaluation (Baseline A vs New C)

Baseline A: 3-class RF (other/concentric/eccentric) + rep parser
New C: Independent Active Detector + 2-class Phase Segmentation + rep parser

Evaluates both pipelines on identical LOSO splits for fair comparison.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, MICRO_LABELS, OTHER_LABEL,
    micro_labels_from_phase, pair_concentric_eccentric_reps, RepDetection,
    SegmentRun, labels_to_runs, truth_reps_from_labels,
)
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.micro_macro_recognition import _load_streams

import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NewCConfig:
    # Active Detector
    active_window_size: int = 100  # 1.0s @100Hz
    active_stride: int = 25  # 0.25s
    active_model: str = "rf"
    # State Control
    min_active_windows: int = 3  # N consecutive active to enter
    min_not_active_windows: int = 3  # M consecutive not_active to exit
    active_threshold: float = 0.5
    # Phase Segmentation
    phase_window_size: int = 100  # 1.0s
    phase_stride: int = 10  # 0.1s
    phase_model: str = "rf"
    # Smoothing
    smoothing_window: int = 15  # samples
    min_phase_samples: int = 5
    # Rep Parser
    max_gap_samples: int = 3
    # Feature columns
    imu_columns: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")


# ---------------------------------------------------------------------------
# Feature Extraction (shared)
# ---------------------------------------------------------------------------

def _extract_window_features(windows: np.ndarray) -> np.ndarray:
    """Extract rich features from windows [N, T, C]."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    window_len = int(arr.shape[1])
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    vmin = np.min(arr, axis=1)
    vmax = np.max(arr, axis=1)
    median = np.median(arr, axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    total_variation = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    
    # Magnitude
    mag = np.sqrt(np.sum(arr ** 2, axis=2))
    mag_stats = np.stack([
        np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1)
    ], axis=1)
    
    per_channel = np.stack([mean, std, vmin, vmax, median, q25, q75, total_variation], axis=-1).reshape(arr.shape[0], -1)
    return np.concatenate([per_channel, mag_stats], axis=1).astype(np.float32, copy=False)


def _build_windows(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows from sequence [T, C]."""
    n = len(x)
    if n <= 0:
        empty = np.zeros((0,), dtype=np.int64)
        return np.zeros((0, 0), dtype=np.float32), empty, empty
    window_size = int(max(1, window_size))
    stride = int(max(1, stride))
    if n < window_size:
        starts = np.asarray([0], dtype=np.int64)
    else:
        starts = np.arange(0, n - window_size + 1, stride, dtype=np.int64)
    pad = max(0, window_size - n)
    padded = np.pad(x, ((0, pad), (0, 0)), mode="edge") if pad > 0 else x
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    selected = windows[starts]
    ends = np.minimum(starts + window_size, n)
    return _extract_window_features(selected), starts, ends


# ---------------------------------------------------------------------------
# Label Preparation
# ---------------------------------------------------------------------------

def _prepare_active_labels(phases: np.ndarray) -> np.ndarray:
    """Convert phase labels to binary active (1) / not_active (0)."""
    labels = micro_labels_from_phase(phases)
    return np.array([1 if str(l) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL} else 0 for l in labels], dtype=np.int64)


def _prepare_phase_labels(phases: np.ndarray) -> np.ndarray:
    """Convert phase labels to concentric (1) / eccentric (0). Other samples get -1 (skip)."""
    labels = micro_labels_from_phase(phases)
    result = np.full(len(labels), -1, dtype=np.int64)
    for i, l in enumerate(labels):
        if str(l) == CONCENTRIC_LABEL:
            result[i] = 1
        elif str(l) == ECCENTRIC_LABEL:
            result[i] = 0
    return result


# ---------------------------------------------------------------------------
# Baseline A: 3-class RF (other / concentric / eccentric)
# ---------------------------------------------------------------------------

def train_baseline_a(train_streams, cfg: NewCConfig):
    """Train 3-class RF for Baseline A."""
    X_all, y_all = [], []
    for _, df in train_streams:
        if "phase" not in df.columns:
            continue
        x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
        labels = micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        
        X_batch, starts, ends = _build_windows(x, cfg.active_window_size, cfg.active_stride)
        if len(X_batch) == 0:
            continue
        
        y_batch = []
        for start, end in zip(starts, ends):
            window_labels = label_idx[int(start):int(end)]
            y_batch.append(int(np.bincount(window_labels, minlength=len(MICRO_LABELS)).argmax()))
        
        X_all.append(X_batch)
        y_all.append(np.asarray(y_batch, dtype=np.int64))
    
    if not X_all:
        return None, None
    
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_s, y)
    return clf, scaler


def predict_baseline_a(clf, scaler, df, cfg: NewCConfig):
    """Predict per-sample probabilities for Baseline A."""
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    X_batch, starts, ends = _build_windows(x, cfg.active_window_size, cfg.active_stride)
    if len(X_batch) == 0:
        return np.zeros((n, len(MICRO_LABELS)))
    
    X_s = scaler.transform(X_batch)
    probs = clf.predict_proba(X_s)
    
    # Map classes to MICRO_LABELS order
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    prob_accum = np.zeros((n, len(MICRO_LABELS)), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    
    full_batch = np.zeros((len(probs), len(MICRO_LABELS)), dtype=np.float64)
    for cls_idx, mi in class_map.items():
        full_batch[:, cls_idx] = probs[:, mi]
    
    for wi, (start, end) in enumerate(zip(starts, ends)):
        prob_accum[int(start):int(end)] += full_batch[wi]
        counts[int(start):int(end)] += 1.0
    
    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts[:, None]


# ---------------------------------------------------------------------------
# New C: Step 1 - Active Detector
# ---------------------------------------------------------------------------

def train_active_detector(train_streams, cfg: NewCConfig):
    """Train binary active detector."""
    X_all, y_all = [], []
    for _, df in train_streams:
        if "phase" not in df.columns:
            continue
        x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
        active_labels = _prepare_active_labels(df["phase"].to_numpy())
        
        X_batch, starts, ends = _build_windows(x, cfg.active_window_size, cfg.active_stride)
        if len(X_batch) == 0:
            continue
        
        y_batch = []
        for start, end in zip(starts, ends):
            window_labels = active_labels[int(start):int(end)]
            y_batch.append(int(np.bincount(window_labels).argmax()))
        
        X_all.append(X_batch)
        y_all.append(np.asarray(y_batch, dtype=np.int64))
    
    if not X_all:
        return None, None
    
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_s, y)
    return clf, scaler


def predict_active(clf, scaler, df, cfg: NewCConfig):
    """Predict per-sample active probability."""
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    X_batch, starts, ends = _build_windows(x, cfg.active_window_size, cfg.active_stride)
    if len(X_batch) == 0:
        return np.zeros(n)
    
    X_s = scaler.transform(X_batch)
    probs = clf.predict_proba(X_s)
    
    # Find active class index (should be 1)
    active_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
    
    prob_accum = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    
    for wi, (start, end) in enumerate(zip(starts, ends)):
        prob_accum[int(start):int(end)] += probs[wi, active_idx]
        counts[int(start):int(end)] += 1.0
    
    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts


def apply_state_control(active_probs, cfg: NewCConfig):
    """Apply hysteresis state control to active probabilities.
    
    Returns:
        active_segments: list of (start, end) tuples
    """
    n = len(active_probs)
    state = "IDLE"  # IDLE or ACTIVE
    consecutive_active = 0
    consecutive_inactive = 0
    active_segments = []
    current_start = None
    
    for i in range(n):
        is_active = active_probs[i] >= cfg.active_threshold
        
        if state == "IDLE":
            if is_active:
                consecutive_active += 1
                consecutive_inactive = 0
                if consecutive_active >= cfg.min_active_windows:
                    state = "ACTIVE"
                    current_start = i - cfg.min_active_windows + 1
            else:
                consecutive_active = 0
        else:  # ACTIVE
            if not is_active:
                consecutive_inactive += 1
                consecutive_active = 0
                if consecutive_inactive >= cfg.min_not_active_windows:
                    state = "IDLE"
                    end = i - cfg.min_not_active_windows + 1
                    if current_start is not None and end > current_start:
                        active_segments.append((current_start, end))
                    current_start = None
            else:
                consecutive_inactive = 0
    
    # Handle trailing ACTIVE state
    if state == "ACTIVE" and current_start is not None:
        active_segments.append((current_start, n))
    
    return active_segments


# ---------------------------------------------------------------------------
# New C: Step 2 - 2-Class Phase Segmentation (concentric/eccentric)
# ---------------------------------------------------------------------------

def train_phase_segmenter(train_streams, cfg: NewCConfig):
    """Train 2-class phase segmenter (concentric/eccentric) using only active samples."""
    X_all, y_all = [], []
    for _, df in train_streams:
        if "phase" not in df.columns:
            continue
        x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
        phase_labels = _prepare_phase_labels(df["phase"].to_numpy())
        
        # Only train on active samples (where phase_labels != -1)
        active_mask = phase_labels >= 0
        if not active_mask.any():
            continue
        
        # Build windows
        X_batch, starts, ends = _build_windows(x, cfg.phase_window_size, cfg.phase_stride)
        if len(X_batch) == 0:
            continue
        
        y_batch = []
        valid = []
        for wi, (start, end) in enumerate(zip(starts, ends)):
            window_labels = phase_labels[int(start):int(end)]
            # Only use window if majority are active and have valid labels
            valid_labels = window_labels[window_labels >= 0]
            if len(valid_labels) > 0 and len(valid_labels) / len(window_labels) > 0.5:
                y_batch.append(int(np.bincount(valid_labels).argmax()))
                valid.append(wi)
        
        if valid:
            X_all.append(X_batch[valid])
            y_all.append(np.asarray(y_batch, dtype=np.int64))
    
    if not X_all:
        return None, None
    
    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_s, y)
    return clf, scaler


def predict_phase(clf, scaler, df, active_segments, cfg: NewCConfig):
    """Predict phase (concentric/eccentric) within active segments only."""
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    
    # Default: uncertain (0.5/0.5)
    phase_probs = np.ones((n, 2)) * 0.5
    
    for seg_start, seg_end in active_segments:
        seg_df = df.iloc[seg_start:seg_end].reset_index(drop=True)
        if len(seg_df) == 0:
            continue
        seg_x = seg_df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
        
        X_batch, starts, ends = _build_windows(seg_x, cfg.phase_window_size, cfg.phase_stride)
        if len(X_batch) == 0:
            continue
        
        X_s = scaler.transform(X_batch)
        probs = clf.predict_proba(X_s)
        
        # Map classes: 0=eccentric, 1=concentric
        class_map = {int(c): i for i, c in enumerate(clf.classes_)}
        full_batch = np.zeros((len(probs), 2), dtype=np.float64)
        for cls_idx, mi in class_map.items():
            if cls_idx < 2:
                full_batch[:, cls_idx] = probs[:, mi]
        
        # Accumulate back to per-sample
        prob_accum = np.zeros((seg_end - seg_start, 2), dtype=np.float64)
        counts = np.zeros(seg_end - seg_start, dtype=np.float64)
        
        for wi, (start, end) in enumerate(zip(starts, ends)):
            prob_accum[int(start):int(end)] += full_batch[wi]
            counts[int(start):int(end)] += 1.0
        
        counts = np.where(counts < 1e-8, 1.0, counts)
        phase_probs[seg_start:seg_end] = prob_accum / counts[:, None]
    
    return phase_probs


def smooth_phase(phase_probs, cfg: NewCConfig):
    """Apply smoothing to phase probabilities."""
    n = len(phase_probs)
    smoothed = np.copy(phase_probs)
    
    # Moving average
    if cfg.smoothing_window > 1:
        window = cfg.smoothing_window
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)
    
    # Minimum phase duration constraint
    pred_labels = np.argmax(smoothed, axis=1)
    runs = labels_to_runs(["eccentric" if p == 0 else "concentric" for p in pred_labels], positive_labels={"eccentric", "concentric"}, min_length=1)
    
    # Filter out runs shorter than min_phase_samples
    filtered = np.copy(pred_labels)
    for run in runs:
        if run.end_idx - run.start_idx < cfg.min_phase_samples:
            # Replace with neighboring label
            before = filtered[run.start_idx - 1] if run.start_idx > 0 else 1 - run.label
            filtered[run.start_idx:run.end_idx] = before
    
    # Convert back to one-hot
    result = np.zeros((n, 2))
    result[np.arange(n), filtered] = 1.0
    return result


# ---------------------------------------------------------------------------
# Rep Parser
# ---------------------------------------------------------------------------

def _merge_adjacent_same_phase(runs):
    """Merge adjacent runs with the same phase label."""
    if not runs:
        return runs
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            # Extend the previous run
            merged[-1] = SegmentRun(
                label=run.label,
                start_idx=merged[-1].start_idx,
                end_idx=run.end_idx,
                confidence=(merged[-1].confidence + run.confidence) / 2,
            )
        else:
            merged.append(run)
    return merged


def _filter_short_runs(runs, min_samples):
    """Remove runs shorter than min_samples, merging them into adjacent runs."""
    if not runs:
        return runs
    filtered = []
    for i, run in enumerate(runs):
        duration = run.end_idx - run.start_idx
        if duration >= min_samples:
            filtered.append(run)
        else:
            # Short run: merge into longer neighbor
            left_dur = runs[i-1].end_idx - runs[i-1].start_idx if i > 0 else 0
            right_dur = runs[i+1].end_idx - runs[i+1].start_idx if i < len(runs) - 1 else 0
            if left_dur >= right_dur and i > 0:
                # Merge into left
                filtered[-1] = SegmentRun(
                    label=filtered[-1].label,
                    start_idx=filtered[-1].start_idx,
                    end_idx=run.end_idx,
                    confidence=filtered[-1].confidence,
                )
            elif i < len(runs) - 1:
                # Will merge into right (mark by not adding, right will handle)
                pass
            elif filtered:
                # Last run, merge into left
                filtered[-1] = SegmentRun(
                    label=filtered[-1].label,
                    start_idx=filtered[-1].start_idx,
                    end_idx=run.end_idx,
                    confidence=filtered[-1].confidence,
                )
    return filtered


def _enforce_alternation(runs):
    """Enforce strict C/E alternation by merging consecutive same phases."""
    if not runs:
        return runs
    # First merge adjacent same phases
    merged = _merge_adjacent_same_phase(runs)
    return merged


def parse_reps_from_phase(phase_probs, cfg: NewCConfig):
    """Parse reps from smoothed phase probabilities with improved post-processing."""
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    
    # Convert to runs
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=1)
    
    # Post-processing pipeline
    # 1. Merge adjacent same-phase runs
    runs = _enforce_alternation(runs)
    
    # 2. Filter short runs (minimum phase duration: 0.3s = 30 samples @100Hz)
    min_phase_dur = max(cfg.min_phase_samples, 30)  # at least 0.3s
    runs = _filter_short_runs(runs, min_phase_dur)
    
    # 3. Re-check alternation after filtering
    runs = _enforce_alternation(runs)
    
    # Pair concentric/eccentric into reps
    reps, diagnostics = pair_concentric_eccentric_reps(runs, micro_source="new_c", max_gap_samples=cfg.max_gap_samples)
    return reps


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_reps(pred_reps, gt_reps):
    """Evaluate rep detection."""
    pred_count = len(pred_reps)
    gt_count = len(gt_reps)
    
    # Match reps by IoU
    tp = 0
    matched_gt = set()
    for pred in pred_reps:
        best_iou = 0
        best_gt = None
        for gi, gt in enumerate(gt_reps):
            if gi in matched_gt:
                continue
            # Compute IoU
            pred_range = set(range(pred.start_idx, pred.end_idx))
            gt_range = set(range(gt.start_idx, gt.end_idx))
            intersection = len(pred_range & gt_range)
            union = len(pred_range | gt_range)
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= 0.5 and best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)
    
    fp = pred_count - tp
    fn = gt_count - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    exact_count = 1 if pred_count == gt_count else 0
    over = 1 if pred_count > gt_count else 0
    under = 1 if pred_count < gt_count else 0
    
    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_count": exact_count,
        "over": over,
        "under": under,
    }


def evaluate_phase(phase_probs, gt_phases, active_segments):
    """Evaluate phase segmentation within active segments only."""
    gt_labels = _prepare_phase_labels(gt_phases)
    pred_labels = np.argmax(phase_probs, axis=1)
    
    # Only evaluate on active samples
    valid = gt_labels >= 0
    if not valid.any():
        return {"accuracy": 0, "macro_f1": 0}
    
    acc = accuracy_score(gt_labels[valid], pred_labels[valid])
    macro_f1 = f1_score(gt_labels[valid], pred_labels[valid], average="macro", zero_division=0)
    
    # Transition MAE
    gt_changes = np.where(np.diff(gt_labels[valid]) != 0)[0]
    pred_changes = np.where(np.diff(pred_labels[valid]) != 0)[0]
    
    mae = None
    if len(gt_changes) > 0 and len(pred_changes) > 0:
        # Match nearest predicted transition to each GT transition
        errors = []
        for gt_c in gt_changes:
            nearest_error = min(abs(gt_c - pc) for pc in pred_changes)
            errors.append(nearest_error)
        mae = np.mean(errors)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "transition_mae_samples": mae,
        "transition_mae_ms": mae * 10 if mae is not None else None,
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_new_c_pipeline(train_streams, test_streams, cfg: NewCConfig):
    """Run full New C pipeline on test streams."""
    print("\n  [New C] Training Active Detector...")
    active_clf, active_scaler = train_active_detector(train_streams, cfg)
    
    print("  [New C] Training Phase Segmenter...")
    phase_clf, phase_scaler = train_phase_segmenter(train_streams, cfg)
    
    if active_clf is None or phase_clf is None:
        return None
    
    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        
        # Active Detection
        active_probs = predict_active(active_clf, active_scaler, df, cfg)
        active_segments = apply_state_control(active_probs, cfg)
        
        # Phase Segmentation
        phase_probs = predict_phase(phase_clf, phase_scaler, df, active_segments, cfg)
        phase_probs_smooth = smooth_phase(phase_probs, cfg)
        
        # Rep Parser
        pred_reps = parse_reps_from_phase(phase_probs_smooth, cfg)
        gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)
        
        # Evaluation
        rep_metrics = evaluate_reps(pred_reps, gt_reps)
        phase_metrics = evaluate_phase(phase_probs_smooth, df["phase"].to_numpy(), active_segments)
        
        all_results.append({
            "stream_id": stream_id,
            **rep_metrics,
            **phase_metrics,
        })
    
    return all_results


def run_baseline_a(train_streams, test_streams, cfg: NewCConfig):
    """Run Baseline A (3-class RF)."""
    print("\n  [Baseline A] Training 3-class RF...")
    clf, scaler = train_baseline_a(train_streams, cfg)
    
    if clf is None:
        return None
    
    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        
        # Predict 3-class probabilities
        probs = predict_baseline_a(clf, scaler, df, cfg)
        pred_labels = np.argmax(probs, axis=1)
        pred_phase = np.array([MICRO_LABELS[i] for i in pred_labels])
        
        # Smooth
        if cfg.smoothing_window > 1:
            smoothed = np.zeros_like(probs)
            for c in range(len(MICRO_LABELS)):
                cumsum = np.cumsum(probs[:, c])
                for i in range(len(probs)):
                    start = max(0, i - cfg.smoothing_window + 1)
                    total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                    smoothed[i, c] = total / (i - start + 1)
            pred_labels = np.argmax(smoothed, axis=1)
            pred_phase = np.array([MICRO_LABELS[i] for i in pred_labels])
        
        # Parse reps
        runs = labels_to_runs(pred_phase, positive_labels={"concentric", "eccentric"}, min_length=cfg.min_phase_samples)
        pred_reps, _ = pair_concentric_eccentric_reps(runs, micro_source="baseline_a", max_gap_samples=cfg.max_gap_samples)
        gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)
        
        # Evaluate
        rep_metrics = evaluate_reps(pred_reps, gt_reps)
        
        # Phase metrics (only on active samples)
        gt_labels = _prepare_phase_labels(df["phase"].to_numpy())
        valid = gt_labels >= 0
        if valid.any():
            # Map baseline predictions to C/E (ignore other)
            pred_ce = np.where(pred_labels == MICRO_LABELS.index(CONCENTRIC_LABEL), 1,
                             np.where(pred_labels == MICRO_LABELS.index(ECCENTRIC_LABEL), 0, -1))
            valid2 = valid & (pred_ce >= 0)
            if valid2.any():
                acc = accuracy_score(gt_labels[valid2], pred_ce[valid2])
                macro_f1 = f1_score(gt_labels[valid2], pred_ce[valid2], average="macro", zero_division=0)
            else:
                acc, macro_f1 = 0, 0
        else:
            acc, macro_f1 = 0, 0
        
        all_results.append({
            "stream_id": stream_id,
            **rep_metrics,
            "accuracy": acc,
            "macro_f1": macro_f1,
        })
    
    return all_results


def aggregate_results(results):
    """Aggregate results across streams."""
    if not results:
        return {}
    
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "streams": len(results),
        "rep_precision": precision,
        "rep_recall": recall,
        "rep_f1": f1,
        "exact_count_acc": sum(r["exact_count"] for r in results) / len(results),
        "over_count": sum(r["over"] for r in results),
        "under_count": sum(r["under"] for r in results),
        "phase_macro_f1": np.mean([r.get("macro_f1", 0) for r in results]),
        "phase_accuracy": np.mean([r.get("accuracy", 0) for r in results]),
        "transition_mae_ms": np.mean([r.get("transition_mae_ms", 0) for r in results if r.get("transition_mae_ms") is not None]),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="New C Pipeline Evaluation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/new_c_pipeline/evaluation"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: kevin only")
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = NewCConfig()
    
    print("="*70)
    print("New C Pipeline: Baseline A vs New C Comparison")
    print("="*70)
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {})
    cfg.imu_columns = tuple(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    print("\n[1/4] Loading streams...")
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"      Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    
    if args.quick:
        test_subjects = ["kevin"]
        print(f"[QUICK] Testing on: {test_subjects}")
    else:
        test_subjects = subjects
    
    baseline_results = []
    new_c_results = []
    
    for test_subject in test_subjects:
        print(f"\n{'='*70}")
        print(f"Fold: test={test_subject}")
        print(f"{'='*70}")
        
        train_streams = [(sid, df) for sid, df in all_streams 
                        if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams 
                       if sid.startswith(f"{test_subject}/")]
        
        print(f"[2/4] train={len(train_streams)}, test={len(test_streams)}")
        
        # Baseline A
        print("\n[3/4] Running Baseline A...")
        baseline_fold = run_baseline_a(train_streams, test_streams, cfg)
        if baseline_fold:
            baseline_results.extend(baseline_fold)
            agg = aggregate_results(baseline_fold)
            print(f"      Baseline A: Rep F1={agg['rep_f1']:.4f}, Phase F1={agg['phase_macro_f1']:.4f}")
        
        # New C
        print("\n[4/4] Running New C...")
        new_c_fold = run_new_c_pipeline(train_streams, test_streams, cfg)
        if new_c_fold:
            new_c_results.extend(new_c_fold)
            agg = aggregate_results(new_c_fold)
            print(f"      New C: Rep F1={agg['rep_f1']:.4f}, Phase F1={agg['phase_macro_f1']:.4f}")
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    if baseline_results:
        baseline_agg = aggregate_results(baseline_results)
        print(f"\nBaseline A (3-class RF):")
        for k, v in baseline_agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    if new_c_results:
        new_c_agg = aggregate_results(new_c_results)
        print(f"\nNew C (Independent Active + 2-class Phase):")
        for k, v in new_c_agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Save results
    summary = {
        "baseline_a": aggregate_results(baseline_results) if baseline_results else {},
        "new_c": aggregate_results(new_c_results) if new_c_results else {},
    }
    output_file = args.output / f"comparison_{'quick' if args.quick else 'full'}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
