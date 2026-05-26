"""
New C Pipeline v2: Full Per-Action + Rich Features + Causal Window

Matches Baseline A (Per-Action Plain RF) settings:
- Per-Action training (independent model per action)
- Rich features (stats + velocity + jerk)
- Causal sliding window (1.0s window, stride=1 for inference, stride=10 for training)
- Proper post-processing (smoothing_window=15, min_phase_samples=3)

Architecture:
Raw IMU
  → Per-Action Independent Active Detector (rich features, causal window)
  → Active Segment State Control (hysteresis)
  → Per-Action 2-Class Phase Segmentation (concentric/eccentric, rich features, causal window)
  → Phase Smoothing (MA + min_phase_duration)
  → Rep Parser / State Machine
  → Rep Count + Rep Boundary + C/E Intervals
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.new_c_pipeline.tcn_phase_segmenter import (
    train_tcn_phase_segmenter,
    predict_tcn_phase,
    extract_active_segments_for_tcn,
)
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
    active_train_stride: int = 10
    active_infer_stride: int = 1
    active_model: str = "rf"
    active_n_estimators: int = 100
    active_max_depth: int = 15
    active_max_samples: float = 0.7
    # State Control
    min_active_windows: int = 3
    min_not_active_windows: int = 3
    active_threshold: float = 0.5
    # Phase Segmentation
    phase_window_size: int = 100  # 1.0s
    phase_train_stride: int = 10
    phase_infer_stride: int = 1
    phase_n_estimators: int = 100
    phase_max_depth: int = 15
    phase_max_samples: float = 0.7
    # Smoothing
    smoothing_window: int = 15
    min_phase_samples: int = 3
    max_phase_gap_samples: int = 3
    # Feature columns
    imu_columns: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
    use_velocity: bool = True
    use_jerk: bool = False


# ---------------------------------------------------------------------------
# Rich Feature Extraction (matching Baseline A)
# ---------------------------------------------------------------------------

def _extract_window_features_batch_base(windows: np.ndarray) -> np.ndarray:
    """Extract baseline statistical features from windows [N, T, C]."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    window_len = int(arr.shape[1])
    norm = float(max(1, window_len - 1))
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    vmin = np.min(arr, axis=1)
    vmax = np.max(arr, axis=1)
    median = np.median(arr, axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    argmax = np.argmax(arr, axis=1).astype(np.float32) / norm
    argmin = np.argmin(arr, axis=1).astype(np.float32) / norm
    total_variation = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    per_channel = np.stack(
        [mean, std, vmin, vmax, median, q25, q75, argmax, argmin, total_variation],
        axis=-1,
    ).reshape(arr.shape[0], -1)

    mag = np.sqrt(np.sum(arr ** 2, axis=2))
    mag_stats = np.stack([np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1)], axis=1)
    return np.concatenate([per_channel, mag_stats], axis=1).astype(np.float32, copy=False)


def _extract_window_features_batch_enhanced(windows: np.ndarray, use_velocity: bool = False, use_jerk: bool = False) -> np.ndarray:
    """Extract statistical features with optional velocity/jerk [N, T, C]."""
    base = _extract_window_features_batch_base(windows)
    features = [base]

    if use_velocity:
        vel = np.diff(windows, axis=1)
        if vel.shape[1] > 0:
            features.append(_extract_window_features_batch_base(vel))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))

    if use_jerk:
        vel = np.diff(windows, axis=1)
        if vel.shape[1] > 1:
            jerk = np.diff(vel, axis=1)
            features.append(_extract_window_features_batch_base(jerk))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _build_start_window_matrix(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal sliding windows from sequence [T, C]."""
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
    return selected, starts, ends


def _extract_features(x: np.ndarray, window_size: int, stride: int, cfg: NewCConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract rich features using causal window."""
    windows, starts, ends = _build_start_window_matrix(x, window_size, stride)
    if len(windows) == 0:
        return np.zeros((0, 0), dtype=np.float32), starts, ends
    features = _extract_window_features_batch_enhanced(windows, cfg.use_velocity, cfg.use_jerk)
    # Flatten features for sklearn
    features = features.reshape(len(features), -1)
    return features, starts, ends


# ---------------------------------------------------------------------------
# Label Preparation
# ---------------------------------------------------------------------------

def _prepare_active_labels(phases: np.ndarray) -> np.ndarray:
    """Convert phase labels to binary active (1) / not_active (0)."""
    labels = micro_labels_from_phase(phases)
    return np.array([1 if str(l) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL} else 0 for l in labels], dtype=np.int64)


def _prepare_phase_labels(phases: np.ndarray) -> np.ndarray:
    """Convert phase labels to concentric (1) / eccentric (0). Other → -1 (skip)."""
    labels = micro_labels_from_phase(phases)
    result = np.full(len(labels), -1, dtype=np.int64)
    for i, l in enumerate(labels):
        if str(l) == CONCENTRIC_LABEL:
            result[i] = 1
        elif str(l) == ECCENTRIC_LABEL:
            result[i] = 0
    return result


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


# ---------------------------------------------------------------------------
# Per-Action Active Detector
# ---------------------------------------------------------------------------

def train_active_detector_per_action(train_streams, cfg: NewCConfig):
    """Train per-action binary active detector."""
    models = {}
    scalers = {}

    # Group streams by action
    action_streams = {}
    for stream_id, df in train_streams:
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_streams:
            action_streams[action] = []
        action_streams[action].append((stream_id, df))

    for action, streams in action_streams.items():
        X_all, y_all = [], []
        for _, df in streams:
            if "phase" not in df.columns:
                continue
            x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
            active_labels = _prepare_active_labels(df["phase"].to_numpy())

            features, starts, ends = _extract_features(x, cfg.active_window_size, cfg.active_train_stride, cfg)
            if len(features) == 0:
                continue

            y_batch = []
            for start, end in zip(starts, ends):
                window_labels = active_labels[int(start):int(end)]
                y_batch.append(int(np.bincount(window_labels).argmax()))

            X_all.append(features)
            y_all.append(np.asarray(y_batch, dtype=np.int64))

        if not X_all:
            continue

        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=cfg.active_n_estimators,
            max_depth=cfg.active_max_depth,
            max_samples=cfg.active_max_samples,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_s, y)
        models[action] = clf
        scalers[action] = scaler

    return models, scalers


def predict_active_per_action(models, scalers, stream_id, df, cfg: NewCConfig):
    """Predict per-sample active probability using per-action model."""
    action = _extract_action_from_stream_id(stream_id)
    if action not in models:
        action = list(models.keys())[0]  # fallback

    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)

    features, starts, ends = _extract_features(x, cfg.active_window_size, cfg.active_infer_stride, cfg)
    if len(features) == 0:
        return np.zeros(n)

    X_s = scalers[action].transform(features)
    probs = models[action].predict_proba(X_s)

    active_idx = list(models[action].classes_).index(1) if 1 in models[action].classes_ else 0

    prob_accum = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for wi, (start, end) in enumerate(zip(starts, ends)):
        prob_accum[int(start):int(end)] += probs[wi, active_idx]
        counts[int(start):int(end)] += 1.0

    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts


# ---------------------------------------------------------------------------
# Active Segment State Control
# ---------------------------------------------------------------------------

def apply_state_control(active_probs, cfg: NewCConfig):
    """Apply hysteresis state control."""
    n = len(active_probs)
    state = "IDLE"
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
        else:
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

    if state == "ACTIVE" and current_start is not None:
        active_segments.append((current_start, n))

    return active_segments


# ---------------------------------------------------------------------------
# Per-Action 2-Class Phase Segmentation
# ---------------------------------------------------------------------------

def train_phase_segmenter_per_action(train_streams, cfg: NewCConfig):
    """Train per-action 2-class phase segmenter (concentric/eccentric)."""
    models = {}
    scalers = {}

    action_streams = {}
    for stream_id, df in train_streams:
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_streams:
            action_streams[action] = []
        action_streams[action].append((stream_id, df))

    for action, streams in action_streams.items():
        X_all, y_all = [], []
        for _, df in streams:
            if "phase" not in df.columns:
                continue
            x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
            phase_labels = _prepare_phase_labels(df["phase"].to_numpy())

            # Only train on active samples
            active_mask = phase_labels >= 0
            if not active_mask.any():
                continue

            features, starts, ends = _extract_features(x, cfg.phase_window_size, cfg.phase_train_stride, cfg)
            if len(features) == 0:
                continue

            y_batch = []
            valid = []
            for wi, (start, end) in enumerate(zip(starts, ends)):
                window_labels = phase_labels[int(start):int(end)]
                valid_labels = window_labels[window_labels >= 0]
                if len(valid_labels) > 0 and len(valid_labels) / len(window_labels) > 0.5:
                    y_batch.append(int(np.bincount(valid_labels).argmax()))
                    valid.append(wi)

            if valid:
                X_all.append(features[valid])
                y_all.append(np.asarray(y_batch, dtype=np.int64))

        if not X_all:
            continue

        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=cfg.phase_n_estimators,
            max_depth=cfg.phase_max_depth,
            max_samples=cfg.phase_max_samples,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_s, y)
        models[action] = clf
        scalers[action] = scaler

    return models, scalers


def predict_phase_per_action(models, scalers, stream_id, df, active_segments, cfg: NewCConfig):
    """Predict phase (concentric/eccentric) within active segments using per-action model."""
    action = _extract_action_from_stream_id(stream_id)
    if action not in models:
        action = list(models.keys())[0]

    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)

    # Default: uncertain
    phase_probs = np.ones((n, 2)) * 0.5

    for seg_start, seg_end in active_segments:
        if seg_start >= seg_end:
            continue
        seg_x = x[seg_start:seg_end]

        features, starts, ends = _extract_features(seg_x, cfg.phase_window_size, cfg.phase_infer_stride, cfg)
        if len(features) == 0:
            continue

        X_s = scalers[action].transform(features)
        probs = models[action].predict_proba(X_s)

        class_map = {int(c): i for i, c in enumerate(models[action].classes_)}
        full_batch = np.zeros((len(probs), 2), dtype=np.float64)
        for cls_idx, mi in class_map.items():
            if cls_idx < 2:
                full_batch[:, cls_idx] = probs[:, mi]

        prob_accum = np.zeros((seg_end - seg_start, 2), dtype=np.float64)
        counts = np.zeros(seg_end - seg_start, dtype=np.float64)

        for wi, (start, end) in enumerate(zip(starts, ends)):
            prob_accum[int(start):int(end)] += full_batch[wi]
            counts[int(start):int(end)] += 1.0

        counts = np.where(counts < 1e-8, 1.0, counts)
        phase_probs[seg_start:seg_end] = prob_accum / counts[:, None]

    return phase_probs


# ---------------------------------------------------------------------------
# Phase Smoothing & Rep Parser
# ---------------------------------------------------------------------------

def smooth_phase_probs(phase_probs, cfg: NewCConfig):
    """Apply smoothing to phase probabilities."""
    n = len(phase_probs)
    smoothed = np.copy(phase_probs)

    # Moving average on probabilities
    if cfg.smoothing_window > 1:
        window = cfg.smoothing_window
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)

    return smoothed


def _merge_adjacent_same_phase(runs):
    """Merge adjacent runs with the same phase label."""
    if not runs:
        return runs
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(
                label=run.label,
                start_idx=merged[-1].start_idx,
                end_idx=run.end_idx,
                confidence=(merged[-1].confidence + run.confidence) / 2,
            )
        else:
            merged.append(run)
    return merged


def parse_reps_from_phase(phase_probs, cfg: NewCConfig):
    """Parse reps from smoothed phase probabilities."""
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])

    # Convert to runs with min_phase_samples
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=cfg.min_phase_samples)

    # Merge adjacent same phases
    runs = _merge_adjacent_same_phase(runs)

    # Pair concentric/eccentric into reps
    reps, diagnostics = pair_concentric_eccentric_reps(runs, micro_source="new_c", max_gap_samples=cfg.max_phase_gap_samples)
    return reps


# ---------------------------------------------------------------------------
# Baseline A: Per-Action 3-Class RF
# ---------------------------------------------------------------------------

def train_baseline_a_per_action(train_streams, cfg: NewCConfig):
    """Train per-action 3-class RF (other/concentric/eccentric)."""
    models = {}
    scalers = {}

    action_streams = {}
    for stream_id, df in train_streams:
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_streams:
            action_streams[action] = []
        action_streams[action].append((stream_id, df))

    for action, streams in action_streams.items():
        X_all, y_all = [], []
        for _, df in streams:
            if "phase" not in df.columns:
                continue
            x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
            labels = micro_labels_from_phase(df["phase"].to_numpy())
            label_idx = np.array([MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)

            features, starts, ends = _extract_features(x, cfg.active_window_size, cfg.active_train_stride, cfg)
            if len(features) == 0:
                continue

            y_batch = []
            for start, end in zip(starts, ends):
                window_labels = label_idx[int(start):int(end)]
                y_batch.append(int(np.bincount(window_labels, minlength=len(MICRO_LABELS)).argmax()))

            X_all.append(features)
            y_all.append(np.asarray(y_batch, dtype=np.int64))

        if not X_all:
            continue

        X = np.concatenate(X_all)
        y = np.concatenate(y_all)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = RandomForestClassifier(
            n_estimators=cfg.active_n_estimators,
            max_depth=cfg.active_max_depth,
            max_samples=cfg.active_max_samples,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_s, y)
        models[action] = clf
        scalers[action] = scaler

    return models, scalers


def predict_baseline_a_per_action(models, scalers, stream_id, df, cfg: NewCConfig):
    """Predict 3-class probabilities using per-action model."""
    action = _extract_action_from_stream_id(stream_id)
    if action not in models:
        action = list(models.keys())[0]

    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)

    features, starts, ends = _extract_features(x, cfg.active_window_size, cfg.active_infer_stride, cfg)
    if len(features) == 0:
        return np.zeros((n, len(MICRO_LABELS)))

    X_s = scalers[action].transform(features)
    probs = models[action].predict_proba(X_s)

    class_map = {int(c): i for i, c in enumerate(models[action].classes_)}
    full_batch = np.zeros((len(probs), len(MICRO_LABELS)), dtype=np.float64)
    for cls_idx, mi in class_map.items():
        full_batch[:, cls_idx] = probs[:, mi]

    prob_accum = np.zeros((n, len(MICRO_LABELS)), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    for wi, (start, end) in enumerate(zip(starts, ends)):
        prob_accum[int(start):int(end)] += full_batch[wi]
        counts[int(start):int(end)] += 1.0

    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts[:, None]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def evaluate_reps(pred_reps, gt_reps):
    """Evaluate rep detection with IoU matching."""
    pred_count = len(pred_reps)
    gt_count = len(gt_reps)

    tp = 0
    matched_gt = set()
    for pred in pred_reps:
        best_iou = 0
        best_gt = None
        for gi, gt in enumerate(gt_reps):
            if gi in matched_gt:
                continue
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

    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "exact_count": 1 if pred_count == gt_count else 0,
        "over": 1 if pred_count > gt_count else 0,
        "under": 1 if pred_count < gt_count else 0,
    }


def evaluate_phase(phase_probs, gt_phases):
    """Evaluate phase segmentation within active samples."""
    gt_labels = _prepare_phase_labels(gt_phases)
    pred_labels = np.argmax(phase_probs, axis=1)

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


def aggregate_results(results):
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
        "transition_mae_ms": np.nanmean([r.get("transition_mae_ms", np.nan) for r in results if r.get("transition_mae_ms") is not None]),
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_new_c_pipeline(train_streams, test_streams, cfg: NewCConfig, use_tcn=False, device='cpu'):
    print("\n  [New C] Training Per-Action Active Detector...")
    active_models, active_scalers = train_active_detector_per_action(train_streams, cfg)

    if not active_models:
        return None

    if use_tcn:
        print("  [New C] Training TCN Phase Segmenter (global, all actions)...")
        # TCN: train one model on all active segments across all actions
        tcn_model = train_tcn_phase_segmenter(train_streams, cfg.imu_columns, device=device, epochs=20, lr=1e-3, batch_size=32)
        if tcn_model is None:
            return None
    else:
        print("  [New C] Training Per-Action RF Phase Segmenter...")
        phase_models, phase_scalers = train_phase_segmenter_per_action(train_streams, cfg)
        if not phase_models:
            return None

    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue

        # Active Detection
        active_probs = predict_active_per_action(active_models, active_scalers, stream_id, df, cfg)
        active_segments = apply_state_control(active_probs, cfg)

        # Phase Segmentation
        if use_tcn:
            phase_probs = predict_tcn_phase(tcn_model, df, active_segments, cfg.imu_columns, device=device)
        else:
            phase_probs = predict_phase_per_action(phase_models, phase_scalers, stream_id, df, active_segments, cfg)
        
        phase_probs_smooth = smooth_phase_probs(phase_probs, cfg)

        # Rep Parser
        pred_reps = parse_reps_from_phase(phase_probs_smooth, cfg)
        gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)

        # Evaluate
        rep_metrics = evaluate_reps(pred_reps, gt_reps)
        phase_metrics = evaluate_phase(phase_probs_smooth, df["phase"].to_numpy())

        all_results.append({"stream_id": stream_id, **rep_metrics, **phase_metrics})

    return all_results


def run_baseline_a(train_streams, test_streams, cfg: NewCConfig):
    print("\n  [Baseline A] Training Per-Action 3-class RF...")
    models, scalers = train_baseline_a_per_action(train_streams, cfg)

    if not models:
        return None

    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue

        probs = predict_baseline_a_per_action(models, scalers, stream_id, df, cfg)

        # Smooth
        if cfg.smoothing_window > 1:
            smoothed = np.zeros_like(probs)
            for c in range(len(MICRO_LABELS)):
                cumsum = np.cumsum(probs[:, c])
                for i in range(len(probs)):
                    start = max(0, i - cfg.smoothing_window + 1)
                    total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                    smoothed[i, c] = total / (i - start + 1)
            probs = smoothed

        pred_labels = np.argmax(probs, axis=1)
        pred_phase = np.array([MICRO_LABELS[i] for i in pred_labels])

        # Parse reps
        runs = labels_to_runs(pred_phase, positive_labels={"concentric", "eccentric"}, min_length=cfg.min_phase_samples)
        pred_reps, _ = pair_concentric_eccentric_reps(runs, micro_source="baseline_a", max_gap_samples=cfg.max_phase_gap_samples)
        gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)

        rep_metrics = evaluate_reps(pred_reps, gt_reps)

        # Phase metrics (only on active samples)
        gt_labels = _prepare_phase_labels(df["phase"].to_numpy())
        valid = gt_labels >= 0
        if valid.any():
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

        all_results.append({"stream_id": stream_id, **rep_metrics, "accuracy": acc, "macro_f1": macro_f1})

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="New C Pipeline v2 (Full Settings)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/new_c_pipeline/evaluation_v2"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: kevin only")
    parser.add_argument("--use-tcn", action="store_true", help="Use TCN for phase segmentation (default: RF)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for TCN (cpu/cuda)")
    parser.add_argument("--full", action="store_true", help="Full 9-fold LOSO")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = NewCConfig()

    print("="*70)
    if args.use_tcn:
        print("New C Pipeline v2: Per-Action Active + TCN Phase")
    else:
        print("New C Pipeline v2: Per-Action + Rich Features + Causal Window")
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

        print("\n[3/4] Running Baseline A...")
        baseline_fold = run_baseline_a(train_streams, test_streams, cfg)
        if baseline_fold:
            baseline_results.extend(baseline_fold)
            agg = aggregate_results(baseline_fold)
            print(f"      Baseline A: Rep F1={agg['rep_f1']:.4f}, Phase F1={agg['phase_macro_f1']:.4f}")

        print("\n[4/4] Running New C...")
        new_c_fold = run_new_c_pipeline(train_streams, test_streams, cfg, use_tcn=args.use_tcn, device=args.device)
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
        print(f"\nBaseline A (Per-Action 3-class RF):")
        for k, v in baseline_agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if new_c_results:
        new_c_agg = aggregate_results(new_c_results)
        phase_model = "TCN" if args.use_tcn else "RF"
        print(f"\nNew C (Per-Action Active + {phase_model} Phase):")
        for k, v in new_c_agg.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    summary = {
        "baseline_a": aggregate_results(baseline_results) if baseline_results else {},
        "new_c": aggregate_results(new_c_results) if new_c_results else {},
    }
    model_suffix = "tcn" if args.use_tcn else "rf"
    output_file = args.output / f"comparison_v2_{model_suffix}_{'quick' if args.quick else 'full'}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
