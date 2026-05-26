"""
New C Pipeline Phase 1: Independent Active Detector

Trains a binary classifier to detect active vs not-active segments.

Label mapping:
- other → not_active (0)
- concentric / eccentric → active (1)

Input: 6-axis IMU window (configurable, default 1.0s @100Hz)
Output: active (1) or not_active (0)

Evaluation: active detection F1, precision, recall
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import MICRO_LABELS, OTHER_LABEL, CONCENTRIC_LABEL, ECCENTRIC_LABEL, micro_labels_from_phase
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.micro_macro_recognition import _load_streams

import yaml


def _extract_window_features_batch(windows: np.ndarray) -> np.ndarray:
    """Extract rich statistical features from windows [N, T, C]."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, T, C], got {arr.shape}")
    if arr.shape[0] == 0:
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
    
    # Skewness and kurtosis
    skew = np.zeros((arr.shape[0], arr.shape[2]), dtype=np.float32)
    kurt = np.zeros((arr.shape[0], arr.shape[2]), dtype=np.float32)
    for c in range(arr.shape[2]):
        col = arr[:, :, c]
        col_mean = np.mean(col, axis=1, keepdims=True)
        col_std = np.std(col, axis=1, keepdims=True)
        col_std = np.where(col_std < 1e-8, 1.0, col_std)
        z = (col - col_mean) / col_std
        skew[:, c] = np.mean(z ** 3, axis=1)
        kurt[:, c] = np.mean(z ** 4, axis=1) - 3.0
    
    per_channel = np.stack(
        [mean, std, vmin, vmax, median, q25, q75, argmax, argmin, total_variation, skew, kurt],
        axis=-1,
    ).reshape(arr.shape[0], -1)
    
    # Magnitude features
    mag = np.sqrt(np.sum(arr ** 2, axis=2))
    mag_stats = np.stack([
        np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1),
        np.min(mag, axis=1), np.median(mag, axis=1)
    ], axis=1)
    
    # Inter-axis correlations
    n_samples = arr.shape[0]
    n_channels = arr.shape[2]
    corrs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            ci = arr[:, :, i]
            cj = arr[:, :, j]
            ci_mean = ci - np.mean(ci, axis=1, keepdims=True)
            cj_mean = cj - np.mean(cj, axis=1, keepdims=True)
            num = np.sum(ci_mean * cj_mean, axis=1)
            den = np.sqrt(np.sum(ci_mean ** 2, axis=1) * np.sum(cj_mean ** 2, axis=1))
            den = np.where(den < 1e-12, 1.0, den)
            corrs.append((num / den).reshape(-1, 1))
    
    if corrs:
        corr_features = np.concatenate(corrs, axis=1)
        return np.concatenate([per_channel, mag_stats, corr_features], axis=1).astype(np.float32, copy=False)
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
    return _extract_window_features_batch(selected), starts, ends


def _prepare_active_labels(phases: np.ndarray) -> np.ndarray:
    """Convert phase labels to active/not-active binary labels."""
    labels = micro_labels_from_phase(phases)
    active = np.array([1 if str(l) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL} else 0 for l in labels], dtype=np.int64)
    return active


def extract_active_training_data(
    streams: List[Tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract sliding window features and binary active labels.
    
    Returns:
        X: (N_windows, N_features)
        y_window: (N_windows,) binary active labels (majority vote within window)
        starts: (N_windows,) start indices
        ends: (N_windows,) end indices
    """
    X_all, y_all = [], []
    for stream_id, df in streams:
        if "phase" not in df.columns:
            continue
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        active_labels = _prepare_active_labels(df["phase"].to_numpy())
        
        X_batch, starts, ends = _build_windows(x, window_size, stride)
        if len(X_batch) == 0:
            continue
        
        # Window label = majority vote of samples within window
        y_batch = []
        for start, end in zip(starts, ends):
            window_labels = active_labels[int(start):int(end)]
            y_batch.append(int(np.bincount(window_labels).argmax()))
        
        X_all.append(X_batch)
        y_all.append(np.asarray(y_batch, dtype=np.int64))
    
    if not X_all:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X, y, np.zeros(len(y), dtype=np.int64), np.zeros(len(y), dtype=np.int64)


def evaluate_active_detector(
    clf,
    scaler: StandardScaler,
    test_streams: List[Tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    window_size: int,
    stride: int,
) -> Dict:
    """Evaluate active detector on test streams."""
    all_y_true = []
    all_y_pred = []
    
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        active_labels = _prepare_active_labels(df["phase"].to_numpy())
        
        X_batch, starts, ends = _build_windows(x, window_size, stride)
        if len(X_batch) == 0:
            continue
        
        X_batch_s = scaler.transform(X_batch)
        y_pred = clf.predict(X_batch_s)
        
        # Expand window predictions back to per-sample (for evaluation)
        for wi, (start, end) in enumerate(zip(starts, ends)):
            sample_labels = active_labels[int(start):int(end)]
            # All samples in window get the window prediction
            all_y_true.extend(sample_labels.tolist())
            all_y_pred.extend([int(y_pred[wi])] * len(sample_labels))
    
    if not all_y_true:
        return {"error": "No test data"}
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "active_ratio": float(y_true.mean()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Independent Active Detector")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--window-size", type=int, default=100, help="Window size in samples (default 100 = 1.0s @100Hz)")
    parser.add_argument("--stride", type=int, default=25, help="Stride in samples (default 25 = 0.25s)")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "logreg"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/new_c_pipeline/active_detector"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: only first 3 subjects")
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("New C Pipeline Phase 1: Independent Active Detector")
    print(f"Window: {args.window_size} samples, Stride: {args.stride} samples")
    print(f"Model: {args.model}")
    print("="*70)
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {})
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    print("\n[1/3] Loading streams...")
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"      Loaded {len(all_streams)} streams from {len(subjects)} subjects: {subjects}")
    
    if args.quick:
        subjects = subjects[:3]
        print(f"[QUICK] Using subjects: {subjects}")
    
    # LOSO evaluation
    all_results = []
    for test_subject in subjects:
        print(f"\n[2/3] Fold: test={test_subject}")
        train_streams = [(sid, df) for sid, df in all_streams 
                        if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams 
                       if sid.startswith(f"{test_subject}/")]
        print(f"      train={len(train_streams)}, test={len(test_streams)}")
        
        # Extract training data
        print("      Extracting window features...")
        X_train, y_train, _, _ = extract_active_training_data(
            train_streams, imu_columns, args.window_size, args.stride
        )
        print(f"      Training windows: {len(X_train)}, Active ratio: {y_train.mean():.2%}")
        
        # Z-score normalization
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        # Train
        print("      Training model...")
        if args.model == "rf":
            clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        
        # Evaluate
        print("      Evaluating...")
        metrics = evaluate_active_detector(clf, scaler, test_streams, imu_columns, args.window_size, args.stride)
        metrics["test_subject"] = test_subject
        all_results.append(metrics)
        
        print(f"      => Acc={metrics['accuracy']:.4f}, Prec={metrics['precision']:.4f}, "
              f"Rec={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Active Detector - LOSO Summary")
    print(f"{'='*70}")
    avg_f1 = np.mean([r["f1"] for r in all_results])
    avg_acc = np.mean([r["accuracy"] for r in all_results])
    avg_prec = np.mean([r["precision"] for r in all_results])
    avg_rec = np.mean([r["recall"] for r in all_results])
    
    print(f"Average Accuracy:  {avg_acc:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall:    {avg_rec:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"\nPer-fold:")
    for r in all_results:
        print(f"  {r['test_subject']:<20}: F1={r['f1']:.4f}, Acc={r['accuracy']:.4f}")
    
    summary = {
        "model_type": args.model,
        "window_size": args.window_size,
        "stride": args.stride,
        "subjects": subjects,
        "results": all_results,
        "average": {
            "accuracy": avg_acc,
            "precision": avg_prec,
            "recall": avg_rec,
            "f1": avg_f1,
        }
    }
    output_file = args.output / f"active_detector_{args.model}_w{args.window_size}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
