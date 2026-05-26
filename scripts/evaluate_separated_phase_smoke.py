"""
Separated Phase Segmentation - Quick Smoke Test (1 subject)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, micro_labels_from_phase, truth_reps_from_labels
)
from train.micro_macro_recognition import _load_streams

import yaml


def extract_rep_sliding_features(df, rep_start, rep_end, imu_cols, window_size=20):
    """Extract sliding window features from a single rep."""
    rep_df = df.iloc[rep_start:rep_end].reset_index(drop=True)
    if len(rep_df) < window_size:
        return np.zeros((0, 0)), np.zeros((0,))
    
    x = rep_df[list(imu_cols)].to_numpy(dtype=np.float32)
    phases = micro_labels_from_phase(rep_df["phase"].to_numpy())
    
    n = len(x)
    n_windows = n - window_size + 1
    if n_windows <= 0:
        return np.zeros((0, 0)), np.zeros((0,))
    
    # Build windows
    windows = np.zeros((n_windows, window_size, len(imu_cols)), dtype=np.float32)
    for i in range(n_windows):
        windows[i] = x[i:i+window_size]
    
    # Simple features: mean, std, min, max, mean_diff, std_diff, position
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    
    diffs = np.diff(windows, axis=1)
    diff_means = diffs.mean(axis=1)
    diff_stds = diffs.std(axis=1)
    
    pos = (np.arange(n_windows) / max(n_windows - 1, 1)).reshape(-1, 1)
    
    X = np.concatenate([means, stds, mins, maxs, diff_means, diff_stds, pos], axis=1)
    
    # Labels: center of window
    y = []
    for i in range(n_windows):
        center = i + window_size // 2
        phase = phases[center]
        y.append(1 if phase == CONCENTRIC_LABEL else 0)
    
    return X, np.array(y)


def main():
    print("="*70)
    print("Separated Phase Segmentation - Smoke Test (kevin)")
    print("="*70)
    
    config_path = Path("config.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_cfg = raw.get("micro_macro", {})
    feature_cfg = raw.get("feature", {})
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    print("\n[1/4] Loading streams...")
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"      Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    
    # Only use kevin as test
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in all_streams 
                      if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams 
                   if sid.startswith(f"{test_subject}/")]
    
    print(f"\n[2/4] train={len(train_streams)}, test={len(test_streams)}")
    
    # Extract training reps
    print("\n[3/4] Extracting training reps...")
    X_train_all, y_train_all = [], []
    for stream_id, df in train_streams:
        if "phase" not in df.columns:
            continue
        gt_reps = truth_reps_from_labels(
            df["phase"].to_numpy(),
            min_phase_samples=mm_cfg.get("min_phase_samples", 3)
        )
        for rep in gt_reps:
            X_rep, y_rep = extract_rep_sliding_features(
                df, rep.start_idx, rep.end_idx, imu_columns, window_size=20
            )
            if len(X_rep) > 0:
                X_train_all.append(X_rep)
                y_train_all.append(y_rep)
    
    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    print(f"      Training windows: {len(X_train)}")
    print(f"      Concentric ratio: {y_train.mean():.2%}")
    
    # Train
    print("\n[4/4] Training & evaluating...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    
    # Evaluate on test
    per_rep_accs = []
    transition_errors = []
    n_reps = 0
    
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        gt_reps = truth_reps_from_labels(
            df["phase"].to_numpy(),
            min_phase_samples=mm_cfg.get("min_phase_samples", 3)
        )
        
        for rep in gt_reps:
            X_rep, y_rep = extract_rep_sliding_features(
                df, rep.start_idx, rep.end_idx, imu_columns, window_size=20
            )
            if len(X_rep) == 0:
                continue
            
            X_rep_s = scaler.transform(X_rep)
            y_pred = clf.predict(X_rep_s)
            
            acc = accuracy_score(y_rep, y_pred)
            per_rep_accs.append(acc)
            
            # Transition error
            gt_changes = np.where(np.diff(y_rep) != 0)[0]
            pred_changes = np.where(np.diff(y_pred) != 0)[0]
            
            if len(gt_changes) > 0 and len(pred_changes) > 0:
                error = abs(gt_changes[0] - pred_changes[0])
                transition_errors.append(error)
            
            n_reps += 1
    
    print(f"\n{'='*70}")
    print("RESULTS (kevin test subject)")
    print(f"{'='*70}")
    print(f"Reps evaluated: {n_reps}")
    print(f"Per-sample accuracy (within rep): {np.mean(per_rep_accs):.4f}")
    if transition_errors:
        print(f"Transition MAE (samples): {np.mean(transition_errors):.1f}")
        print(f"Transition Median (samples): {np.median(transition_errors):.1f}")
        print(f"Transition MAE (ms @100Hz): {np.mean(transition_errors) * 10:.1f}ms")
    
    # Compare with Direct approach from existing results
    print(f"\n[Comparison with Direct approach]")
    print(f"  Direct (Per-Action RF) Transition MAE: ~235ms (from previous results)")
    print(f"  Separated (this test) Transition MAE: {np.mean(transition_errors) * 10:.1f}ms" if transition_errors else "  N/A")


if __name__ == "__main__":
    main()
