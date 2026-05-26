"""
Separated Phase Segmentation (Strategy A)

Step 1: Use GT rep boundaries (or Per-Action RF predicted boundaries)
Step 2: Inside each rep, train a 2-class model to distinguish concentric vs eccentric

This script evaluates whether separating rep detection and phase segmentation
improves transition accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.micro_macro_segments import (
    MICRO_LABELS,
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    OTHER_LABEL,
    micro_labels_from_phase,
    truth_reps_from_labels,
)
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.micro_macro_recognition import _load_streams

import yaml


def _extract_rep_features(
    df: pd.DataFrame,
    rep_start: int,
    rep_end: int,
    imu_columns: Sequence[str],
    window_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract sliding-window features from within a single rep.
    
    Returns:
        X: (N_windows, features) feature matrix
        y: (N_windows,) labels (1=concentric, 0=eccentric)
    """
    rep_df = df.iloc[rep_start:rep_end].reset_index(drop=True)
    if len(rep_df) < window_size:
        return np.zeros((0, 0)), np.zeros((0,))
    
    x = rep_df[list(imu_columns)].to_numpy(dtype=np.float32)
    phases = micro_labels_from_phase(rep_df["phase"].to_numpy())
    
    # Build sliding windows
    n = len(x)
    n_windows = n - window_size + 1
    if n_windows <= 0:
        return np.zeros((0, 0)), np.zeros((0,))
    
    # Features: window-level stats
    windows = np.zeros((n_windows, window_size, len(imu_columns)), dtype=np.float32)
    for i in range(n_windows):
        windows[i] = x[i:i+window_size]
    
    # Compute rich features per window
    feats = []
    # Mean, std, min, max per axis
    means = windows.mean(axis=1)  # (N, C)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    
    # Concatenate features
    feat_list = [means, stds, mins, maxs]
    
    # Add velocity (simple diff-based)
    diffs = np.diff(windows, axis=1)  # (N, W-1, C)
    feat_list.extend([diffs.mean(axis=1), diffs.std(axis=1)])
    
    # Add position within rep (normalized 0-1)
    pos = np.arange(n_windows) / max(n_windows - 1, 1)
    feat_list.append(pos.reshape(-1, 1))
    
    X = np.concatenate(feat_list, axis=1)
    
    # Labels: center sample of each window
    y = []
    for i in range(n_windows):
        center = i + window_size // 2
        phase = phases[center]
        if phase == CONCENTRIC_LABEL:
            y.append(1)
        elif phase == ECCENTRIC_LABEL:
            y.append(0)
        else:
            y.append(-1)  # Skip
    
    y = np.array(y)
    valid = y >= 0
    return X[valid], y[valid]


def _find_transition_point(predictions: np.ndarray) -> int:
    """Find the transition point from eccentric to concentric (or vice versa).
    
    Assumes predictions are for a single rep: 0=eccentric, 1=concentric
    Returns the index where the first transition occurs.
    """
    if len(predictions) == 0:
        return 0
    # Find first change in prediction
    changes = np.where(np.diff(predictions) != 0)[0]
    if len(changes) > 0:
        return changes[0] + 1
    return len(predictions) // 2  # fallback: midpoint


def evaluate_separated_phase_segmentation(
    config_path: Path,
    window_size: int = 20,
    model_type: str = "rf",  # "rf" or "logreg"
) -> Dict:
    """Evaluate separated phase segmentation."""
    print("="*70)
    print("Separated Phase Segmentation (Strategy A)")
    print(f"Window size: {window_size}, Model: {model_type}")
    print("="*70)
    
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_cfg = raw.get("micro_macro", {})
    feature_cfg = raw.get("feature", {})
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    # Load set-level streams
    print("\n[1/5] Loading streams...")
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"      Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    
    # For each subject, do LOSO
    all_results = []
    for test_subject in subjects:
        print(f"\n[2/5] Fold: test={test_subject}")
        train_streams = [(sid, df) for sid, df in all_streams 
                        if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams 
                       if sid.startswith(f"{test_subject}/")]
        
        print(f"      train={len(train_streams)}, test={len(test_streams)}")
        
        # Extract rep-level training data
        print("      [3/5] Extracting training reps...")
        X_train_all, y_train_all = [], []
        for stream_id, df in train_streams:
            if "phase" not in df.columns:
                continue
            gt_reps = truth_reps_from_labels(
                df["phase"].to_numpy(),
                min_phase_samples=mm_cfg.get("min_phase_samples", 3)
            )
            for rep in gt_reps:
                X_rep, y_rep = _extract_rep_features(
                    df, rep.start_idx, rep.end_idx, imu_columns, window_size
                )
                if len(X_rep) > 0:
                    X_train_all.append(X_rep)
                    y_train_all.append(y_rep)
        
        if not X_train_all:
            print(f"      [WARN] No training reps found")
            continue
        
        X_train = np.concatenate(X_train_all, axis=0)
        y_train = np.concatenate(y_train_all, axis=0)
        print(f"      Training samples: {len(X_train)}")
        
        # Train model
        print("      [4/5] Training 2-class model...")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        if model_type == "rf":
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        
        clf.fit(X_train_s, y_train)
        
        # Evaluate on test streams
        print("      [5/5] Evaluating test streams...")
        per_rep_errors = []
        per_rep_accuracies = []
        n_reps = 0
        
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            gt_reps = truth_reps_from_labels(
                df["phase"].to_numpy(),
                min_phase_samples=mm_cfg.get("min_phase_samples", 3)
            )
            
            for rep in gt_reps:
                X_rep, y_rep = _extract_rep_features(
                    df, rep.start_idx, rep.end_idx, imu_columns, window_size
                )
                if len(X_rep) == 0:
                    continue
                
                X_rep_s = scaler.transform(X_rep)
                y_pred = clf.predict(X_rep_s)
                
                # Per-sample accuracy within this rep
                acc = accuracy_score(y_rep, y_pred)
                per_rep_accuracies.append(acc)
                
                # Transition error
                gt_transition = np.where(np.diff(y_rep) != 0)[0]
                pred_transition = np.where(np.diff(y_pred) != 0)[0]
                
                if len(gt_transition) > 0 and len(pred_transition) > 0:
                    error = abs(gt_transition[0] - pred_transition[0])
                    per_rep_errors.append(error)
                
                n_reps += 1
        
        avg_acc = np.mean(per_rep_accuracies) if per_rep_accuracies else 0
        avg_error = np.mean(per_rep_errors) if per_rep_errors else float('nan')
        median_error = np.median(per_rep_errors) if per_rep_errors else float('nan')
        
        print(f"      => Reps: {n_reps}, Avg acc: {avg_acc:.4f}, "
              f"Transition MAE: {avg_error:.1f} samples, Median: {median_error:.1f}")
        
        all_results.append({
            "test_subject": test_subject,
            "n_reps": n_reps,
            "avg_per_sample_acc": avg_acc,
            "transition_mae_samples": avg_error,
            "transition_median_samples": median_error,
        })
    
    # Aggregate
    print(f"\n{'='*70}")
    print("SUMMARY (Separated Phase Segmentation)")
    print(f"{'='*70}")
    
    if all_results:
        overall_acc = np.mean([r["avg_per_sample_acc"] for r in all_results])
        overall_mae = np.nanmean([r["transition_mae_samples"] for r in all_results])
        overall_median = np.nanmedian([r["transition_median_samples"] for r in all_results])
        
        print(f"Per-sample accuracy (within rep): {overall_acc:.4f}")
        print(f"Transition MAE (samples):         {overall_mae:.1f}")
        print(f"Transition Median (samples):      {overall_median:.1f}")
        print(f"Transition MAE (ms @100Hz):         {overall_mae * 10:.1f}ms")
        print(f"\nPer-fold:")
        for r in all_results:
            print(f"  {r['test_subject']:<20}: acc={r['avg_per_sample_acc']:.4f}, "
                  f"MAE={r['transition_mae_samples']:.1f}samp")
    
    return {
        "model_type": model_type,
        "window_size": window_size,
        "results": all_results,
        "overall": {
            "avg_per_sample_acc": overall_acc if all_results else 0,
            "transition_mae_samples": overall_mae if all_results else float('nan'),
            "transition_mae_ms": overall_mae * 10 if all_results else float('nan'),
        }
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "logreg"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/separated_phase_segmentation"))
    args = parser.parse_args()
    
    results = evaluate_separated_phase_segmentation(args.config, args.window_size, args.model)
    
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"results_{args.model}_w{args.window_size}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
