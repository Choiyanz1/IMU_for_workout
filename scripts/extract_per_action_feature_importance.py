"""Extract per-action feature importance for each of the 8 actions.

This reveals which features are most important for detecting reps of each specific action,
enabling per-action feature subset customization.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

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


def _build_trailing_feature_matrix(x, window_size, stride):
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


def _prepare_train_data(train_streams, imu_columns, window_size, stride):
    X_all, y_all = [], []
    for _, df in train_streams:
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(X_batch):
            X_all.append(X_batch)
            y_all.append(label_idx[np.maximum(0, ends - 1)])
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    return X_all, y_all


def _get_feature_names(imu_columns):
    """Generate feature names matching _extract_window_features_batch."""
    names = []
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "argmax", "argmin", "tv"]
    for col in imu_columns:
        for stat in stats:
            names.append(f"{col}_{stat}")
    names.extend(["mag_mean", "mag_std", "mag_max"])
    return names


def extract_per_action_importance(config_path: Path, subjects: List[str], actions: List[str], window_size: int = 100):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    
    feature_names = _get_feature_names(imu_columns)
    
    # Per-action importance storage
    per_action_importance: Dict[str, List[np.ndarray]] = {action: [] for action in actions}
    
    # For each fold (leave-one-subject-out)
    for test_subject in subjects[:3]:  # Use first 3 subjects to save time
        print(f"\n{'='*60}")
        print(f"[Fold] test={test_subject}")
        print(f"{'='*60}")
        
        train_subjects = [s for s in subjects if s != test_subject]
        
        # For each action, train a per-action model using only that action's data
        for action in actions:
            print(f"\n  [Action] {action}")
            
            # Load all streams for this action from train subjects
            train_streams = []
            for subject in train_subjects:
                subject_dir = data_dir / subject
                if not subject_dir.exists():
                    continue
                for session_dir in subject_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    action_dir = session_dir / action
                    if not action_dir.exists():
                        continue
                    for set_dir in action_dir.iterdir():
                        if not set_dir.is_dir():
                            continue
                        # Load all CSVs in this set
                        for csv_path in sorted(set_dir.glob("*.csv")):
                            if "whole_session" in str(csv_path) or csv_path.name.endswith("_w"):
                                continue
                            try:
                                df = pd.read_csv(csv_path)
                                if "phase" not in df.columns:
                                    continue
                                train_streams.append((f"{subject}/{session_dir.name}/{action}/{set_dir.name}/{csv_path.stem}", df))
                            except Exception:
                                continue
            
            if len(train_streams) < 5:
                print(f"    [WARN] Only {len(train_streams)} streams for {action}, skipping")
                continue
            
            # Compute z-score stats from train streams
            stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
            train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
            
            # Train per-action Causal RF
            print(f"    Training on {len(train_z)} streams...")
            try:
                rf_clf = crf.train_causal_rf(train_z, imu_columns, window_size=window_size, stride=10,
                                               n_estimators=100, max_depth=15, max_samples=0.7)
                
                importance = rf_clf.feature_importances_
                per_action_importance[action].append(importance)
                
                # Print top 10 for this action
                idx = np.argsort(importance)[::-1]
                print(f"    Top 10 features for {action}:")
                for i in idx[:10]:
                    print(f"      {feature_names[i]:25s} = {importance[i]:.4f}")
                
            except Exception as e:
                print(f"    [ERROR] Training failed: {e}")
                continue
    
    # Average across folds
    print(f"\n{'='*60}")
    print("PER-ACTION FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*60}")
    
    summary = {}
    for action in actions:
        if not per_action_importance[action]:
            continue
        
        mean_imp = np.mean(per_action_importance[action], axis=0)
        
        # Group by axis
        axis_groups = {}
        for i, name in enumerate(feature_names):
            axis = name.split("_")[0]
            if axis not in axis_groups:
                axis_groups[axis] = 0
            axis_groups[axis] += mean_imp[i]
        
        print(f"\n{action}:")
        print(f"  By axis: {dict(sorted(axis_groups.items(), key=lambda x: -x[1])[:4])}")
        
        # Top 5 features
        idx = np.argsort(mean_imp)[::-1]
        print(f"  Top 5 features:")
        for i in idx[:5]:
            print(f"    {feature_names[i]:25s} = {mean_imp[i]:.4f}")
        
        summary[action] = {
            "by_axis": dict(sorted(axis_groups.items(), key=lambda x: -x[1])),
            "top_features": [(feature_names[i], float(mean_imp[i])) for i in idx[:30]],
        }
    
    # Save
    out_dir = Path("artifacts/baseline_comparison/per_action_feature_importance")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_action_importance.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Saved to {out_dir / 'per_action_importance.json'}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--subjects", type=str, default="haoyu,hsianshun,kevin,thomas,yoru,yushuan,yanz")
    parser.add_argument("--actions", type=str, default="db_bench_press,db_biceps_curl,db_rdl,db_shoulder_press,db_squat,db_triceps_curl,db_weighted_crunch,one_arm_db_row")
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    extract_per_action_importance(args.config, subjects, actions, window_size=args.window_size)
