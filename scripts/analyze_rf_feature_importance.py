"""Extract feature importance from trained Causal RF model.

Analyzes which features are most important for rep boundary detection.
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


def _get_feature_names(imu_columns, window_size):
    """Generate feature names matching _extract_window_features_batch."""
    names = []
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "argmax", "argmin", "tv"]
    for col in imu_columns:
        for stat in stats:
            names.append(f"{col}_{stat}")
    # mag stats
    names.extend(["mag_mean", "mag_std", "mag_max"])
    return names


def analyze_feature_importance(config_path: Path, subjects: List[str], window_size: int = 100):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, all_subjects, actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)
    
    feature_names = _get_feature_names(imu_columns, window_size)
    
    all_importance = []
    
    # Test on first 2 subjects (to save time, enough for feature analysis)
    test_subjects = subjects[:2]
    
    for test_subject in test_subjects:
        print(f"\n[Fold] test={test_subject}")
        
        train_subjects = [s for s in subjects if s != test_subject]
        train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
        
        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        
        # Train RF
        print("  Training Causal RF...")
        rf_clf = crf.train_causal_rf(train_z, imu_columns, window_size=window_size, stride=10,
                                      n_estimators=100, max_depth=15, max_samples=0.7)
        
        # Extract feature importance
        importance = rf_clf.feature_importances_
        all_importance.append(importance)
        
        # Print top 15
        idx = np.argsort(importance)[::-1]
        print(f"  Top 15 features (by importance):")
        for i in idx[:15]:
            print(f"    {i:3d}: {feature_names[i]:25s} = {importance[i]:.4f}")
    
    # Average across folds
    mean_importance = np.mean(all_importance, axis=0)
    std_importance = np.std(all_importance, axis=0)
    
    # Create summary dataframe
    df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "importance_pct": mean_importance / np.sum(mean_importance) * 100,
    })
    df = df.sort_values("mean_importance", ascending=False)
    
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE SUMMARY (averaged across folds)")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    # Group by sensor axis
    axis_groups = {}
    for _, row in df.iterrows():
        feat = row["feature"]
        axis = feat.split("_")[0]  # e.g., "ax_mean" -> "ax"
        if axis not in axis_groups:
            axis_groups[axis] = 0
        axis_groups[axis] += row["mean_importance"]
    
    print(f"\n{'='*60}")
    print("IMPORTANCE BY SENSOR AXIS")
    print(f"{'='*60}")
    axis_df = pd.DataFrame([
        {"axis": k, "importance": v, "pct": v / np.sum(mean_importance) * 100}
        for k, v in sorted(axis_groups.items(), key=lambda x: -x[1])
    ])
    print(axis_df.to_string(index=False))
    
    # Group by statistic type
    stat_groups = {}
    for _, row in df.iterrows():
        feat = row["feature"]
        parts = feat.split("_")
        stat = parts[-1] if len(parts) > 1 else "unknown"
        if stat not in stat_groups:
            stat_groups[stat] = 0
        stat_groups[stat] += row["mean_importance"]
    
    print(f"\n{'='*60}")
    print("IMPORTANCE BY STATISTIC TYPE")
    print(f"{'='*60}")
    stat_df = pd.DataFrame([
        {"statistic": k, "importance": v, "pct": v / np.sum(mean_importance) * 100}
        for k, v in sorted(stat_groups.items(), key=lambda x: -x[1])
    ])
    print(stat_df.to_string(index=False))
    
    # Save to file
    out_dir = Path("artifacts/baseline_comparison/feature_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "feature_importance.csv", index=False)
    
    with open(out_dir / "feature_analysis_summary.json", "w") as f:
        json.dump({
            "by_feature": df.to_dict("records"),
            "by_axis": axis_df.to_dict("records"),
            "by_statistic": stat_df.to_dict("records"),
        }, f, indent=2)
    
    print(f"\n[OK] Saved to {out_dir}")
    
    return df, axis_df, stat_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--subjects", type=str, default="haoyu,hsianshun,kevin,thomas,yoru,yushuan,yanz")
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    analyze_feature_importance(args.config, subjects, window_size=args.window_size)
