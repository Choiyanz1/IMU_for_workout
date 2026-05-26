"""Test NEW FEATURES for Per-Action RF: velocity, jerk, delta.

Compares:
- Baseline: original 63-dim trailing-window features
- +Velocity: add diff(IMU) statistics (+63 dim)
- +Velocity+Jerk: add diff(diff(IMU)) statistics (+63 dim)

Usage:
    python scripts/test_new_features_rf.py --config config.yaml --subjects haoyu,kevin,yoru
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")

# ---------------------------------------------------------------------------
# Enhanced Feature Extraction
# ---------------------------------------------------------------------------

def _stats(arr: np.ndarray) -> np.ndarray:
    """Compute statistics along axis=1 for array [N, T, C]."""
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


def _extract_enhanced_features(windows: np.ndarray, use_velocity: bool = False, use_jerk: bool = False) -> np.ndarray:
    """Extract features with optional velocity/jerk."""
    base = _stats(windows)
    features = [base]
    
    if use_velocity:
        vel = np.diff(windows, axis=1)  # [N, T-1, C]
        if vel.shape[1] > 0:
            features.append(_stats(vel))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))
    
    if use_jerk:
        vel = np.diff(windows, axis=1)
        if vel.shape[1] > 1:
            jerk = np.diff(vel, axis=1)  # [N, T-2, C]
            features.append(_stats(jerk))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))
    
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _build_feature_matrix(x, window_size, stride, use_velocity=False, use_jerk=False):
    """Build enhanced trailing feature matrix."""
    n = len(x)
    starts = np.arange(0, n, int(max(1, stride)), dtype=np.int64)
    ends = np.minimum(starts + window_size, n)
    
    windows = []
    for s, e in zip(starts, ends):
        w = x[s:e]
        if len(w) < window_size:
            pad = window_size - len(w)
            w = np.pad(w, ((0, pad), (0, 0)), mode='edge')
        windows.append(w)
    
    windows_arr = np.stack(windows, axis=0) if windows else np.zeros((0, window_size, x.shape[1]), dtype=np.float32)
    features = _extract_enhanced_features(windows_arr, use_velocity=use_velocity, use_jerk=use_jerk)
    return features, starts, ends


def _prepare_train_data_enhanced(train_streams, imu_columns, window_size, stride, use_velocity=False, use_jerk=False):
    """Prepare training data with enhanced features."""
    X_all, y_all = [], []
    for _, df in train_streams:
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, starts, ends = _build_feature_matrix(x, window_size, stride, use_velocity, use_jerk)
        if len(X_batch):
            y_batch = []
            for s, e in zip(starts, ends):
                window_labels = label_idx[int(s):int(e)]
                y_batch.append(int(np.argmax(np.bincount(window_labels, minlength=len(cb.MICRO_LABELS)))))
            X_all.append(X_batch)
            y_all.append(np.asarray(y_batch, dtype=np.int64))
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    return X_all, y_all


def train_per_action_rf_enhanced(train_streams, action, imu_columns, window_size=100, stride=10, use_velocity=False, use_jerk=False):
    """Train per-action RF with enhanced features."""
    X_train, y_train = _prepare_train_data_enhanced(train_streams, imu_columns, window_size, stride, use_velocity, use_jerk)
    if len(X_train) == 0:
        return None
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def predict_per_action_rf_enhanced(clf, df, imu_columns, window_size=100, stride=1, use_velocity=False, use_jerk=False):
    """Predict with enhanced features."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    X_batch, starts, ends = _build_feature_matrix(x, window_size, stride, use_velocity, use_jerk)
    
    probs = np.zeros((n, len(cb.MICRO_LABELS)), dtype=np.float32)
    if len(X_batch) == 0:
        return probs
    
    raw_probs = clf.predict_proba(X_batch)
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    full_batch = np.zeros((len(raw_probs), len(cb.MICRO_LABELS)), dtype=np.float32)
    for cls_idx, mi in class_map.items():
        full_batch[:, cls_idx] = raw_probs[:, mi]
    
    for wi, (s, e) in enumerate(zip(starts, ends)):
        probs[int(s):int(e)] += full_batch[wi]
    
    counts = np.zeros(n, dtype=np.int32)
    for s, e in zip(starts, ends):
        counts[int(s):int(e)] += 1
    counts = np.maximum(counts, 1)
    return probs / counts[:, None]


# ---------------------------------------------------------------------------
# LOSO Evaluation
# ---------------------------------------------------------------------------

def run_loso_variant(config_path, subjects, variant_name, use_velocity=False, use_jerk=False):
    """Run smoke test for one feature variant."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, "sensor_ts", target_sample_rate)
    
    if subjects is None:
        subjects = sorted(set(all_subjects))
    
    print(f"\n{'='*60}")
    print(f"VARIANT: {variant_name} | subjects={subjects}")
    print(f"Velocity={use_velocity}, Jerk={use_jerk}")
    print(f"{'='*60}")
    
    all_results = []
    for test_subject in subjects:
        action_streams = {}
        for sid, df in all_streams:
            action = sid.split("/")[-2] if len(sid.split("/")) >= 3 else "unknown"
            if action not in action_streams:
                action_streams[action] = []
            action_streams[action].append((sid, df))
        
        subject_results = []
        for action, streams in action_streams.items():
            train_streams = [(sid, df) for sid, df in streams if sid.split("/")[0] != test_subject]
            test_streams = [(sid, df) for sid, df in streams if sid.split("/")[0] == test_subject]
            
            if not test_streams or not train_streams:
                continue
            
            stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
            train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
            test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
            
            clf = train_per_action_rf_enhanced(train_z, action, imu_columns, 
                                               window_size=100, stride=10,
                                               use_velocity=use_velocity, use_jerk=use_jerk)
            if clf is None:
                continue
            
            raw_prob_cache = []
            for sid, df in test_z:
                probs = predict_per_action_rf_enhanced(clf, df, imu_columns, 
                                                      window_size=100, stride=1,
                                                      use_velocity=use_velocity, use_jerk=use_jerk)
                raw_prob_cache.append((sid, df, probs))
            
            # Smooth
            smoothed = []
            for sid, df, probs in raw_prob_cache:
                smooth = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                sw = 15
                for i in range(len(probs)):
                    start = max(0, i - sw + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    smooth[i] = total / float(i - start + 1)
                smoothed.append((sid, df, smooth))
            
            def predict_fn(df, _cache_iter=iter(smoothed)):
                _, _, cached = next(_cache_iter)
                return cached, None
            
            actions = cb._available_actions(Path("./datasets/raw_data"), None)
            macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]
            results = cb.evaluate_all_streams(predict_fn, test_z, macro_classes, mm_cfg)
            subject_results.append(results)
        
        if subject_results:
            total_tp = sum(r["tp"] for r in subject_results)
            total_fp = sum(r["fp"] for r in subject_results)
            total_fn = sum(r["fn"] for r in subject_results)
            p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
            r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            print(f"  {test_subject}: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
            all_results.append({"f1": f1, "p": p, "r": r, "tp": total_tp, "fp": total_fp, "fn": total_fn})
    
    if all_results:
        total_tp = sum(r["tp"] for r in all_results)
        total_fp = sum(r["fp"] for r in all_results)
        total_fn = sum(r["fn"] for r in all_results)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"\n  >>> {variant_name} GRAND: F1={f1:.4f}, P={p:.4f}, R={r:.4f}")
        return {"variant": variant_name, "f1": f1, "p": p, "r": r}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--subjects", default="haoyu,kevin,yoru")
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    
    results = []
    results.append(run_loso_variant(args.config, subjects, "BASELINE (63-dim)", False, False))
    results.append(run_loso_variant(args.config, subjects, "+VELOCITY (126-dim)", True, False))
    results.append(run_loso_variant(args.config, subjects, "+VELOCITY+JERK (189-dim)", True, True))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r:
            print(f"{r['variant']:25s} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f}")


if __name__ == "__main__":
    main()
