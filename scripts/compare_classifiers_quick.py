"""Quick head-to-head comparison: RF vs XGBoost vs CatBoost on Causal RF features.

Tests 3 folds to save time. Same features, same window, different classifier.
"""
from __future__ import annotations

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


def _extract_subject(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


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
    """Prepare training data (same as Causal RF)."""
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


def train_xgboost(X, y, n_estimators=100, max_depth=6):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    clf.fit(X, y)
    return clf


def train_catboost(X, y, n_estimators=100, max_depth=6):
    import catboost as cbt
    clf = cbt.CatBoostClassifier(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=0.1,
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )
    clf.fit(X, y)
    return clf


def predict_with_clf(clf, df, imu_columns, window_size):
    """Generic prediction for any sklearn-like classifier."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    probs = np.zeros((n, len(cb.MICRO_LABELS)), dtype=np.float32)
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), 1)
    raw_batch = clf.predict_proba(X_batch) if len(X_batch) else np.zeros((0, len(class_map)), dtype=np.float32)
    if len(raw_batch):
        full_batch = np.zeros((len(raw_batch), len(cb.MICRO_LABELS)), dtype=np.float32)
        for cls_idx, mi in class_map.items():
            full_batch[:, cls_idx] = raw_batch[:, mi]
        probs[np.maximum(0, ends - 1)] = full_batch
    return probs


def run_quick_compare(config_path: Path, subjects: List[str], window_size: int = 100):
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
    
    macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]
    
    results = {
        "Random Forest": [],
        "XGBoost": [],
        "CatBoost": [],
    }
    train_times = {
        "Random Forest": [],
        "XGBoost": [],
        "CatBoost": [],
    }
    
    # Only test first 3 subjects to save time
    test_subjects = subjects[:3]
    
    for test_subject in test_subjects:
        print(f"\n{'='*60}")
        print(f"[Fold] test={test_subject}")
        print(f"{'='*60}")
        
        train_subjects = [s for s in subjects if s != test_subject]
        train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
        test_streams = cb._filter_subjects(streams, [test_subject], subject_column)
        
        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
        
        # Prepare data once
        print("[Prep] Extracting features...")
        X_train, y_train = _prepare_train_data(train_z, imu_columns, window_size, 10)
        print(f"  Training data: {len(X_train)} samples, {X_train.shape[1]} features")
        
        # --- Random Forest ---
        print("\n[Training] Random Forest...")
        t0 = time.time()
        rf_clf = crf.train_causal_rf(train_z, imu_columns, window_size=window_size, stride=10,
                                      n_estimators=100, max_depth=15, max_samples=0.7)
        train_times["Random Forest"].append(time.time() - t0)
        
        def rf_predict(df):
            return crf.predict_causal_rf(rf_clf, df, imu_columns, window_size=window_size), None
        
        rf_results = cb.evaluate_all_streams(rf_predict, test_z, macro_classes, mm_cfg)
        print(f"  RF: F1={rf_results['rep_f1']:.4f}")
        results["Random Forest"].append(rf_results['rep_f1'])
        
        # --- XGBoost ---
        print("\n[Training] XGBoost...")
        t0 = time.time()
        xgb_clf = train_xgboost(X_train, y_train, n_estimators=100, max_depth=6)
        train_times["XGBoost"].append(time.time() - t0)
        
        def xgb_predict(df):
            return predict_with_clf(xgb_clf, df, imu_columns, window_size=window_size), None
        
        xgb_results = cb.evaluate_all_streams(xgb_predict, test_z, macro_classes, mm_cfg)
        print(f"  XGBoost: F1={xgb_results['rep_f1']:.4f}")
        results["XGBoost"].append(xgb_results['rep_f1'])
        
        # --- CatBoost ---
        print("\n[Training] CatBoost...")
        t0 = time.time()
        cb_clf = train_catboost(X_train, y_train, n_estimators=100, max_depth=6)
        train_times["CatBoost"].append(time.time() - t0)
        
        def cb_predict(df):
            return predict_with_clf(cb_clf, df, imu_columns, window_size=window_size), None
        
        cb_results = cb.evaluate_all_streams(cb_predict, test_z, macro_classes, mm_cfg)
        print(f"  CatBoost: F1={cb_results['rep_f1']:.4f}")
        results["CatBoost"].append(cb_results['rep_f1'])
    
    print(f"\n{'='*60}")
    print("HEAD-TO-HEAD COMPARISON SUMMARY (3 folds)")
    print(f"{'='*60}")
    for method in ["Random Forest", "XGBoost", "CatBoost"]:
        f1s = results[method]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        mean_time = np.mean(train_times[method])
        print(f"{method:20s}: F1 = {mean_f1:.4f} ± {std_f1:.4f}  |  train = {mean_time:.1f}s  |  folds = {[f'{f:.3f}' for f in f1s]}")
    
    best = max(results.items(), key=lambda x: np.mean(x[1]))
    print(f"\n[Winner] {best[0]} (F1 = {np.mean(best[1]):.4f})")
    
    # Save results
    out_dir = Path("artifacts/baseline_comparison/classifier_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "quick_compare_3fold.json", "w") as f:
        json.dump({
            "results": results,
            "train_times": {k: [float(t) for t in v] for k, v in train_times.items()},
            "config": {"window_size": window_size, "subjects_tested": test_subjects},
        }, f, indent=2)
    print(f"\n[OK] Saved to {out_dir / 'quick_compare_3fold.json'}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--subjects", type=str, default="haoyu,hsianshun,kevin,thomas,yoru,yushuan,yanz")
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    run_quick_compare(args.config, subjects, window_size=args.window_size)
