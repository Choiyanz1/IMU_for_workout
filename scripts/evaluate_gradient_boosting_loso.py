"""Gradient Boosting Baselines (XGBoost / CatBoost) with 7-fold LOSO.

Same trailing-window features as Causal RF, but with XGBoost/CatBoost classifier.
Usage:
    # Smoke test (3 subjects)
    python scripts/evaluate_gradient_boosting_loso.py --config config.yaml \
        --classifier catboost --output artifacts/catboost_smoke --subjects haoyu,kevin,yoru

    # Full 7-fold
    python scripts/evaluate_gradient_boosting_loso.py --config config.yaml \
        --classifier catboost --output artifacts/catboost_7fold

    # XGBoost
    python scripts/evaluate_gradient_boosting_loso.py --config config.yaml \
        --classifier xgboost --output artifacts/xgboost_7fold
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
    RepDetection, match_segments, rep_metrics, truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz


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


def train_classifier(X, y, classifier: str, n_estimators=100, max_depth=6):
    """Train specified gradient boosting classifier."""
    if classifier == "xgboost":
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
    elif classifier == "catboost":
        import catboost as cbt
        clf = cbt.CatBoostClassifier(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
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


def run_loso(
    config_path: Path,
    output_dir: Path,
    classifier: str,
    subjects: List[str] | None = None,
    window_size: int = 100,
    train_stride: int = 10,
    n_estimators: int = 100,
    max_depth: int = 6,
    smoothing_window: int = 15,
) -> Dict:
    """Run full LOSO evaluation."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))

    print(f"[INFO] Loading streams...")
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)

    if subjects is None:
        subjects = sorted(set(all_subjects))

    print(f"[INFO] {classifier.upper()} Baseline | subjects={subjects} | actions={available_actions}")
    print(f"[INFO] window_size={window_size}, n_estimators={n_estimators}, max_depth={max_depth}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for test_subject in subjects:
        fold_file = output_dir / f"fold_{test_subject}.json"
        if fold_file.exists():
            print(f"[Resume] Loading {fold_file}")
            with open(fold_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        train_streams = [(sid, df) for sid, df in all_streams
                        if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams
                       if sid.split("/")[0] == test_subject]

        if not test_streams:
            print(f"[Skip] No test streams for {test_subject}")
            continue

        print(f"\n[Fold] test={test_subject} train={len(train_streams)} test={len(test_streams)}")

        # Prepare training data
        t0 = time.time()
        X_train, y_train = _prepare_train_data(train_streams, imu_columns, window_size, train_stride)
        print(f"  Prepared {len(X_train)} training windows")

        # Train
        clf = train_classifier(X_train, y_train, classifier, n_estimators=n_estimators, max_depth=max_depth)
        train_time = time.time() - t0

        # Predict all test streams
        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_streams, start=1):
            probs = predict_with_clf(clf, df, imu_columns, window_size)
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 10 == 0 or stream_idx == len(test_streams):
                print(f"  Predicted {stream_idx}/{len(test_streams)} test streams", flush=True)

        # Apply smoothing
        smoothed = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smoothing_window) > 1:
                smooth = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smoothing_window) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    count = i - start + 1
                    smooth[i] = total / float(count)
                cur = smooth
            smoothed.append((stream_id, df, cur))

        # Evaluate each stream using compare_baselines evaluation pipeline
        actions = cb._available_actions(
            Path(data_cfg.get("data_dir", "./datasets/raw_data")),
            data_cfg.get("include_actions"),
        )
        macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]

        def predict_fn_from_cache(df, _cache_iter=iter(smoothed)):
            sid, _, cached_probs = next(_cache_iter)
            return cached_probs, None

        results = cb.evaluate_all_streams(
            predict_fn_from_cache, test_streams, macro_classes, mm_cfg
        )
        results["model_name"] = f"{classifier.upper()} Causal (7-fold LOSO)"
        results["evaluation_protocol"] = "loso"
        results["test_subject"] = test_subject
        results["train_time_s"] = train_time
        results["config"] = {
            "window_size": int(window_size), "train_stride": int(train_stride),
            "smoothing_window": int(smoothing_window), "n_estimators": int(n_estimators),
            "max_depth": int(max_depth), "classifier": classifier,
        }

        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved: {fold_file} | Rep F1={results.get('rep_f1', 0):.4f}")
        all_results.append(results)

    if not all_results:
        return {"error": "No results"}

    # Grand summary
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    summary = {
        "model": f"{classifier.upper()} Causal (7-fold LOSO)",
        "n_folds": len(all_results),
        "subjects": subjects,
        "overall": {
            "precision": p, "recall": r, "rep_f1": f1,
            "n_true": sum(r["n_true"] for r in all_results),
            "n_pred": sum(r["n_pred"] for r in all_results),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"{classifier.upper()} GRAND SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\n[OK] Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--classifier", choices=["xgboost", "catboost"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="",
                        help="Comma-separated subjects (default: all). Use subset for smoke test.")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--smoothing-window", type=int, default=15)
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None

    run_loso(
        Path(args.config),
        Path(args.output),
        classifier=args.classifier,
        subjects=subjects,
        window_size=args.window_size,
        train_stride=args.train_stride,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
