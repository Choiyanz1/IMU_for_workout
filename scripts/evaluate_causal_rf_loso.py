"""LOSO evaluation of plain Causal RF (no refiner).

Usage:
    python scripts/evaluate_causal_rf_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/causal_rf_loso \
        --window-size 50
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


def _extract_subject(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def run_loso(config_path: Path, output_dir: Path, subjects: List[str] | None = None,
             window_size: int = 50, train_stride: int = 10, n_estimators: int = 50,
             max_depth: int = 15, max_samples: float = 0.7, smoothing_window: int = 15):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw = raw.get("train", {}) or {}
    data_cfg = raw.get("data", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, all_subjects, actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)
    
    macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]
    
    if subjects is None:
        subjects = sorted(set(all_subjects))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    completed = set()
    for subject in subjects:
        if (output_dir / f"fold_{subject}.json").exists():
            completed.add(subject)
            print(f"[Resume] Found completed fold: {subject}")
    
    all_results = []
    for test_subject in subjects:
        if test_subject in completed:
            with open(output_dir / f"fold_{test_subject}.json", "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue
        
        train_subjects = [s for s in subjects if s != test_subject]
        train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
        test_streams = cb._filter_subjects(streams, [test_subject], subject_column)
        
        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
        
        print(f"\n[Fold] test={test_subject} train={train_subjects} train_streams={len(train_z)} test_streams={len(test_z)}")
        
        # Train causal RF
        t0 = time.time()
        clf = crf.train_causal_rf(train_z, imu_columns, window_size=window_size, stride=train_stride,
                                   n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples)
        train_time = time.time() - t0
        
        # Predict and smooth
        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
            probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 25 == 0 or stream_idx == len(test_z):
                print(f"  [CausalRF] predicted {stream_idx}/{len(test_z)} test streams", flush=True)
        
        # Apply smoothing and evaluate
        smooth_w = smoothing_window
        smoothed_streams = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smooth_w) > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smooth_w) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    smoothed[i] = total / float(i - start + 1)
                cur = smoothed
            smoothed_streams.append((stream_id, df, cur))
        
        def predict_fn(df, _cache_iter=iter(smoothed_streams)):
            _, _, cached_probs = next(_cache_iter)
            return cached_probs, None
        
        results = cb.evaluate_all_streams(predict_fn, test_z, macro_classes, mm_cfg)
        results["model_name"] = "Causal Random Forest"
        results["evaluation_protocol"] = "loso"
        results["test_subject"] = test_subject
        results["train_time_s"] = train_time
        results["smoothing_window"] = smooth_w
        results["config"] = {
            "window_size": window_size, "train_stride": train_stride,
            "smoothing_window": smooth_w, "n_estimators": n_estimators,
            "max_depth": max_depth, "max_samples": max_samples,
        }
        
        # Clean stream_rows to avoid circular JSON issues
        results.pop("stream_rows", None)
        
        with open(output_dir / f"fold_{test_subject}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved fold: fold_{test_subject}.json | Rep F1={results.get('rep_f1', 0):.4f}")
        all_results.append(results)
        
        if len(all_results) % 3 == 0 or len(all_results) == len(subjects):
            f1s = [r["rep_f1"] for r in all_results if "rep_f1" in r]
            if f1s:
                print(f"\n[Progress] Completed {len(all_results)}/{len(subjects)} folds")
                print(f"  Rep F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    
    overall = {
        "n_folds": len(all_results), "subjects": subjects,
        "streams": sum(r.get("stream_count", 0) for r in all_results),
        "n_true": sum(r["n_true"] for r in all_results), "n_pred": sum(r["n_pred"] for r in all_results),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": p, "recall": r, "rep_f1": f1,
        "start_mae_ms": float(np.mean([r.get("start_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("start_mae_ms", float("nan")))])),
        "end_mae_ms": float(np.mean([r.get("end_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("end_mae_ms", float("nan")))])),
        "transition_mae_ms": float(np.mean([r.get("transition_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("transition_mae_ms", float("nan")))])),
    }
    
    summary = {"model": "Causal Random Forest (LOSO)", "overall": overall, "fold_results": all_results}
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}\nLOSO SUMMARY\n{'='*60}")
    print(json.dumps(overall, indent=2))
    print(f"\n[OK] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/causal_rf_loso")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    run_loso(Path(args.config), Path(args.output), subjects=subjects, window_size=args.window_size,
             train_stride=args.train_stride, n_estimators=args.n_estimators, max_depth=args.max_depth,
             max_samples=args.max_samples, smoothing_window=args.smoothing_window)


if __name__ == "__main__":
    main()
