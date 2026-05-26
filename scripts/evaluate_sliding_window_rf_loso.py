"""LOSO evaluation of Sliding-window Random Forest (non-causal upper bound).

This baseline uses fixed sliding windows and leaks future information
at window edges. It is NOT deployable but provides a theoretical upper bound.

Usage:
    python scripts/evaluate_sliding_window_rf_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/sliding_window_rf_loso \
        --window-size 50 --stride 25
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


def _extract_subject(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def run_loso(config_path: Path, output_dir: Path, subjects: List[str] | None = None,
             window_size: int = 50, stride: int = 25, n_estimators: int = 100, max_depth: int = 15):
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
    
    all_stream_results: List[Dict] = []
    fold_summaries: List[Dict] = []
    
    for test_subject in subjects:
        if test_subject in completed:
            with open(output_dir / f"fold_{test_subject}.json") as f:
                fold_data = json.load(f)
            fold_summaries.append(fold_data["fold_summary"])
            all_stream_results.extend(fold_data["stream_results"])
            print(f"[Resume] Loaded fold: {test_subject}")
            continue
        
        train_streams = [(sid, df) for sid, df in streams if _extract_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if _extract_subject(sid) == test_subject]
        
        if not train_streams or not test_streams:
            print(f"[WARN] Skip {test_subject}: train={len(train_streams)}, test={len(test_streams)}")
            continue
        
        print(f"\n[Fold] test={test_subject}, train={len(train_streams)} streams, test={len(test_streams)} streams")
        
        t0 = time.time()
        clf = cb.train_rf_baseline(train_streams, imu_columns, window_size=window_size, stride=stride)
        train_time = time.time() - t0
        print(f"  Train time: {train_time:.1f}s")
        
        def rf_predict(df):
            probs = cb.predict_rf(clf, df, imu_columns, window_size=window_size, stride=stride)
            return probs, None
        
        t0 = time.time()
        fold_results = cb.evaluate_all_streams(rf_predict, test_streams, macro_classes, mm_cfg)
        eval_time = time.time() - t0
        print(f"  Eval time: {eval_time:.1f}s")
        print(f"  Rep F1: {fold_results['rep_f1']:.4f}  Precision: {fold_results['precision']:.4f}  Recall: {fold_results['recall']:.4f}")
        
        stream_results = fold_results.pop("stream_results", [])
        fold_summary = {
            "test_subject": test_subject,
            "n_train_streams": len(train_streams),
            "n_test_streams": len(test_streams),
            "train_time_s": train_time,
            "eval_time_s": eval_time,
            **{k: v for k, v in fold_results.items() if isinstance(v, (int, float, str))},
        }
        fold_summaries.append(fold_summary)
        all_stream_results.extend(stream_results)
        
        with open(output_dir / f"fold_{test_subject}.json", "w") as f:
            json.dump({"fold_summary": fold_summary, "stream_results": stream_results}, f, indent=2)
        print(f"[OK] Saved fold: {output_dir / f'fold_{test_subject}.json'}")
        
        # Save running summary
        summary = {
            "method": "Sliding-window Random Forest (non-causal upper bound)",
            "window_size": window_size,
            "stride": stride,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "subjects": subjects,
            "n_folds": len(fold_summaries),
            "rep_f1_mean": float(np.mean([f["rep_f1"] for f in fold_summaries])),
            "rep_f1_std": float(np.std([f["rep_f1"] for f in fold_summaries])),
            "precision_mean": float(np.mean([f["precision"] for f in fold_summaries])),
            "recall_mean": float(np.mean([f["recall"] for f in fold_summaries])),
            "start_mae_ms_mean": float(np.mean([f.get("start_mae_ms", 0) for f in fold_summaries])),
            "end_mae_ms_mean": float(np.mean([f.get("end_mae_ms", 0) for f in fold_summaries])),
            "fold_summaries": fold_summaries,
            "stream_results": all_stream_results,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("SLIDING-WINDOW RF BASELINE SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2, default=str))
    print(f"[OK] Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_comparison/sliding_window_rf_loso"))
    parser.add_argument("--subjects", type=str, default="haoyu,hsianshun,kevin,thomas,yoru,yushuan,yanz",
                        help="Comma-separated subject list")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    run_loso(
        config_path=args.config,
        output_dir=args.output,
        subjects=subjects,
        window_size=args.window_size,
        stride=args.stride,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
