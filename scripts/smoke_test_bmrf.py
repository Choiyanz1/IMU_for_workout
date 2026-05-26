#!/usr/bin/env python3
"""Smoke test: evaluate ONE action with browse_model_replay.py RF model."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from preprocessing.micro_macro_segments import rep_metrics
import compare_baselines as cb
import importlib.util


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_mod(ROOT / "scripts" / "train_rf_boundary_refiner.py", "train_rf_boundary_refiner_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")

DATA_ROOT = ROOT / "datasets" / "raw_data"
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def _load_action_streams(action: str):
    streams = []
    for csv_path in sorted(DATA_ROOT.rglob("*.csv")):
        rel = csv_path.relative_to(DATA_ROOT)
        parts = rel.parts
        if len(parts) >= 4 and parts[2] == action:
            try:
                df = pd.read_csv(csv_path)
                sid = "/".join(parts)
                streams.append((sid, df))
            except Exception as e:
                print(f"  Skip {csv_path}: {e}")
    return streams


def main():
    action = "db_biceps_curl"
    print(f"Smoke test: {action}")
    
    all_streams = _load_action_streams(action)
    print(f"  Loaded {len(all_streams)} streams")
    
    subjects = sorted(set(sid.split("/")[0] for sid, _ in all_streams))
    print(f"  Subjects: {subjects}")
    
    mm_cfg = cb.MicroMacroConfig()
    
    for test_subject in subjects[:1]:  # Just first subject
        print(f"\n  Test subject: {test_subject}")
        train_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] == test_subject]
        print(f"    Train: {len(train_streams)} streams, Test: {len(test_streams)} streams")
        
        if not test_streams:
            continue
        
        stats = compute_train_stats([df for _, df in train_streams], IMU_COLUMNS)
        print(f"    Stats computed")
        
        train_z = [(sid, apply_zscore(df, IMU_COLUMNS, stats)) for sid, df in train_streams]
        test_z = [(sid, apply_zscore(df, IMU_COLUMNS, stats)) for sid, df in test_streams]
        
        # Train RF
        print(f"    Training RF...")
        clf = crf.train_causal_rf(
            train_z, IMU_COLUMNS,
            window_size=100, stride=10,
            n_estimators=100, max_depth=15, max_samples=0.7,
        )
        print(f"    RF trained")
        
        # Predict one test stream
        sid, df = test_z[0]
        probs = crf.predict_causal_rf(clf, df, IMU_COLUMNS, window_size=100, stride=1)
        print(f"    Predicted {sid}: probs shape={probs.shape}")
        
        gt_reps = cb.truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        print(f"    Coarse reps: {len(coarse_reps)}, GT reps: {len(gt_reps)}")
        
        metrics = rep_metrics(coarse_reps, gt_reps, sample_rate_hz=cb.infer_sample_rate_hz(df))
        print(f"    Metrics: F1={metrics['f1']:.3f}, Pred={metrics['n_pred']}, True={metrics['n_true']}")
        
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
