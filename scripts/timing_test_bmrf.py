#!/usr/bin/env python3
"""Quick batch eval with timing — ONE action only."""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Tuple

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


def main():
    action = "db_biceps_curl"
    print(f"Quick eval: {action}")
    
    streams = []
    for csv_path in sorted(DATA_ROOT.rglob("*.csv")):
        rel = csv_path.relative_to(DATA_ROOT)
        parts = rel.parts
        if len(parts) >= 4 and parts[2] == action and not parts[0].startswith("_"):
            streams.append(("/".join(parts), pd.read_csv(csv_path)))
    
    subjects = sorted(set(sid.split("/")[0] for sid, _ in streams))
    print(f"Streams: {len(streams)}, Subjects: {subjects}")
    
    mm_cfg = cb.MicroMacroConfig()
    
    for test_subject in subjects[:2]:
        print(f"\nTest subject: {test_subject}")
        t0 = time.time()
        
        train_streams = [(sid, df) for sid, df in streams if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in streams if sid.split("/")[0] == test_subject]
        
        stats = compute_train_stats([df for _, df in train_streams], IMU_COLUMNS)
        train_z = [(sid, apply_zscore(df, IMU_COLUMNS, stats)) for sid, df in train_streams]
        test_z = [(sid, apply_zscore(df, IMU_COLUMNS, stats)) for sid, df in test_streams]
        t1 = time.time()
        print(f"  Prep: {t1-t0:.1f}s")
        
        clf = crf.train_causal_rf(train_z, IMU_COLUMNS, window_size=100, stride=10, n_estimators=100, max_depth=15, max_samples=0.7)
        t2 = time.time()
        print(f"  Train: {t2-t1:.1f}s")
        
        # Cache train probs
        train_prob_cache = {}
        for sid, df in train_z:
            probs = crf.predict_causal_rf(clf, df, IMU_COLUMNS, window_size=100, stride=1)
            train_prob_cache[sid] = probs
        t3 = time.time()
        print(f"  Predict train: {t3-t2:.1f}s")
        
        # Fit refiner
        refiner = None
        # ... skip for timing test
        
        # Predict test
        for sid, df in test_z[:1]:
            probs = crf.predict_causal_rf(clf, df, IMU_COLUMNS, window_size=100, stride=1)
            gt_reps = cb.truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=mm_cfg.min_phase_samples)
            coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
            metrics = rep_metrics(coarse_reps, gt_reps, sample_rate_hz=cb.infer_sample_rate_hz(df))
            print(f"  Test {sid}: GT={metrics['n_true']}, Pred={metrics['n_pred']}, F1={metrics['f1']:.3f}")
        t4 = time.time()
        print(f"  Predict test: {t4-t3:.1f}s")


if __name__ == "__main__":
    main()
