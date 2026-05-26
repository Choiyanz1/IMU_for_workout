"""Ultra-fast SDTW test: 1 subject, 1 stream to check runtime."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.sdtw_rep_segmentation import (
    SDTWConfig,
    detect_reps_sdtw_templates,
    fit_sdtw_templates,
    infer_sample_rate_hz,
)
from preprocessing.micro_macro_segments import truth_reps_from_labels

# Load 1 train stream and 1 test stream
data_dir = ROOT / "datasets" / "raw_data"
train_stream = list((data_dir / "kevin" / "kevin" / "db_bench_press" / "set0").glob("*.csv"))[0]
test_stream = list((data_dir / "haoyu" / "haoyu0512workout" / "db_bench_press" / "set0").glob("*.csv"))[0]

train_df = pd.read_csv(train_stream)
test_df = pd.read_csv(test_stream)

print(f"Train stream: {len(train_df)} samples")
print(f"Test stream: {len(test_df)} samples")

imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
sdtw_cfg = SDTWConfig()

# Fit template
print("\n[Fitting template...]")
t0 = time.time()
templates = fit_sdtw_templates("db_bench_press", [train_df], imu_cols, sdtw_cfg)
print(f"  Templates: {len(templates)}, Time: {time.time()-t0:.1f}s")

# Detect
print("\n[Detecting reps...]")
t0 = time.time()
detections = detect_reps_sdtw_templates(test_df, templates, imu_cols, sdtw_cfg)
print(f"  Detections: {len(detections)}, Time: {time.time()-t0:.1f}s")

# GT
sample_rate = infer_sample_rate_hz(test_df)
gt_reps = truth_reps_from_labels(test_df["phase"].to_numpy(), min_phase_samples=1)
print(f"  GT reps: {len(gt_reps)}")

# Quick metrics
from preprocessing.sdtw_rep_segmentation import summarize_detection_metrics
truth = [(int(r.start_idx), int(r.end_idx)) for r in gt_reps]
metrics = summarize_detection_metrics(detections, truth, sample_rate)
print(f"\n  F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
print(f"  n_true={metrics['n_true']:.0f}, n_pred={metrics['n_pred']:.0f}")
