"""Diagnostic script: compare coarse vs refined predictions on a specific fold.

Usage:
    python scripts/diagnose_refiner_failure.py --config config.yaml --action db_bench_press --test-subject hsianshun
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
base = _load_mod(ROOT / "scripts" / "train_rf_boundary_refiner.py", "train_rf_boundary_refiner_mod")

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, OTHER_LABEL,
    RepDetection, SegmentRun, labels_to_runs, match_segments, rep_metrics,
    segment_iou_f1, truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz


_build_edge_features = base._build_edge_features
_train_refiners = base._train_refiners
_refine_reps = base._refine_reps
_evaluate_stream = base._evaluate_stream
_aggregate_rows = base._aggregate_rows


def _coarse_predict_reps(df: pd.DataFrame, probs: np.ndarray, mm_cfg) -> Tuple[List[RepDetection], list]:
    labels = cb._decode_phase_labels(probs, mm_cfg)
    runs = labels_to_runs(
        labels,
        positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
        probabilities=probs,
        min_length=mm_cfg.min_phase_samples,
    )
    reps, _ = cb.pair_concentric_eccentric_reps(runs, micro_source="causal_rf", max_gap_samples=mm_cfg.max_phase_gap_samples)
    reps = cb._filter_predicted_reps(
        reps,
        sample_rate_hz=infer_sample_rate_hz(df),
        min_duration_seconds=mm_cfg.min_rep_duration_seconds,
        min_confidence=mm_cfg.min_rep_confidence,
    )
    return reps, runs


def _extract_action(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def _extract_subject(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def diagnose_fold(
    config_path: Path,
    action: str,
    test_subject: str,
    window_size: int = 50,
    edge_window: int = 20,
    n_estimators: int = 50,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    match_iou: float = 0.3,
    max_shift: int = 20,
    target_matched_reps: int = 1200,
    max_refiner_train_streams: int = 100,
):
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
    all_streams, all_subjects, actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)
    
    # Filter for action
    action_streams = [(sid, df) for sid, df in all_streams if _extract_action(sid) == action]
    
    train_streams = [(sid, df) for sid, df in action_streams if _extract_subject(sid) != test_subject]
    test_streams = [(sid, df) for sid, df in action_streams if _extract_subject(sid) == test_subject]
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: action={action}, test_subject={test_subject}")
    print(f"{'='*60}")
    print(f"Train streams: {len(train_streams)}")
    print(f"Test streams: {len(test_streams)}")
    
    # Compute stats
    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    
    # Train RF
    print("\n[1] Training Causal RF...")
    clf = crf.train_causal_rf(
        train_z, imu_columns,
        window_size=int(window_size), stride=int(train_stride),
        n_estimators=int(n_estimators), max_depth=int(max_depth), max_samples=float(max_samples),
    )
    
    # Get coarse predictions on test
    print("\n[2] Coarse predictions on test streams:")
    coarse_rows = []
    for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
        coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
        sample_rate = infer_sample_rate_hz(df)
        truth, metrics = _evaluate_stream(df, probs, coarse_reps, sample_rate)
        coarse_rows.append({**metrics, "stream_id": stream_id, "n_coarse": len(coarse_reps), "n_truth": len(truth)})
        print(f"  {stream_id}: coarse_reps={len(coarse_reps)}, truth={len(truth)}, F1={metrics.get('f1', 0):.4f}")
    
    coarse_summary = _aggregate_rows(coarse_rows)
    print(f"\n[Coarse Summary] F1={coarse_summary.get('rep_f1', 0):.4f}, P={coarse_summary.get('precision', 0):.4f}, R={coarse_summary.get('recall', 0):.4f}")
    
    # Train refiner
    print("\n[3] Training Refiner...")
    refiner = _train_refiners(
        train_z, clf, imu_columns, mm_cfg,
        window_size=int(window_size), edge_window=int(edge_window),
        match_iou=float(match_iou), max_shift=int(max_shift),
        target_matched_reps=int(target_matched_reps),
        max_refiner_train_streams=int(max_refiner_train_streams),
    )
    
    # Get refined predictions
    print("\n[4] Refined predictions on test streams:")
    refined_rows = []
    for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
        coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
        refined_reps = _refine_reps(
            df, probs, coarse_reps, refiner, imu_columns,
            edge_window=int(edge_window), max_shift=int(max_shift),
        )
        sample_rate = infer_sample_rate_hz(df)
        truth, metrics = _evaluate_stream(df, probs, refined_reps, sample_rate)
        refined_rows.append({**metrics, "stream_id": stream_id, "n_coarse": len(coarse_reps), "n_refined": len(refined_reps), "n_truth": len(truth)})
        print(f"  {stream_id}: coarse={len(coarse_reps)}, refined={len(refined_reps)}, truth={len(truth)}, F1={metrics.get('f1', 0):.4f}")
    
    refined_summary = _aggregate_rows(refined_rows)
    print(f"\n[Refined Summary] F1={refined_summary.get('rep_f1', 0):.4f}, P={refined_summary.get('precision', 0):.4f}, R={refined_summary.get('recall', 0):.4f}")
    
    print(f"\n{'='*60}")
    print("COMPARISON:")
    print(f"  Coarse  F1: {coarse_summary.get('rep_f1', 0):.4f}")
    print(f"  Refined F1: {refined_summary.get('rep_f1', 0):.4f}")
    print(f"  Delta:      {refined_summary.get('rep_f1', 0) - coarse_summary.get('rep_f1', 0):+.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--action", required=True)
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--edge-window", type=int, default=20)
    args = parser.parse_args()
    
    diagnose_fold(
        Path(args.config), args.action, args.test_subject,
        window_size=args.window_size, edge_window=args.edge_window,
    )


if __name__ == "__main__":
    main()
