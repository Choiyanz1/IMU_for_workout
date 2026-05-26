"""LOSO evaluation of Causal RF + Boundary Refiner with fixed hyperparameters.

This is a simplified LOSO wrapper over train_rf_boundary_refiner.py,
removing all inner tuning (no trailing_window_candidates or edge_window_candidates).

Usage:
    python scripts/evaluate_rf_refiner_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/rf_refiner_loso \
        --window-size 50 \
        --edge-window 20 \
        --n-estimators 50 \
        --max-depth 15
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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from existing modules
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

from evaluation.rep_segmentation import _write_segmentation_svg
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, OTHER_LABEL,
    RepDetection, SegmentRun, labels_to_runs, match_segments, rep_metrics,
    segment_iou_f1, truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import SegmentDetection, infer_sample_rate_hz


# Reuse functions from train_rf_boundary_refiner.py
_build_edge_features = base._build_edge_features
_train_refiners = base._train_refiners
_refine_reps = base._refine_reps
_evaluate_stream = base._evaluate_stream
_aggregate_rows = base._aggregate_rows
_phase_labels_from_reps = base._phase_labels_from_reps
_coarse_predict_reps = base._coarse_predict_reps


def _extract_subject_from_stream_id(stream_id: str) -> str:
    """Extract subject name from stream_id like 'kevin/session1/action/set1'."""
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def load_all_streams(config: dict, mm_cfg) -> List[Tuple[str, pd.DataFrame]]:
    """Load all streams from config."""
    data_cfg = config.get("data", {})
    feature_cfg = config.get("feature", {})
    window_cfg = config.get("window", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(config, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)
    return streams


def run_loso_fold(
    train_streams: List[Tuple[str, pd.DataFrame]],
    test_streams: List[Tuple[str, pd.DataFrame]],
    test_subject: str,
    imu_columns: Sequence[str],
    mm_cfg,
    window_size: int,
    edge_window: int,
    n_estimators: int,
    max_depth: int,
    max_samples: float,
    train_stride: int,
    match_iou: float,
    max_shift: int,
    target_matched_reps: int,
    max_refiner_train_streams: int,
) -> Dict:
    """Run one LOSO fold: train on train_streams, evaluate on test_streams."""
    print(f"\n[Fold] test_subject={test_subject} train_streams={len(train_streams)} test_streams={len(test_streams)}")
    
    # Compute z-score stats
    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    
    # Train causal RF
    t0 = time.time()
    clf = crf.train_causal_rf(
        train_z,
        imu_columns,
        window_size=int(window_size),
        stride=int(train_stride),
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples),
    )
    rf_time = time.time() - t0
    
    # Train refiner
    t0 = time.time()
    refiner = _train_refiners(
        train_z,
        clf,
        imu_columns,
        mm_cfg,
        window_size=int(window_size),
        edge_window=int(edge_window),
        match_iou=float(match_iou),
        max_shift=int(max_shift),
        target_matched_reps=int(target_matched_reps),
        max_refiner_train_streams=int(max_refiner_train_streams),
    )
    refiner_time = time.time() - t0
    
    # Evaluate test streams
    rows = []
    for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
        coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
        refined_reps = _refine_reps(
            df, probs, coarse_reps, refiner, imu_columns,
            edge_window=int(edge_window), max_shift=int(max_shift),
        )
        sample_rate = infer_sample_rate_hz(df)
        truth, metrics = _evaluate_stream(df, probs, refined_reps, sample_rate)
        row = {
            **metrics,
            "stream_id": stream_id,
            "count_diff": float(metrics.get("n_pred", 0.0) - metrics.get("n_true", 0.0)),
        }
        rows.append(row)
        if stream_idx % 10 == 0 or stream_idx == len(test_z):
            print(f"  [RFRefiner] evaluated {stream_idx}/{len(test_z)} test streams", flush=True)
    
    results = _aggregate_rows(rows)
    results["model_name"] = "Causal RF + Boundary Refiner"
    results["evaluation_protocol"] = "loso"
    results["test_subject"] = test_subject
    results["rf_train_time_s"] = rf_time
    results["refiner_train_time_s"] = refiner_time
    results["config"] = {
        "window_size": int(window_size),
        "train_stride": int(train_stride),
        "edge_window": int(edge_window),
        "match_iou_train": float(match_iou),
        "max_shift": int(max_shift),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "max_samples": float(max_samples),
    }
    
    # Extract per-action breakdown if available
    if "action_type" in test_z[0][1].columns:
        action_rows = {}
        for stream_id, df in test_z:
            action = str(df["action_type"].iloc[0]) if "action_type" in df.columns else "unknown"
            if action not in action_rows:
                action_rows[action] = []
        # Re-evaluate per action
        for action in action_rows:
            action_test = [(sid, df) for sid, df in test_z 
                          if "action_type" in df.columns and str(df["action_type"].iloc[0]) == action]
            if not action_test:
                continue
            a_rows = []
            for stream_id, df in action_test:
                probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
                coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
                refined_reps = _refine_reps(
                    df, probs, coarse_reps, refiner, imu_columns,
                    edge_window=int(edge_window), max_shift=int(max_shift),
                )
                sample_rate = infer_sample_rate_hz(df)
                truth, metrics = _evaluate_stream(df, probs, refined_reps, sample_rate)
                a_rows.append({**metrics, "stream_id": stream_id})
            action_rows[action] = _aggregate_rows(a_rows)
        results["by_action"] = action_rows
    
    return results


def run_loso(
    config_path: Path,
    output_dir: Path,
    subjects: List[str] | None = None,
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
    """Run full LOSO evaluation."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    # Load all data
    print("[INFO] Loading all streams...")
    all_streams = load_all_streams(raw, mm_cfg)
    print(f"[INFO] Loaded {len(all_streams)} streams")
    
    # Get unique subjects
    all_subjects = sorted({_extract_subject_from_stream_id(sid) for sid, _ in all_streams})
    if subjects is None:
        subjects = all_subjects
    print(f"[INFO] LOSO folds: {subjects}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for completed folds (resume support)
    completed = set()
    for subject in subjects:
        fold_file = output_dir / f"fold_{subject}.json"
        if fold_file.exists():
            completed.add(subject)
            print(f"[Resume] Found completed fold: {subject}")
    
    # Run each fold
    all_results = []
    for test_subject in subjects:
        if test_subject in completed:
            # Load existing result
            fold_file = output_dir / f"fold_{test_subject}.json"
            with open(fold_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            all_results.append(results)
            continue
        
        # Split streams
        train_streams = [(sid, df) for sid, df in all_streams 
                        if _extract_subject_from_stream_id(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams 
                       if _extract_subject_from_stream_id(sid) == test_subject]
        
        if not test_streams:
            print(f"[WARN] No test streams for subject {test_subject}, skipping")
            continue
        
        # Run fold
        results = run_loso_fold(
            train_streams, test_streams, test_subject,
            imu_columns, mm_cfg,
            window_size=window_size,
            edge_window=edge_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            train_stride=train_stride,
            match_iou=match_iou,
            max_shift=max_shift,
            target_matched_reps=target_matched_reps,
            max_refiner_train_streams=max_refiner_train_streams,
        )
        
        # Save fold result immediately
        fold_file = output_dir / f"fold_{test_subject}.json"
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved fold result: {fold_file}")
        
        all_results.append(results)
        
        # Print real-time summary every 3 folds
        if len(all_results) % 3 == 0 or len(all_results) == len(subjects):
            f1s = [r["rep_f1"] for r in all_results if "rep_f1" in r]
            if f1s:
                print(f"\n[Progress] Completed {len(all_results)}/{len(subjects)} folds")
                print(f"  Rep F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
                print(f"  Individual: {', '.join(f'{s}={f:.3f}' for s, f in zip(subjects[:len(f1s)], f1s))}")
    
    # Aggregate across all folds
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    overall = {
        "n_folds": len(all_results),
        "subjects": subjects,
        "streams": sum(r["stream_count"] for r in all_results),
        "n_true": sum(r["n_true"] for r in all_results),
        "n_pred": sum(r["n_pred"] for r in all_results),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "rep_f1": f1,
        "start_mae_ms": float(np.mean([r.get("start_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("start_mae_ms", float("nan")))])),
        "end_mae_ms": float(np.mean([r.get("end_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("end_mae_ms", float("nan")))])),
        "transition_mae_ms": float(np.mean([r.get("transition_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("transition_mae_ms", float("nan")))])),
    }
    
    # Save final summary
    summary = {
        "model": "Causal RF + Boundary Refiner (LOSO)",
        "overall": overall,
        "fold_results": all_results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("LOSO SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(overall, indent=2))
    print(f"\n[OK] Results saved to {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="LOSO evaluation of Causal RF + Boundary Refiner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/rf_refiner_loso")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--edge-window", type=int, default=20)
    parser.add_argument("--match-iou-train", type=float, default=0.3)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--target-matched-reps", type=int, default=1200)
    parser.add_argument("--max-refiner-train-streams", type=int, default=100)
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    
    run_loso(
        Path(args.config),
        Path(args.output),
        subjects=subjects,
        window_size=args.window_size,
        edge_window=args.edge_window,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        train_stride=args.train_stride,
        match_iou=args.match_iou_train,
        max_shift=args.max_shift,
        target_matched_reps=args.target_matched_reps,
        max_refiner_train_streams=args.max_refiner_train_streams,
    )


if __name__ == "__main__":
    main()
