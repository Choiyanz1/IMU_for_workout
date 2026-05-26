"""
Seq2Seq Post-processing Parameter Search: Find optimal smoothing + parser params
for reducing over-segmentation while maintaining high phase accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    RepDetection, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, _prepare_phase_labels, evaluate_reps, evaluate_phase,
    extract_active_segments, predict_active, predict_seq2seq_phase,
    smooth_phase_probs, parse_reps_from_phase, train_active_detector, train_seq2seq_phase,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Config Space
# ---------------------------------------------------------------------------

SMOOTHING_WINDOWS = [15, 20, 25, 30, 40]
MIN_PHASE_SAMPLES_LIST = [3, 5, 7, 10]
MAX_GAP_SAMPLES_LIST = [0, 3, 5]


def _median_smooth_phase_probs(phase_probs: np.ndarray, window: int = 15) -> np.ndarray:
    """Apply median filter per class channel."""
    from scipy.ndimage import median_filter
    smoothed = np.copy(phase_probs)
    for c in range(2):
        smoothed[:, c] = median_filter(phase_probs[:, c], size=window, mode='reflect')
    return smoothed


def parse_reps_with_params(phase_probs, min_phase_samples, max_gap_samples):
    """Parse reps with given parameters."""
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase_samples)
    
    # Merge adjacent same phase
    if not runs:
        return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(
                label=run.label,
                start_idx=merged[-1].start_idx,
                end_idx=run.end_idx,
                confidence=(merged[-1].confidence + run.confidence) / 2,
            )
        else:
            merged.append(run)
    
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="phase", max_gap_samples=max_gap_samples)
    return reps


def run_param_search(train_streams, test_streams, cfg: PhaseCompareConfig, output_dir: Path):
    """Grid search over smoothing and parser parameters for Seq2Seq."""
    print("=" * 70)
    print("Seq2Seq Post-processing Parameter Search")
    print("=" * 70)
    
    # Train models once
    print("\n[1/2] Training models...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    seq2seq_model, seq2seq_mean, seq2seq_std = train_seq2seq_phase(train_streams, cfg.imu_columns, cfg)
    print("      Done.")
    
    if seq2seq_model is None:
        print("ERROR: Seq2Seq model training failed!")
        return {}
    
    # Pre-compute all test stream predictions
    print("\n[2/2] Running grid search...")
    
    all_results = []
    
    for sw in SMOOTHING_WINDOWS:
        for mps in MIN_PHASE_SAMPLES_LIST:
            for mgs in MAX_GAP_SAMPLES_LIST:
                variant_name = f"sw{sw}_mps{mps}_mgs{mgs}"
                print(f"  Testing: {variant_name}...")
                
                results = []
                for stream_id, df in test_streams:
                    if "phase" not in df.columns:
                        continue
                    
                    # Active detection
                    active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
                    active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
                    
                    # Seq2Seq prediction
                    sq_phase_probs = predict_seq2seq_phase(
                        seq2seq_model, df, active_segments, cfg.imu_columns,
                        seq2seq_mean, seq2seq_std, cfg
                    )
                    
                    # Smoothing
                    sq_phase_probs_smooth = smooth_phase_probs(sq_phase_probs, sw)
                    
                    # Ground truth
                    gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)
                    
                    # Parse reps
                    sq_pred_reps = parse_reps_with_params(sq_phase_probs_smooth, mps, mgs)
                    
                    # Evaluate
                    sq_rep_metrics = evaluate_reps(sq_pred_reps, gt_reps)
                    sq_phase_metrics = evaluate_phase(sq_phase_probs_smooth, df["phase"].to_numpy())
                    
                    results.append({
                        "stream_id": stream_id,
                        **sq_rep_metrics,
                        **sq_phase_metrics,
                    })
                
                # Aggregate
                valid = [r for r in results if "f1" in r]
                if not valid:
                    continue
                
                total_tp = sum(r["tp"] for r in valid)
                total_fp = sum(r["fp"] for r in valid)
                total_fn = sum(r["fn"] for r in valid)
                p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                
                exact = sum(r["exact_count"] for r in valid)
                over = sum(r["over"] for r in valid)
                under = sum(r["under"] for r in valid)
                n = len(valid)
                
                phase_macro_f1_values = [r["phase_macro_f1"] for r in valid]
                phase_acc_values = [r["phase_accuracy"] for r in valid]
                trans_mae_values = [r["transition_mae_ms"] for r in valid if r.get("transition_mae_ms") is not None]
                
                agg = {
                    "variant": variant_name,
                    "smoothing_window": sw,
                    "min_phase_samples": mps,
                    "max_gap_samples": mgs,
                    "streams": n,
                    "rep_precision": p,
                    "rep_recall": r,
                    "rep_f1": f1,
                    "exact_count_acc": exact / n if n > 0 else 0,
                    "over_count": over,
                    "under_count": under,
                    "over_rate": over / n if n > 0 else 0,
                    "under_rate": under / n if n > 0 else 0,
                    "phase_macro_f1": np.mean(phase_macro_f1_values) if phase_macro_f1_values else 0,
                    "phase_accuracy": np.mean(phase_acc_values) if phase_acc_values else 0,
                    "transition_mae_ms": np.mean(trans_mae_values) if trans_mae_values else None,
                }
                
                all_results.append(agg)
                print(f"    Rep F1: {f1:.4f}, Exact: {agg['exact_count_acc']:.4f}, "
                      f"Over/Under: {over}/{under}, Phase F1: {agg['phase_macro_f1']:.4f}, "
                      f"Trans MAE: {agg['transition_mae_ms']:.1f}ms")
    
    # Sort by different criteria
    print(f"\n{'=' * 70}")
    print("TOP CONFIGURATIONS BY DIFFERENT CRITERIA")
    print(f"{'=' * 70}")
    
    # Best Rep F1
    best_rep_f1 = max(all_results, key=lambda x: x["rep_f1"])
    print(f"\n[Best Rep F1] {best_rep_f1['variant']}")
    print(f"  Rep F1: {best_rep_f1['rep_f1']:.4f}")
    print(f"  Exact Count: {best_rep_f1['exact_count_acc']:.4f}")
    print(f"  Over/Under: {best_rep_f1['over_count']}/{best_rep_f1['under_count']}")
    print(f"  Phase F1: {best_rep_f1['phase_macro_f1']:.4f}")
    
    # Best Exact Count
    best_exact = max(all_results, key=lambda x: x["exact_count_acc"])
    print(f"\n[Best Exact Count] {best_exact['variant']}")
    print(f"  Rep F1: {best_exact['rep_f1']:.4f}")
    print(f"  Exact Count: {best_exact['exact_count_acc']:.4f}")
    print(f"  Over/Under: {best_exact['over_count']}/{best_exact['under_count']}")
    print(f"  Phase F1: {best_exact['phase_macro_f1']:.4f}")
    
    # Best Balance (minimize |over - under|)
    best_balance = min(all_results, key=lambda x: abs(x["over_count"] - x["under_count"]))
    print(f"\n[Best Balance] {best_balance['variant']}")
    print(f"  Rep F1: {best_balance['rep_f1']:.4f}")
    print(f"  Exact Count: {best_balance['exact_count_acc']:.4f}")
    print(f"  Over/Under: {best_balance['over_count']}/{best_balance['under_count']}")
    print(f"  Phase F1: {best_balance['phase_macro_f1']:.4f}")
    
    # Best Phase F1
    best_phase = max(all_results, key=lambda x: x["phase_macro_f1"])
    print(f"\n[Best Phase F1] {best_phase['variant']}")
    print(f"  Rep F1: {best_phase['rep_f1']:.4f}")
    print(f"  Exact Count: {best_phase['exact_count_acc']:.4f}")
    print(f"  Over/Under: {best_phase['over_count']}/{best_phase['under_count']}")
    print(f"  Phase F1: {best_phase['phase_macro_f1']:.4f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "seq2seq_param_search.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved to {out_path}")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Seq2Seq Post-processing Parameter Search")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/seq2seq_param_search"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: kevin only")
    args = parser.parse_args()
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    
    test_subjects = ["kevin"] if args.quick else subjects
    train_streams = [(sid, df) for sid, df in all_streams if not any(sid.startswith(f"{ts}/") for ts in test_subjects)]
    test_streams = [(sid, df) for sid, df in all_streams if any(sid.startswith(f"{ts}/") for ts in test_subjects)]
    
    cfg = PhaseCompareConfig()
    cfg.seq2seq_epochs = 30  # Use early stopping model
    
    run_param_search(train_streams, test_streams, cfg, args.output)


if __name__ == "__main__":
    main()
