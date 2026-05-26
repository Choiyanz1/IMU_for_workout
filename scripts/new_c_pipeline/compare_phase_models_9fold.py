"""
9-Fold LOSO Phase Model Comparison: RF Window-based vs CNN Seq2Seq (FIXED architecture)

Fixed parameters (no fold-specific tuning):
  - Seq2Seq architecture: 5-layer dilated CNN with GroupNorm, residual connections
  - Training: Adam, lr=1e-3, early stopping (patience=10), 30 epochs max
  - Post-processing tested at TWO fixed configs:
    * Default (fair vs RF):  sw=15, mps=3, mgs=3
    * Conservative (seq2seq-appropriate): sw=30, mps=7, mgs=3

Outputs per-fold and summary with C/E ratio analysis.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import accuracy_score, f1_score

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    RepDetection, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig,
    evaluate_phase,
    evaluate_reps,
    extract_active_segments,
    predict_active,
    predict_rf_phase,
    predict_seq2seq_phase,
    smooth_phase_probs,
    train_active_detector,
    train_rf_phase,
    train_seq2seq_phase,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Fixed configs
# ---------------------------------------------------------------------------

@dataclass
class PostProcConfig:
    smoothing_window: int
    min_phase_samples: int
    max_gap_samples: int

DEFAULT_PP = PostProcConfig(smoothing_window=15, min_phase_samples=3, max_gap_samples=3)
CONSERVATIVE_PP = PostProcConfig(smoothing_window=30, min_phase_samples=7, max_gap_samples=3)


def parse_reps_with_config(phase_probs: np.ndarray, pp: PostProcConfig) -> List[RepDetection]:
    """Parse reps with a fixed post-processing config."""
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    runs = labels_to_runs(
        pred_phase,
        positive_labels={"eccentric", "concentric"},
        min_length=pp.min_phase_samples,
    )
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
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="phase", max_gap_samples=pp.max_gap_samples)
    return reps


# ---------------------------------------------------------------------------
# C/E Ratio helpers
# ---------------------------------------------------------------------------

def compute_rep_ce_ratios(reps: List[RepDetection], phase_labels: np.ndarray) -> List[float]:
    """Compute concentric/eccentric duration ratio for each rep.
    phase_labels: array of 'concentric'/'eccentric' strings.
    Returns list of ratios (C_duration / E_duration).
    """
    ratios = []
    for rep in reps:
        seg = phase_labels[rep.start_idx:rep.end_idx]
        if len(seg) == 0:
            ratios.append(float('nan'))
            continue
        c_count = np.sum(seg == CONCENTRIC_LABEL)
        e_count = np.sum(seg == ECCENTRIC_LABEL)
        if e_count == 0:
            ratios.append(float('inf'))
        else:
            ratios.append(c_count / e_count)
    return ratios


def compute_ce_ratio_metrics(pred_ratios: List[float], gt_ratios: List[float]) -> dict:
    """Compute MAE, RMSE, bias of C/E ratios. Only on finite, non-NaN pairs."""
    valid_pairs = []
    for p, g in zip(pred_ratios, gt_ratios):
        if np.isfinite(p) and np.isfinite(g) and p != float('inf') and g != float('inf'):
            valid_pairs.append((p, g))
    if not valid_pairs:
        return {"ce_ratio_mae": None, "ce_ratio_rmse": None, "ce_ratio_bias": None, "n_valid": 0}
    pred_arr = np.array([p for p, _ in valid_pairs])
    gt_arr = np.array([g for _, g in valid_pairs])
    errors = pred_arr - gt_arr
    return {
        "ce_ratio_mae": float(np.mean(np.abs(errors))),
        "ce_ratio_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ce_ratio_bias": float(np.mean(errors)),
        "n_valid": len(valid_pairs),
    }


# ---------------------------------------------------------------------------
# Per-stream evaluation
# ---------------------------------------------------------------------------

def evaluate_stream(
    stream_id: str,
    df: pd.DataFrame,
    phase_probs: np.ndarray,
    pp: PostProcConfig,
    gt_reps: List[RepDetection],
    gt_phases: np.ndarray,
) -> dict:
    """Evaluate a single stream with given phase predictions and post-processing config."""
    phase_probs_smooth = smooth_phase_probs(phase_probs, pp.smoothing_window)
    pred_reps = parse_reps_with_config(phase_probs_smooth, pp)
    
    rep_metrics = evaluate_reps(pred_reps, gt_reps)
    phase_metrics = evaluate_phase(phase_probs_smooth, gt_phases)
    
    # C/E ratios
    pred_ratios = compute_rep_ce_ratios(pred_reps, np.array(
        ["eccentric" if p == 0 else "concentric" for p in np.argmax(phase_probs_smooth, axis=1)]
    ))
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)
    
    # Count error
    count_error = abs(rep_metrics["pred_count"] - rep_metrics["gt_count"])
    
    return {
        "stream_id": stream_id,
        "pred_count": rep_metrics["pred_count"],
        "gt_count": rep_metrics["gt_count"],
        "count_error": count_error,
        **{k: v for k, v in rep_metrics.items() if k not in ["pred_count", "gt_count"]},
        **phase_metrics,
        **ce_metrics,
    }


def aggregate_fold_results(results: List[dict]) -> dict:
    """Aggregate results across all test streams in a fold."""
    if not results:
        return {}
    
    n = len(results)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    exact_count = sum(r["exact_count"] for r in results)
    over_count = sum(r["over"] for r in results)
    under_count = sum(r["under"] for r in results)
    count_errors = [r["count_error"] for r in results]
    
    trans_mae_list = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    phase_f1_list = [r["phase_macro_f1"] for r in results]
    phase_acc_list = [r["phase_accuracy"] for r in results]
    
    ce_mae_list = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]
    
    return {
        "streams": n,
        "rep_precision": p,
        "rep_recall": r,
        "rep_f1": f1,
        "exact_count_acc": exact_count / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": over_count,
        "under_count": under_count,
        "phase_macro_f1": np.mean(phase_f1_list) if phase_f1_list else 0,
        "phase_accuracy": np.mean(phase_acc_list) if phase_acc_list else 0,
        "transition_mae_ms": np.mean(trans_mae_list) if trans_mae_list else None,
        "ce_ratio_mae": np.mean(ce_mae_list) if ce_mae_list else None,
    }


# ---------------------------------------------------------------------------
# Main 9-fold evaluation
# ---------------------------------------------------------------------------

def run_9fold_evaluation(all_streams, subjects, cfg: PhaseCompareConfig, output_dir: Path):
    print("=" * 80)
    print("9-Fold LOSO Phase Model Comparison: RF vs Seq2Seq")
    print("=" * 80)
    print(f"Subjects ({len(subjects)}): {subjects}")
    
    rf_fold_results = []
    seq2seq_default_fold_results = []
    seq2seq_conservative_fold_results = []
    
    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/9: test={test_subject}")
        print(f"{'=' * 80}")
        
        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
        print(f"  train={len(train_streams)}, test={len(test_streams)}")
        
        # Train all models once per fold
        print("  Training Active Detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        
        print("  Training RF Phase Model...")
        rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, cfg)
        
        print("  Training Seq2Seq Phase Model...")
        seq2seq_model, seq2seq_mean, seq2seq_std = train_seq2seq_phase(train_streams, cfg.imu_columns, cfg)
        
        # Evaluate each test stream
        rf_stream_results = []
        sq_def_stream_results = []
        sq_con_stream_results = []
        
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=cfg.min_phase_samples)
            
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            # RF
            rf_phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, cfg)
            rf_stream_results.append(evaluate_stream(stream_id, df, rf_phase_probs, DEFAULT_PP, gt_reps, gt_phases))
            
            # Seq2Seq (only if trained successfully)
            if seq2seq_model is not None:
                sq_phase_probs = predict_seq2seq_phase(
                    seq2seq_model, df, active_segments, cfg.imu_columns,
                    seq2seq_mean, seq2seq_std, cfg
                )
                sq_def_stream_results.append(evaluate_stream(stream_id, df, sq_phase_probs, DEFAULT_PP, gt_reps, gt_phases))
                sq_con_stream_results.append(evaluate_stream(stream_id, df, sq_phase_probs, CONSERVATIVE_PP, gt_reps, gt_phases))
        
        # Aggregate fold
        rf_fold = aggregate_fold_results(rf_stream_results)
        rf_fold["fold"] = fold_idx + 1
        rf_fold["test_subject"] = test_subject
        rf_fold_results.append(rf_fold)
        
        if seq2seq_model is not None:
            sq_def_fold = aggregate_fold_results(sq_def_stream_results)
            sq_def_fold["fold"] = fold_idx + 1
            sq_def_fold["test_subject"] = test_subject
            seq2seq_default_fold_results.append(sq_def_fold)
            
            sq_con_fold = aggregate_fold_results(sq_con_stream_results)
            sq_con_fold["fold"] = fold_idx + 1
            sq_con_fold["test_subject"] = test_subject
            seq2seq_conservative_fold_results.append(sq_con_fold)
        
        print(f"  RF:  RepF1={rf_fold['rep_f1']:.4f} Exact={rf_fold['exact_count_acc']:.3f} "
              f"PhaseF1={rf_fold['phase_macro_f1']:.4f} TransMAE={rf_fold.get('transition_mae_ms', 0):.0f}ms")
        if seq2seq_model is not None:
            print(f"  SqD: RepF1={sq_def_fold['rep_f1']:.4f} Exact={sq_def_fold['exact_count_acc']:.3f} "
                  f"PhaseF1={sq_def_fold['phase_macro_f1']:.4f} TransMAE={sq_def_fold.get('transition_mae_ms', 0):.0f}ms")
            print(f"  SqC: RepF1={sq_con_fold['rep_f1']:.4f} Exact={sq_con_fold['exact_count_acc']:.3f} "
                  f"PhaseF1={sq_con_fold['phase_macro_f1']:.4f} TransMAE={sq_con_fold.get('transition_mae_ms', 0):.0f}ms")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("9-FOLD SUMMARY")
    print(f"{'=' * 80}")
    
    summary = {}
    for name, fold_results in [
        ("RF", rf_fold_results),
        ("Seq2Seq_Default", seq2seq_default_fold_results),
        ("Seq2Seq_Conservative", seq2seq_conservative_fold_results),
    ]:
        if not fold_results:
            continue
        
        print(f"\n[{name}] ({len(fold_results)} folds)")
        
        metrics = ["rep_f1", "exact_count_acc", "mean_abs_count_error", "over_count", "under_count",
                   "phase_macro_f1", "phase_accuracy", "transition_mae_ms", "ce_ratio_mae"]
        
        for metric in metrics:
            values = [f[metric] for f in fold_results if f.get(metric) is not None]
            if not values:
                continue
            mean = np.mean(values)
            std = np.std(values)
            best = np.max(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.min(values)
            worst = np.min(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.max(values)
            print(f"  {metric}: mean={mean:.4f} std={std:.4f} best={best:.4f} worst={worst:.4f}")
            summary[f"{name}_{metric}_mean"] = mean
            summary[f"{name}_{metric}_std"] = std
        
        # Per-fold details
        for f in fold_results:
            print(f"    Fold {f['fold']} ({f['test_subject']}): "
                  f"RepF1={f['rep_f1']:.4f} Exact={f['exact_count_acc']:.3f} "
                  f"PhaseF1={f['phase_macro_f1']:.4f} TransMAE={f.get('transition_mae_ms', 0):.0f}ms "
                  f"Over/Under={f['over_count']}/{f['under_count']}")
    
    # Compare Seq2Seq vs RF
    if seq2seq_default_fold_results and rf_fold_results:
        print(f"\n{'=' * 80}")
        print("SEQ2SEQ vs RF COMPARISON")
        print(f"{'=' * 80}")
        
        for label, sq_folds in [
            ("Seq2Seq (Default PP)", seq2seq_default_fold_results),
            ("Seq2Seq (Conservative PP)", seq2seq_conservative_fold_results),
        ]:
            if not sq_folds:
                continue
            print(f"\n{label}:")
            
            comparisons = {
                "rep_f1": "higher is better",
                "phase_macro_f1": "higher is better",
                "phase_accuracy": "higher is better",
                "exact_count_acc": "higher is better",
                "transition_mae_ms": "lower is better",
                "mean_abs_count_error": "lower is better",
            }
            
            for metric, direction in comparisons.items():
                sq_wins = 0
                for sq_f, rf_f in zip(sq_folds, rf_fold_results):
                    sq_val = sq_f.get(metric)
                    rf_val = rf_f.get(metric)
                    if sq_val is None or rf_val is None:
                        continue
                    if direction == "higher is better":
                        if sq_val > rf_val:
                            sq_wins += 1
                    else:
                        if sq_val < rf_val:
                            sq_wins += 1
                print(f"  {metric}: Seq2Seq wins in {sq_wins}/{len(sq_folds)} folds")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_9fold.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "rf_per_fold": rf_fold_results,
            "seq2seq_default_per_fold": seq2seq_default_fold_results,
            "seq2seq_conservative_per_fold": seq2seq_conservative_fold_results,
        }, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}")
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="9-Fold LOSO Phase Model Comparison")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_model_comparison_9fold"))
    args = parser.parse_args()
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects: {subjects}")
    
    cfg = PhaseCompareConfig()
    cfg.seq2seq_epochs = 30
    
    run_9fold_evaluation(all_streams, subjects, cfg, args.output)


if __name__ == "__main__":
    main()
