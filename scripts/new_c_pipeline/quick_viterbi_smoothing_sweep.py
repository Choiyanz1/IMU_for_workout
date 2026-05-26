"""
Quick test: Viterbi with larger smoothing windows on trained models.
Reuses models from boundary_head_experiment_v2.
Tests: MA15, MA25, MA40 with Viterbi penalty=0.3
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from preprocessing.micro_macro_segments import (
    labels_to_runs, pair_concentric_eccentric_reps, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, evaluate_phase, evaluate_reps,
    extract_active_segments, predict_active, train_active_detector,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# Reuse models from v2 experiment
from scripts.new_c_pipeline.boundary_head_experiment_v2 import (
    CausalCNN_PhaseOnly, CausalCNN_BoundaryAware,
    predict_phase, predict_boundary,
)


def viterbi_decode(phase_probs, penalty=0.3):
    n = len(phase_probs)
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2)); dp[0] = log_probs[0]
    for i in range(1, n):
        for s in range(2):
            stay = dp[i-1, s]
            switch = dp[i-1, 1-s] - penalty
            dp[i, s] = log_probs[i, s] + max(stay, switch)
    pred = np.zeros(n, dtype=np.int64)
    pred[-1] = np.argmax(dp[-1])
    for i in range(n-2, -1, -1):
        s = pred[i+1]
        stay = dp[i, s]
        switch = dp[i, 1-s] - penalty
        pred[i] = s if stay >= switch else (1-s)
    result = np.zeros((n, 2))
    result[pred == 0, 0] = 1.0
    result[pred == 1, 1] = 1.0
    return result


def smooth_ma(phase_probs, window):
    n = len(phase_probs); smoothed = np.copy(phase_probs)
    if window > 1:
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)
    return smoothed


def parse_reps(hard_labels, min_phase=3, max_gap=3):
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in hard_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase)
    if not runs: return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(label=run.label, start_idx=merged[-1].start_idx, end_idx=run.end_idx, confidence=(merged[-1].confidence + run.confidence) / 2)
        else:
            merged.append(run)
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="phase", max_gap_samples=max_gap)
    return reps


def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
    
    cfg = PhaseCompareConfig()
    
    # Load trained models
    print("Loading models...")
    import json
    import pickle
    
    # We need to re-train quickly or load from file - let's just re-train fast
    # Actually, let's load from the previous experiment's results
    # But we don't have saved model files... let's re-train quickly (20s each)
    
    from scripts.new_c_pipeline.boundary_head_experiment_v2 import (
        extract_segments, normalize, SimpleDataset, BndDataset,
        train_model,
    )
    from torch.utils.data import DataLoader
    
    segments, labels, boundaries = extract_segments(train_streams, cfg.imu_columns)
    mean, std, norm_segments = normalize(segments)
    
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    print("\nTraining Phase-Only CNN...")
    phase_model = CausalCNN_PhaseOnly(6, 64)
    train_ds = SimpleDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = SimpleDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    phase_model = train_model(phase_model, train_loader, val_loader, epochs=20, patience=8, is_boundary=False)
    
    print("\nTraining Boundary-Aware CNN...")
    bnd_model = CausalCNN_BoundaryAware(6, 64)
    train_ds_b = BndDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], [boundaries[i] for i in train_idx])
    val_ds_b = BndDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], [boundaries[i] for i in val_idx])
    train_loader_b = DataLoader(train_ds_b, batch_size=32, shuffle=True, drop_last=True)
    val_loader_b = DataLoader(val_ds_b, batch_size=32, shuffle=False, drop_last=False)
    bnd_model = train_model(bnd_model, train_loader_b, val_loader_b, epochs=20, patience=8, is_boundary=True, lam=0.5)
    
    # Evaluate with different smoothing + Viterbi combos
    print(f"\n{'='*60}")
    print("SMOOTHING + VITERBI SWEEP")
    print(f"{'='*60}")
    
    configs = [
        ("PhaseOnly", phase_model, None, [15, 25, 40]),
        ("BoundaryAware", bnd_model, "boundary", [15, 25, 40]),
    ]
    
    all_results = {}
    
    for model_name, model, model_type, windows in configs:
        for window in windows:
            results = []
            for stream_id, df in test_streams:
                if "phase" not in df.columns: continue
                gt_phases = df["phase"].to_numpy()
                gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
                
                active_probs = predict_active(train_active_detector(train_streams, cfg)[0], train_active_detector(train_streams, cfg)[1], stream_id, df, cfg)
                active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
                
                if model_type == "boundary":
                    phase_probs, boundary_probs = predict_boundary(model, df, active_segments, cfg.imu_columns, mean, std)
                    # For now just test phase-only decoding on boundary model
                    # (hybrid with boundary is more complex, skip for now)
                else:
                    phase_probs = predict_phase(model, df, active_segments, cfg.imu_columns, mean, std)
                
                # Apply smoothing then Viterbi
                smoothed = smooth_ma(phase_probs, window)
                decoded = viterbi_decode(smoothed, penalty=0.3)
                pred_reps = parse_reps(np.argmax(decoded, axis=1))
                rep_m = evaluate_reps(pred_reps, gt_reps)
                phase_m = evaluate_phase(decoded, gt_phases)
                results.append({**rep_m, **phase_m})
            
            # Aggregate
            n = len(results)
            total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            exact = sum(r["exact_count"] for r in results)
            over = sum(r["over"] for r in results); under = sum(r["under"] for r in results)
            phase_f1 = np.mean([r["phase_macro_f1"] for r in results])
            trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
            
            key = f"{model_name}_MA{window}_Viterbi"
            all_results[key] = {
                "rep_f1": f1, "exact_count_acc": exact / n if n > 0 else 0,
                "over_count": over, "under_count": under,
                "phase_macro_f1": phase_f1,
                "transition_mae_ms": np.mean(trans) if trans else None,
            }
            print(f"{key:35s}: RepF1={f1:.4f} Exact={exact/n:.3f} Over/Under={over}/{under} PhaseF1={phase_f1:.4f} TransMAE={np.mean(trans) if trans else 0:.0f}ms")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Best by Rep F1
    print("\n[Top 5 by Rep F1]")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["rep_f1"], reverse=True)[:5]:
        print(f"  {name:35s}: RepF1={m['rep_f1']:.4f} Exact={m['exact_count_acc']:.3f} Over/Under={m['over_count']}/{m['under_count']}")
    
    # Best by balance
    print("\n[Top 5 by Balance (Over+Under)]")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["over_count"] + x[1]["under_count"])[:5]:
        print(f"  {name:35s}: Over+Under={m['over_count']+m['under_count']} RepF1={m['rep_f1']:.4f} Exact={m['exact_count_acc']:.3f}")


if __name__ == "__main__":
    main()
