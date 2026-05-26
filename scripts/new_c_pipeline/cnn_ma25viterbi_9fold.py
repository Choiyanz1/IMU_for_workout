"""
9-Fold LOSO: RF (Per-Action Window) vs Causal CNN + MA25+Viterbi (Global Seq)

Fixed architecture (no fold-specific tuning):
  - Causal CNN: 5-layer dilated, hidden=64, GroupNorm, residual
  - Training: Adam, lr=1e-3, 20 epochs, early stopping (best val loss)
  - Decoder: MA25 smoothing + Viterbi (penalty=0.3)
  - RF: Per-Action, window=100, 100 trees, depth=15

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    train_active_detector,
    train_rf_phase,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Causal CNN Model (same as fast_viterbi_sweep.py)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0), mode='reflect'))


class SharedEncoder(nn.Module):
    def __init__(self, in_ch=6, hidden=64, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, hidden, 5, 1); self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = CausalConv1d(hidden, hidden, 5, 2); self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = CausalConv1d(hidden, hidden, 5, 4); self.gn3 = nn.GroupNorm(8, hidden)
        self.conv4 = CausalConv1d(hidden, hidden, 5, 8); self.gn4 = nn.GroupNorm(8, hidden)
        self.conv5 = CausalConv1d(hidden, hidden, 5, 16); self.gn5 = nn.GroupNorm(8, hidden)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(in_ch, hidden, 1) if in_ch != hidden else None
    def forward(self, x):
        identity = x if self.res_proj is None else self.res_proj(x)
        x = F.relu(self.gn1(self.conv1(x))); x = self.dropout(x)
        x = F.relu(self.gn2(self.conv2(x))); x = self.dropout(x)
        x = F.relu(self.gn3(self.conv3(x))); x = self.dropout(x)
        x = F.relu(self.gn4(self.conv4(x))); x = self.dropout(x)
        x = F.relu(self.gn5(self.conv5(x))); x = self.dropout(x)
        if x.shape[2] == identity.shape[2]:
            x = x + identity[:, :x.shape[1], :]
        return x


class CausalCNN_PhaseOnly(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, num_classes, 1)
    def forward(self, x):
        return self.phase_head(self.encoder(x))


# ---------------------------------------------------------------------------
# Data utils
# ---------------------------------------------------------------------------

def extract_segments(train_streams, imu_columns):
    segments, labels = [], []
    for _, df in train_streams:
        if "phase" not in df.columns: continue
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        phase_arr = df["phase"].to_numpy()
        active_mask = np.array([str(p) in {"concentric", "eccentric"} for p in phase_arr])
        in_active = False; seg_start = 0
        for i, is_active in enumerate(active_mask):
            if is_active and not in_active:
                seg_start = i; in_active = True
            elif not is_active and in_active:
                if i - seg_start >= 10:
                    segments.append(x[seg_start:i])
                    labels.append(np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:i]]))
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            segments.append(x[seg_start:])
            labels.append(np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]]))
    return segments, labels


def normalize(segments):
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std, [(seg - mean) / std for seg in segments]


class PhaseDataset(Dataset):
    def __init__(self, segments, labels, slice_len=300):
        self.samples = []
        for seq, lab in zip(segments, labels):
            n = len(seq)
            if n <= slice_len:
                pad_len = max(0, slice_len - n)
                seq_pad = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
                lab_pad = np.pad(lab, (0, pad_len), constant_values=-1)
                mask = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
                self.samples.append((seq_pad[:slice_len], lab_pad[:slice_len], mask[:slice_len]))
            else:
                stride = slice_len // 2
                starts = list(range(0, n - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < n:
                    starts.append(n - slice_len)
                for start in starts:
                    self.samples.append((seq[start:start+slice_len], lab[start:start+slice_len], np.ones(slice_len, dtype=np.float32)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, lab, mask = self.samples[idx]
        return (torch.from_numpy(seq).float().transpose(0, 1),
                torch.from_numpy(lab).long(),
                torch.from_numpy(mask).float())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_causal_cnn(model, train_loader, val_loader, max_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_val = float('inf'); best_state = None
    
    for epoch in range(max_epochs):
        model.train(); train_loss = 0; n_batches = 0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            logits = model(x)
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y.reshape(B * T)
            mask_flat = m.reshape(B * T)
            valid = (labels_flat >= 0) & (mask_flat > 0)
            if valid.sum() == 0: continue
            loss = criterion(logits_flat[valid], labels_flat[valid])
            loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        
        model.eval(); val_loss = 0; val_batches = 0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                logits = model(x)
                B, C, T = logits.shape
                logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
                labels_flat = y.reshape(B * T)
                mask_flat = m.reshape(B * T)
                valid = (labels_flat >= 0) & (mask_flat > 0)
                if valid.sum() == 0: continue
                loss = criterion(logits_flat[valid], labels_flat[valid])
                val_loss += loss.item(); val_batches += 1
        
        avg_val = val_loss / val_batches if val_batches > 0 else float('inf')
        if avg_val < best_val:
            best_val = avg_val; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_cnn_phase(train_streams, imu_columns):
    segments, labels = extract_segments(train_streams, imu_columns)
    if len(segments) == 0:
        return None, None, None
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    model = CausalCNN_PhaseOnly(6, 64)
    train_ds = PhaseDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    model = train_causal_cnn(model, train_loader, val_loader, max_epochs=20)
    return model, mean, std


# ---------------------------------------------------------------------------
# Inference with MA25+Viterbi
# ---------------------------------------------------------------------------

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


def predict_cnn_phase(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    if model is None: return phase_probs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x); seg_x_norm = (seg_x - mean) / std
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    gs = seg_start + start
                    phase_probs[gs:gs + 300, :] += probs.T
                    phase_counts[gs:gs + 300] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    
    # MA25 + Viterbi
    phase_probs = smooth_ma(phase_probs, 25)
    phase_probs = viterbi_decode(phase_probs, 0.3)
    return phase_probs


# ---------------------------------------------------------------------------
# Rep parsing (same as fast script)
# ---------------------------------------------------------------------------

def parse_reps_viterbi(hard_labels, min_phase=3, max_gap=3):
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in hard_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase)
    if not runs: return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(label=run.label, start_idx=merged[-1].start_idx, end_idx=run.end_idx,
                                    confidence=(merged[-1].confidence + run.confidence) / 2)
        else:
            merged.append(run)
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="phase", max_gap_samples=max_gap)
    return reps


# ---------------------------------------------------------------------------
# C/E Ratio helpers
# ---------------------------------------------------------------------------

def compute_rep_ce_ratios(reps, phase_labels):
    ratios = []
    for rep in reps:
        seg = phase_labels[rep.start_idx:rep.end_idx]
        if len(seg) == 0: ratios.append(float('nan')); continue
        c_count = np.sum(seg == CONCENTRIC_LABEL)
        e_count = np.sum(seg == ECCENTRIC_LABEL)
        ratios.append(float('inf') if e_count == 0 else c_count / e_count)
    return ratios


def compute_ce_ratio_metrics(pred_ratios, gt_ratios):
    valid_pairs = [(p, g) for p, g in zip(pred_ratios, gt_ratios)
                   if np.isfinite(p) and np.isfinite(g) and p != float('inf') and g != float('inf')]
    if not valid_pairs:
        return {"ce_ratio_mae": None, "ce_ratio_rmse": None, "ce_ratio_bias": None, "n_valid": 0}
    pred_arr = np.array([p for p, _ in valid_pairs]); gt_arr = np.array([g for _, g in valid_pairs])
    errors = pred_arr - gt_arr
    return {"ce_ratio_mae": float(np.mean(np.abs(errors))), "ce_ratio_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "ce_ratio_bias": float(np.mean(errors)), "n_valid": len(valid_pairs)}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases):
    pred_reps = parse_reps_viterbi(np.argmax(phase_probs, axis=1))
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(phase_probs, gt_phases)
    
    pred_phase_arr = np.array(["eccentric" if p == 0 else "concentric" for p in np.argmax(phase_probs, axis=1)])
    pred_ratios = compute_rep_ce_ratios(pred_reps, pred_phase_arr)
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)
    
    count_error = abs(rep_m["pred_count"] - rep_m["gt_count"])
    
    return {"stream_id": stream_id, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"],
            "count_error": count_error,
            **{k: v for k, v in rep_m.items() if k not in ["pred_count", "gt_count"]},
            **phase_m, **ce_metrics}


def aggregate_fold_results(results):
    if not results: return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    exact_count = sum(r["exact_count"] for r in results)
    over_count = sum(r["over"] for r in results); under_count = sum(r["under"] for r in results)
    count_errors = [r["count_error"] for r in results]
    trans_mae_list = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    phase_f1_list = [r["phase_macro_f1"] for r in results]
    phase_acc_list = [r["phase_accuracy"] for r in results]
    ce_mae_list = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]
    
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": exact_count / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": over_count, "under_count": under_count,
        "phase_macro_f1": np.mean(phase_f1_list) if phase_f1_list else 0,
        "phase_accuracy": np.mean(phase_acc_list) if phase_acc_list else 0,
        "transition_mae_ms": np.mean(trans_mae_list) if trans_mae_list else None,
        "ce_ratio_mae": np.mean(ce_mae_list) if ce_mae_list else None,
    }


# ---------------------------------------------------------------------------
# Main 9-fold
# ---------------------------------------------------------------------------

def run_9fold(all_streams, subjects, cfg, output_dir):
    print("=" * 80)
    print("9-Fold LOSO: RF (Per-Action Window) vs CausalCNN + MA25+Viterbi (Global Seq)")
    print("=" * 80)
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"GPU: {torch.cuda.is_available()}")
    
    rf_fold_results = []
    cnn_fold_results = []
    
    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/9: test={test_subject}")
        print(f"{'=' * 80}")
        
        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
        print(f"  train={len(train_streams)}, test={len(test_streams)}")
        
        # Train
        print("  Training Active Detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        
        print("  Training RF Phase Model...")
        rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, cfg)
        
        print("  Training Causal CNN Phase Model...")
        cnn_model, cnn_mean, cnn_std = train_cnn_phase(train_streams, cfg.imu_columns)
        
        # Evaluate
        rf_stream_results = []
        cnn_stream_results = []
        
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            # RF
            rf_phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, cfg)
            rf_stream_results.append(evaluate_stream(stream_id, df, rf_phase_probs, gt_reps, gt_phases))
            
            # CNN + MA25+Viterbi
            if cnn_model is not None:
                cnn_phase_probs = predict_cnn_phase(cnn_model, df, active_segments, cfg.imu_columns, cnn_mean, cnn_std)
                cnn_stream_results.append(evaluate_stream(stream_id, df, cnn_phase_probs, gt_reps, gt_phases))
        
        # Aggregate
        rf_fold = aggregate_fold_results(rf_stream_results)
        rf_fold["fold"] = fold_idx + 1; rf_fold["test_subject"] = test_subject
        rf_fold_results.append(rf_fold)
        
        if cnn_model is not None:
            cnn_fold = aggregate_fold_results(cnn_stream_results)
            cnn_fold["fold"] = fold_idx + 1; cnn_fold["test_subject"] = test_subject
            cnn_fold_results.append(cnn_fold)
        
        print(f"  RF:   RepF1={rf_fold['rep_f1']:.4f} Exact={rf_fold['exact_count_acc']:.3f} "
              f"PhaseF1={rf_fold['phase_macro_f1']:.4f} TransMAE={rf_fold.get('transition_mae_ms', 0):.0f}ms "
              f"Over/Under={rf_fold['over_count']}/{rf_fold['under_count']}")
        if cnn_model is not None:
            print(f"  CNN:  RepF1={cnn_fold['rep_f1']:.4f} Exact={cnn_fold['exact_count_acc']:.3f} "
                  f"PhaseF1={cnn_fold['phase_macro_f1']:.4f} TransMAE={cnn_fold.get('transition_mae_ms', 0):.0f}ms "
                  f"Over/Under={cnn_fold['over_count']}/{cnn_fold['under_count']}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("9-FOLD SUMMARY")
    print(f"{'=' * 80}")
    
    summary = {}
    for name, fold_results in [("RF", rf_fold_results), ("CNN_MA25Viterbi", cnn_fold_results)]:
        if not fold_results: continue
        print(f"\n[{name}] ({len(fold_results)} folds)")
        
        metrics = ["rep_f1", "exact_count_acc", "mean_abs_count_error", "over_count", "under_count",
                   "phase_macro_f1", "phase_accuracy", "transition_mae_ms", "ce_ratio_mae"]
        
        for metric in metrics:
            values = [f[metric] for f in fold_results if f.get(metric) is not None]
            if not values: continue
            mean = np.mean(values); std = np.std(values)
            best = np.max(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.min(values)
            worst = np.min(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.max(values)
            print(f"  {metric}: mean={mean:.4f} std={std:.4f} best={best:.4f} worst={worst:.4f}")
            summary[f"{name}_{metric}_mean"] = mean
            summary[f"{name}_{metric}_std"] = std
        
        for f in fold_results:
            print(f"    Fold {f['fold']} ({f['test_subject']}): "
                  f"RepF1={f['rep_f1']:.4f} Exact={f['exact_count_acc']:.3f} "
                  f"PhaseF1={f['phase_macro_f1']:.4f} TransMAE={f.get('transition_mae_ms', 0):.0f}ms "
                  f"Over/Under={f['over_count']}/{f['under_count']}")
    
    # Head-to-head comparison
    if cnn_fold_results and rf_fold_results:
        print(f"\n{'=' * 80}")
        print("CNN vs RF COMPARISON")
        print(f"{'=' * 80}")
        
        comparisons = {
            "rep_f1": "higher is better",
            "phase_macro_f1": "higher is better",
            "phase_accuracy": "higher is better",
            "exact_count_acc": "higher is better",
            "transition_mae_ms": "lower is better",
            "mean_abs_count_error": "lower is better",
        }
        
        for metric, direction in comparisons.items():
            cnn_wins = 0
            for cnn_f, rf_f in zip(cnn_fold_results, rf_fold_results):
                cnn_val = cnn_f.get(metric); rf_val = rf_f.get(metric)
                if cnn_val is None or rf_val is None: continue
                if direction == "higher is better":
                    if cnn_val > rf_val: cnn_wins += 1
                else:
                    if cnn_val < rf_val: cnn_wins += 1
            print(f"  {metric}: CNN wins in {cnn_wins}/{len(cnn_fold_results)} folds")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_9fold_cnn_ma25viterbi.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "rf_per_fold": rf_fold_results,
            "cnn_per_fold": cnn_fold_results,
        }, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}")
    
    return summary


def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects: {subjects}")
    
    cfg = PhaseCompareConfig()
    cfg.seq2seq_epochs = 20
    
    run_9fold(all_streams, subjects, cfg, Path("artifacts/cnn_ma25viterbi_9fold"))


if __name__ == "__main__":
    main()
