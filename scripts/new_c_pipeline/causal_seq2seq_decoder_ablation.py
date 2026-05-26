"""
Seq2Seq Causal Model + Decoder Ablation on Kevin Single-Fold

Tests the decoder/post-processing hypothesis for over-segmentation:
1. Causal dilated CNN (no future peeking)
2. Multiple smoothing methods (moving avg, median, Gaussian)
3. Hysteresis thresholding (asymmetric C↔E switching)
4. Viterbi-style decoding with transition penalty
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import median_filter, gaussian_filter1d

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    RepDetection, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, evaluate_phase, evaluate_reps,
    extract_active_segments, predict_active,
    train_active_detector, _extract_action_from_stream_id,
    _prepare_phase_labels,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Causal SimpleSeq2Seq
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal convolution: output[t] only sees input[:t+1]."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
    
    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0), mode='reflect')
        return self.conv(x)


class CausalSimpleSeq2Seq(nn.Module):
    """Causal 1D-CNN for per-sample C/E segmentation.
    No future peeking. Receptive field grows with dilation.
    """
    def __init__(self, input_channels=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(input_channels, hidden, 5, dilation=1)   # RF=5
        self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = CausalConv1d(hidden, hidden, 5, dilation=2)          # RF += 8
        self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = CausalConv1d(hidden, hidden, 5, dilation=4)        # RF += 16
        self.gn3 = nn.GroupNorm(8, hidden)
        self.conv4 = CausalConv1d(hidden, hidden, 5, dilation=8)        # RF += 32
        self.gn4 = nn.GroupNorm(8, hidden)
        self.conv5 = CausalConv1d(hidden, hidden, 5, dilation=16)       # RF += 64
        self.gn5 = nn.GroupNorm(8, hidden)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Conv1d(hidden, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.gn5(self.conv5(x)))
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ---------------------------------------------------------------------------
# Re-use data loading from compare_phase_models
# ---------------------------------------------------------------------------

def extract_active_segments_for_seq2seq(train_streams, imu_columns):
    """Extract active segments and per-sample C/E labels."""
    segments = []; labels = []
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
                    seg_x = x[seg_start:i]
                    seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:i]])
                    segments.append(seg_x); labels.append(seg_lab)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            seg_x = x[seg_start:]
            seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]])
            segments.append(seg_x); labels.append(seg_lab)
    return segments, labels


def compute_normalization_stats(segments):
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_segments(segments, mean, std):
    return [(seg - mean) / std for seg in segments]


class ActiveSegmentDataset(Dataset):
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
                    self.samples.append((
                        seq[start:start+slice_len],
                        lab[start:start+slice_len],
                        np.ones(slice_len, dtype=np.float32)
                    ))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        seq, lab, mask = self.samples[idx]
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        m = torch.from_numpy(mask).float()
        return x, y, m


def train_causal_seq2seq(train_streams, imu_columns, cfg):
    """Train causal seq2seq with early stopping."""
    segments, labels = extract_active_segments_for_seq2seq(train_streams, imu_columns)
    if not segments:
        print("      [CausalSeq2Seq] No active segments")
        return None, None, None
    
    mean, std = compute_normalization_stats(segments)
    norm_segments = normalize_segments(segments, mean, std)
    
    # Train/val split
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_dataset = ActiveSegmentDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], slice_len=cfg.seq2seq_slice_len)
    val_dataset = ActiveSegmentDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], slice_len=cfg.seq2seq_slice_len)
    
    if len(train_dataset) == 0:
        return None, None, None
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=False, drop_last=False)
    
    print(f"      [CausalSeq2Seq] Train: {len(train_dataset)} slices, Val: {len(val_dataset)} slices")
    
    device = torch.device(cfg.seq2seq_device)
    model = CausalSimpleSeq2Seq(input_channels=len(imu_columns), hidden=cfg.seq2seq_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.seq2seq_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(cfg.seq2seq_epochs):
        model.train()
        train_loss = 0; n_batches = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device); y_batch = y_batch.to(device); mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y_batch.reshape(B * T)
            mask_flat = mask_batch.reshape(B * T)
            valid = (labels_flat >= 0) & (mask_flat > 0)
            if valid.sum() == 0: continue
            loss = criterion(logits_flat[valid], labels_flat[valid])
            loss.backward()
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        
        model.eval()
        val_loss = 0; val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, mask_batch in val_loader:
                x_batch = x_batch.to(device); y_batch = y_batch.to(device); mask_batch = mask_batch.to(device)
                logits = model(x_batch)
                B, C, T = logits.shape
                logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
                labels_flat = y_batch.reshape(B * T)
                mask_flat = mask_batch.reshape(B * T)
                valid = (labels_flat >= 0) & (mask_flat > 0)
                if valid.sum() == 0: continue
                loss = criterion(logits_flat[valid], labels_flat[valid])
                val_loss += loss.item(); val_batches += 1
        
        avg_train = train_loss / n_batches if n_batches > 0 else float('inf')
        avg_val = val_loss / val_batches if val_batches > 0 else float('inf')
        scheduler.step(avg_val)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      [CausalSeq2Seq] Epoch {epoch+1}/{cfg.seq2seq_epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"      [CausalSeq2Seq] Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, mean, std


def predict_causal_seq2seq(model, df, active_segments, imu_columns, mean, std, cfg):
    """Predict with causal seq2seq using sliding window + overlap averaging."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    
    if model is None:
        return phase_probs
    
    device = torch.device(cfg.seq2seq_device)
    model.eval()
    slice_len = cfg.seq2seq_slice_len
    
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]
            seg_len = len(seg_x)
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= slice_len:
                pad_len = slice_len - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = slice_len // 2
                starts = list(range(0, seg_len - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < seg_len:
                    starts.append(seg_len - slice_len)
                
                for start in starts:
                    window = seg_x_norm[start:start + slice_len]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    global_start = seg_start + start
                    global_end = global_start + slice_len
                    phase_probs[global_start:global_end, :] += probs.T
                    phase_counts[global_start:global_end] += 1.0
    
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    return phase_probs


# ---------------------------------------------------------------------------
# Decoder variants
# ---------------------------------------------------------------------------

def smooth_moving_avg(phase_probs, window):
    n = len(phase_probs); smoothed = np.copy(phase_probs)
    if window > 1:
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)
    return smoothed


def smooth_median(phase_probs, window=15):
    """Median filter per channel."""
    smoothed = np.copy(phase_probs)
    for c in range(2):
        smoothed[:, c] = median_filter(phase_probs[:, c], size=window, mode='reflect')
    return smoothed


def smooth_gaussian(phase_probs, sigma=3.0):
    """Gaussian smoothing per channel."""
    smoothed = np.copy(phase_probs)
    for c in range(2):
        smoothed[:, c] = gaussian_filter1d(phase_probs[:, c], sigma=sigma)
    return smoothed


def decode_hysteresis(phase_probs, enter_c=0.6, exit_c=0.4):
    """Hysteresis thresholding: need high confidence to switch phases.
    
    State machine:
    - Start in state=0 (E) if P(C) < enter_c
    - Switch to C only if P(C) >= enter_c
    - Switch to E only if P(C) <= exit_c (i.e., P(E) >= 1-exit_c=0.6)
    - In between: stay in current state
    """
    n = len(phase_probs)
    pred = np.zeros(n, dtype=np.int64)
    state = 0  # 0=E, 1=C
    
    for i in range(n):
        p_c = phase_probs[i, 1]  # P(concentric)
        if state == 0:  # Currently E
            if p_c >= enter_c:
                state = 1
        else:  # Currently C
            if p_c <= exit_c:
                state = 0
        pred[i] = state
    
    # Convert to phase_probs format
    result = np.zeros((n, 2))
    result[pred == 0, 0] = 1.0
    result[pred == 1, 1] = 1.0
    return result


def decode_viterbi_simple(phase_probs, transition_penalty=0.1):
    """Simple Viterbi-like decoding with transition penalty.
    Penalizes rapid C↔E switching.
    """
    n = len(phase_probs)
    # Log probabilities
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    
    # DP tables
    dp = np.zeros((n, 2))
    dp[0] = log_probs[0]
    
    for i in range(1, n):
        for s in range(2):
            # Cost of staying in same state
            stay = dp[i-1, s]
            # Cost of switching
            switch = dp[i-1, 1-s] - transition_penalty
            dp[i, s] = log_probs[i, s] + max(stay, switch)
    
    # Backtrack
    pred = np.zeros(n, dtype=np.int64)
    pred[-1] = np.argmax(dp[-1])
    for i in range(n-2, -1, -1):
        s = pred[i+1]
        stay = dp[i, s]
        switch = dp[i, 1-s] - transition_penalty
        pred[i] = s if stay >= switch else (1-s)
    
    result = np.zeros((n, 2))
    result[pred == 0, 0] = 1.0
    result[pred == 1, 1] = 1.0
    return result


def parse_reps_from_hard_labels(hard_labels, min_phase_samples=3, max_gap_samples=3):
    """Parse reps from hard labels (0=E, 1=C)."""
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in hard_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase_samples)
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


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_decoder_ablation(train_streams, test_streams, cfg: PhaseCompareConfig, output_dir: Path):
    print("=" * 70)
    print("Seq2Seq Causal + Decoder Ablation (Kevin Single-Fold)")
    print("=" * 70)
    
    # Train models
    print("\n[1/2] Training models...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    print("  Training Causal Seq2Seq...")
    causal_model, causal_mean, causal_std = train_causal_seq2seq(train_streams, cfg.imu_columns, cfg)
    print("      Done.")
    
    if causal_model is None:
        print("ERROR: Causal Seq2Seq training failed!")
        return
    
    # Test configurations
    configs = [
        # Baseline: just argmax on raw predictions
        ("Raw_argmax", lambda p: p, {"min_phase": 3, "max_gap": 3}),
        
        # Moving average only
        ("MA_sw15", lambda p: smooth_moving_avg(p, 15), {"min_phase": 3, "max_gap": 3}),
        ("MA_sw25", lambda p: smooth_moving_avg(p, 25), {"min_phase": 3, "max_gap": 3}),
        ("MA_sw40", lambda p: smooth_moving_avg(p, 40), {"min_phase": 3, "max_gap": 3}),
        
        # Median filter
        ("Median_w11", lambda p: smooth_median(p, 11), {"min_phase": 3, "max_gap": 3}),
        ("Median_w21", lambda p: smooth_median(p, 21), {"min_phase": 3, "max_gap": 3}),
        
        # Gaussian smoothing
        ("Gaussian_s2", lambda p: smooth_gaussian(p, 2.0), {"min_phase": 3, "max_gap": 3}),
        ("Gaussian_s3", lambda p: smooth_gaussian(p, 3.0), {"min_phase": 3, "max_gap": 3}),
        
        # Hysteresis (different thresholds)
        ("Hyst_0.6_0.4", lambda p: decode_hysteresis(p, 0.6, 0.4), {"min_phase": 3, "max_gap": 3}),
        ("Hyst_0.7_0.3", lambda p: decode_hysteresis(p, 0.7, 0.3), {"min_phase": 3, "max_gap": 3}),
        ("Hyst_0.8_0.2", lambda p: decode_hysteresis(p, 0.8, 0.2), {"min_phase": 3, "max_gap": 3}),
        
        # Viterbi
        ("Viterbi_0.1", lambda p: decode_viterbi_simple(p, 0.1), {"min_phase": 3, "max_gap": 3}),
        ("Viterbi_0.3", lambda p: decode_viterbi_simple(p, 0.3), {"min_phase": 3, "max_gap": 3}),
        
        # Combinations
        ("MA_sw25+Hyst_0.7_0.3", lambda p: decode_hysteresis(smooth_moving_avg(p, 25), 0.7, 0.3), {"min_phase": 5, "max_gap": 3}),
        ("Median_w21+Hyst_0.7_0.3", lambda p: decode_hysteresis(smooth_median(p, 21), 0.7, 0.3), {"min_phase": 5, "max_gap": 3}),
    ]
    
    all_results = {}
    
    for name, decoder_fn, parser_params in configs:
        print(f"\n  [{name}]")
        results = []
        
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            # Predict
            sq_probs = predict_causal_seq2seq(causal_model, df, active_segments, cfg.imu_columns, causal_mean, causal_std, cfg)
            
            # Decode
            decoded = decoder_fn(sq_probs)
            
            # Ground truth
            gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)
            gt_phases = df["phase"].to_numpy()
            
            # Parse reps
            hard_labels = np.argmax(decoded, axis=1)
            pred_reps = parse_reps_from_hard_labels(hard_labels, parser_params["min_phase"], parser_params["max_gap"])
            
            # Evaluate
            rep_metrics = evaluate_reps(pred_reps, gt_reps)
            phase_metrics = evaluate_phase(decoded, gt_phases)
            
            results.append({
                "stream_id": stream_id,
                **rep_metrics,
                **phase_metrics,
            })
        
        # Aggregate
        valid = [r for r in results if "f1" in r]
        n = len(valid)
        total_tp = sum(r["tp"] for r in valid)
        total_fp = sum(r["fp"] for r in valid)
        total_fn = sum(r["fn"] for r in valid)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        exact = sum(r["exact_count"] for r in valid)
        over = sum(r["over"] for r in valid)
        under = sum(r["under"] for r in valid)
        
        phase_f1_list = [r["phase_macro_f1"] for r in valid]
        trans_mae_list = [r["transition_mae_ms"] for r in valid if r.get("transition_mae_ms") is not None]
        
        agg = {
            "variant": name,
            "streams": n,
            "rep_precision": p,
            "rep_recall": r,
            "rep_f1": f1,
            "exact_count_acc": exact / n if n > 0 else 0,
            "over_count": over,
            "under_count": under,
            "phase_macro_f1": np.mean(phase_f1_list) if phase_f1_list else 0,
            "transition_mae_ms": np.mean(trans_mae_list) if trans_mae_list else None,
        }
        
        all_results[name] = agg
        print(f"    RepF1={f1:.4f} Exact={agg['exact_count_acc']:.3f} "
              f"Over/Under={over}/{under} PhaseF1={agg['phase_macro_f1']:.4f} "
              f"TransMAE={agg.get('transition_mae_ms', 0):.0f}ms")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    # Sort by Rep F1
    sorted_by_f1 = sorted(all_results.items(), key=lambda x: x[1]["rep_f1"], reverse=True)
    print("\n[Top 5 by Rep F1]")
    for name, agg in sorted_by_f1[:5]:
        print(f"  {name:25s}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
              f"Over/Under={agg['over_count']}/{agg['under_count']}")
    
    # Sort by Exact Count
    sorted_by_exact = sorted(all_results.items(), key=lambda x: x[1]["exact_count_acc"], reverse=True)
    print("\n[Top 5 by Exact Count Acc]")
    for name, agg in sorted_by_exact[:5]:
        print(f"  {name:25s}: Exact={agg['exact_count_acc']:.3f} RepF1={agg['rep_f1']:.4f} "
              f"Over/Under={agg['over_count']}/{agg['under_count']}")
    
    # Sort by balance (min over+under)
    sorted_by_balance = sorted(all_results.items(), key=lambda x: x[1]["over_count"] + x[1]["under_count"])
    print("\n[Top 5 by Balance (Over+Under)]")
    for name, agg in sorted_by_balance[:5]:
        print(f"  {name:25s}: Over+Under={agg['over_count']+agg['under_count']} "
              f"RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "causal_seq2seq_decoder_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Causal Seq2Seq Decoder Ablation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/causal_seq2seq_ablation"))
    args = parser.parse_args()
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
    
    print(f"Train streams: {len(train_streams)}, Test streams: {len(test_streams)}")
    
    cfg = PhaseCompareConfig()
    cfg.seq2seq_epochs = 30
    
    run_decoder_ablation(train_streams, test_streams, cfg, args.output)


if __name__ == "__main__":
    main()
