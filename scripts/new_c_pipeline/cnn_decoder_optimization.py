"""
Causal CNN Decoder Optimization: Find best post-processing for over-segmentation.

Strategies:
1. Larger MA windows (15, 20, 25, 30, 40, 50, 60)
2. Hysteresis thresholding (enter_c: 0.55~0.75, exit_c: 0.25~0.45)
3. MA + Hysteresis combinations
4. Min phase duration filter (after decoding)

Test on kevin single-fold, find best config for 9-fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, evaluate_phase, evaluate_reps,
    extract_active_segments, predict_active, train_active_detector,
    _extract_action_from_stream_id,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Causal CNN Model (same as before)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode='reflect')
        return self.conv(x)


class CausalSimpleSeq2Seq(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, hidden, 5, 1); self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = CausalConv1d(hidden, hidden, 5, 2); self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = CausalConv1d(hidden, hidden, 5, 4); self.gn3 = nn.GroupNorm(8, hidden)
        self.conv4 = CausalConv1d(hidden, hidden, 5, 8); self.gn4 = nn.GroupNorm(8, hidden)
        self.conv5 = CausalConv1d(hidden, hidden, 5, 16); self.gn5 = nn.GroupNorm(8, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Conv1d(hidden, num_classes, 1)
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
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data helpers (same)
# ---------------------------------------------------------------------------

def extract_active_segments_for_seq2seq(train_streams, imu_columns):
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


class SegmentDataset(Dataset):
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
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        m = torch.from_numpy(mask).float()
        return x, y, m


def train_causal_cnn(train_streams, imu_columns, cfg):
    segments, labels = extract_active_segments_for_seq2seq(train_streams, imu_columns)
    if not segments:
        return None, None, None
    mean, std = compute_normalization_stats(segments)
    norm_segments = normalize_segments(segments, mean, std)
    
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_dataset = SegmentDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], slice_len=300)
    val_dataset = SegmentDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], slice_len=300)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalSimpleSeq2Seq(in_ch=6, hidden=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(20):
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
            print(f"      Epoch {epoch+1}/20, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"      Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, mean, std


def predict_causal_cnn(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    
    if model is None:
        return phase_probs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    slice_len = 300
    
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


def decode_hysteresis(phase_probs, enter_c=0.6, exit_c=0.4):
    """State machine: need high confidence to switch."""
    n = len(phase_probs)
    pred = np.zeros(n, dtype=np.int64)
    state = 0  # 0=E
    for i in range(n):
        p_c = phase_probs[i, 1]
        if state == 0:  # E
            if p_c >= enter_c:
                state = 1
        else:  # C
            if p_c <= exit_c:
                state = 0
        pred[i] = state
    result = np.zeros((n, 2))
    result[pred == 0, 0] = 1.0
    result[pred == 1, 1] = 1.0
    return result


def decode_viterbi(phase_probs, penalty=0.1):
    """Simple Viterbi with transition penalty."""
    n = len(phase_probs)
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2))
    dp[0] = log_probs[0]
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


def eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, sw=15, mps=3, mgs=3):
    # Test different decoding strategies on same phase_probs
    results = {}
    
    # Strategy 1: MA only
    for w in [sw]:
        smoothed = smooth_ma(phase_probs, w)
        pred_reps = parse_reps(np.argmax(smoothed, axis=1), mps, mgs)
        rep_m = evaluate_reps(pred_reps, gt_reps)
        phase_m = evaluate_phase(smoothed, gt_phases)
        results[f"MA{w}"] = {**rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"]}
    
    # Strategy 2: Hysteresis on raw (no MA)
    for enter, exit in [(0.55, 0.45), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3)]:
        decoded = decode_hysteresis(phase_probs, enter, exit)
        pred_reps = parse_reps(np.argmax(decoded, axis=1), mps, mgs)
        rep_m = evaluate_reps(pred_reps, gt_reps)
        phase_m = evaluate_phase(decoded, gt_phases)
        results[f"Hyst_{enter:.1f}_{exit:.1f}"] = {**rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"]}
    
    # Strategy 3: Viterbi
    for penalty in [0.1, 0.2, 0.3]:
        decoded = decode_viterbi(phase_probs, penalty)
        pred_reps = parse_reps(np.argmax(decoded, axis=1), mps, mgs)
        rep_m = evaluate_reps(pred_reps, gt_reps)
        phase_m = evaluate_phase(decoded, gt_phases)
        results[f"Viterbi_{penalty}"] = {**rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"]}
    
    # Strategy 4: MA + Hysteresis
    for w in [15, 25, 40]:
        for enter, exit in [(0.6, 0.4), (0.65, 0.35), (0.7, 0.3)]:
            smoothed = smooth_ma(phase_probs, w)
            decoded = decode_hysteresis(smoothed, enter, exit)
            pred_reps = parse_reps(np.argmax(decoded, axis=1), mps, mgs)
            rep_m = evaluate_reps(pred_reps, gt_reps)
            phase_m = evaluate_phase(decoded, gt_phases)
            results[f"MA{w}+Hyst_{enter:.1f}_{exit:.1f}"] = {**rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"]}
    
    return results


def aggregate_results(results_list):
    """results_list: list of dicts {variant_name: metrics} per stream"""
    if not results_list:
        return {}
    
    variants = list(results_list[0].keys())
    agg = {}
    for v in variants:
        n = len(results_list)
        total_tp = sum(r[v]["tp"] for r in results_list)
        total_fp = sum(r[v]["fp"] for r in results_list)
        total_fn = sum(r[v]["fn"] for r in results_list)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        exact = sum(r[v]["exact_count"] for r in results_list)
        over = sum(r[v]["over"] for r in results_list)
        under = sum(r[v]["under"] for r in results_list)
        phase_f1 = np.mean([r[v]["phase_macro_f1"] for r in results_list])
        trans = [r[v]["transition_mae_ms"] for r in results_list if r[v].get("transition_mae_ms") is not None]
        agg[v] = {
            "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
            "exact_count_acc": exact / n if n > 0 else 0,
            "over_count": over, "under_count": under,
            "phase_macro_f1": phase_f1,
            "transition_mae_ms": np.mean(trans) if trans else None,
        }
    return agg


def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
    
    print(f"Train: {len(train_streams)}, Test: {len(test_streams)}")
    print(f"GPU: {torch.cuda.is_available()}")
    
    # Train models
    print("\nTraining models...")
    print("  Active Detector...")
    active_models, active_scalers = train_active_detector(train_streams, PhaseCompareConfig())
    
    print("  Causal CNN...")
    cnn_model, cnn_mean, cnn_std = train_causal_cnn(train_streams, ("ax", "ay", "az", "gx", "gy", "gz"), PhaseCompareConfig())
    
    # Evaluate with all decoder variants
    print(f"\n{'='*60}")
    print("DECODER OPTIMIZATION")
    print(f"{'='*60}")
    
    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns: continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        
        active_probs = predict_active(active_models, active_scalers, stream_id, df, PhaseCompareConfig())
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        
        cnn_probs = predict_causal_cnn(cnn_model, df, active_segments, ("ax", "ay", "az", "gx", "gy", "gz"), cnn_mean, cnn_std)
        
        stream_results = eval_stream(stream_id, df, cnn_probs, gt_reps, gt_phases)
        all_results.append(stream_results)
    
    # Aggregate
    agg = aggregate_results(all_results)
    
    # Sort by Rep F1
    sorted_by_f1 = sorted(agg.items(), key=lambda x: x[1]["rep_f1"], reverse=True)
    print("\n[Top 10 by Rep F1]")
    for name, m in sorted_by_f1[:10]:
        print(f"  {name:25s}: RepF1={m['rep_f1']:.4f} Exact={m['exact_count_acc']:.3f} "
              f"Over/Under={m['over_count']}/{m['under_count']} "
              f"PhaseF1={m['phase_macro_f1']:.4f} TransMAE={m.get('transition_mae_ms', 0):.0f}ms")
    
    # Sort by Exact Count
    sorted_by_exact = sorted(agg.items(), key=lambda x: x[1]["exact_count_acc"], reverse=True)
    print("\n[Top 10 by Exact Count Acc]")
    for name, m in sorted_by_exact[:10]:
        print(f"  {name:25s}: Exact={m['exact_count_acc']:.3f} RepF1={m['rep_f1']:.4f} "
              f"Over/Under={m['over_count']}/{m['under_count']}")
    
    # Sort by balance (over+under)
    sorted_by_balance = sorted(agg.items(), key=lambda x: x[1]["over_count"] + x[1]["under_count"])
    print("\n[Top 10 by Balance (Over+Under)]")
    for name, m in sorted_by_balance[:10]:
        print(f"  {name:25s}: Over+Under={m['over_count']+m['under_count']} "
              f"RepF1={m['rep_f1']:.4f} Exact={m['exact_count_acc']:.3f}")
    
    # Save
    output_dir = Path("artifacts/cnn_decoder_opt")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "decoder_comparison.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
