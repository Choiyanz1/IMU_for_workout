"""
Boundary Head Experiment v2: Fair comparison with Viterbi decoder.

Compares:
  A. Phase-Only CNN + Viterbi (penalty=0.3) — baseline
  B. Boundary-Aware CNN + Viterbi (penalty=0.3) — same decoder, check if boundary loss improves phase head
  C. Boundary-Aware CNN + Viterbi + boundary threshold — hybrid decoding

Fixed config: Kevin 1-fold, 20 epochs max, Viterbi penalty=0.3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Models
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


class CausalCNN_BoundaryAware(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, num_classes, 1)
        self.boundary_head = nn.Conv1d(hidden, 1, 1)
    def forward(self, x):
        f = self.encoder(x)
        return self.phase_head(f), self.boundary_head(f)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_segments(train_streams, imu_columns):
    segments, labels, boundaries = [], [], []
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
                    seg_bnd = np.zeros(len(seg_lab), dtype=np.float32)
                    for c in np.where(np.diff(seg_lab) != 0)[0]:
                        seg_bnd[max(0,c-10):min(len(seg_lab),c+1+10)] = 1.0
                    segments.append(seg_x); labels.append(seg_lab); boundaries.append(seg_bnd)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            seg_x = x[seg_start:]
            seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]])
            seg_bnd = np.zeros(len(seg_lab), dtype=np.float32)
            for c in np.where(np.diff(seg_lab) != 0)[0]:
                seg_bnd[max(0,c-10):min(len(seg_lab),c+1+10)] = 1.0
            segments.append(seg_x); labels.append(seg_lab); boundaries.append(seg_bnd)
    return segments, labels, boundaries


def normalize(segments):
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std, [(seg - mean) / std for seg in segments]


class SimpleDataset(Dataset):
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
        return torch.from_numpy(seq).float().transpose(0, 1), torch.from_numpy(lab).long(), torch.from_numpy(mask).float()


class BndDataset(Dataset):
    def __init__(self, segments, labels, boundaries, slice_len=300):
        self.samples = []
        for seq, lab, bnd in zip(segments, labels, boundaries):
            n = len(seq)
            if n <= slice_len:
                pad_len = max(0, slice_len - n)
                seq_pad = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
                lab_pad = np.pad(lab, (0, pad_len), constant_values=-1)
                bnd_pad = np.pad(bnd, (0, pad_len), mode='constant', constant_values=0)
                mask = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
                self.samples.append((seq_pad[:slice_len], lab_pad[:slice_len], bnd_pad[:slice_len], mask[:slice_len]))
            else:
                stride = slice_len // 2
                starts = list(range(0, n - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < n:
                    starts.append(n - slice_len)
                for start in starts:
                    self.samples.append((seq[start:start+slice_len], lab[start:start+slice_len], bnd[start:start+slice_len], np.ones(slice_len, dtype=np.float32)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, lab, bnd, mask = self.samples[idx]
        return torch.from_numpy(seq).float().transpose(0, 1), torch.from_numpy(lab).long(), torch.from_numpy(bnd).float(), torch.from_numpy(mask).float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, epochs=20, patience=8, is_boundary=False, lam=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    phase_crit = nn.CrossEntropyLoss(ignore_index=-1)
    bnd_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    
    best_val = float('inf'); best_state = None; patience_cnt = 0
    
    for epoch in range(epochs):
        model.train(); train_loss = 0; n_batches = 0
        for batch in train_loader:
            if is_boundary:
                x, y, b, m = batch; x, y, b, m = x.to(device), y.to(device), b.to(device), m.to(device)
            else:
                x, y, m = batch; x, y, m = x.to(device), y.to(device), m.to(device)
            
            optimizer.zero_grad()
            if is_boundary:
                phase_logits, bnd_logits = model(x)
            else:
                phase_logits = model(x)
            
            B, C, T = phase_logits.shape
            logits_flat = phase_logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y.reshape(B * T)
            mask_flat = m.reshape(B * T)
            valid = (labels_flat >= 0) & (mask_flat > 0)
            p_loss = phase_crit(logits_flat[valid], labels_flat[valid]) if valid.sum() > 0 else 0
            
            if is_boundary:
                bnd_flat = bnd_logits.reshape(B * T)
                b_flat = b.reshape(B * T)
                b_loss = bnd_crit(bnd_flat[valid], b_flat[valid]) if valid.sum() > 0 else 0
                loss = p_loss + lam * b_loss
            else:
                loss = p_loss
            
            loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        
        model.eval(); val_loss = 0; val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if is_boundary:
                    x, y, b, m = batch; x, y, b, m = x.to(device), y.to(device), b.to(device), m.to(device)
                    phase_logits, bnd_logits = model(x)
                else:
                    x, y, m = batch; x, y, m = x.to(device), y.to(device), m.to(device)
                    phase_logits = model(x)
                
                B, C, T = phase_logits.shape
                logits_flat = phase_logits.permute(0, 2, 1).reshape(B * T, C)
                labels_flat = y.reshape(B * T)
                mask_flat = m.reshape(B * T)
                valid = (labels_flat >= 0) & (mask_flat > 0)
                p_loss = phase_crit(logits_flat[valid], labels_flat[valid]) if valid.sum() > 0 else 0
                
                if is_boundary:
                    bnd_flat = bnd_logits.reshape(B * T)
                    b_flat = b.reshape(B * T)
                    b_loss = bnd_crit(bnd_flat[valid], b_flat[valid]) if valid.sum() > 0 else 0
                    loss = p_loss + lam * b_loss
                else:
                    loss = p_loss
                val_loss += loss.item(); val_batches += 1
        
        avg_train = train_loss / n_batches if n_batches > 0 else float('inf')
        avg_val = val_loss / val_batches if val_batches > 0 else float('inf')
        scheduler.step(avg_val)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val:
            best_val = avg_val; best_state = model.state_dict(); patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_phase(model, df, active_segments, imu_columns, mean, std):
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
    return phase_probs


def predict_boundary(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    boundary_probs = np.zeros(n, dtype=np.float32); boundary_counts = np.zeros(n, dtype=np.float32)
    if model is None: return phase_probs, boundary_probs
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
                phase_logits, bnd_logits = model(x_tensor)
                phase_probs_batch = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                boundary_probs_batch = torch.sigmoid(bnd_logits).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += phase_probs_batch[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
                boundary_probs[seg_start:seg_end] += boundary_probs_batch[0, :seg_len]
                boundary_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    phase_logits, bnd_logits = model(x_tensor)
                    phase_probs_batch = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                    boundary_probs_batch = torch.sigmoid(bnd_logits).cpu().numpy()[0]
                    gs = seg_start + start; ge = gs + 300
                    phase_probs[gs:ge, :] += phase_probs_batch.T
                    phase_counts[gs:ge] += 1.0
                    boundary_probs[gs:ge] += boundary_probs_batch[0, :]
                    boundary_counts[gs:ge] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    boundary_probs[valid] /= boundary_counts[valid]
    return phase_probs, boundary_probs


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def viterbi_decode(phase_probs, penalty=0.3):
    """Viterbi-like decoding with transition penalty."""
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


def viterbi_with_boundary(phase_probs, boundary_probs, penalty=0.3, b_weight=0.2):
    """Viterbi that also uses boundary probability to encourage transitions."""
    n = len(phase_probs)
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2)); dp[0] = log_probs[0]
    for i in range(1, n):
        b_bonus = boundary_probs[i] * b_weight  # bonus for switching at boundary
        for s in range(2):
            stay = dp[i-1, s]
            switch = dp[i-1, 1-s] - penalty + b_bonus
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


def eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, method_name, boundary_probs=None, b_weight=0.0):
    results = {}
    
    # Viterbi (penalty=0.3) on phase_probs
    decoded = viterbi_decode(phase_probs, penalty=0.3)
    pred_reps = parse_reps(np.argmax(decoded, axis=1))
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(decoded, gt_phases)
    results[f"{method_name}_Viterbi"] = {**rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"]}
    
    # Viterbi + Boundary (if available)
    if boundary_probs is not None:
        decoded_b = viterbi_with_boundary(phase_probs, boundary_probs, penalty=0.3, b_weight=b_weight)
        pred_reps_b = parse_reps(np.argmax(decoded_b, axis=1))
        rep_m_b = evaluate_reps(pred_reps_b, gt_reps)
        phase_m_b = evaluate_phase(decoded_b, gt_phases)
        results[f"{method_name}_ViterbiBoundary{b_weight}"] = {**rep_m_b, **phase_m_b, "pred_count": rep_m_b["pred_count"], "gt_count": rep_m_b["gt_count"]}
    
    return results


def aggregate(results_list):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
    
    print(f"Train: {len(train_streams)}, Test: {len(test_streams)}")
    print(f"GPU: {torch.cuda.is_available()}")
    
    cfg = PhaseCompareConfig()
    
    # Data
    segments, labels, boundaries = extract_segments(train_streams, cfg.imu_columns)
    mean, std, norm_segments = normalize(segments)
    
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    # Train Phase-Only
    print("\n[1/2] Training Phase-Only CNN...")
    phase_model = CausalCNN_PhaseOnly(6, 64)
    train_ds = SimpleDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = SimpleDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    phase_model = train_model(phase_model, train_loader, val_loader, epochs=20, patience=8, is_boundary=False)
    
    # Train Boundary-Aware
    print("\n[2/2] Training Boundary-Aware CNN...")
    bnd_model = CausalCNN_BoundaryAware(6, 64)
    train_ds_b = BndDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], [boundaries[i] for i in train_idx])
    val_ds_b = BndDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], [boundaries[i] for i in val_idx])
    train_loader_b = DataLoader(train_ds_b, batch_size=32, shuffle=True, drop_last=True)
    val_loader_b = DataLoader(val_ds_b, batch_size=32, shuffle=False, drop_last=False)
    bnd_model = train_model(bnd_model, train_loader_b, val_loader_b, epochs=20, patience=8, is_boundary=True, lam=0.5)
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns: continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        
        active_probs = predict_active(train_active_detector(train_streams, cfg)[0], train_active_detector(train_streams, cfg)[1], stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        
        # Phase-only
        phase_probs = predict_phase(phase_model, df, active_segments, cfg.imu_columns, mean, std)
        r1 = eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, "PhaseOnly")
        
        # Boundary-aware
        b_phase_probs, boundary_probs = predict_boundary(bnd_model, df, active_segments, cfg.imu_columns, mean, std)
        r2 = eval_stream(stream_id, df, b_phase_probs, gt_reps, gt_phases, "BoundaryAware", boundary_probs, b_weight=0.2)
        
        merged = {**r1, **r2}
        all_results.append(merged)
    
    agg = aggregate(all_results)
    
    print(f"\n{'='*60}")
    print("RESULTS (Kevin Single-Fold)")
    print(f"{'='*60}")
    for name, m in sorted(agg.items(), key=lambda x: x[1]["rep_f1"], reverse=True):
        print(f"\n{name}:")
        print(f"  Rep F1:        {m['rep_f1']:.4f}")
        print(f"  Exact Count:   {m['exact_count_acc']:.4f}")
        print(f"  Over/Under:    {m['over_count']}/{m['under_count']}")
        print(f"  Phase F1:      {m['phase_macro_f1']:.4f}")
        print(f"  Trans MAE:     {m.get('transition_mae_ms', 0):.0f}ms")
    
    output_dir = Path("artifacts/boundary_head_experiment_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n[OK] Saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
