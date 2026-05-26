"""
Minimal Experiment: CNN Seq2Seq + Boundary Head for C/E Phase Segmentation

Compares:
  A. Original Causal CNN Seq2Seq
  B. Boundary-aware Causal CNN Seq2Seq (shared encoder + 2 heads)

Fixed config:
  - boundary positive region: transition ±100ms (±10 samples @100Hz)
  - lambda = 0.5
  - boundary threshold = 0.5
  - Kevin 1-fold only
"""
from __future__ import annotations

import json
import sys
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

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    RepDetection, SegmentRun,
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
    """Shared 5-layer dilated causal CNN encoder."""
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
    """Original: shared encoder + phase head only."""
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, num_classes, 1)
    def forward(self, x):
        features = self.encoder(x)
        return self.phase_head(features)


class CausalCNN_BoundaryAware(nn.Module):
    """Boundary-aware: shared encoder + phase head + boundary head."""
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, num_classes, 1)
        self.boundary_head = nn.Conv1d(hidden, 1, 1)
    def forward(self, x):
        features = self.encoder(x)
        phase_logits = self.phase_head(features)
        boundary_logits = self.boundary_head(features)
        return phase_logits, boundary_logits


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_segments_with_boundary(train_streams, imu_columns, boundary_margin_samples=10):
    """Extract active segments with per-sample C/E labels and boundary labels."""
    segments, labels, boundary_labels = [], [], []
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
                    seg_phase = phase_arr[seg_start:i]
                    seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in seg_phase])
                    seg_boundary = np.zeros(len(seg_lab), dtype=np.float32)
                    # Mark transitions
                    changes = np.where(np.diff(seg_lab) != 0)[0]
                    for c in changes:
                        start = max(0, c - boundary_margin_samples)
                        end = min(len(seg_lab), c + 1 + boundary_margin_samples)
                        seg_boundary[start:end] = 1.0
                    segments.append(seg_x); labels.append(seg_lab); boundary_labels.append(seg_boundary)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            seg_x = x[seg_start:]
            seg_phase = phase_arr[seg_start:]
            seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in seg_phase])
            seg_boundary = np.zeros(len(seg_lab), dtype=np.float32)
            changes = np.where(np.diff(seg_lab) != 0)[0]
            for c in changes:
                start = max(0, c - boundary_margin_samples)
                end = min(len(seg_lab), c + 1 + boundary_margin_samples)
                seg_boundary[start:end] = 1.0
            segments.append(seg_x); labels.append(seg_lab); boundary_labels.append(seg_boundary)
    return segments, labels, boundary_labels


def compute_normalization_stats(segments):
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class SegmentDatasetWithBoundary(Dataset):
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
                    self.samples.append((
                        seq[start:start+slice_len],
                        lab[start:start+slice_len],
                        bnd[start:start+slice_len],
                        np.ones(slice_len, dtype=np.float32)
                    ))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, lab, bnd, mask = self.samples[idx]
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        b = torch.from_numpy(bnd).float()
        m = torch.from_numpy(mask).float()
        return x, y, b, m


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_phase_only(train_streams, imu_columns, cfg):
    segments, labels, _ = extract_segments_with_boundary(train_streams, imu_columns)
    if not segments: return None, None, None
    mean, std = compute_normalization_stats(segments)
    norm_segments = [(seg - mean) / std for seg in segments]
    
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    # Simple dataset without boundary
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
    
    train_dataset = SimpleDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_dataset = SimpleDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalCNN_PhaseOnly(in_ch=6, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf'); best_state = None; patience_counter = 0
    
    for epoch in range(20):
        model.train()
        train_loss = 0; n_batches = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y_batch.reshape(B * T)
            mask_flat = mask_batch.reshape(B * T)
            valid = (labels_flat >= 0) & (mask_flat > 0)
            if valid.sum() == 0: continue
            loss = criterion(logits_flat[valid], labels_flat[valid])
            loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        
        model.eval()
        val_loss = 0; val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, mask_batch in val_loader:
                x_batch, y_batch, mask_batch = x_batch.to(device), y_batch.to(device), mask_batch.to(device)
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
            print(f"      [PhaseOnly] Epoch {epoch+1}/20, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val; best_state = model.state_dict(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"      [PhaseOnly] Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def train_boundary_aware(train_streams, imu_columns, cfg, lam=0.5, pos_weight=10.0):
    segments, labels, boundaries = extract_segments_with_boundary(train_streams, imu_columns, boundary_margin_samples=10)
    if not segments: return None, None, None
    mean, std = compute_normalization_stats(segments)
    norm_segments = [(seg - mean) / std for seg in segments]
    
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_dataset = SegmentDatasetWithBoundary([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], [boundaries[i] for i in train_idx])
    val_dataset = SegmentDatasetWithBoundary([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], [boundaries[i] for i in val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalCNN_BoundaryAware(in_ch=6, hidden=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    boundary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    best_val_loss = float('inf'); best_state = None; patience_counter = 0
    
    for epoch in range(20):
        model.train()
        train_loss = 0; n_batches = 0
        for x_batch, y_batch, b_batch, mask_batch in train_loader:
            x_batch, y_batch, b_batch, mask_batch = x_batch.to(device), y_batch.to(device), b_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            phase_logits, boundary_logits = model(x_batch)
            
            # Phase loss
            B, C, T = phase_logits.shape
            phase_logits_flat = phase_logits.permute(0, 2, 1).reshape(B * T, C)
            labels_flat = y_batch.reshape(B * T)
            mask_flat = mask_batch.reshape(B * T)
            valid = (labels_flat >= 0) & (mask_flat > 0)
            phase_loss = phase_criterion(phase_logits_flat[valid], labels_flat[valid]) if valid.sum() > 0 else 0
            
            # Boundary loss
            boundary_logits_flat = boundary_logits.reshape(B * T)
            b_flat = b_batch.reshape(B * T)
            boundary_loss = boundary_criterion(boundary_logits_flat[valid], b_flat[valid]) if valid.sum() > 0 else 0
            
            loss = phase_loss + lam * boundary_loss
            loss.backward(); optimizer.step()
            train_loss += loss.item(); n_batches += 1
        
        model.eval()
        val_loss = 0; val_batches = 0
        with torch.no_grad():
            for x_batch, y_batch, b_batch, mask_batch in val_loader:
                x_batch, y_batch, b_batch, mask_batch = x_batch.to(device), y_batch.to(device), b_batch.to(device), mask_batch.to(device)
                phase_logits, boundary_logits = model(x_batch)
                
                B, C, T = phase_logits.shape
                phase_logits_flat = phase_logits.permute(0, 2, 1).reshape(B * T, C)
                labels_flat = y_batch.reshape(B * T)
                mask_flat = mask_batch.reshape(B * T)
                valid = (labels_flat >= 0) & (mask_flat > 0)
                phase_loss = phase_criterion(phase_logits_flat[valid], labels_flat[valid]) if valid.sum() > 0 else 0
                
                boundary_logits_flat = boundary_logits.reshape(B * T)
                b_flat = b_batch.reshape(B * T)
                boundary_loss = boundary_criterion(boundary_logits_flat[valid], b_flat[valid]) if valid.sum() > 0 else 0
                
                loss = phase_loss + lam * boundary_loss
                val_loss += loss.item(); val_batches += 1
        
        avg_train = train_loss / n_batches if n_batches > 0 else float('inf')
        avg_val = val_loss / val_batches if val_batches > 0 else float('inf')
        scheduler.step(avg_val)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      [BoundaryAware] Epoch {epoch+1}/20, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val; best_state = model.state_dict(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"      [BoundaryAware] Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_phase_only(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    
    if model is None: return phase_probs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x)
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150
                starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len:
                    starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    global_start = seg_start + start
                    phase_probs[global_start:global_start + 300, :] += probs.T
                    phase_counts[global_start:global_start + 300] += 1.0
    
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    return phase_probs


def predict_boundary_aware(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    boundary_probs = np.zeros(n, dtype=np.float32); boundary_counts = np.zeros(n, dtype=np.float32)
    
    if model is None: return phase_probs, boundary_probs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x)
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                phase_logits, boundary_logits = model(x_tensor)
                phase_probs_batch = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                boundary_probs_batch = torch.sigmoid(boundary_logits).cpu().numpy()[0]
                
                phase_probs[seg_start:seg_end, :] += phase_probs_batch[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
                boundary_probs[seg_start:seg_end] += boundary_probs_batch[0, :seg_len]
                boundary_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150
                starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len:
                    starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    phase_logits, boundary_logits = model(x_tensor)
                    phase_probs_batch = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                    boundary_probs_batch = torch.sigmoid(boundary_logits).cpu().numpy()[0]
                    
                    global_start = seg_start + start
                    global_end = global_start + 300
                    phase_probs[global_start:global_end, :] += phase_probs_batch.T
                    phase_counts[global_start:global_end] += 1.0
                    boundary_probs[global_start:global_end] += boundary_probs_batch[0, :]
                    boundary_counts[global_start:global_end] += 1.0
    
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    boundary_probs[valid_mask] /= boundary_counts[valid_mask]
    return phase_probs, boundary_probs


# ---------------------------------------------------------------------------
# Decoding strategies
# ---------------------------------------------------------------------------

def smooth_ma(phase_probs, window=15):
    n = len(phase_probs); smoothed = np.copy(phase_probs)
    if window > 1:
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)
    return smoothed


def decode_with_boundary(phase_probs, boundary_probs, boundary_threshold=0.5, min_phase_samples=3):
    """Phase + Boundary-aware decoding.
    Transition accepted only if phase changes AND boundary_prob > threshold.
    """
    n = len(phase_probs)
    pred_labels = np.zeros(n, dtype=np.int64)
    
    # First pass: argmax phase
    hard_labels = np.argmax(phase_probs, axis=1)
    
    state = hard_labels[0]
    pred_labels[0] = state
    
    for i in range(1, n):
        if hard_labels[i] != state:
            # Potential transition - check boundary
            if boundary_probs[i] >= boundary_threshold:
                state = hard_labels[i]
        pred_labels[i] = state
    
    # Convert to phase_probs format
    result = np.zeros((n, 2))
    result[pred_labels == 0, 0] = 1.0
    result[pred_labels == 1, 1] = 1.0
    return result


def parse_reps_from_probs(phase_probs, min_phase=3, max_gap=3):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
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


def compute_ce_ratios(reps, phase_arr):
    ratios = []
    for rep in reps:
        seg = phase_arr[rep.start_idx:rep.end_idx]
        c = np.sum(seg == CONCENTRIC_LABEL); e = np.sum(seg == ECCENTRIC_LABEL)
        ratios.append(c / e if e > 0 else float('inf'))
    return ratios


def compute_ce_ratio_metrics(pred_ratios, gt_ratios):
    valid = [(p, g) for p, g in zip(pred_ratios, gt_ratios) if np.isfinite(p) and np.isfinite(g) and p != float('inf') and g != float('inf')]
    if not valid: return {"mae": None, "rmse": None, "bias": None, "n": 0}
    pred = np.array([p for p, _ in valid]); gt = np.array([g for _, g in valid])
    errors = pred - gt
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))), "bias": float(np.mean(errors)), "n": len(valid)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, method_name, boundary_probs=None, boundary_threshold=0.5):
    results = {}
    
    # Baseline: MA15 + argmax
    smoothed = smooth_ma(phase_probs, 15)
    pred_reps = parse_reps_from_probs(smoothed)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(smoothed, gt_phases)
    pred_ratios = compute_ce_ratios(pred_reps, np.array(["eccentric" if p == 0 else "concentric" for p in np.argmax(smoothed, axis=1)]))
    gt_ratios = compute_ce_ratios(gt_reps, gt_phases)
    ce_m = compute_ce_ratio_metrics(pred_ratios, gt_ratios)
    results[f"{method_name}_MA15"] = {
        **rep_m, **phase_m, "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"],
        **ce_m,
        "phase_seg_count": len(pred_reps),
    }
    
    # If boundary available, try boundary-aware decoding
    if boundary_probs is not None:
        decoded = decode_with_boundary(smoothed, boundary_probs, boundary_threshold)
        pred_reps_b = parse_reps_from_probs(decoded)
        rep_m_b = evaluate_reps(pred_reps_b, gt_reps)
        phase_m_b = evaluate_phase(decoded, gt_phases)
        pred_ratios_b = compute_ce_ratios(pred_reps_b, np.array(["eccentric" if p == 0 else "concentric" for p in np.argmax(decoded, axis=1)]))
        ce_m_b = compute_ce_ratio_metrics(pred_ratios_b, gt_ratios)
        results[f"{method_name}_MA15_Boundary{boundary_threshold}"] = {
            **rep_m_b, **phase_m_b, "pred_count": rep_m_b["pred_count"], "gt_count": rep_m_b["gt_count"],
            **ce_m_b,
            "phase_seg_count": len(pred_reps_b),
        }
    
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
        ce_mae = np.mean([r[v]["mae"] for r in results_list if r[v].get("mae") is not None])
        phase_seg_ratio = np.mean([r[v]["phase_seg_count"] / max(r[v]["gt_count"], 1) for r in results_list])
        agg[v] = {
            "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
            "exact_count_acc": exact / n if n > 0 else 0,
            "over_count": over, "under_count": under,
            "phase_macro_f1": phase_f1,
            "transition_mae_ms": np.mean(trans) if trans else None,
            "ce_ratio_mae": ce_mae if ce_mae is not None else None,
            "phase_seg_count_ratio": phase_seg_ratio,
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
    
    # Train models
    print("\n[1/3] Training Active Detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    
    print("\n[2/3] Training Phase-Only CNN...")
    phase_model, phase_mean, phase_std = train_phase_only(train_streams, cfg.imu_columns, cfg)
    
    print("\n[3/3] Training Boundary-Aware CNN (lambda=0.5)...")
    boundary_model, boundary_mean, boundary_std = train_boundary_aware(train_streams, cfg.imu_columns, cfg, lam=0.5)
    
    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")
    
    all_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns: continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        
        # Phase-only
        phase_probs = predict_phase_only(phase_model, df, active_segments, cfg.imu_columns, phase_mean, phase_std)
        r1 = eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, "PhaseOnly")
        
        # Boundary-aware
        b_phase_probs, boundary_probs = predict_boundary_aware(boundary_model, df, active_segments, cfg.imu_columns, boundary_mean, boundary_std)
        r2 = eval_stream(stream_id, df, b_phase_probs, gt_reps, gt_phases, "BoundaryAware", boundary_probs, 0.5)
        
        merged = {**r1, **r2}
        all_results.append(merged)
    
    # Aggregate
    agg = aggregate(all_results)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    for name, m in agg.items():
        print(f"\n{name}:")
        print(f"  Rep F1:           {m['rep_f1']:.4f}")
        print(f"  Exact Count:      {m['exact_count_acc']:.4f}")
        print(f"  Over/Under:       {m['over_count']}/{m['under_count']}")
        print(f"  Phase F1:         {m['phase_macro_f1']:.4f}")
        print(f"  Trans MAE:        {m.get('transition_mae_ms', 0):.0f}ms")
        print(f"  C/E Ratio MAE:    {m.get('ce_ratio_mae', 0):.3f}")
        print(f"  Phase Seg Ratio:  {m.get('phase_seg_count_ratio', 0):.3f}")
    
    # Save
    output_dir = Path("artifacts/boundary_head_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\n[OK] Saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
