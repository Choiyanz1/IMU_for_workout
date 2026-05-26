"""
Phase Segmentation Model Comparison: RF window-based vs Sequence Labeling

Purpose: Compare phase segmentation quality on identical active segments.

Pipeline:
Raw IMU
→ Per-Action Active Detector (RF, rich features, causal window)
→ Active Segment Extraction
→ [A] RF window-based C/E phase model
→ [B] 1D-CNN seq2seq C/E phase model
→ Phase Smoothing
→ Rep Parser
→ Evaluation

Metrics: phase macro F1, transition MAE, rep IoU-F1@50, rep count error, etc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, MICRO_LABELS, OTHER_LABEL,
    micro_labels_from_phase, pair_concentric_eccentric_reps, RepDetection,
    SegmentRun, labels_to_runs, truth_reps_from_labels,
)
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.micro_macro_recognition import _load_streams

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PhaseCompareConfig:
    imu_columns: Tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
    # Active Detector
    active_window_size: int = 100
    active_stride: int = 10
    active_n_estimators: int = 100
    active_max_depth: int = 15
    active_max_samples: float = 0.7
    # RF Phase
    phase_window_size: int = 100
    phase_stride: int = 10
    phase_n_estimators: int = 100
    phase_max_depth: int = 15
    phase_max_samples: float = 0.7
    # Smoothing & Rep Parser
    smoothing_window: int = 15
    min_phase_samples: int = 3
    max_phase_gap_samples: int = 3
    # Seq2Seq
    seq2seq_epochs: int = 30
    seq2seq_lr: float = 1e-3
    seq2seq_batch_size: int = 16
    seq2seq_hidden: int = 64
    seq2seq_slice_len: int = 300
    seq2seq_device: str = "cpu"


# ---------------------------------------------------------------------------
# Feature Extraction (shared)
# ---------------------------------------------------------------------------

def _extract_window_features_batch(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    window_len = int(arr.shape[1])
    mean = np.mean(arr, axis=1); std = np.std(arr, axis=1); vmin = np.min(arr, axis=1)
    vmax = np.max(arr, axis=1); median = np.median(arr, axis=1); q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1); total_variation = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    mag = np.sqrt(np.sum(arr ** 2, axis=2))
    mag_stats = np.stack([np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1)], axis=1)
    per_channel = np.stack([mean, std, vmin, vmax, median, q25, q75, total_variation], axis=-1).reshape(arr.shape[0], -1)
    return np.concatenate([per_channel, mag_stats], axis=1).astype(np.float32, copy=False)


def _build_start_window_matrix(x: np.ndarray, window_size: int, stride: int):
    n = len(x)
    if n <= 0:
        empty = np.zeros((0,), dtype=np.int64)
        return np.zeros((0, 0), dtype=np.float32), empty, empty
    window_size = int(max(1, window_size)); stride = int(max(1, stride))
    if n < window_size:
        starts = np.asarray([0], dtype=np.int64)
    else:
        starts = np.arange(0, n - window_size + 1, stride, dtype=np.int64)
    pad = max(0, window_size - n)
    padded = np.pad(x, ((0, pad), (0, 0)), mode="edge") if pad > 0 else x
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    selected = windows[starts]
    ends = np.minimum(starts + window_size, n)
    return _extract_window_features_batch(selected), starts, ends


# ---------------------------------------------------------------------------
# Action Utils
# ---------------------------------------------------------------------------

def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def _prepare_active_labels(phases: np.ndarray) -> np.ndarray:
    labels = micro_labels_from_phase(phases)
    return np.array([1 if str(l) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL} else 0 for l in labels], dtype=np.int64)


def _prepare_phase_labels(phases: np.ndarray) -> np.ndarray:
    labels = micro_labels_from_phase(phases)
    result = np.full(len(labels), -1, dtype=np.int64)
    for i, l in enumerate(labels):
        if str(l) == CONCENTRIC_LABEL: result[i] = 1
        elif str(l) == ECCENTRIC_LABEL: result[i] = 0
    return result


# ---------------------------------------------------------------------------
# Active Detector (unchanged)
# ---------------------------------------------------------------------------

def train_active_detector(train_streams, cfg):
    action_streams = {}
    for stream_id, df in train_streams:
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_streams: action_streams[action] = []
        action_streams[action].append((stream_id, df))
    models, scalers = {}, {}
    for action, streams in action_streams.items():
        X_all, y_all = [], []
        for _, df in streams:
            if "phase" not in df.columns: continue
            x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
            active_labels = _prepare_active_labels(df["phase"].to_numpy())
            features, starts, ends = _build_start_window_matrix(x, cfg.active_window_size, cfg.active_stride)
            if len(features) == 0: continue
            y_batch = [int(np.bincount(active_labels[int(s):int(e)]).argmax()) for s, e in zip(starts, ends)]
            X_all.append(features); y_all.append(np.asarray(y_batch, dtype=np.int64))
        if not X_all: continue
        X, y = np.concatenate(X_all), np.concatenate(y_all)
        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=cfg.active_n_estimators, max_depth=cfg.active_max_depth,
                                     max_samples=cfg.active_max_samples, random_state=42, n_jobs=-1)
        clf.fit(X_s, y); models[action] = clf; scalers[action] = scaler
    return models, scalers


def predict_active(models, scalers, stream_id, df, cfg):
    action = _extract_action_from_stream_id(stream_id)
    if action not in models: action = list(models.keys())[0]
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    features, starts, ends = _build_start_window_matrix(x, cfg.active_window_size, cfg.active_stride)
    if len(features) == 0: return np.zeros(n)
    X_s = scalers[action].transform(features)
    probs = models[action].predict_proba(X_s)
    active_idx = list(models[action].classes_).index(1) if 1 in models[action].classes_ else 0
    prob_accum = np.zeros(n, dtype=np.float64); counts = np.zeros(n, dtype=np.float64)
    for wi, (s, e) in enumerate(zip(starts, ends)):
        prob_accum[int(s):int(e)] += probs[wi, active_idx]; counts[int(s):int(e)] += 1.0
    counts = np.where(counts < 1e-8, 1.0, counts); return prob_accum / counts


# ---------------------------------------------------------------------------
# Active Segment Extraction
# ---------------------------------------------------------------------------

def extract_active_segments(active_probs, threshold=0.5, min_consecutive=3):
    """Extract active segments from per-sample active probabilities."""
    n = len(active_probs); state = "IDLE"; consecutive = 0; segments = []; current_start = None
    for i in range(n):
        is_active = active_probs[i] >= threshold
        if state == "IDLE":
            if is_active:
                consecutive += 1
                if consecutive >= min_consecutive:
                    state = "ACTIVE"; current_start = i - min_consecutive + 1
            else:
                consecutive = 0
        else:
            if not is_active:
                consecutive += 1
                if consecutive >= min_consecutive:
                    state = "IDLE"; end = i - min_consecutive + 1
                    if current_start is not None and end > current_start:
                        segments.append((current_start, end))
                    current_start = None
            else:
                consecutive = 0
    if state == "ACTIVE" and current_start is not None:
        segments.append((current_start, n))
    return segments


# ---------------------------------------------------------------------------
# [A] RF Window-based Phase Model (existing)
# ---------------------------------------------------------------------------

def train_rf_phase(train_streams, cfg):
    action_streams = {}
    for stream_id, df in train_streams:
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_streams: action_streams[action] = []
        action_streams[action].append((stream_id, df))
    models, scalers = {}, {}
    for action, streams in action_streams.items():
        X_all, y_all = [], []
        for _, df in streams:
            if "phase" not in df.columns: continue
            x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
            phase_labels = _prepare_phase_labels(df["phase"].to_numpy())
            features, starts, ends = _build_start_window_matrix(x, cfg.phase_window_size, cfg.phase_stride)
            if len(features) == 0: continue
            y_batch = []; valid = []
            for s, e in zip(starts, ends):
                wl = phase_labels[int(s):int(e)]
                valid_labels = wl[wl >= 0]
                if len(valid_labels) > 0 and len(valid_labels) / len(wl) > 0.5:
                    y_batch.append(int(np.bincount(valid_labels).argmax())); valid.append(True)
                else:
                    valid.append(False)
            valid_idx = np.where(valid)[0]
            if len(valid_idx) > 0:
                X_all.append(features[valid_idx]); y_all.append(np.asarray(y_batch, dtype=np.int64))
        if not X_all: continue
        X, y = np.concatenate(X_all), np.concatenate(y_all)
        scaler = StandardScaler(); X_s = scaler.fit_transform(X)
        clf = RandomForestClassifier(n_estimators=cfg.phase_n_estimators, max_depth=cfg.phase_max_depth,
                                     max_samples=cfg.phase_max_samples, random_state=42, n_jobs=-1)
        clf.fit(X_s, y); models[action] = clf; scalers[action] = scaler
    return models, scalers


def predict_rf_phase(models, scalers, stream_id, df, active_segments, cfg):
    action = _extract_action_from_stream_id(stream_id)
    if action not in models: action = list(models.keys())[0]
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5
    for seg_start, seg_end in active_segments:
        if seg_start >= seg_end: continue
        seg_x = x[seg_start:seg_end]
        features, starts, ends = _build_start_window_matrix(seg_x, cfg.phase_window_size, cfg.phase_stride)
        if len(features) == 0: continue
        X_s = scalers[action].transform(features)
        probs = models[action].predict_proba(X_s)
        class_map = {int(c): i for i, c in enumerate(models[action].classes_)}
        full_batch = np.zeros((len(probs), 2), dtype=np.float64)
        for cls_idx, mi in class_map.items():
            if cls_idx < 2: full_batch[:, cls_idx] = probs[:, mi]
        prob_accum = np.zeros((seg_end - seg_start, 2), dtype=np.float64); counts = np.zeros(seg_end - seg_start, dtype=np.float64)
        for wi, (s, e) in enumerate(zip(starts, ends)):
            prob_accum[int(s):int(e)] += full_batch[wi]; counts[int(s):int(e)] += 1.0
        counts = np.where(counts < 1e-8, 1.0, counts)
        phase_probs[seg_start:seg_end] = prob_accum / counts[:, None]
    return phase_probs


# ---------------------------------------------------------------------------
# [B] 1D-CNN Seq2Seq Phase Model (NEW)
# ---------------------------------------------------------------------------

class SimpleSeq2Seq(nn.Module):
    """Lightweight 1D-CNN for per-sample C/E segmentation.
    
    Architecture fixes:
    - GroupNorm instead of BatchNorm for stable batch_size=1 inference
    - Dilated convolutions with larger receptive field (RF ~1.3s @100Hz)
    - Residual connections for better gradient flow
    
    Input: [B, 6, T]
    Output: [B, 2, T] (logits for eccentric/concentric)
    """
    def __init__(self, input_channels=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(8, hidden)
        # dilation=2: RF += 8 (total 13 samples = 130ms)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2)
        self.gn2 = nn.GroupNorm(8, hidden)
        # dilation=4: RF += 16 (total 29 samples = 290ms)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4)
        self.gn3 = nn.GroupNorm(8, hidden)
        # dilation=8: RF += 32 (total 61 samples = 610ms)
        self.conv4 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=16, dilation=8)
        self.gn4 = nn.GroupNorm(8, hidden)
        # dilation=16: RF += 64 (total 125 samples = 1.25s)
        self.conv5 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=32, dilation=16)
        self.gn5 = nn.GroupNorm(8, hidden)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Conv1d(hidden, num_classes, kernel_size=1)
        
        # Residual projection if needed
        self.res_proj = nn.Conv1d(input_channels, hidden, kernel_size=1) if input_channels != hidden else None
    
    def forward(self, x):
        # x: [B, C, T]
        identity = x if self.res_proj is None else self.res_proj(x)
        
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
        
        # Add residual (trim if size mismatch from padding)
        if x.shape[2] == identity.shape[2]:
            x = x + identity[:, :x.shape[1], :]
        
        logits = self.fc(x)  # [B, 2, T]
        return logits


class ActiveSegmentDataset(Dataset):
    """Dataset of active segments with per-sample C/E labels."""
    def __init__(self, segments, labels, slice_len=300):
        """
        segments: List of [T, C] numpy arrays (IMU active segments)
        labels: List of [T] numpy arrays (0=E, 1=C, -1=pad/invalid)
        slice_len: Fixed sequence length
        """
        self.samples = []
        for seq, lab in zip(segments, labels):
            n = len(seq)
            if n <= slice_len:
                # Pad short sequences
                pad_len = max(0, slice_len - n)
                seq_pad = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
                lab_pad = np.pad(lab, (0, pad_len), constant_values=-1)
                mask = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
                self.samples.append((seq_pad[:slice_len], lab_pad[:slice_len], mask[:slice_len]))
            else:
                # Sliding windows with 50% overlap, NO duplicate last window
                stride = slice_len // 2
                starts = list(range(0, n - slice_len + 1, stride))
                # If last window is already covered, don't add it again
                if not starts or starts[-1] + slice_len < n:
                    starts.append(n - slice_len)
                for start in starts:
                    self.samples.append((
                        seq[start:start+slice_len],
                        lab[start:start+slice_len],
                        np.ones(slice_len, dtype=np.float32)
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, lab, mask = self.samples[idx]
        # [T, C] -> [C, T]
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        m = torch.from_numpy(mask).float()
        return x, y, m


def extract_active_segments_for_seq2seq(train_streams, imu_columns):
    """Extract active segments and per-sample C/E labels."""
    segments = []; labels = []
    for _, df in train_streams:
        if "phase" not in df.columns: continue
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        phase_arr = df["phase"].to_numpy()
        
        # Find active segments
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
    """Compute per-channel mean/std across all segments."""
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_segments(segments, mean, std):
    return [(seg - mean) / std for seg in segments]


def train_seq2seq_phase(train_streams, imu_columns, cfg):
    """Train 1D-CNN seq2seq phase segmenter with validation split and early stopping."""
    segments, labels = extract_active_segments_for_seq2seq(train_streams, imu_columns)
    if not segments:
        print("      [Seq2Seq] No active segments found for training")
        return None, None, None
    
    print(f"      [Seq2Seq] Training on {len(segments)} active segments")
    
    # Normalize using training stats
    mean, std = compute_normalization_stats(segments)
    norm_segments = normalize_segments(segments, mean, std)
    
    # Train/val split: hold out 15% of segments
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_segments = [norm_segments[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_segments = [norm_segments[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    # Create datasets
    train_dataset = ActiveSegmentDataset(train_segments, train_labels, slice_len=cfg.seq2seq_slice_len)
    val_dataset = ActiveSegmentDataset(val_segments, val_labels, slice_len=cfg.seq2seq_slice_len)
    
    if len(train_dataset) == 0:
        print("      [Seq2Seq] Empty training dataset after slicing")
        return None, None, None
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=False, drop_last=False)
    
    print(f"      [Seq2Seq] Train: {len(train_dataset)} slices, Val: {len(val_dataset)} slices")
    
    # Model
    device = torch.device(cfg.seq2seq_device)
    model = SimpleSeq2Seq(input_channels=len(imu_columns), hidden=cfg.seq2seq_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.seq2seq_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    # Training loop
    for epoch in range(cfg.seq2seq_epochs):
        # Train
        model.train()
        train_loss = 0; n_batches = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device); y_batch = y_batch.to(device); mask_batch = mask_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)  # [B, 2, T]
            
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
        
        # Validate
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
            print(f"      [Seq2Seq] Epoch {epoch+1}/{cfg.seq2seq_epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"      [Seq2Seq] Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, mean, std


def predict_seq2seq_phase(model, df, active_segments, imu_columns, mean, std, cfg):
    """Predict per-sample C/E probability using seq2seq model with sliding window + overlap averaging.
    
    CRITICAL FIX: For long segments, use overlapping windows and average predictions in overlap regions.
    """
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5  # default uncertain
    phase_counts = np.zeros(n, dtype=np.float32)  # for averaging overlapping predictions
    
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
            
            # Normalize
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= slice_len:
                # Short segment: pad and predict once
                pad_len = slice_len - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)  # [1, C, T]
                logits = model(x_tensor)  # [1, 2, T]
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [2, T]
                
                # Only use valid portion (before padding)
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                # Long segment: sliding window with 50% overlap, then average
                stride = slice_len // 2
                
                for start in range(0, seg_len - slice_len + 1, stride):
                    window = seg_x_norm[start:start + slice_len]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [2, T]
                    
                    # Add to accumulator
                    global_start = seg_start + start
                    global_end = global_start + slice_len
                    phase_probs[global_start:global_end, :] += probs.T
                    phase_counts[global_start:global_end] += 1.0
                
                # Last window: ensure coverage to end
                last_start = seg_len - slice_len
                if last_start % stride != 0:
                    window = seg_x_norm[last_start:last_start + slice_len]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    global_start = seg_start + last_start
                    global_end = global_start + slice_len
                    phase_probs[global_start:global_end, :] += probs.T
                    phase_counts[global_start:global_end] += 1.0
    
    # Average overlapping predictions
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    
    return phase_probs


# ---------------------------------------------------------------------------
# Shared Post-processing
# ---------------------------------------------------------------------------

def smooth_phase_probs(phase_probs, smoothing_window=15):
    n = len(phase_probs); smoothed = np.copy(phase_probs)
    if smoothing_window > 1:
        for c in range(2):
            cumsum = np.cumsum(phase_probs[:, c])
            for i in range(n):
                start = max(0, i - smoothing_window + 1)
                total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
                smoothed[i, c] = total / (i - start + 1)
    return smoothed


def _merge_adjacent_same_phase(runs):
    if not runs: return runs
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(label=run.label, start_idx=merged[-1].start_idx,
                                     end_idx=run.end_idx, confidence=(merged[-1].confidence + run.confidence) / 2)
        else:
            merged.append(run)
    return merged


def parse_reps_from_phase(phase_probs, min_phase_samples=3, max_gap_samples=3):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase_samples)
    runs = _merge_adjacent_same_phase(runs)
    reps, _ = pair_concentric_eccentric_reps(runs, micro_source="phase", max_gap_samples=max_gap_samples)
    return reps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_reps(pred_reps, gt_reps):
    pred_count, gt_count = len(pred_reps), len(gt_reps)
    tp = 0; matched_gt = set()
    for pred in pred_reps:
        best_iou = 0; best_gt = None
        for gi, gt in enumerate(gt_reps):
            if gi in matched_gt: continue
            pred_range = set(range(pred.start_idx, pred.end_idx))
            gt_range = set(range(gt.start_idx, gt.end_idx))
            inter = len(pred_range & gt_range); union = len(pred_range | gt_range)
            iou = inter / union if union > 0 else 0
            if iou > best_iou: best_iou = iou; best_gt = gi
        if best_iou >= 0.5 and best_gt is not None:
            tp += 1; matched_gt.add(best_gt)
    fp = pred_count - tp; fn = gt_count - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "pred_count": pred_count, "gt_count": gt_count, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "exact_count": 1 if pred_count == gt_count else 0,
        "over": 1 if pred_count > gt_count else 0, "under": 1 if pred_count < gt_count else 0,
    }


def evaluate_phase(phase_probs, gt_phases):
    gt_labels = _prepare_phase_labels(gt_phases); pred_labels = np.argmax(phase_probs, axis=1)
    valid = gt_labels >= 0
    if not valid.any(): return {"accuracy": 0, "macro_f1": 0, "transition_mae_ms": None}
    acc = accuracy_score(gt_labels[valid], pred_labels[valid])
    macro_f1 = f1_score(gt_labels[valid], pred_labels[valid], average="macro", zero_division=0)
    
    gt_changes = np.where(np.diff(gt_labels[valid]) != 0)[0]
    pred_changes = np.where(np.diff(pred_labels[valid]) != 0)[0]
    mae = None
    if len(gt_changes) > 0 and len(pred_changes) > 0:
        errors = [min(abs(gt_c - pc) for pc in pred_changes) for gt_c in gt_changes]
        mae = np.mean(errors) * 10  # convert to ms @100Hz
    return {"phase_accuracy": acc, "phase_macro_f1": macro_f1, "transition_mae_ms": mae}


def aggregate_results(results):
    if not results: return {}
    valid = [r for r in results if "f1" in r]
    if not valid: return {}
    total_tp = sum(r["tp"] for r in valid); total_fp = sum(r["fp"] for r in valid); total_fn = sum(r["fn"] for r in valid)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    phase_results = [r for r in valid if "phase_macro_f1" in r]
    phase_macro_f1 = np.mean([r["phase_macro_f1"] for r in phase_results]) if phase_results else 0
    phase_acc = np.mean([r["phase_accuracy"] for r in phase_results]) if phase_results else 0
    trans_mae = [r["transition_mae_ms"] for r in phase_results if r.get("transition_mae_ms") is not None]
    
    return {
        "streams": len(valid), "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": sum(r["exact_count"] for r in valid) / len(valid),
        "over_count": sum(r["over"] for r in valid), "under_count": sum(r["under"] for r in valid),
        "phase_macro_f1": phase_macro_f1,
        "phase_accuracy": phase_acc,
        "transition_mae_ms": np.nanmean(trans_mae) if trans_mae else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase Model Comparison: RF vs Seq2Seq")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_model_comparison"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: kevin only")
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = PhaseCompareConfig()
    
    print("="*70)
    print("Phase Segmentation Model Comparison")
    print("A: RF Window-based  vs  B: 1D-CNN Seq2Seq")
    print("="*70)
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {})
    cfg.imu_columns = tuple(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    
    print("\n[1/3] Loading streams...")
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"      Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    
    test_subjects = ["kevin"] if args.quick else subjects
    if args.quick:
        print(f"[QUICK] Testing on: {test_subjects}")
    
    rf_results = []; seq2seq_results = []
    
    for test_subject in test_subjects:
        print(f"\n{'='*70}")
        print(f"Fold: test={test_subject}")
        print(f"{'='*70}")
        
        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
        print(f"      train={len(train_streams)}, test={len(test_streams)}")
        
        # Train models
        print("\n  Training Active Detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        print("  Training RF Phase Model...")
        rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, cfg)
        print("  Training Seq2Seq Phase Model...")
        seq2seq_model, seq2seq_mean, seq2seq_std = train_seq2seq_phase(train_streams, cfg.imu_columns, cfg)
        
        # Evaluate on test streams
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            
            # Active Detection (shared)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            # Ground truth
            gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=cfg.min_phase_samples)
            
            # [A] RF Phase
            rf_phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, cfg)
            rf_phase_probs_smooth = smooth_phase_probs(rf_phase_probs, cfg.smoothing_window)
            rf_pred_reps = parse_reps_from_phase(rf_phase_probs_smooth, cfg.min_phase_samples, cfg.max_phase_gap_samples)
            rf_rep_metrics = evaluate_reps(rf_pred_reps, gt_reps)
            rf_phase_metrics = evaluate_phase(rf_phase_probs_smooth, df["phase"].to_numpy())
            rf_results.append({"stream_id": stream_id, **rf_rep_metrics, **rf_phase_metrics})
            
            # [B] Seq2Seq Phase
            if seq2seq_model is not None:
                sq_phase_probs = predict_seq2seq_phase(seq2seq_model, df, active_segments, cfg.imu_columns,
                                                          seq2seq_mean, seq2seq_std, cfg)
                sq_phase_probs_smooth = smooth_phase_probs(sq_phase_probs, cfg.smoothing_window)
                sq_pred_reps = parse_reps_from_phase(sq_phase_probs_smooth, cfg.min_phase_samples, cfg.max_phase_gap_samples)
                sq_rep_metrics = evaluate_reps(sq_pred_reps, gt_reps)
                sq_phase_metrics = evaluate_phase(sq_phase_probs_smooth, df["phase"].to_numpy())
                seq2seq_results.append({"stream_id": stream_id, **sq_rep_metrics, **sq_phase_metrics})
            else:
                seq2seq_results.append({"stream_id": stream_id, "error": "seq2seq not trained"})
    
    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    
    rf_agg = aggregate_results([r for r in rf_results if "f1" in r])
    print(f"\n[A] RF Window-based Phase Model:")
    for k, v in rf_agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    if seq2seq_results and "f1" in seq2seq_results[0]:
        sq_agg = aggregate_results([r for r in seq2seq_results if "f1" in r])
        print(f"\n[B] 1D-CNN Seq2Seq Phase Model:")
        for k, v in sq_agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        print(f"\n[B] 1D-CNN Seq2Seq Phase Model: NOT AVAILABLE (training failed)")
    
    summary = {"rf": rf_agg, "seq2seq": aggregate_results([r for r in seq2seq_results if "f1" in r]) if seq2seq_results else {}}
    output_file = args.output / f"comparison_{'quick' if args.quick else 'full'}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
