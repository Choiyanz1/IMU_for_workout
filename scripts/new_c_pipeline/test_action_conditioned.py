"""
Kevin Fold Quick Test: Action-Conditioned CNN vs Global 2-class CNN

Action-Conditioned Architecture:
  - Shared dilated causal encoder (trained on ALL segments)
  - 8 independent 2-class heads (one per action)
  - Inference: pick head for the known action

Strict comparison rules:
  - Same data exclusion (light-weight sessions removed)
  - Same decoder (MA25 + Viterbi p=0.3)
  - Same random seed
  - Train both models from scratch in same script
  - Report per-action breakdown
"""
from __future__ import annotations

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
    RepDetection, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig,
    evaluate_phase,
    evaluate_reps,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Data Exclusion Config
# ---------------------------------------------------------------------------

EXCLUDED_SESSIONS = {
    "yanz": ["1000"],
    "thomas": ["thomas", "thomas_2"],
    "kevin": ["kevin"],
}


def should_exclude(stream_id: str) -> bool:
    parts = stream_id.split("/")
    if len(parts) < 2:
        return False
    return parts[0] in EXCLUDED_SESSIONS and parts[1] in EXCLUDED_SESSIONS[parts[0]]


# ---------------------------------------------------------------------------
# Fixed Seeds
# ---------------------------------------------------------------------------

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Action Mapping
# ---------------------------------------------------------------------------

ACTION_LIST = [
    'db_bench_press', 'db_biceps_curl', 'db_rdl', 'db_shoulder_press',
    'db_squat', 'db_triceps_curl', 'db_weighted_crunch', 'one_arm_db_row'
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def get_action_idx(stream_id):
    parts = [p for p in str(stream_id).split("/") if p]
    action = parts[-2] if len(parts) >= 3 else "unknown"
    return ACTION_TO_IDX.get(action, 0)


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


# ---------------------------------------------------------------------------
# Architectures
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


class Global2ClassCNN(nn.Module):
    """Baseline: shared encoder + single 2-class head."""
    def __init__(self, in_ch=6, hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.head = nn.Conv1d(hidden, 2, 1)
    def forward(self, x):
        return self.head(self.encoder(x))


class ActionConditionedCNN(nn.Module):
    """Shared encoder + 8 independent 2-class heads.
    
    forward(x, action_idx):
      x: [B, in_ch, T]
      action_idx: [B] (integers 0-7)
    """
    def __init__(self, in_ch=6, hidden=64, num_actions=8, dropout=0.2):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.heads = nn.ModuleList([nn.Conv1d(hidden, 2, 1) for _ in range(num_actions)])
    def forward(self, x, action_idx):
        B, _, T = x.shape
        features = self.encoder(x)  # [B, hidden, T]
        output = torch.zeros(B, 2, T, device=x.device, dtype=x.dtype)
        for a in range(self.num_actions):
            mask = (action_idx == a)
            if mask.sum() > 0:
                output[mask] = self.heads[a](features[mask])
        return output


# ---------------------------------------------------------------------------
# Data Utils
# ---------------------------------------------------------------------------

def extract_segments_with_action(train_streams, imu_columns):
    """Extract segments with action_idx labels."""
    segments, labels, actions = [], [], []
    for stream_id, df in train_streams:
        if "phase" not in df.columns: continue
        action_idx = get_action_idx(stream_id)
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
                    actions.append(action_idx)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            segments.append(x[seg_start:])
            labels.append(np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]]))
            actions.append(action_idx)
    return segments, labels, actions


def normalize(segments):
    if len(segments) == 0: return None, None, []
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std, [(seg - mean) / std for seg in segments]


class ActionPhaseDataset(Dataset):
    """Dataset that returns (x, y, mask, action_idx)."""
    def __init__(self, segments, labels, actions, slice_len=300):
        self.samples = []
        for seq, lab, act in zip(segments, labels, actions):
            n = len(seq)
            if n <= slice_len:
                pad_len = max(0, slice_len - n)
                seq_pad = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
                lab_pad = np.pad(lab, (0, pad_len), constant_values=-1)
                mask = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
                self.samples.append((seq_pad[:slice_len], lab_pad[:slice_len], mask[:slice_len], act))
            else:
                stride = slice_len // 2
                starts = list(range(0, n - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < n:
                    starts.append(n - slice_len)
                for start in starts:
                    self.samples.append((seq[start:start+slice_len], lab[start:start+slice_len], np.ones(slice_len, dtype=np.float32), act))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq, lab, mask, act = self.samples[idx]
        return (torch.from_numpy(seq).float().transpose(0, 1),
                torch.from_numpy(lab).long(),
                torch.from_numpy(mask).float(),
                torch.tensor(act, dtype=torch.long))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_global_model(model, train_segments, train_labels, max_epochs=20):
    if len(train_segments) == 0: return None, None, None
    mean, std, norm_segments = normalize(train_segments)
    if mean is None: return None, None, None
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    # Simple dataset (no action), sliced to 300 samples
    def _prepare_sample(seq, lab):
        mask = np.ones(len(seq), dtype=np.float32)
        if len(seq) < 300:
            pad_len = 300 - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
            lab = np.pad(lab, (0, pad_len), constant_values=-1)
            mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.float32)])
            return seq[:300], lab[:300], mask[:300]
        else:
            # Random crop to 300
            start = np.random.randint(0, len(seq) - 300 + 1)
            return seq[start:start+300], lab[start:start+300], mask[:300]
    
    train_ds = []
    for i in train_idx:
        seq, lab, mask = _prepare_sample(norm_segments[i], train_labels[i])
        train_ds.append((torch.from_numpy(seq).float().transpose(0, 1), torch.from_numpy(lab).long(), torch.from_numpy(mask).float()))
    
    val_ds = []
    for i in val_idx:
        seq, lab, mask = _prepare_sample(norm_segments[i], train_labels[i])
        val_ds.append((torch.from_numpy(seq).float().transpose(0, 1), torch.from_numpy(lab).long(), torch.from_numpy(mask).float()))
    
    from torch.utils.data import TensorDataset, DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    
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
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


def train_action_conditioned_model(model, train_segments, train_labels, train_actions, max_epochs=20):
    if len(train_segments) == 0: return None, None, None
    mean, std, norm_segments = normalize(train_segments)
    if mean is None: return None, None, None
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_ds = ActionPhaseDataset([norm_segments[i] for i in train_idx], [train_labels[i] for i in train_idx], [train_actions[i] for i in train_idx])
    val_ds = ActionPhaseDataset([norm_segments[i] for i in val_idx], [train_labels[i] for i in val_idx], [train_actions[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_val = float('inf'); best_state = None
    
    for epoch in range(max_epochs):
        model.train(); train_loss = 0; n_batches = 0
        for x, y, m, a in train_loader:
            x, y, m, a = x.to(device), y.to(device), m.to(device), a.to(device)
            optimizer.zero_grad()
            logits = model(x, a)
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
            for x, y, m, a in val_loader:
                x, y, m, a = x.to(device), y.to(device), m.to(device), a.to(device)
                logits = model(x, a)
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
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std


# ---------------------------------------------------------------------------
# Inference
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


def predict_global(model, df, active_segments, imu_columns, mean, std):
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
    phase_probs = smooth_ma(phase_probs, 25)
    phase_probs = viterbi_decode(phase_probs, 0.3)
    return phase_probs


def predict_action_conditioned(model, df, active_segments, imu_columns, mean, std, action_idx):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    if model is None: return phase_probs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    action_tensor = torch.tensor([action_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x); seg_x_norm = (seg_x - mean) / std
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor, action_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor, action_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    gs = seg_start + start
                    phase_probs[gs:gs + 300, :] += probs.T
                    phase_counts[gs:gs + 300] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    phase_probs = smooth_ma(phase_probs, 25)
    phase_probs = viterbi_decode(phase_probs, 0.3)
    return phase_probs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_reps(hard_labels, min_phase=3, max_gap=3):
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


def evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_reps = parse_reps(pred_labels)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(phase_probs, gt_phases)
    return {
        "stream_id": stream_id,
        "action": _extract_action_from_stream_id(stream_id),
        "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"],
        "count_error": abs(rep_m["pred_count"] - rep_m["gt_count"]),
        **{k: v for k, v in rep_m.items() if k not in ["pred_count", "gt_count"]},
        **phase_m,
    }


def aggregate(results):
    if not results: return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    count_errors = [r["count_error"] for r in results]
    trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": sum(r["exact_count"] for r in results) / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": sum(r["over"] for r in results), "under_count": sum(r["under"] for r in results),
        "phase_macro_f1": np.mean([r["phase_macro_f1"] for r in results]),
        "transition_mae_ms": np.mean(trans) if trans else None,
    }


def aggregate_by_action(results):
    action_results = {}
    for r in results:
        action = r["action"]
        if action not in action_results:
            action_results[action] = []
        action_results[action].append(r)
    aggregated = {}
    for action, a_results in action_results.items():
        n = len(a_results)
        total_tp = sum(r["tp"] for r in a_results); total_fp = sum(r["fp"] for r in a_results); total_fn = sum(r["fn"] for r in a_results)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        count_errors = [r["count_error"] for r in a_results]
        aggregated[action] = {
            "streams": n, "rep_f1": f1,
            "exact_count_acc": sum(r["exact_count"] for r in a_results) / n if n > 0 else 0,
            "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
            "over_count": sum(r["over"] for r in a_results), "under_count": sum(r["under"] for r in a_results),
            "phase_macro_f1": np.mean([r["phase_macro_f1"] for r in a_results]),
        }
    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    raw = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    filtered_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    print(f"Excluded: {len(all_streams) - len(filtered_streams)}, Remaining: {len(filtered_streams)}")
    print(f"GPU: {torch.cuda.is_available()}")

    cfg = PhaseCompareConfig()
    test_subject = "kevin"
    train_streams = [(sid, df) for sid, df in filtered_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in filtered_streams if sid.startswith(f"{test_subject}/")]
    print(f"\nTrain: {len(train_streams)}, Test: {len(test_streams)}")

    # Extract segments
    segs_g, labs_g = extract_segments_with_action(train_streams, cfg.imu_columns)[:2]
    segs_ac, labs_ac, acts_ac = extract_segments_with_action(train_streams, cfg.imu_columns)

    # Train Global Baseline
    print("\n[1/2] Training Global 2-class CNN...")
    global_model, mean_g, std_g = train_global_model(Global2ClassCNN(6, 64), segs_g, labs_g, max_epochs=20)
    print(f"    Done. Val loss optimized.")

    # Train Action-Conditioned
    print("\n[2/2] Training Action-Conditioned CNN...")
    ac_model, mean_ac, std_ac = train_action_conditioned_model(ActionConditionedCNN(6, 64, 8), segs_ac, labs_ac, acts_ac, max_epochs=20)
    print(f"    Done. Val loss optimized.")

    # Evaluate both on same test data
    print(f"\n{'='*70}")
    print("KEVIN FOLD COMPARISON")
    print(f"{'='*70}")

    active_models, active_scalers = train_active_detector(train_streams, cfg)

    variants = {
        "Global_2class": lambda sid, df, segs: predict_global(global_model, df, segs, cfg.imu_columns, mean_g, std_g),
        "Action_Conditioned": lambda sid, df, segs: predict_action_conditioned(ac_model, df, segs, cfg.imu_columns, mean_ac, std_ac, get_action_idx(sid)),
    }

    all_results = {}
    for name, predict_fn in variants.items():
        results = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fn(stream_id, df, active_segments)
            results.append(evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases))

        agg = aggregate(results)
        by_action = aggregate_by_action(results)
        all_results[name] = {"overall": agg, "per_action": by_action}

        print(f"\n{name}")
        print(f"  Overall: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
              f"Over/Under={agg['over_count']}/{agg['under_count']} CountMAE={agg['mean_abs_count_error']:.2f} "
              f"PhaseF1={agg['phase_macro_f1']:.4f}")
        print(f"  Per-Action:")
        for action in sorted(by_action.keys()):
            a = by_action[action]
            print(f"    {action:<25s}: RepF1={a['rep_f1']:.4f} Exact={a['exact_count_acc']:.3f} "
                  f"Over/Under={a['over_count']}/{a['under_count']} CountMAE={a['mean_abs_count_error']:.2f}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    g = all_results["Global_2class"]["overall"]
    ac = all_results["Action_Conditioned"]["overall"]
    if ac["rep_f1"] > g["rep_f1"] and ac["exact_count_acc"] > g["exact_count_acc"]:
        print("ACTION-CONDITIONED WINS on RepF1 AND ExactCount")
        print(f"  RepF1: {g['rep_f1']:.4f} -> {ac['rep_f1']:.4f} (+{ac['rep_f1']-g['rep_f1']:.4f})")
        print(f"  Exact: {g['exact_count_acc']:.3f} -> {ac['exact_count_acc']:.3f} (+{ac['exact_count_acc']-g['exact_count_acc']:.3f})")
    elif ac["rep_f1"] > g["rep_f1"]:
        print("ACTION-CONDITIONED wins on RepF1 but NOT ExactCount")
    elif ac["exact_count_acc"] > g["exact_count_acc"]:
        print("ACTION-CONDITIONED wins on ExactCount but NOT RepF1")
    else:
        print("GLOBAL BASELINE still better. No improvement.")
        print(f"  RepF1: Global={g['rep_f1']:.4f}, AC={ac['rep_f1']:.4f}")
        print(f"  Exact: Global={g['exact_count_acc']:.3f}, AC={ac['exact_count_acc']:.3f}")


if __name__ == "__main__":
    main()
