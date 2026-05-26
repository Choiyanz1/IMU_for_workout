"""
9-Fold LOSO: Compare three CNN variants
  A) Global 2-class CNN
  B) Per-Action 2-class CNN (8 models)
  C) Action-Phase 16-class CNN (1 model, 16 = 8 actions x 2 phases)

Metrics:
  - Rep F1 (IoU@0.50), Count MAE, Exact Count, Over/Under
  - Phase Macro F1, Phase Accuracy, Transition MAE
  - Concentric/Eccentric Phase Segment F1@0.50
  - C/E Ratio MAE
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
# Model architectures
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
    def __init__(self, in_ch=6, hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.head = nn.Conv1d(hidden, 2, 1)
    def forward(self, x):
        return self.head(self.encoder(x))


class Action16ClassCNN(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_actions=8, dropout=0.2):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.head = nn.Conv1d(hidden, num_actions * 2, 1)
    def forward(self, x):
        return self.head(self.encoder(x))
    def predict_for_action(self, logits, action_idx):
        B, C, T = logits.shape
        c_idx = action_idx * 2
        e_idx = action_idx * 2 + 1
        relevant = torch.stack([logits[:, c_idx, :], logits[:, e_idx, :]], dim=1)
        return F.softmax(relevant, dim=1)


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

ACTION_LIST = [
    'db_bench_press', 'db_biceps_curl', 'db_rdl', 'db_shoulder_press',
    'db_squat', 'db_triceps_curl', 'db_weighted_crunch', 'one_arm_db_row'
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def get_action_idx(stream_id):
    action = _extract_action_from_stream_id(stream_id)
    return ACTION_TO_IDX.get(action, 0)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_segments_global(train_streams, imu_columns):
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


def extract_segments_per_action(train_streams, imu_columns):
    action_data = {}
    for stream_id, df in train_streams:
        if "phase" not in df.columns: continue
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_data:
            action_data[action] = ([], [])
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        phase_arr = df["phase"].to_numpy()
        active_mask = np.array([str(p) in {"concentric", "eccentric"} for p in phase_arr])
        in_active = False; seg_start = 0
        for i, is_active in enumerate(active_mask):
            if is_active and not in_active:
                seg_start = i; in_active = True
            elif not is_active and in_active:
                if i - seg_start >= 10:
                    action_data[action][0].append(x[seg_start:i])
                    action_data[action][1].append(np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:i]]))
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            action_data[action][0].append(x[seg_start:])
            action_data[action][1].append(np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]]))
    return action_data


def extract_segments_16class(train_streams, imu_columns):
    segments, labels = [], []
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
                    lab = np.array([action_idx * 2 + (1 if str(p) == "concentric" else 0) for p in phase_arr[seg_start:i]])
                    labels.append(lab)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            segments.append(x[seg_start:])
            lab = np.array([action_idx * 2 + (1 if str(p) == "concentric" else 0) for p in phase_arr[seg_start:]])
            labels.append(lab)
    return segments, labels


def normalize(segments):
    if len(segments) == 0: return None, None, []
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

def train_model(model, train_segments, train_labels, max_epochs=20):
    if len(train_segments) == 0: return None, None, None
    mean, std, norm_segments = normalize(train_segments)
    if mean is None: return None, None, None
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    train_ds = PhaseDataset([norm_segments[i] for i in train_idx], [train_labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_segments[i] for i in val_idx], [train_labels[i] for i in val_idx])
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


def predict_global2class(model, df, active_segments, imu_columns, mean, std):
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


def predict_peraction(action_models, df, active_segments, imu_columns, action):
    if action not in action_models:
        action = list(action_models.keys())[0]
    model, mean, std = action_models[action]
    return predict_global2class(model, df, active_segments, imu_columns, mean, std)


def predict_16class(model, df, active_segments, imu_columns, mean, std, action_idx):
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
                probs = model.predict_for_action(logits, action_idx).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = model.predict_for_action(logits, action_idx).cpu().numpy()[0]
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


def extract_phase_segments(phase_arr, label):
    segments = []
    in_seg = False; seg_start = 0
    for i, p in enumerate(phase_arr):
        if str(p) == label:
            if not in_seg:
                seg_start = i; in_seg = True
        else:
            if in_seg:
                segments.append((seg_start, i)); in_seg = False
    if in_seg:
        segments.append((seg_start, len(phase_arr)))
    return segments


def _iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def evaluate_phase_segments(pred_segments, gt_segments, iou_threshold=0.50):
    tp = 0; matched_gt = set()
    for ps, pe in pred_segments:
        best_iou = 0; best_gt = None
        for gi, (gs, ge) in enumerate(gt_segments):
            if gi in matched_gt: continue
            iou = _iou(ps, pe, gs, ge)
            if iou > best_iou:
                best_iou = iou; best_gt = gi
        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1; matched_gt.add(best_gt)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - len(matched_gt)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1}


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
        return {"ce_ratio_mae": None, "n_valid": 0}
    pred_arr = np.array([p for p, _ in valid_pairs]); gt_arr = np.array([g for _, g in valid_pairs])
    errors = pred_arr - gt_arr
    return {"ce_ratio_mae": float(np.mean(np.abs(errors))), "n_valid": len(valid_pairs)}


def evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_reps = parse_reps(pred_labels)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(phase_probs, gt_phases)

    pred_phase_arr = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    gt_c_segs = extract_phase_segments(gt_phases, "concentric")
    gt_e_segs = extract_phase_segments(gt_phases, "eccentric")
    pred_c_segs = extract_phase_segments(pred_phase_arr, "concentric")
    pred_e_segs = extract_phase_segments(pred_phase_arr, "eccentric")
    c_seg = evaluate_phase_segments(pred_c_segs, gt_c_segs)
    e_seg = evaluate_phase_segments(pred_e_segs, gt_e_segs)

    pred_ratios = compute_rep_ce_ratios(pred_reps, pred_phase_arr)
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)

    return {
        "stream_id": stream_id,
        "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"],
        "count_error": abs(rep_m["pred_count"] - rep_m["gt_count"]),
        **{k: v for k, v in rep_m.items() if k not in ["pred_count", "gt_count"]},
        **phase_m,
        "concentric_seg_f1": c_seg["f1"],
        "eccentric_seg_f1": e_seg["f1"],
        **ce_metrics,
    }


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
    c_seg_f1 = [r["concentric_seg_f1"] for r in results]
    e_seg_f1 = [r["eccentric_seg_f1"] for r in results]
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": exact_count / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": over_count, "under_count": under_count,
        "phase_macro_f1": np.mean(phase_f1_list) if phase_f1_list else 0,
        "phase_accuracy": np.mean(phase_acc_list) if phase_acc_list else 0,
        "transition_mae_ms": np.mean(trans_mae_list) if trans_mae_list else None,
        "concentric_seg_f1": np.mean(c_seg_f1) if c_seg_f1 else 0,
        "eccentric_seg_f1": np.mean(e_seg_f1) if e_seg_f1 else 0,
        "ce_ratio_mae": np.mean(ce_mae_list) if ce_mae_list else None,
    }


# ---------------------------------------------------------------------------
# Main 9-fold
# ---------------------------------------------------------------------------

def run_9fold(all_streams, subjects, cfg, output_dir):
    print("=" * 80)
    print("9-Fold LOSO: Global 2-class vs Per-Action 2-class vs Action-Phase 16-class")
    print("=" * 80)
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"GPU: {torch.cuda.is_available()}")

    fold_results = {
        "global": [],
        "per_action": [],
        "action_phase_16": [],
    }

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/9: test={test_subject}")
        print(f"{'=' * 80}")

        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
        print(f"  train={len(train_streams)}, test={len(test_streams)}")

        # Train active detector once per fold
        print("  Training Active Detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)

        # --- Global 2-class ---
        print("  Training Global 2-class CNN...")
        segs_g, labs_g = extract_segments_global(train_streams, cfg.imu_columns)
        model_g, mean_g, std_g = train_model(Global2ClassCNN(6, 64), segs_g, labs_g, max_epochs=20)

        # --- Per-Action 2-class ---
        print("  Training Per-Action 2-class CNNs...")
        action_data = extract_segments_per_action(train_streams, cfg.imu_columns)
        action_models = {}
        for action, (segs, labs) in action_data.items():
            if len(segs) == 0: continue
            m, mn, st = train_model(Global2ClassCNN(6, 64), segs, labs, max_epochs=20)
            if m is not None:
                action_models[action] = (m, mn, st)
        print(f"    Trained {len(action_models)} action models")

        # --- 16-class Action-Phase ---
        print("  Training 16-class Action-Phase CNN...")
        segs_16, labs_16 = extract_segments_16class(train_streams, cfg.imu_columns)
        model_16, mean_16, std_16 = train_model(Action16ClassCNN(6, 64, num_actions=8), segs_16, labs_16, max_epochs=20)

        # Evaluate all variants
        variants = {
            "global": lambda sid, df, segs: predict_global2class(model_g, df, segs, cfg.imu_columns, mean_g, std_g),
            "per_action": lambda sid, df, segs: predict_peraction(action_models, df, segs, cfg.imu_columns, _extract_action_from_stream_id(sid)),
            "action_phase_16": lambda sid, df, segs: predict_16class(model_16, df, segs, cfg.imu_columns, mean_16, std_16, get_action_idx(sid)),
        }

        for variant_name, predict_fn in variants.items():
            stream_results = []
            for stream_id, df in test_streams:
                if "phase" not in df.columns: continue
                gt_phases = df["phase"].to_numpy()
                gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)

                active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
                active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)

                phase_probs = predict_fn(stream_id, df, active_segments)
                if phase_probs is None: continue
                stream_results.append(evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases))

            if stream_results:
                fold_agg = aggregate_fold_results(stream_results)
                fold_agg["fold"] = fold_idx + 1
                fold_agg["test_subject"] = test_subject
                fold_results[variant_name].append(fold_agg)
                print(f"  {variant_name:20s}: RepF1={fold_agg['rep_f1']:.4f} Exact={fold_agg['exact_count_acc']:.3f} "
                      f"Over/Under={fold_agg['over_count']}/{fold_agg['under_count']} "
                      f"PhaseF1={fold_agg['phase_macro_f1']:.4f} TransMAE={fold_agg.get('transition_mae_ms', 0):.0f}ms "
                      f"CSegF1={fold_agg.get('concentric_seg_f1', 0):.3f} ESegF1={fold_agg.get('eccentric_seg_f1', 0):.3f}")

    # Summary
    print(f"\n{'=' * 80}")
    print("9-FOLD SUMMARY")
    print(f"{'=' * 80}")

    summary = {}
    for name, results in [
        ("Global_2class", fold_results["global"]),
        ("PerAction_2class", fold_results["per_action"]),
        ("ActionPhase_16class", fold_results["action_phase_16"]),
    ]:
        if not results: continue
        print(f"\n[{name}] ({len(results)} folds)")

        metrics = ["rep_f1", "exact_count_acc", "mean_abs_count_error", "over_count", "under_count",
                   "phase_macro_f1", "phase_accuracy", "transition_mae_ms", "ce_ratio_mae",
                   "concentric_seg_f1", "eccentric_seg_f1"]

        for metric in metrics:
            values = [f[metric] for f in results if f.get(metric) is not None]
            if not values: continue
            mean = np.mean(values); std = np.std(values)
            best = np.max(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.min(values)
            worst = np.min(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.max(values)
            print(f"  {metric}: mean={mean:.4f} std={std:.4f} best={best:.4f} worst={worst:.4f}")
            summary[f"{name}_{metric}_mean"] = mean
            summary[f"{name}_{metric}_std"] = std

        for f in results:
            print(f"    Fold {f['fold']} ({f['test_subject']}): "
                  f"RepF1={f['rep_f1']:.4f} Exact={f['exact_count_acc']:.3f} "
                  f"PhaseF1={f['phase_macro_f1']:.4f} TransMAE={f.get('transition_mae_ms', 0):.0f}ms "
                  f"CSegF1={f.get('concentric_seg_f1', 0):.3f} ESegF1={f.get('eccentric_seg_f1', 0):.3f} "
                  f"Over/Under={f['over_count']}/{f['under_count']}")

    # Head-to-head comparison
    print(f"\n{'=' * 80}")
    print("WIN/LOSS COMPARISON")
    print(f"{'=' * 80}")

    comparisons = {
        "rep_f1": "higher",
        "exact_count_acc": "higher",
        "phase_macro_f1": "higher",
        "transition_mae_ms": "lower",
        "mean_abs_count_error": "lower",
        "concentric_seg_f1": "higher",
        "eccentric_seg_f1": "higher",
    }

    for metric, direction in comparisons.items():
        g = [f.get(metric) for f in fold_results["global"]]
        p = [f.get(metric) for f in fold_results["per_action"]]
        a16 = [f.get(metric) for f in fold_results["action_phase_16"]]

        if not g or not p or not a16: continue

        pa_wins = sum(1 for gv, pv in zip(g, p) if pv is not None and gv is not None and (pv > gv if direction == "higher" else pv < gv))
        a16_wins_pa = sum(1 for pv, a16v in zip(p, a16) if pv is not None and a16v is not None and (a16v > pv if direction == "higher" else a16v < pv))
        a16_wins_g = sum(1 for gv, a16v in zip(g, a16) if gv is not None and a16v is not None and (a16v > gv if direction == "higher" else a16v < gv))

        print(f"  {metric}:")
        print(f"    PerAction vs Global: PerAction wins in {pa_wins}/{len(g)} folds")
        print(f"    ActionPhase16 vs PerAction: ActionPhase16 wins in {a16_wins_pa}/{len(g)} folds")
        print(f"    ActionPhase16 vs Global: ActionPhase16 wins in {a16_wins_g}/{len(g)} folds")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_9fold_cnn_variants.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "global_per_fold": fold_results["global"],
            "per_action_per_fold": fold_results["per_action"],
            "action_phase_16_per_fold": fold_results["action_phase_16"],
        }, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}")

    return summary


def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects: {subjects}")

    cfg = PhaseCompareConfig()

    run_9fold(all_streams, subjects, cfg, Path("artifacts/cnn_variant_comparison"))


if __name__ == "__main__":
    main()
