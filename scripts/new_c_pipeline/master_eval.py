"""
MASTER EVALUATION SCRIPT: Strict Baseline + Single-Variable Ablation

RULES:
  1. Train ONCE per fold, save checkpoint to artifacts/baselines/
  2. All decoder experiments load the SAME checkpoint, only change post-processing
  3. A new config is "accepted" only if it beats the current baseline on BOTH:
     - Rep IoU-F1@0.50
     - Exact Count Accuracy
  4. If accepted, it becomes the new baseline checkpoint for subsequent experiments
  5. If rejected, we keep the old baseline and record the failure

Current Golden Baseline (2026-05-18):
  - Model: Global 2-class Causal CNN (hidden=64, 5-layer, GroupNorm, residual)
  - Data: EXCLUDE light-weight sessions (yanz/1000, thomas/thomas, thomas/thomas_2, kevin/kevin)
  - Decoder: MA25 + Viterbi penalty=0.3
  - 9-Fold LOSO Result: Rep F1=0.848, Exact Count=49.5%, Count MAE=1.72

Usage:
  # First time: establish baseline
  python master_eval.py --mode establish_baseline

  # Test new decoder (loads baseline checkpoint, only changes decoder):
  python master_eval.py --mode test_decoder --decoder_config path/to/new_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Strict Seeds for Determinism
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These slow down training but improve reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# Model (Fixed Architecture)
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


# ---------------------------------------------------------------------------
# Fixed Decoder Config
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    name: str
    ma_window: int
    viterbi_penalty: float


GOLDEN_BASELINE_DECODER = DecoderConfig("baseline_ma25_p03", 25, 0.3)


# ---------------------------------------------------------------------------
# Data Utils
# ---------------------------------------------------------------------------

def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


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


def train_model(model, train_segments, train_labels, max_epochs=20, device=None):
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
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_val = float('inf'); best_state = None; best_epoch = 0
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
            best_val = avg_val; best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    [Train] Best val loss={best_val:.4f} at epoch {best_epoch+1}")
    return model, mean, std


# ---------------------------------------------------------------------------
# Fixed Inference Pipeline
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


def predict_cnn_phase(model, df, active_segments, imu_columns, mean, std, decoder: DecoderConfig):
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
    phase_probs = smooth_ma(phase_probs, decoder.ma_window)
    phase_probs = viterbi_decode(phase_probs, decoder.viterbi_penalty)
    return phase_probs


# ---------------------------------------------------------------------------
# Evaluation (Fixed Metrics)
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
    segments = []; in_seg = False; seg_start = 0
    for i, p in enumerate(phase_arr):
        if str(p) == label:
            if not in_seg: seg_start = i; in_seg = True
        else:
            if in_seg: segments.append((seg_start, i)); in_seg = False
    if in_seg: segments.append((seg_start, len(phase_arr)))
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
            if iou > best_iou: best_iou = iou; best_gt = gi
        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1; matched_gt.add(best_gt)
    fp = len(pred_segments) - tp; fn = len(gt_segments) - len(matched_gt)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1}


def compute_rep_ce_ratios(reps, phase_labels):
    ratios = []
    for rep in reps:
        seg = phase_labels[rep.start_idx:rep.end_idx]
        if len(seg) == 0: ratios.append(float('nan')); continue
        c_count = np.sum(seg == CONCENTRIC_LABEL); e_count = np.sum(seg == ECCENTRIC_LABEL)
        ratios.append(float('inf') if e_count == 0 else c_count / e_count)
    return ratios


def compute_ce_ratio_metrics(pred_ratios, gt_ratios):
    valid_pairs = [(p, g) for p, g in zip(pred_ratios, gt_ratios)
                   if np.isfinite(p) and np.isfinite(g) and p != float('inf') and g != float('inf')]
    if not valid_pairs: return {"ce_ratio_mae": None, "n_valid": 0}
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


def aggregate(results):
    if not results: return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    exact_count = sum(r["exact_count"] for r in results)
    over_count = sum(r["over"] for r in results); under_count = sum(r["under"] for r in results)
    count_errors = [r["count_error"] for r in results]
    trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    ce = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": exact_count / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": over_count, "under_count": under_count,
        "phase_macro_f1": np.mean([r["phase_macro_f1"] for r in results]),
        "phase_accuracy": np.mean([r["phase_accuracy"] for r in results]),
        "transition_mae_ms": np.mean(trans) if trans else None,
        "concentric_seg_f1": np.mean([r["concentric_seg_f1"] for r in results]),
        "eccentric_seg_f1": np.mean([r["eccentric_seg_f1"] for r in results]),
        "ce_ratio_mae": np.mean(ce) if ce else None,
    }


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------

def save_baseline(fold_idx, test_subject, model, mean, std, output_dir):
    """Save a trained model checkpoint as the baseline for this fold."""
    fold_dir = output_dir / f"fold{fold_idx:02d}_{test_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "fold": fold_idx,
        "test_subject": test_subject,
    }
    torch.save(ckpt, fold_dir / "baseline.pt")
    np.savez(fold_dir / "norm.npz", mean=mean, std=std)
    print(f"    [Baseline] Saved to {fold_dir / 'baseline.pt'}")
    return fold_dir


def load_baseline(fold_dir, device):
    """Load a baseline checkpoint."""
    ckpt = torch.load(fold_dir / "baseline.pt", map_location=device)
    model = Global2ClassCNN(6, 64).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    mean = ckpt["mean"]
    std = ckpt["std"]
    return model, mean, std


def run_fold(fold_idx, test_subject, all_streams, cfg, output_dir, mode="establish", decoder=None):
    """Run a single fold. If mode='establish', train and save baseline.
    If mode='test', load baseline and test new decoder."""
    train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
    test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
    print(f"  Fold {fold_idx+1}: test={test_subject}, train={len(train_streams)}, test={len(test_streams)}")

    fold_dir = output_dir / f"fold{fold_idx:02d}_{test_subject}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == "establish":
        # Train from scratch and save
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        segs, labs = extract_segments(train_streams, cfg.imu_columns)
        model, mean, std = train_model(Global2ClassCNN(6, 64), segs, labs, max_epochs=20, device=device)
        save_baseline(fold_idx, test_subject, model, mean, std, output_dir)
    else:
        # Load existing baseline
        if not (fold_dir / "baseline.pt").exists():
            print(f"    [ERROR] Baseline not found for fold {fold_idx}. Run establish first.")
            return None
        model, mean, std = load_baseline(fold_dir, device)
        active_models, active_scalers = train_active_detector(train_streams, cfg)

    # Evaluate
    results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns: continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        phase_probs = predict_cnn_phase(model, df, active_segments, cfg.imu_columns, mean, std,
                                         decoder if decoder else GOLDEN_BASELINE_DECODER)
        results.append(evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases))

    return aggregate(results)


def compare_results(new_results, baseline_results):
    """Compare new results against baseline. Returns True if new is strictly better."""
    if not new_results or not baseline_results:
        return False
    new_f1 = new_results.get("rep_f1", 0)
    base_f1 = baseline_results.get("rep_f1", 0)
    new_exact = new_results.get("exact_count_acc", 0)
    base_exact = baseline_results.get("exact_count_acc", 0)
    return new_f1 > base_f1 and new_exact > base_exact


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Script")
    parser.add_argument("--mode", choices=["establish_baseline", "test_decoder"], default="establish_baseline")
    parser.add_argument("--decoder_name", type=str, default="", help="Name for decoder experiment")
    parser.add_argument("--ma_window", type=int, default=25)
    parser.add_argument("--viterbi_penalty", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/master_baseline"))
    args = parser.parse_args()

    set_seed(42)

    raw = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    filtered_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    print(f"Total: {len(all_streams)}, After exclusion: {len(filtered_streams)}")
    print(f"Subjects: {subjects}")
    print(f"GPU: {torch.cuda.is_available()}")

    cfg = PhaseCompareConfig()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load or establish baseline results
    baseline_path = args.output / "baseline_results.json"
    baseline_results = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        print(f"\n[LOADED] Existing baseline: RepF1={baseline_results['summary']['rep_f1']:.4f} "
              f"Exact={baseline_results['summary']['exact_count_acc']:.3f}")

    if args.mode == "establish_baseline":
        print(f"\n{'='*60}")
        print("ESTABLISHING GOLDEN BASELINE")
        print(f"{'='*60}")
        print(f"Decoder: {GOLDEN_BASELINE_DECODER.name}")

        fold_results = []
        for fold_idx, test_subject in enumerate(subjects):
            print(f"\n  Fold {fold_idx+1}/9: {test_subject}")
            fold_agg = run_fold(fold_idx, test_subject, filtered_streams, cfg, args.output,
                                mode="establish", decoder=GOLDEN_BASELINE_DECODER)
            if fold_agg:
                fold_agg["fold"] = fold_idx + 1
                fold_agg["test_subject"] = test_subject
                fold_results.append(fold_agg)
                print(f"    RepF1={fold_agg['rep_f1']:.4f} Exact={fold_agg['exact_count_acc']:.3f} "
                      f"Over/Under={fold_agg['over_count']}/{fold_agg['under_count']}")

        summary = aggregate(fold_results)
        with open(baseline_path, "w") as f:
            json.dump({"summary": summary, "per_fold": fold_results}, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print("BASELINE ESTABLISHED")
        print(f"{'='*60}")
        print(f"Rep IoU-F1@0.50: {summary['rep_f1']:.4f}")
        print(f"Exact Count:     {summary['exact_count_acc']:.3f}")
        print(f"Count MAE:       {summary['mean_abs_count_error']:.2f}")
        print(f"Saved to: {baseline_path}")

    elif args.mode == "test_decoder":
        if not baseline_results:
            print("[ERROR] No baseline found. Run --mode establish_baseline first.")
            return

        new_decoder = DecoderConfig(
            name=args.decoder_name or f"ma{args.ma_window}_p{args.viterbi_penalty}",
            ma_window=args.ma_window,
            viterbi_penalty=args.viterbi_penalty,
        )

        print(f"\n{'='*60}")
        print("TESTING NEW DECODER (loading baseline checkpoints)")
        print(f"{'='*60}")
        print(f"New Decoder: {new_decoder.name}")
        print(f"Baseline:    {GOLDEN_BASELINE_DECODER.name}")

        fold_results = []
        for fold_idx, test_subject in enumerate(subjects):
            print(f"\n  Fold {fold_idx+1}/9: {test_subject}")
            fold_agg = run_fold(fold_idx, test_subject, filtered_streams, cfg, args.output,
                                mode="test", decoder=new_decoder)
            if fold_agg:
                fold_agg["fold"] = fold_idx + 1
                fold_agg["test_subject"] = test_subject
                fold_results.append(fold_agg)
                base_fold = next((f for f in baseline_results["per_fold"] if f["test_subject"] == test_subject), None)
                if base_fold:
                    print(f"    New:  RepF1={fold_agg['rep_f1']:.4f} Exact={fold_agg['exact_count_acc']:.3f} "
                          f"Over/Under={fold_agg['over_count']}/{fold_agg['under_count']}")
                    print(f"    Base: RepF1={base_fold['rep_f1']:.4f} Exact={base_fold['exact_count_acc']:.3f} "
                          f"Over/Under={base_fold['over_count']}/{base_fold['under_count']}")

        summary = aggregate(fold_results)
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        base_summary = baseline_results["summary"]
        print(f"Metric          {'Baseline':>12s} {'New':>12s} {'Delta':>10s}")
        print("-" * 50)
        for metric in ["rep_f1", "exact_count_acc", "mean_abs_count_error", "phase_macro_f1", "transition_mae_ms"]:
            b = base_summary.get(metric, 0) if base_summary.get(metric) is not None else 0
            n = summary.get(metric, 0) if summary.get(metric) is not None else 0
            delta = n - b
            print(f"{metric:<15s} {b:>12.4f} {n:>12.4f} {delta:>+10.4f}")

        is_better = compare_results(summary, base_summary)
        print(f"\n{'='*60}")
        if is_better:
            print("VERDICT: ACCEPTED - New decoder beats baseline on RepF1 AND ExactCount")
            print("ACTION: Update baseline checkpoint to new decoder config")
            # Save as new baseline
            for fold_idx, test_subject in enumerate(subjects):
                fold_dir = args.output / f"fold{fold_idx:02d}_{test_subject}"
                if (fold_dir / "baseline.pt").exists():
                    # Rename old baseline
                    old = fold_dir / "baseline.pt"
                    backup = fold_dir / f"baseline_{GOLDEN_BASELINE_DECODER.name}.pt"
                    old.rename(backup)
                    print(f"  Backed up old baseline to {backup.name}")
            # Note: we don't actually re-train, just record the new decoder as accepted
            with open(args.output / f"accepted_decoder_{new_decoder.name}.json", "w") as f:
                json.dump({
                    "decoder": {"name": new_decoder.name, "ma_window": new_decoder.ma_window, "viterbi_penalty": new_decoder.viterbi_penalty},
                    "summary": summary,
                    "per_fold": fold_results,
                }, f, indent=2, default=str)
            print(f"  Saved accepted decoder to {args.output / f'accepted_decoder_{new_decoder.name}.json'}")
        else:
            print("VERDICT: REJECTED - New decoder does not beat baseline")
            print("ACTION: Keep existing baseline, discard this experiment")


if __name__ == "__main__":
    main()
