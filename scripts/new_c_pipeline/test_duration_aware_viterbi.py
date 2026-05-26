"""
Quick test: Duration-Aware Viterbi on kevin fold.
Compares: Standard Viterbi vs Duration-Aware Viterbi (min_duration=30 samples = 0.3s).
Metrics: Rep F1, Exact Count, Phase F1, Transition MAE, Count MAE, C/E Segment F1, C/E Ratio MAE.
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
# Model (same as before)
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
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_cnn_phase(train_streams, imu_columns):
    segments, labels = extract_segments(train_streams, imu_columns)
    if len(segments) == 0: return None, None, None
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
# Inference (MA25 + Viterbi variants)
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


def viterbi_standard(phase_probs, penalty=0.3):
    """Standard Viterbi decoding."""
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


def viterbi_duration_aware(phase_probs, penalty=0.3, min_c_duration=30, min_e_duration=30):
    """
    Two-pass duration-aware Viterbi:
    1. Standard Viterbi
    2. For segments shorter than min_duration, greedily merge with
       the neighbor that gives higher total log-probability.
    This is principled: it uses the model's own probabilities to decide
    whether a short segment is spurious (better to merge) or genuine
    (better to keep).
    """
    # Pass 1: Standard Viterbi
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

    # Pass 2: Duration repair
    # Find runs
    if n == 0:
        return np.zeros((n, 2))

    runs = []
    current = pred[0]
    start = 0
    for i in range(1, n):
        if pred[i] != current:
            runs.append((current, start, i))
            current = pred[i]
            start = i
    runs.append((current, start, n))

    # Determine min duration per label
    min_dur = {0: min_e_duration, 1: min_c_duration}
    repaired = pred.copy()

    for idx, (label, s, e) in enumerate(runs):
        dur = e - s
        if dur < min_dur[label]:
            # Try merging with left or right, choosing by higher total log-prob
            costs = []

            # Option 1: keep as-is (negative log-prob = cost)
            cost_keep = -np.sum(log_probs[s:e, label])
            costs.append((cost_keep, 'keep'))

            # Option 2: merge with left neighbor
            if idx > 0:
                left_label = runs[idx-1][0]
                cost_left = -np.sum(log_probs[s:e, left_label])
                costs.append((cost_left, 'left'))

            # Option 3: merge with right neighbor
            if idx < len(runs) - 1:
                right_label = runs[idx+1][0]
                cost_right = -np.sum(log_probs[s:e, right_label])
                costs.append((cost_right, 'right'))

            # Choose minimum cost (highest probability)
            best_cost, best_action = min(costs, key=lambda x: x[0])

            if best_action == 'left':
                left_label = runs[idx-1][0]
                repaired[s:e] = left_label
            elif best_action == 'right':
                right_label = runs[idx+1][0]
                repaired[s:e] = right_label
            # else: keep as-is

    result = np.zeros((n, 2))
    result[repaired == 0, 0] = 1.0
    result[repaired == 1, 1] = 1.0
    return result


def predict_cnn_phase(model, df, active_segments, imu_columns, mean, std, decoder_fn):
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
    phase_probs = decoder_fn(phase_probs)
    return phase_probs


# ---------------------------------------------------------------------------
# Parsing
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


# ---------------------------------------------------------------------------
# C/E Phase Segment F1@0.50
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# C/E Ratio
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
        return {"ce_ratio_mae": None, "n_valid": 0}
    pred_arr = np.array([p for p, _ in valid_pairs]); gt_arr = np.array([g for _, g in valid_pairs])
    errors = pred_arr - gt_arr
    return {"ce_ratio_mae": float(np.mean(np.abs(errors))), "n_valid": len(valid_pairs)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_reps = parse_reps(pred_labels)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(phase_probs, gt_phases)

    # C/E Phase Segment F1
    pred_phase_arr = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    gt_c_segs = extract_phase_segments(gt_phases, "concentric")
    gt_e_segs = extract_phase_segments(gt_phases, "eccentric")
    pred_c_segs = extract_phase_segments(pred_phase_arr, "concentric")
    pred_e_segs = extract_phase_segments(pred_phase_arr, "eccentric")
    c_seg = evaluate_phase_segments(pred_c_segs, gt_c_segs)
    e_seg = evaluate_phase_segments(pred_e_segs, gt_e_segs)

    # C/E Ratio
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
    count_errors = [r["count_error"] for r in results]
    trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    ce = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": sum(r["exact_count"] for r in results) / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": sum(r["over"] for r in results), "under_count": sum(r["under"] for r in results),
        "phase_macro_f1": np.mean([r["phase_macro_f1"] for r in results]),
        "phase_accuracy": np.mean([r["phase_accuracy"] for r in results]),
        "transition_mae_ms": np.mean(trans) if trans else None,
        "concentric_seg_f1": np.mean([r["concentric_seg_f1"] for r in results]),
        "eccentric_seg_f1": np.mean([r["eccentric_seg_f1"] for r in results]),
        "ce_ratio_mae": np.mean(ce) if ce else None,
    }


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
    print("\nTraining CNN...")
    cnn_model, cnn_mean, cnn_std = train_cnn_phase(train_streams, cfg.imu_columns)
    print("Training Active Detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)

    # Test multiple decoders
    decoders = [
        ("Standard Viterbi", lambda p: viterbi_standard(p, 0.3)),
        ("DurAware_Viterbi_min30", lambda p: viterbi_duration_aware(p, 0.3, 30, 30)),
        ("DurAware_Viterbi_min50", lambda p: viterbi_duration_aware(p, 0.3, 50, 50)),
        ("DurAware_Viterbi_min20", lambda p: viterbi_duration_aware(p, 0.3, 20, 20)),
    ]

    print(f"\n{'='*60}")
    print("DECODER COMPARISON (kevin fold)")
    print(f"{'='*60}")

    all_results = {}
    for name, decoder_fn in decoders:
        results = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_cnn_phase(cnn_model, df, active_segments, cfg.imu_columns, cnn_mean, cnn_std, decoder_fn)
            results.append(evaluate_stream(stream_id, df, phase_probs, gt_reps, gt_phases))
        agg = aggregate(results)
        all_results[name] = agg
        print(f"{name:30s}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
              f"Over/Under={agg['over_count']}/{agg['under_count']} "
              f"PhaseF1={agg['phase_macro_f1']:.4f} TransMAE={agg.get('transition_mae_ms', 0):.0f}ms "
              f"CSegF1={agg['concentric_seg_f1']:.3f} ESegF1={agg['eccentric_seg_f1']:.3f} "
              f"CERatioMAE={agg.get('ce_ratio_mae', 0):.3f}")

    print(f"\n{'='*60}")
    print("SUMMARY (sorted by Rep F1)")
    print(f"{'='*60}")
    for name, agg in sorted(all_results.items(), key=lambda x: x[1]["rep_f1"], reverse=True):
        print(f"  {name:30s}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
              f"CountMAE={agg['mean_abs_count_error']:.2f} Over+Under={agg['over_count']+agg['under_count']}")


if __name__ == "__main__":
    main()
