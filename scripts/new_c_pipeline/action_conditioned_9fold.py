"""
9-Fold LOSO: Global 2-class vs Action-Conditioned CNN
Optimized: fixed seeds, excluded light-weight, shared active detector.
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

import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# Model
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


class ActionConditionedCNN(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_actions=8, dropout=0.2):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.heads = nn.ModuleList([nn.Conv1d(hidden, 2, 1) for _ in range(num_actions)])
    def forward(self, x, action_idx):
        B, _, T = x.shape
        features = self.encoder(x)
        output = torch.zeros(B, 2, T, device=x.device, dtype=x.dtype)
        for a in range(self.num_actions):
            mask = (action_idx == a)
            if mask.sum() > 0:
                output[mask] = self.heads[a](features[mask])
        return output


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def extract_segments_with_action(train_streams, imu_columns):
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


def prepare_sample(seq, lab):
    mask = np.ones(len(seq), dtype=np.float32)
    if len(seq) < 300:
        pad_len = 300 - len(seq)
        seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='edge')
        lab = np.pad(lab, (0, pad_len), constant_values=-1)
        mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.float32)])
        return seq[:300], lab[:300], mask[:300]
    else:
        start = np.random.randint(0, len(seq) - 300 + 1)
        return seq[start:start+300], lab[start:start+300], mask[:300]


class SimpleDataset(Dataset):
    def __init__(self, segments, labels):
        self.samples = []
        for seq, lab in zip(segments, labels):
            s, l, m = prepare_sample(seq, lab)
            self.samples.append((torch.from_numpy(s).float().transpose(0, 1), torch.from_numpy(l).long(), torch.from_numpy(m).float()))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


class ActionDataset(Dataset):
    def __init__(self, segments, labels, actions):
        self.samples = []
        for seq, lab, act in zip(segments, labels, actions):
            s, l, m = prepare_sample(seq, lab)
            self.samples.append((torch.from_numpy(s).float().transpose(0, 1), torch.from_numpy(l).long(), torch.from_numpy(m).float(), torch.tensor(act, dtype=torch.long)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Fast Training
# ---------------------------------------------------------------------------

def train_fast(model, loader, val_loader, device, max_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_val = float('inf'); best_state = None
    for epoch in range(max_epochs):
        model.train(); train_loss = 0; n_batches = 0
        for batch in loader:
            x = batch[0].to(device); y = batch[1].to(device); m = batch[2].to(device)
            if len(batch) > 3:
                a = batch[3].to(device)
            optimizer.zero_grad()
            if len(batch) > 3:
                logits = model(x, a)
            else:
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
            for batch in val_loader:
                x = batch[0].to(device); y = batch[1].to(device); m = batch[2].to(device)
                if len(batch) > 3:
                    a = batch[3].to(device)
                if len(batch) > 3:
                    logits = model(x, a)
                else:
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
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x); seg_x_norm = (seg_x - mean) / std
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                probs = F.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    probs = F.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
                    gs = seg_start + start
                    phase_probs[gs:gs + 300, :] += probs.T
                    phase_counts[gs:gs + 300] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    phase_probs = smooth_ma(phase_probs, 25)
    phase_probs = viterbi_decode(phase_probs, 0.3)
    return phase_probs


def predict_ac(model, df, active_segments, imu_columns, mean, std, action_idx):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5; phase_counts = np.zeros(n, dtype=np.float32)
    if model is None: return phase_probs
    device = next(model.parameters()).device
    model.eval()
    a_tensor = torch.tensor([action_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]; seg_len = len(seg_x); seg_x_norm = (seg_x - mean) / std
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                probs = F.softmax(model(x_tensor, a_tensor), dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150; starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len: starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x_norm[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    probs = F.softmax(model(x_tensor, a_tensor), dim=1).cpu().numpy()[0]
                    gs = seg_start + start
                    phase_probs[gs:gs + 300, :] += probs.T
                    phase_counts[gs:gs + 300] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    phase_probs = smooth_ma(phase_probs, 25)
    phase_probs = viterbi_decode(phase_probs, 0.3)
    return phase_probs


# ---------------------------------------------------------------------------
# Evaluation
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
        **{k: v for k, v in rep_m.items()}, **phase_m,
    }


def aggregate(results):
    """Aggregate stream-level results into fold-level metrics."""
    if not results: return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    count_errors = [abs(r["pred_count"] - r["gt_count"]) for r in results]
    return {
        "streams": n, "rep_f1": f1,
        "exact_count_acc": sum(r["exact_count"] for r in results) / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": sum(r["over"] for r in results), "under_count": sum(r["under"] for r in results),
        "phase_macro_f1": np.mean([r["phase_macro_f1"] for r in results]),
        "transition_mae_ms": np.mean([r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]) if any(r.get("transition_mae_ms") is not None for r in results) else None,
    }


def summarize_folds(fold_results):
    """Aggregate fold-level results into overall summary."""
    if not fold_results: return {}
    n = len(fold_results)
    return {
        "folds": n,
        "rep_f1": np.mean([f["rep_f1"] for f in fold_results]),
        "exact_count_acc": np.mean([f["exact_count_acc"] for f in fold_results]),
        "mean_abs_count_error": np.mean([f["mean_abs_count_error"] for f in fold_results]),
        "over_count": sum(f["over_count"] for f in fold_results),
        "under_count": sum(f["under_count"] for f in fold_results),
        "phase_macro_f1": np.mean([f["phase_macro_f1"] for f in fold_results]),
        "transition_mae_ms": np.mean([f["transition_mae_ms"] for f in fold_results if f.get("transition_mae_ms") is not None]) if any(f.get("transition_mae_ms") is not None for f in fold_results) else None,
    }


# ---------------------------------------------------------------------------
# Main 9-fold
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    raw = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    filtered_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    print(f"Excluded: {len(all_streams) - len(filtered_streams)}, Remaining: {len(filtered_streams)}")
    print(f"Subjects: {subjects}")
    print(f"GPU: {torch.cuda.is_available()}")
    cfg = PhaseCompareConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    global_fold_results = []
    ac_fold_results = []
    
    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/9: test={test_subject}")
        print(f"{'='*60}")
        train_streams = [(sid, df) for sid, df in filtered_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in filtered_streams if sid.startswith(f"{test_subject}/")]
        if len(train_streams) == 0 or len(test_streams) == 0:
            print("  Skipping (no data)")
            continue
        
        print(f"  train={len(train_streams)}, test={len(test_streams)}")
        
        # Active detector (shared)
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        
        # Extract data
        segs_g, labs_g, _ = extract_segments_with_action(train_streams, cfg.imu_columns)
        segs_ac, labs_ac, acts_ac = extract_segments_with_action(train_streams, cfg.imu_columns)
        mean_g, std_g, norm_g = normalize(segs_g)
        mean_ac, std_ac, norm_ac = normalize(segs_ac)
        
        n_total = len(norm_g); n_val = max(1, int(n_total * 0.15))
        indices = np.random.RandomState(42).permutation(n_total)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]
        
        # Global
        print("  Training Global CNN...", end=" ")
        g_train_ds = SimpleDataset([norm_g[i] for i in train_idx], [labs_g[i] for i in train_idx])
        g_val_ds = SimpleDataset([norm_g[i] for i in val_idx], [labs_g[i] for i in val_idx])
        g_train_loader = DataLoader(g_train_ds, batch_size=32, shuffle=True, drop_last=True)
        g_val_loader = DataLoader(g_val_ds, batch_size=32, shuffle=False, drop_last=False)
        global_model = Global2ClassCNN(6, 64).to(device)
        global_model = train_fast(global_model, g_train_loader, g_val_loader, device, max_epochs=20)
        print("Done")
        
        # Action-Conditioned
        print("  Training Action-Conditioned CNN...", end=" ")
        ac_train_ds = ActionDataset([norm_ac[i] for i in train_idx], [labs_ac[i] for i in train_idx], [acts_ac[i] for i in train_idx])
        ac_val_ds = ActionDataset([norm_ac[i] for i in val_idx], [labs_ac[i] for i in val_idx], [acts_ac[i] for i in val_idx])
        ac_train_loader = DataLoader(ac_train_ds, batch_size=32, shuffle=True, drop_last=True)
        ac_val_loader = DataLoader(ac_val_ds, batch_size=32, shuffle=False, drop_last=False)
        ac_model = ActionConditionedCNN(6, 64, 8).to(device)
        ac_model = train_fast(ac_model, ac_train_loader, ac_val_loader, device, max_epochs=20)
        print("Done")
        
        # Evaluate both
        g_results = []
        ac_results = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            g_probs = predict_global(global_model, df, active_segments, cfg.imu_columns, mean_g, std_g)
            g_results.append(evaluate_stream(stream_id, df, g_probs, gt_reps, gt_phases))
            
            a_idx = get_action_idx(stream_id)
            ac_probs = predict_ac(ac_model, df, active_segments, cfg.imu_columns, mean_ac, std_ac, a_idx)
            ac_results.append(evaluate_stream(stream_id, df, ac_probs, gt_reps, gt_phases))
        
        g_agg = aggregate(g_results)
        g_agg["fold"] = fold_idx + 1; g_agg["test_subject"] = test_subject
        global_fold_results.append(g_agg)
        
        ac_agg = aggregate(ac_results)
        ac_agg["fold"] = fold_idx + 1; ac_agg["test_subject"] = test_subject
        ac_fold_results.append(ac_agg)
        
        print(f"  Global:    RepF1={g_agg['rep_f1']:.4f} Exact={g_agg['exact_count_acc']:.3f} Over/Under={g_agg['over_count']}/{g_agg['under_count']}")
        print(f"  AC:        RepF1={ac_agg['rep_f1']:.4f} Exact={ac_agg['exact_count_acc']:.3f} Over/Under={ac_agg['over_count']}/{ac_agg['under_count']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("9-FOLD SUMMARY")
    print(f"{'='*60}")
    
    g_summary = summarize_folds(global_fold_results)
    ac_summary = summarize_folds(ac_fold_results)
    
    print(f"\n{'Metric':<25s} {'Global':>10s} {'ActionCond':>12s} {'Delta':>10s}")
    print("-" * 60)
    for metric in ["rep_f1", "exact_count_acc", "mean_abs_count_error", "phase_macro_f1", "transition_mae_ms"]:
        gv = g_summary.get(metric, 0) if g_summary.get(metric) is not None else 0
        av = ac_summary.get(metric, 0) if ac_summary.get(metric) is not None else 0
        delta = av - gv
        print(f"{metric:<25s} {gv:>10.4f} {av:>12.4f} {delta:>+10.4f}")
    
    print(f"\n{'='*60}")
    g_wins = sum(1 for g, a in zip(global_fold_results, ac_fold_results) if g['rep_f1'] > a['rep_f1'])
    ac_wins = sum(1 for g, a in zip(global_fold_results, ac_fold_results) if a['rep_f1'] > g['rep_f1'])
    print(f"Rep F1: Global wins in {g_wins}/{len(global_fold_results)} folds, AC wins in {ac_wins}/{len(ac_fold_results)} folds")
    
    g_wins_e = sum(1 for g, a in zip(global_fold_results, ac_fold_results) if g['exact_count_acc'] > a['exact_count_acc'])
    ac_wins_e = sum(1 for g, a in zip(global_fold_results, ac_fold_results) if a['exact_count_acc'] > g['exact_count_acc'])
    print(f"Exact:  Global wins in {g_wins_e}/{len(global_fold_results)} folds, AC wins in {ac_wins_e}/{len(ac_fold_results)} folds")
    
    # Save
    import json
    out_dir = Path("artifacts/cnn_variant_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "action_conditioned_9fold.json", "w") as f:
        json.dump({"global": g_summary, "global_per_fold": global_fold_results,
                   "action_conditioned": ac_summary, "ac_per_fold": ac_fold_results}, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir / 'action_conditioned_9fold.json'}")


if __name__ == "__main__":
    main()
