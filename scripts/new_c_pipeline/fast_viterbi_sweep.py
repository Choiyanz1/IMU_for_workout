"""
Fast test: Train once, then test multiple decoders (MA15/25/40 + Viterbi).
No model saving needed - trains in ~1 minute total.
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

from preprocessing.micro_macro_segments import labels_to_runs, pair_concentric_eccentric_reps, SegmentRun
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, evaluate_phase, evaluate_reps,
    extract_active_segments, predict_active, train_active_detector,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


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
                    seg_x = x[seg_start:i]
                    seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:i]])
                    segments.append(seg_x); labels.append(seg_lab)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            seg_x = x[seg_start:]
            seg_lab = np.array([1 if str(p) == "concentric" else 0 for p in phase_arr[seg_start:]])
            segments.append(seg_x); labels.append(seg_lab)
    return segments, labels


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


def train_fast(model, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    best_val = float('inf'); best_state = None
    
    for epoch in range(20):
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
        
        avg_train = train_loss / n_batches if n_batches > 0 else float('inf')
        avg_val = val_loss / val_batches if val_batches > 0 else float('inf')
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val:
            best_val = avg_val; best_state = model.state_dict()
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_fast(model, df, active_segments, imu_columns, mean, std):
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
    segments, labels = extract_segments(train_streams, cfg.imu_columns)
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments); n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    # Train
    print("\nTraining Phase-Only CNN (fast)...")
    model = CausalCNN_PhaseOnly(6, 64)
    train_ds = SimpleDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = SimpleDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)
    model = train_fast(model, train_loader, val_loader)
    
    # Test multiple decoders
    print(f"\n{'='*60}")
    print("DECODER SWEEP (same model)")
    print(f"{'='*60}")
    
    decoders = [
        ("Viterbi_raw", lambda p: viterbi_decode(p, 0.3)),
        ("MA15+Viterbi", lambda p: viterbi_decode(smooth_ma(p, 15), 0.3)),
        ("MA25+Viterbi", lambda p: viterbi_decode(smooth_ma(p, 25), 0.3)),
        ("MA40+Viterbi", lambda p: viterbi_decode(smooth_ma(p, 40), 0.3)),
        ("MA60+Viterbi", lambda p: viterbi_decode(smooth_ma(p, 60), 0.3)),
    ]
    
    all_results = {}
    
    # Train active detector ONCE
    print("Training Active Detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    
    for decoder_name, decoder_fn in decoders:
        results = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std)
            decoded = decoder_fn(phase_probs)
            pred_reps = parse_reps(np.argmax(decoded, axis=1))
            rep_m = evaluate_reps(pred_reps, gt_reps)
            phase_m = evaluate_phase(decoded, gt_phases)
            results.append({**rep_m, **phase_m})
        
        n = len(results)
        total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        exact = sum(r["exact_count"] for r in results)
        over = sum(r["over"] for r in results); under = sum(r["under"] for r in results)
        phase_f1 = np.mean([r["phase_macro_f1"] for r in results])
        trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
        
        all_results[decoder_name] = {
            "rep_f1": f1, "exact_count_acc": exact / n if n > 0 else 0,
            "over_count": over, "under_count": under,
            "phase_macro_f1": phase_f1,
            "transition_mae_ms": np.mean(trans) if trans else None,
        }
        print(f"{decoder_name:20s}: RepF1={f1:.4f} Exact={exact/n:.3f} Over/Under={over}/{under} PhaseF1={phase_f1:.4f}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\n[By Rep F1]")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["rep_f1"], reverse=True):
        print(f"  {name:20s}: RepF1={m['rep_f1']:.4f} Exact={m['exact_count_acc']:.3f} Over/Under={m['over_count']}/{m['under_count']}")
    
    print("\n[By Balance]")
    for name, m in sorted(all_results.items(), key=lambda x: x[1]["over_count"] + x[1]["under_count"]):
        print(f"  {name:20s}: Over+Under={m['over_count']+m['under_count']} RepF1={m['rep_f1']:.4f}")


if __name__ == "__main__":
    main()
