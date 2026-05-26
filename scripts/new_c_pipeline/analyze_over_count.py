"""
Analyze over-count sources for Global CNN (excluded light-weight).

Reports:
  - Per-subject over-count distribution
  - Per-action over-count distribution
  - Subject x Action heatmap
  - Example streams with worst over-count (for inspection)
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


EXCLUDED_SESSIONS = {
    "yanz": ["1000"],
    "thomas": ["thomas", "thomas_2"],
    "kevin": ["kevin"],
}


def should_exclude(stream_id: str) -> bool:
    parts = stream_id.split("/")
    if len(parts) < 2:
        return False
    subject = parts[0]
    session = parts[1]
    if subject in EXCLUDED_SESSIONS and session in EXCLUDED_SESSIONS[subject]:
        return True
    return False


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


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


def predict_cnn_phase(model, df, active_segments, imu_columns, mean, std):
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


def evaluate_stream_detailed(stream_id, df, phase_probs, gt_reps, gt_phases):
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_reps = parse_reps(pred_labels)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    return {
        "stream_id": stream_id,
        "action": _extract_action_from_stream_id(stream_id),
        "subject": stream_id.split("/")[0],
        "pred_count": rep_m["pred_count"],
        "gt_count": rep_m["gt_count"],
        "count_error": rep_m["pred_count"] - rep_m["gt_count"],
        "is_over": 1 if rep_m["pred_count"] > rep_m["gt_count"] else 0,
        "is_under": 1 if rep_m["pred_count"] < rep_m["gt_count"] else 0,
        "is_exact": 1 if rep_m["pred_count"] == rep_m["gt_count"] else 0,
        "over_amount": max(0, rep_m["pred_count"] - rep_m["gt_count"]),
        "under_amount": max(0, rep_m["gt_count"] - rep_m["pred_count"]),
    }


def main():
    raw = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])

    # Exclude light-weight sessions
    filtered_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    print(f"Excluded: {len(all_streams) - len(filtered_streams)}, Remaining: {len(filtered_streams)}")

    cfg = PhaseCompareConfig()
    all_stream_results = []

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\nFold {fold_idx + 1}/9: test={test_subject}")
        train_streams = [(sid, df) for sid, df in filtered_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in filtered_streams if sid.startswith(f"{test_subject}/")]
        if len(train_streams) == 0 or len(test_streams) == 0:
            print(f"  Skipping fold")
            continue

        active_models, active_scalers = train_active_detector(train_streams, cfg)
        segs, labs = extract_segments(train_streams, cfg.imu_columns)
        model, mean, std = train_model(Global2ClassCNN(6, 64), segs, labs, max_epochs=20)

        for stream_id, df in test_streams:
            if "phase" not in df.columns: continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_cnn_phase(model, df, active_segments, cfg.imu_columns, mean, std)
            all_stream_results.append(evaluate_stream_detailed(stream_id, df, phase_probs, gt_reps, gt_phases))

    # Analysis
    print(f"\n{'='*80}")
    print("OVER-COUNT SOURCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total streams evaluated: {len(all_stream_results)}")

    over_streams = [r for r in all_stream_results if r["is_over"] == 1]
    under_streams = [r for r in all_stream_results if r["is_under"] == 1]
    exact_streams = [r for r in all_stream_results if r["is_exact"] == 1]

    print(f"Over-counted: {len(over_streams)} ({len(over_streams)/len(all_stream_results)*100:.1f}%)")
    print(f"Under-counted: {len(under_streams)} ({len(under_streams)/len(all_stream_results)*100:.1f}%)")
    print(f"Exact: {len(exact_streams)} ({len(exact_streams)/len(all_stream_results)*100:.1f}%)")

    # Per-subject over-count
    print(f"\n{'='*60}")
    print("PER-SUBJECT OVER-COUNT")
    print(f"{'='*60}")
    print(f"{'Subject':<15s} {'Total':>6s} {'Over':>6s} {'%Over':>8s} {'AvgOver':>10s} {'Exact':>6s} {'%Exact':>8s}")
    print("-" * 75)
    subject_stats = {}
    for subj in sorted(set(r["subject"] for r in all_stream_results)):
        subj_streams = [r for r in all_stream_results if r["subject"] == subj]
        over = [r for r in subj_streams if r["is_over"] == 1]
        exact = [r for r in subj_streams if r["is_exact"] == 1]
        avg_over = np.mean([r["over_amount"] for r in over]) if over else 0
        subject_stats[subj] = {
            "total": len(subj_streams),
            "over": len(over),
            "pct_over": len(over) / len(subj_streams) * 100,
            "avg_over": avg_over,
            "exact": len(exact),
            "pct_exact": len(exact) / len(subj_streams) * 100,
        }
        print(f"{subj:<15s} {len(subj_streams):>6d} {len(over):>6d} {len(over)/len(subj_streams)*100:>7.1f}% {avg_over:>9.1f} {len(exact):>6d} {len(exact)/len(subj_streams)*100:>7.1f}%")

    # Per-action over-count
    print(f"\n{'='*60}")
    print("PER-ACTION OVER-COUNT")
    print(f"{'='*60}")
    print(f"{'Action':<25s} {'Total':>6s} {'Over':>6s} {'%Over':>8s} {'AvgOver':>10s} {'Exact':>6s} {'%Exact':>8s}")
    print("-" * 85)
    action_stats = {}
    for action in sorted(set(r["action"] for r in all_stream_results)):
        act_streams = [r for r in all_stream_results if r["action"] == action]
        over = [r for r in act_streams if r["is_over"] == 1]
        exact = [r for r in act_streams if r["is_exact"] == 1]
        avg_over = np.mean([r["over_amount"] for r in over]) if over else 0
        action_stats[action] = {
            "total": len(act_streams),
            "over": len(over),
            "pct_over": len(over) / len(act_streams) * 100,
            "avg_over": avg_over,
            "exact": len(exact),
            "pct_exact": len(exact) / len(act_streams) * 100,
        }
        print(f"{action:<25s} {len(act_streams):>6d} {len(over):>6d} {len(over)/len(act_streams)*100:>7.1f}% {avg_over:>9.1f} {len(exact):>6d} {len(exact)/len(act_streams)*100:>7.1f}%")

    # Subject x Action heatmap (over-count)
    print(f"\n{'='*60}")
    print("SUBJECT x ACTION OVER-COUNT MATRIX (% over-counted)")
    print(f"{'='*60}")
    subjects_list = sorted(set(r["subject"] for r in all_stream_results))
    actions_list = sorted(set(r["action"] for r in all_stream_results))
    print(f"{'Subject':<15s}", end="")
    for action in actions_list:
        print(f" {action[:12]:>12s}", end="")
    print()
    print("-" * (15 + 13 * len(actions_list)))
    for subj in subjects_list:
        print(f"{subj:<15s}", end="")
        for action in actions_list:
            cell = [r for r in all_stream_results if r["subject"] == subj and r["action"] == action]
            if not cell:
                print(f" {'N/A':>12s}", end="")
            else:
                over = sum(1 for r in cell if r["is_over"] == 1)
                print(f" {over/len(cell)*100:>11.1f}%", end="")
        print()

    # Worst over-count examples
    print(f"\n{'='*60}")
    print("TOP 20 WORST OVER-COUNT STREAMS")
    print(f"{'='*60}")
    sorted_over = sorted(over_streams, key=lambda x: x["over_amount"], reverse=True)[:20]
    print(f"{'Stream ID':<50s} {'Pred':>5s} {'GT':>5s} {'Over':>5s}")
    print("-" * 70)
    for r in sorted_over:
        print(f"{r['stream_id']:<50s} {r['pred_count']:>5d} {r['gt_count']:>5d} {r['over_amount']:>5d}")

    # Save
    out_dir = Path("artifacts/cnn_variant_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "over_count_analysis.json", "w") as f:
        json.dump({
            "stream_results": all_stream_results,
            "subject_stats": subject_stats,
            "action_stats": action_stats,
        }, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_dir / 'over_count_analysis.json'}")


if __name__ == "__main__":
    main()
