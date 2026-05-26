"""
1-Fold Quick Compare: RF vs Causal CNN Seq2Seq vs TCN-lite (Kevin only)

Optimized for speed:
  - Batch size: 32
  - Max epochs: 20
  - Early stopping patience: 8
  - GPU only
  - Single test subject (kevin)

All three models use GLOBAL (non-per-action) training for fair comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

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
    train_rf_phase, predict_rf_phase, smooth_phase_probs,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class FastConfig:
    imu_columns = ("ax", "ay", "az", "gx", "gy", "gz")
    active_window_size = 100; active_stride = 10
    active_n_estimators = 100; active_max_depth = 15; active_max_samples = 0.7
    phase_window_size = 100; phase_stride = 10
    phase_n_estimators = 100; phase_max_depth = 15; phase_max_samples = 0.7
    smoothing_window = 15
    min_phase_samples = 3; max_phase_gap_samples = 3
    # Shared
    slice_len = 300
    batch_size = 32
    lr = 1e-3
    max_epochs = 20
    patience = 8
    device_str = "cuda"


# ---------------------------------------------------------------------------
# Causal CNN Seq2Seq (5-layer, from previous)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode='reflect')
        return self.conv(x)


class CausalSimpleSeq2Seq(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, hidden, 5, 1); self.gn1 = nn.GroupNorm(8, hidden)
        self.conv2 = CausalConv1d(hidden, hidden, 5, 2); self.gn2 = nn.GroupNorm(8, hidden)
        self.conv3 = CausalConv1d(hidden, hidden, 5, 4); self.gn3 = nn.GroupNorm(8, hidden)
        self.conv4 = CausalConv1d(hidden, hidden, 5, 8); self.gn4 = nn.GroupNorm(8, hidden)
        self.conv5 = CausalConv1d(hidden, hidden, 5, 16); self.gn5 = nn.GroupNorm(8, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Conv1d(hidden, num_classes, 1)
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
        return self.fc(x)


# ---------------------------------------------------------------------------
# TCN-lite (5 residual blocks)
# ---------------------------------------------------------------------------

class CausalConv1dTcn(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=0, dilation=dilation)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0), mode='reflect')
        return self.conv(x)


class TCNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1dTcn(in_ch, out_ch, k, dilation)
        self.wn1 = nn.utils.parametrizations.weight_norm(self.conv1.conv, name='weight')
        self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1dTcn(out_ch, out_ch, k, dilation)
        self.wn2 = nn.utils.parametrizations.weight_norm(self.conv2.conv, name='weight')
        self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_PhaseSegmenter(nn.Module):
    def __init__(self, in_ch=6, hidden=64, num_classes=2, k=5, num_layers=5, dropout=0.2):
        super(TCN_PhaseSegmenter, self).__init__()
        layers = []
        in_ch_cur = in_ch
        for i in range(num_layers):
            layers.append(TCNResBlock(in_ch_cur, hidden, k, 2**i, dropout))
            in_ch_cur = hidden
        self.network = nn.Sequential(*layers)
        self.fc = nn.Conv1d(hidden, num_classes, 1)
    def forward(self, x):
        return self.fc(self.network(x))


# ---------------------------------------------------------------------------
# TCN-micro: smaller, simpler
# ---------------------------------------------------------------------------

class TCNMicroResBlock(nn.Module):
    """Simplified residual block: single causal conv + ReLU + dropout."""
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.2):
        super().__init__()
        self.conv = CausalConv1dTcn(in_ch, out_ch, k, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TCNMicro(nn.Module):
    """Ultra-lightweight TCN: 3 layers, hidden=32, no weight norm."""
    def __init__(self, in_ch=6, hidden=32, num_classes=2, k=5, num_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch_cur = in_ch
        for i in range(num_layers):
            layers.append(TCNMicroResBlock(in_ch_cur, hidden, k, 2**i, dropout))
            in_ch_cur = hidden
        self.network = nn.Sequential(*layers)
        self.fc = nn.Conv1d(hidden, num_classes, 1)
    def forward(self, x):
        return self.fc(self.network(x))


# ---------------------------------------------------------------------------
# Data helpers (shared)
# ---------------------------------------------------------------------------

def extract_active_segments_for_seq2seq(train_streams, imu_columns):
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


def compute_normalization_stats(segments):
    all_data = np.concatenate([seg for seg in segments], axis=0)
    mean = np.mean(all_data, axis=0); std = np.std(all_data, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_segments(segments, mean, std):
    return [(seg - mean) / std for seg in segments]


class SegmentDataset(Dataset):
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
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        m = torch.from_numpy(mask).float()
        return x, y, m


# ---------------------------------------------------------------------------
# Training helper (shared)
# ---------------------------------------------------------------------------

def train_seq_model(model_class, model_kwargs, train_streams, imu_columns, cfg: FastConfig):
    """Generic trainer for CNN/TCN. Returns model, mean, std."""
    segments, labels = extract_active_segments_for_seq2seq(train_streams, imu_columns)
    if not segments:
        print("      No active segments")
        return None, None, None
    
    mean, std = compute_normalization_stats(segments)
    norm_segments = normalize_segments(segments, mean, std)
    
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_dataset = SegmentDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], slice_len=cfg.slice_len)
    val_dataset = SegmentDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], slice_len=cfg.slice_len)
    
    if len(train_dataset) == 0:
        return None, None, None
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    
    device = torch.device(cfg.device_str if torch.cuda.is_available() else 'cpu')
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0; n_batches = 0
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device); y_batch = y_batch.to(device); mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
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
            print(f"      Epoch {epoch+1}/{cfg.max_epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, mean, std


def predict_seq_model(model, df, active_segments, imu_columns, mean, std, cfg: FastConfig):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    
    if model is None:
        return phase_probs
    
    device = torch.device(cfg.device_str if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]
            seg_len = len(seg_x)
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= cfg.slice_len:
                pad_len = cfg.slice_len - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = cfg.slice_len // 2
                starts = list(range(0, seg_len - cfg.slice_len + 1, stride))
                if not starts or starts[-1] + cfg.slice_len < seg_len:
                    starts.append(seg_len - cfg.slice_len)
                for start in starts:
                    window = seg_x_norm[start:start + cfg.slice_len]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    global_start = seg_start + start
                    global_end = global_start + cfg.slice_len
                    phase_probs[global_start:global_end, :] += probs.T
                    phase_counts[global_start:global_end] += 1.0
    
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    return phase_probs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def parse_reps(phase_probs, min_phase=3, max_gap=3):
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


def eval_stream(stream_id, df, phase_probs, gt_reps, gt_phases, sw=15):
    smoothed = smooth_phase_probs(phase_probs, sw)
    pred_reps = parse_reps(smoothed)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(smoothed, gt_phases)
    return {
        "stream_id": stream_id, **rep_m, **phase_m,
        "pred_count": rep_m["pred_count"], "gt_count": rep_m["gt_count"],
    }


def aggregate(results):
    if not results: return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results); total_fp = sum(r["fp"] for r in results); total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    exact = sum(r["exact_count"] for r in results)
    over = sum(r["over"] for r in results); under = sum(r["under"] for r in results)
    phase_f1 = np.mean([r["phase_macro_f1"] for r in results])
    trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    return {
        "streams": n, "rep_precision": p, "rep_recall": r, "rep_f1": f1,
        "exact_count_acc": exact / n if n > 0 else 0,
        "over_count": over, "under_count": under,
        "phase_macro_f1": phase_f1,
        "transition_mae_ms": np.mean(trans) if trans else None,
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
    print(f"GPU: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    cfg = FastConfig()
    
    # Train Active Detector (shared)
    print("\n[1/4] Training Active Detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    
    # Train RF
    print("\n[2/4] Training RF Phase Model...")
    t0 = time.time()
    rf_models, rf_scalers = train_rf_phase(train_streams, cfg)
    print(f"      RF trained in {time.time()-t0:.1f}s")
    
    # Train Causal CNN
    print("\n[3/4] Training Causal CNN Seq2Seq...")
    t0 = time.time()
    cnn_model, cnn_mean, cnn_std = train_seq_model(
        CausalSimpleSeq2Seq, {"in_ch": 6, "hidden": 64, "num_classes": 2},
        train_streams, cfg.imu_columns, cfg
    )
    print(f"      CNN trained in {time.time()-t0:.1f}s")
    
    # Train TCN-lite
    print("\n[4/5] Training TCN-lite...")
    t0 = time.time()
    tcn_model, tcn_mean, tcn_std = train_seq_model(
        TCN_PhaseSegmenter, {"in_ch": 6, "hidden": 64, "num_classes": 2, "k": 5, "num_layers": 5},
        train_streams, cfg.imu_columns, cfg
    )
    print(f"      TCN-lite trained in {time.time()-t0:.1f}s")
    
    # Train TCN-micro
    print("\n[5/5] Training TCN-micro...")
    t0 = time.time()
    tcn_micro_model, tcn_micro_mean, tcn_micro_std = train_seq_model(
        TCNMicro, {"in_ch": 6, "hidden": 32, "num_classes": 2, "k": 5, "num_layers": 3},
        train_streams, cfg.imu_columns, cfg
    )
    print(f"      TCN-micro trained in {time.time()-t0:.1f}s")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    rf_results, cnn_results, tcn_results, tcn_micro_results = [], [], [], []
    
    for stream_id, df in test_streams:
        if "phase" not in df.columns: continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=cfg.min_phase_samples)
        
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        
        # RF
        rf_probs = predict_rf_phase(rf_models, rf_scalers, stream_id, df, active_segments, cfg)
        rf_results.append(eval_stream(stream_id, df, rf_probs, gt_reps, gt_phases, sw=cfg.smoothing_window))
        
        # CNN
        if cnn_model is not None:
            cnn_probs = predict_seq_model(cnn_model, df, active_segments, cfg.imu_columns, cnn_mean, cnn_std, cfg)
            cnn_results.append(eval_stream(stream_id, df, cnn_probs, gt_reps, gt_phases, sw=cfg.smoothing_window))
        
        # TCN-lite
        if tcn_model is not None:
            tcn_probs = predict_seq_model(tcn_model, df, active_segments, cfg.imu_columns, tcn_mean, tcn_std, cfg)
            tcn_results.append(eval_stream(stream_id, df, tcn_probs, gt_reps, gt_phases, sw=cfg.smoothing_window))
        
        # TCN-micro
        if tcn_micro_model is not None:
            tcn_micro_probs = predict_seq_model(tcn_micro_model, df, active_segments, cfg.imu_columns, tcn_micro_mean, tcn_micro_std, cfg)
            tcn_micro_results.append(eval_stream(stream_id, df, tcn_micro_probs, gt_reps, gt_phases, sw=cfg.smoothing_window))
    
    # Aggregate
    rf_agg = aggregate(rf_results)
    cnn_agg = aggregate(cnn_results)
    tcn_agg = aggregate(tcn_results)
    tcn_micro_agg = aggregate(tcn_micro_results)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for name, agg in [("RF", rf_agg), ("CausalCNN", cnn_agg), ("TCN-lite", tcn_agg), ("TCN-micro", tcn_micro_agg)]:
        print(f"\n{name}:")
        print(f"  Rep F1:        {agg['rep_f1']:.4f}")
        print(f"  Exact Count:   {agg['exact_count_acc']:.4f}")
        print(f"  Over/Under:    {agg['over_count']}/{agg['under_count']}")
        print(f"  Phase F1:      {agg['phase_macro_f1']:.4f}")
        print(f"  Trans MAE:     {agg.get('transition_mae_ms', 0):.0f}ms")
    
    # Save
    output_dir = Path("artifacts/quick_compare_1fold")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "rf": rf_agg, "cnn": cnn_agg, "tcn": tcn_agg, "tcn_micro": tcn_micro_agg,
            "rf_per_stream": rf_results, "cnn_per_stream": cnn_results, 
            "tcn_per_stream": tcn_results, "tcn_micro_per_stream": tcn_micro_results,
        }, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
