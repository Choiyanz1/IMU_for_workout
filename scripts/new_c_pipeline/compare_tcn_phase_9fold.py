"""
9-Fold LOSO Phase Model Comparison: RF Window-based vs TCN-lite (CAUSAL)

Fixed parameters (no fold-specific tuning):
  - TCN architecture: Causal dilated conv, residual connections, weight norm
  - Training: Adam, lr=1e-3, early stopping (patience=10), 30 epochs max
  - Post-processing at TWO fixed configs (not tuned per-fold):
    * Default:  sw=15, mps=3, mgs=3
    * Conservative (TCN-appropriate): sw=30, mps=7, mgs=3

Outputs per-fold and summary with C/E ratio analysis.

Difference from previous seq2seq:
  - TCN is strictly causal (no future peeking)
  - Exponentially growing receptive field via dilated causal convolutions
  - Residual connections for stable deep networks
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

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
    predict_rf_phase,
    smooth_phase_probs,
    train_active_detector,
    train_rf_phase,
    _extract_action_from_stream_id,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# TCN-lite Model (Causal)
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal convolution: output[t] only sees input[:t+1].
    Left-pads with exactly (kernel_size-1)*dilation so output length == input length.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0), mode='reflect')
        return self.conv(x)


class ResidualBlock(nn.Module):
    """TCN Residual Block with Causal Conv -> Weight Norm -> ReLU -> Dropout."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        # Layer 1
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.wn1 = nn.utils.parametrizations.weight_norm(self.conv1.conv, name='weight')
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.wn2 = nn.utils.parametrizations.weight_norm(self.conv2.conv, name='weight')
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN_PhaseSegmenter(nn.Module):
    """Lightweight TCN for per-sample C/E segmentation.
    
    Architecture:
      Input [B, C, T]
      -> 1x1 Conv projection to hidden_channels
      -> Stack of ResidualBlocks with exponentially increasing dilation
      -> 1x1 Conv to num_classes
      -> Output [B, num_classes, T]
    """
    def __init__(self, input_channels=6, hidden_channels=64, num_classes=2, 
                 kernel_size=5, num_layers=5, dropout=0.2):
        super(TCN_PhaseSegmenter, self).__init__()
        
        layers = []
        # Input projection
        in_ch = input_channels
        
        # Stacked residual blocks with doubling dilation
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_channels
            layers.append(ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # x: [B, C, T]
        y = self.network(x)
        return self.fc(y)


# ---------------------------------------------------------------------------
# Data loading (re-use from compare_phase_models)
# ---------------------------------------------------------------------------

def extract_active_segments_for_seq2seq(train_streams, imu_columns):
    """Extract active segments and per-sample C/E labels."""
    segments = []; labels = []
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


class ActiveSegmentDataset(Dataset):
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
                    self.samples.append((
                        seq[start:start+slice_len],
                        lab[start:start+slice_len],
                        np.ones(slice_len, dtype=np.float32)
                    ))
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        seq, lab, mask = self.samples[idx]
        x = torch.from_numpy(seq).float().transpose(0, 1)
        y = torch.from_numpy(lab).long()
        m = torch.from_numpy(mask).float()
        return x, y, m


# ---------------------------------------------------------------------------
# TCN Training
# ---------------------------------------------------------------------------

def train_tcn_phase(train_streams, imu_columns, cfg: PhaseCompareConfig):
    """Train TCN phase segmenter with early stopping."""
    segments, labels = extract_active_segments_for_seq2seq(train_streams, imu_columns)
    if not segments:
        print("      [TCN] No active segments found for training")
        return None, None, None
    
    print(f"      [TCN] Training on {len(segments)} active segments")
    
    mean, std = compute_normalization_stats(segments)
    norm_segments = normalize_segments(segments, mean, std)
    
    # Train/val split
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    
    train_dataset = ActiveSegmentDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], slice_len=cfg.seq2seq_slice_len)
    val_dataset = ActiveSegmentDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], slice_len=cfg.seq2seq_slice_len)
    
    if len(train_dataset) == 0:
        print("      [TCN] Empty training dataset after slicing")
        return None, None, None
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.seq2seq_batch_size, shuffle=False, drop_last=False)
    
    print(f"      [TCN] Train: {len(train_dataset)} slices, Val: {len(val_dataset)} slices")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      [TCN] Using device: {device}")
    model = TCN_PhaseSegmenter(
        input_channels=len(imu_columns), 
        hidden_channels=cfg.seq2seq_hidden,
        num_classes=2,
        kernel_size=5,
        num_layers=5,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.seq2seq_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(cfg.seq2seq_epochs):
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
            print(f"      [TCN] Epoch {epoch+1}/{cfg.seq2seq_epochs}, Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"      [TCN] Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, mean, std


# ---------------------------------------------------------------------------
# TCN Inference (Sliding window + overlap averaging)
# ---------------------------------------------------------------------------

def predict_tcn_phase(model, df, active_segments, imu_columns, mean, std, cfg: PhaseCompareConfig):
    """Predict per-sample C/E probability using TCN with sliding window + overlap averaging."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32); n = len(x)
    phase_probs = np.ones((n, 2)) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    
    if model is None:
        return phase_probs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    slice_len = cfg.seq2seq_slice_len
    
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end: continue
            seg_x = x[seg_start:seg_end]
            seg_len = len(seg_x)
            seg_x_norm = (seg_x - mean) / std
            
            if seg_len <= slice_len:
                pad_len = slice_len - seg_len
                padded = np.pad(seg_x_norm, ((0, pad_len), (0, 0)), mode='edge')
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                logits = model(x_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = slice_len // 2
                starts = list(range(0, seg_len - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < seg_len:
                    starts.append(seg_len - slice_len)
                
                for start in starts:
                    window = seg_x_norm[start:start + slice_len]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    logits = model(x_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    global_start = seg_start + start
                    global_end = global_start + slice_len
                    phase_probs[global_start:global_end, :] += probs.T
                    phase_counts[global_start:global_end] += 1.0
    
    valid_mask = phase_counts > 0
    phase_probs[valid_mask] /= phase_counts[valid_mask][:, None]
    return phase_probs


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

@dataclass
class PostProcConfig:
    smoothing_window: int
    min_phase_samples: int
    max_gap_samples: int

DEFAULT_PP = PostProcConfig(smoothing_window=15, min_phase_samples=3, max_gap_samples=3)
CONSERVATIVE_PP = PostProcConfig(smoothing_window=30, min_phase_samples=7, max_gap_samples=3)


def parse_reps_with_config(phase_probs: np.ndarray, pp: PostProcConfig) -> List[RepDetection]:
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=pp.min_phase_samples)
    if not runs:
        return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(
                label=run.label, start_idx=merged[-1].start_idx,
                end_idx=run.end_idx, confidence=(merged[-1].confidence + run.confidence) / 2,
            )
        else:
            merged.append(run)
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="phase", max_gap_samples=pp.max_gap_samples)
    return reps


def compute_rep_ce_ratios(reps: List[RepDetection], phase_labels: np.ndarray) -> List[float]:
    ratios = []
    for rep in reps:
        seg = phase_labels[rep.start_idx:rep.end_idx]
        if len(seg) == 0:
            ratios.append(float('nan'))
            continue
        c_count = np.sum(seg == CONCENTRIC_LABEL)
        e_count = np.sum(seg == ECCENTRIC_LABEL)
        if e_count == 0:
            ratios.append(float('inf'))
        else:
            ratios.append(c_count / e_count)
    return ratios


def compute_ce_ratio_metrics(pred_ratios: List[float], gt_ratios: List[float]) -> dict:
    valid_pairs = []
    for p, g in zip(pred_ratios, gt_ratios):
        if np.isfinite(p) and np.isfinite(g) and p != float('inf') and g != float('inf'):
            valid_pairs.append((p, g))
    if not valid_pairs:
        return {"ce_ratio_mae": None, "ce_ratio_rmse": None, "ce_ratio_bias": None, "n_valid": 0}
    pred_arr = np.array([p for p, _ in valid_pairs])
    gt_arr = np.array([g for _, g in valid_pairs])
    errors = pred_arr - gt_arr
    return {
        "ce_ratio_mae": float(np.mean(np.abs(errors))),
        "ce_ratio_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ce_ratio_bias": float(np.mean(errors)),
        "n_valid": len(valid_pairs),
    }


def evaluate_stream(
    stream_id: str,
    df: pd.DataFrame,
    phase_probs: np.ndarray,
    pp: PostProcConfig,
    gt_reps: List[RepDetection],
    gt_phases: np.ndarray,
) -> dict:
    phase_probs_smooth = smooth_phase_probs(phase_probs, pp.smoothing_window)
    pred_reps = parse_reps_with_config(phase_probs_smooth, pp)
    
    rep_metrics = evaluate_reps(pred_reps, gt_reps)
    phase_metrics = evaluate_phase(phase_probs_smooth, gt_phases)
    
    pred_ratios = compute_rep_ce_ratios(pred_reps, np.array(
        ["eccentric" if p == 0 else "concentric" for p in np.argmax(phase_probs_smooth, axis=1)]
    ))
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)
    
    count_error = abs(rep_metrics["pred_count"] - rep_metrics["gt_count"])
    
    return {
        "stream_id": stream_id,
        "pred_count": rep_metrics["pred_count"],
        "gt_count": rep_metrics["gt_count"],
        "count_error": count_error,
        **{k: v for k, v in rep_metrics.items() if k not in ["pred_count", "gt_count"]},
        **phase_metrics,
        **ce_metrics,
    }


def aggregate_fold_results(results: List[dict]) -> dict:
    if not results:
        return {}
    n = len(results)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    exact_count = sum(r["exact_count"] for r in results)
    over_count = sum(r["over"] for r in results)
    under_count = sum(r["under"] for r in results)
    count_errors = [r["count_error"] for r in results]
    
    trans_mae_list = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    phase_f1_list = [r["phase_macro_f1"] for r in results]
    phase_acc_list = [r["phase_accuracy"] for r in results]
    ce_mae_list = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]
    
    return {
        "streams": n,
        "rep_precision": p,
        "rep_recall": r,
        "rep_f1": f1,
        "exact_count_acc": exact_count / n if n > 0 else 0,
        "mean_abs_count_error": np.mean(count_errors) if count_errors else 0,
        "over_count": over_count,
        "under_count": under_count,
        "phase_macro_f1": np.mean(phase_f1_list) if phase_f1_list else 0,
        "phase_accuracy": np.mean(phase_acc_list) if phase_acc_list else 0,
        "transition_mae_ms": np.mean(trans_mae_list) if trans_mae_list else None,
        "ce_ratio_mae": np.mean(ce_mae_list) if ce_mae_list else None,
    }


# ---------------------------------------------------------------------------
# Main 9-fold evaluation
# ---------------------------------------------------------------------------

def run_9fold_evaluation(all_streams, subjects, cfg: PhaseCompareConfig, output_dir: Path):
    print("=" * 80)
    print("9-Fold LOSO Phase Model Comparison: RF vs TCN-lite (CAUSAL)")
    print("=" * 80)
    print(f"Subjects ({len(subjects)}): {subjects}")
    
    rf_fold_results = []
    tcn_default_fold_results = []
    tcn_conservative_fold_results = []
    
    for fold_idx, test_subject in enumerate(subjects):
        print(f"\n{'=' * 80}")
        print(f"Fold {fold_idx + 1}/9: test={test_subject}")
        print(f"{'=' * 80}")
        
        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in all_streams if sid.startswith(f"{test_subject}/")]
        print(f"  train={len(train_streams)}, test={len(test_streams)}")
        
        # Train all models once per fold
        print("  Training Active Detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        
        print("  Training RF Phase Model...")
        rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, cfg)
        
        print("  Training TCN Phase Model...")
        tcn_model, tcn_mean, tcn_std = train_tcn_phase(train_streams, cfg.imu_columns, cfg)
        
        # Evaluate each test stream
        rf_stream_results = []
        tcn_def_stream_results = []
        tcn_con_stream_results = []
        
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=cfg.min_phase_samples)
            
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            
            # RF
            rf_phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, cfg)
            rf_stream_results.append(evaluate_stream(stream_id, df, rf_phase_probs, DEFAULT_PP, gt_reps, gt_phases))
            
            # TCN (only if trained successfully)
            if tcn_model is not None:
                tcn_phase_probs = predict_tcn_phase(
                    tcn_model, df, active_segments, cfg.imu_columns,
                    tcn_mean, tcn_std, cfg
                )
                tcn_def_stream_results.append(evaluate_stream(stream_id, df, tcn_phase_probs, DEFAULT_PP, gt_reps, gt_phases))
                tcn_con_stream_results.append(evaluate_stream(stream_id, df, tcn_phase_probs, CONSERVATIVE_PP, gt_reps, gt_phases))
        
        # Aggregate fold
        rf_fold = aggregate_fold_results(rf_stream_results)
        rf_fold["fold"] = fold_idx + 1
        rf_fold["test_subject"] = test_subject
        rf_fold_results.append(rf_fold)
        
        if tcn_model is not None:
            tcn_def_fold = aggregate_fold_results(tcn_def_stream_results)
            tcn_def_fold["fold"] = fold_idx + 1
            tcn_def_fold["test_subject"] = test_subject
            tcn_default_fold_results.append(tcn_def_fold)
            
            tcn_con_fold = aggregate_fold_results(tcn_con_stream_results)
            tcn_con_fold["fold"] = fold_idx + 1
            tcn_con_fold["test_subject"] = test_subject
            tcn_conservative_fold_results.append(tcn_con_fold)
        
        print(f"  RF:  RepF1={rf_fold['rep_f1']:.4f} Exact={rf_fold['exact_count_acc']:.3f} "
              f"PhaseF1={rf_fold['phase_macro_f1']:.4f} TransMAE={rf_fold.get('transition_mae_ms', 0):.0f}ms")
        if tcn_model is not None:
            print(f"  TCNd: RepF1={tcn_def_fold['rep_f1']:.4f} Exact={tcn_def_fold['exact_count_acc']:.3f} "
                  f"PhaseF1={tcn_def_fold['phase_macro_f1']:.4f} TransMAE={tcn_def_fold.get('transition_mae_ms', 0):.0f}ms")
            print(f"  TCNc: RepF1={tcn_con_fold['rep_f1']:.4f} Exact={tcn_con_fold['exact_count_acc']:.3f} "
                  f"PhaseF1={tcn_con_fold['phase_macro_f1']:.4f} TransMAE={tcn_con_fold.get('transition_mae_ms', 0):.0f}ms")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("9-FOLD SUMMARY")
    print(f"{'=' * 80}")
    
    summary = {}
    for name, fold_results in [
        ("RF", rf_fold_results),
        ("TCN_Default", tcn_default_fold_results),
        ("TCN_Conservative", tcn_conservative_fold_results),
    ]:
        if not fold_results:
            continue
        
        print(f"\n[{name}] ({len(fold_results)} folds)")
        
        metrics = ["rep_f1", "exact_count_acc", "mean_abs_count_error", "over_count", "under_count",
                   "phase_macro_f1", "phase_accuracy", "transition_mae_ms", "ce_ratio_mae"]
        
        for metric in metrics:
            values = [f[metric] for f in fold_results if f.get(metric) is not None]
            if not values:
                continue
            mean = np.mean(values)
            std = np.std(values)
            best = np.max(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.min(values)
            worst = np.min(values) if metric not in ["mean_abs_count_error", "transition_mae_ms", "ce_ratio_mae", "over_count", "under_count"] else np.max(values)
            print(f"  {metric}: mean={mean:.4f} std={std:.4f} best={best:.4f} worst={worst:.4f}")
            summary[f"{name}_{metric}_mean"] = mean
            summary[f"{name}_{metric}_std"] = std
        
        for f in fold_results:
            print(f"    Fold {f['fold']} ({f['test_subject']}): "
                  f"RepF1={f['rep_f1']:.4f} Exact={f['exact_count_acc']:.3f} "
                  f"PhaseF1={f['phase_macro_f1']:.4f} TransMAE={f.get('transition_mae_ms', 0):.0f}ms "
                  f"Over/Under={f['over_count']}/{f['under_count']}")
    
    # Compare TCN vs RF
    if tcn_conservative_fold_results and rf_fold_results:
        print(f"\n{'=' * 80}")
        print("TCN vs RF COMPARISON")
        print(f"{'=' * 80}")
        
        for label, tcn_folds in [
            ("TCN (Default PP)", tcn_default_fold_results),
            ("TCN (Conservative PP)", tcn_conservative_fold_results),
        ]:
            if not tcn_folds:
                continue
            print(f"\n{label}:")
            
            comparisons = {
                "rep_f1": "higher is better",
                "phase_macro_f1": "higher is better",
                "phase_accuracy": "higher is better",
                "exact_count_acc": "higher is better",
                "transition_mae_ms": "lower is better",
                "mean_abs_count_error": "lower is better",
            }
            
            for metric, direction in comparisons.items():
                tcn_wins = 0
                for tcn_f, rf_f in zip(tcn_folds, rf_fold_results):
                    tcn_val = tcn_f.get(metric)
                    rf_val = rf_f.get(metric)
                    if tcn_val is None or rf_val is None:
                        continue
                    if direction == "higher is better":
                        if tcn_val > rf_val:
                            tcn_wins += 1
                    else:
                        if tcn_val < rf_val:
                            tcn_wins += 1
                print(f"  {metric}: TCN wins in {tcn_wins}/{len(tcn_folds)} folds")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_9fold_tcn.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "rf_per_fold": rf_fold_results,
            "tcn_default_per_fold": tcn_default_fold_results,
            "tcn_conservative_per_fold": tcn_conservative_fold_results,
        }, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_path}")
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="9-Fold LOSO Phase Model Comparison: RF vs TCN-lite")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase_model_comparison_9fold_tcn"))
    args = parser.parse_args()
    
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects: {subjects}")
    
    cfg = PhaseCompareConfig()
    cfg.seq2seq_epochs = 30
    
    run_9fold_evaluation(all_streams, subjects, cfg, args.output)


if __name__ == "__main__":
    main()
