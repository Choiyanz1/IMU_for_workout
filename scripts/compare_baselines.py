"""Compare our proposed DS-MS-TCN pipeline against common baseline models.

Baselines:
  1. Sliding-window Random Forest (classic HAR)
  2. Bidirectional LSTM
  3. Simple 1D CNN (no multi-stage)

All models share the same:
  - Train/test split (LOSO by subject)
  - Z-score normalization from training data
  - Rep decoding pipeline (pair_concentric_eccentric + aggregate_action)
  - Evaluation metrics (rep F1, micro_f1_at_50, etc.)

Usage:
  python scripts/compare_baselines.py --config configs/micro_macro_recognition_3act_40ep_dualhead_viterbi_testkevin.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Ensure project root is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset

from models.ds_ms_tcn import DSMSTCN, DSMSTCNConfig, SingleStageTCN, ds_ms_tcn_loss
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    OTHER_LABEL,
    aggregate_action_for_reps,
    edit_score,
    labels_to_runs,
    macro_labels_from_action,
    match_segments,
    micro_labels_from_phase,
    pair_concentric_eccentric_reps,
    rep_metrics,
    sample_classification_metrics,
    segment_iou_f1,
    truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz
from preprocessing.window_pipeline import compute_train_stats, apply_zscore, set_seed
from train.micro_macro_recognition import (
    MicroMacroConfig,
    TrainConfig,
    SequenceSliceDataset,
    _available_actions,
    _collapse_micro_probs_to_phase,
    _decode_phase_labels,
    _filter_predicted_reps,
    _filter_subjects,
    _load_streams,
    _macro_runs_from_probs,
    _micro_classes_for_mode,
    _predict_full_sequence,
    _resample_streams_to_rate,
    _resolve_device,
    _resolve_num_workers,
    _resolve_pin_memory,
    _median_sample_rate,
    _is_all_subjects_mode,
)

import yaml


def _make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


# ---------------------------------------------------------------------------
# 1. Sliding-window Random Forest
# ---------------------------------------------------------------------------

def _extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract statistical features from a single window [T, C]."""
    return _extract_window_features_batch(window[None, :, :])[0]


def _extract_window_features_batch_base(windows: np.ndarray) -> np.ndarray:
    """Extract baseline statistical features from windows [N, T, C]."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected windows with shape [N, T, C], got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    window_len = int(arr.shape[1])
    norm = float(max(1, window_len - 1))
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    vmin = np.min(arr, axis=1)
    vmax = np.max(arr, axis=1)
    median = np.median(arr, axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    argmax = np.argmax(arr, axis=1).astype(np.float32) / norm
    argmin = np.argmin(arr, axis=1).astype(np.float32) / norm
    total_variation = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    per_channel = np.stack(
        [mean, std, vmin, vmax, median, q25, q75, argmax, argmin, total_variation],
        axis=-1,
    ).reshape(arr.shape[0], -1)

    mag = np.sqrt(np.sum(arr ** 2, axis=2))
    mag_stats = np.stack([np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1)], axis=1)
    return np.concatenate([per_channel, mag_stats], axis=1).astype(np.float32, copy=False)


def _extract_window_features_batch(windows: np.ndarray) -> np.ndarray:
    """Extract statistical features from windows [N, T, C]."""
    mode = os.environ.get("FEATURE_MODE", "baseline")
    if mode == "velocity":
        return _extract_window_features_batch_enhanced(windows, use_velocity=True, use_jerk=False)
    if mode == "velocity_jerk":
        return _extract_window_features_batch_enhanced(windows, use_velocity=True, use_jerk=True)
    return _extract_window_features_batch_base(windows)


def _extract_window_features_batch_enhanced(windows: np.ndarray, use_velocity: bool = False, use_jerk: bool = False) -> np.ndarray:
    """Extract statistical features with optional velocity/jerk [N, T, C]."""
    base = _extract_window_features_batch_base(windows)
    features = [base]

    if use_velocity:
        vel = np.diff(windows, axis=1)
        if vel.shape[1] > 0:
            features.append(_extract_window_features_batch_base(vel))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))

    if use_jerk:
        vel = np.diff(windows, axis=1)
        if vel.shape[1] > 1:
            jerk = np.diff(vel, axis=1)
            features.append(_extract_window_features_batch_base(jerk))
        else:
            features.append(np.zeros((windows.shape[0], base.shape[1]), dtype=np.float32))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _build_start_window_matrix(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if n <= 0:
        empty = np.zeros((0,), dtype=np.int64)
        return np.zeros((0, 0), dtype=np.float32), empty, empty
    window_size = int(max(1, window_size))
    stride = int(max(1, stride))
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


def train_rf_baseline(
    train_streams: List[Tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 25,
) -> RandomForestClassifier:
    """Train a sliding-window Random Forest for phase classification."""
    X_all, y_all = [], []
    for _, df in train_streams:
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, starts, ends = _build_start_window_matrix(x, int(window_size), int(stride))
        if not len(X_batch):
            continue
        y_batch = []
        for start, end in zip(starts, ends):
            window_labels = label_idx[int(start):int(end)]
            y_batch.append(int(np.argmax(np.bincount(window_labels, minlength=len(MICRO_LABELS)))))
        X_all.append(X_batch)
        y_all.append(np.asarray(y_batch, dtype=np.int64))
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    print(f"  [RF] Training on {len(X_all)} windows ({window_size} samples, stride {stride})")
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X_all, y_all)
    return clf


def predict_rf(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 10,
) -> np.ndarray:
    """Predict per-sample micro probs using sliding-window RF."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    X_batch, starts, ends = _build_start_window_matrix(x, int(window_size), int(stride))
    all_probs = clf.predict_proba(X_batch)  # [num_windows, num_classes]
    # Map RF classes to MICRO_LABELS order
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    prob_accum = np.zeros((n, len(MICRO_LABELS)), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    if len(all_probs):
        full_batch = np.zeros((len(all_probs), len(MICRO_LABELS)), dtype=np.float64)
        for cls_idx, mi in class_map.items():
            full_batch[:, cls_idx] = all_probs[:, mi]
        for wi, (start, end) in enumerate(zip(starts, ends)):
            prob_accum[int(start):int(end)] += full_batch[wi]
            counts[int(start):int(end)] += 1.0
    counts = np.maximum(counts, 1.0)
    return (prob_accum / counts[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Bidirectional LSTM
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        out, _ = self.lstm(x)
        out = self.dropout(out)
        logits = self.fc(out)  # [B, T, num_classes]
        return logits


class SimpleCNN(nn.Module):
    """Simple 1D CNN for temporal classification (no multi-stage)."""
    def __init__(self, input_dim: int, num_filters: int, num_classes: int, kernel_size: int = 5, num_conv_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_ch, num_filters, kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = num_filters
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Conv1d(num_filters, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> conv expects [B, C, T]
        h = self.conv(x.transpose(1, 2))
        logits = self.fc(h).transpose(1, 2)  # [B, T, num_classes]
        return logits


class PhaseOnlyTCN(nn.Module):
    """Single-stage phase-only TCN baseline."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_filters: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.tcn = SingleStageTCN(
            input_channels=input_dim,
            num_classes=num_classes,
            num_filters=num_filters,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


# ---------------------------------------------------------------------------
# Dataset for simple models (micro labels only, no macro)
# ---------------------------------------------------------------------------

class SimpleSliceDataset(Dataset):
    def __init__(self, streams, imu_columns, slice_len, stride_len):
        self.items = []
        for _, df in streams:
            x = df[list(imu_columns)].to_numpy(dtype=np.float32)
            labels = micro_labels_from_phase(df["phase"].to_numpy())
            label_idx = np.array([MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
            for start in range(0, max(1, len(x) - slice_len + 1), stride_len):
                end = min(start + slice_len, len(x))
                xi = x[start:end]
                yi = label_idx[start:end]
                if len(xi) < slice_len:
                    pad = slice_len - len(xi)
                    xi = np.pad(xi, ((0, pad), (0, 0)), mode="constant")
                    yi = np.pad(yi, (0, pad), mode="constant", constant_values=-100)
                self.items.append((xi, yi))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_simple_model(
    model: nn.Module,
    train_streams: List[Tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    slice_len: int,
    stride_len: int,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str,
):
    ds = SimpleSliceDataset(train_streams, imu_columns, slice_len, stride_len)
    print(f"  [{model_name}] Training on {len(ds)} slices, {epochs} epochs")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            with _autocast_context(use_amp):
                logits = model(x)  # [B, T, C]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{model_name}] epoch={epoch:03d}/{epochs:03d} loss={total_loss / max(1, count):.4f}")


def predict_simple_model(
    model: nn.Module,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    device: torch.device,
    smoothing_window: int = 15,
) -> np.ndarray:
    """Predict per-sample micro probs from a simple model."""
    model.eval()
    x = torch.from_numpy(df[list(imu_columns)].to_numpy(dtype=np.float32))[None].to(device)
    with torch.no_grad():
        logits = model(x)  # [1, T, C]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    # Causal smoothing
    if smoothing_window > 1:
        smoothed = np.zeros_like(probs)
        csum = np.cumsum(probs, axis=0)
        for i in range(len(probs)):
            start = max(0, i - smoothing_window + 1)
            total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
            smoothed[i] = total / float(i - start + 1)
        probs = smoothed
    return probs


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

def evaluate_micro_probs(
    micro_probs: np.ndarray,
    macro_probs: np.ndarray | None,
    df: pd.DataFrame,
    macro_classes: Sequence[str],
    mm_cfg: MicroMacroConfig,
    sample_rate: float,
) -> Dict:
    """Evaluate predictions given micro probs and optional macro probs."""
    pred_micro_labels = _decode_phase_labels(micro_probs, mm_cfg)
    gt_micro_labels = micro_labels_from_phase(df["phase"].to_numpy())
    gt_macro_labels = macro_labels_from_action(
        df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else [OTHER_LABEL] * len(df),
        gt_micro_labels,
    )
    # For baselines without macro head, use uniform macro probs
    if macro_probs is None:
        macro_probs = np.ones((len(df), len(macro_classes)), dtype=np.float32) / len(macro_classes)
    pred_macro_labels = [macro_classes[int(i)] for i in np.argmax(macro_probs, axis=1)]
    pred_micro_runs = labels_to_runs(
        pred_micro_labels,
        positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
        probabilities=micro_probs,
        min_length=mm_cfg.min_phase_samples,
    )
    pred_reps, diagnostics = pair_concentric_eccentric_reps(
        pred_micro_runs, micro_source="tcn", max_gap_samples=mm_cfg.max_phase_gap_samples,
    )
    pred_reps = _filter_predicted_reps(
        pred_reps, sample_rate_hz=sample_rate,
        min_duration_seconds=mm_cfg.min_rep_duration_seconds,
        min_confidence=mm_cfg.min_rep_confidence,
    )
    pred_reps = aggregate_action_for_reps(pred_reps, macro_probs, macro_classes)
    truth_reps = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=mm_cfg.min_phase_samples,
    )
    gt_micro_runs = labels_to_runs(gt_micro_labels, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=mm_cfg.min_phase_samples)
    gt_macro_runs = labels_to_runs(gt_macro_labels, positive_labels=[l for l in macro_classes if l != OTHER_LABEL], min_length=mm_cfg.min_phase_samples)
    pred_macro_runs = _macro_runs_from_probs(macro_probs, macro_classes, min_length=mm_cfg.min_phase_samples)

    metrics = rep_metrics(pred_reps, truth_reps, sample_rate_hz=sample_rate)
    micro_sample = sample_classification_metrics(gt_micro_labels, pred_micro_labels, MICRO_LABELS)
    macro_sample = sample_classification_metrics(gt_macro_labels, pred_macro_labels, macro_classes)
    micro_iou = segment_iou_f1(gt_micro_runs, pred_micro_runs)
    macro_iou = segment_iou_f1(gt_macro_runs, pred_macro_runs)

    true_actions, pred_actions = [], []
    for pi, ti, _ in match_segments(
        [(r.start_idx, r.end_idx) for r in pred_reps],
        [(r.start_idx, r.end_idx) for r in truth_reps],
    ):
        true_actions.append(truth_reps[ti].pred_action_type)
        pred_actions.append(pred_reps[pi].pred_action_type)

    return {
        **metrics,
        "micro_sample_accuracy": micro_sample["accuracy"],
        "micro_sample_macro_f1": micro_sample["macro_f1"],
        "macro_sample_accuracy": macro_sample["accuracy"],
        "macro_sample_macro_f1": macro_sample["macro_f1"],
        "micro_f1_at_10": micro_iou["f1_at_10"],
        "micro_f1_at_25": micro_iou["f1_at_25"],
        "micro_f1_at_50": micro_iou["f1_at_50"],
        "macro_f1_at_10": macro_iou["f1_at_10"],
        "macro_f1_at_25": macro_iou["f1_at_25"],
        "macro_f1_at_50": macro_iou["f1_at_50"],
    }


def evaluate_all_streams(
    predict_fn,  # callable(df) -> (micro_probs, macro_probs_or_None)
    test_streams: List[Tuple[str, pd.DataFrame]],
    macro_classes: Sequence[str],
    mm_cfg: MicroMacroConfig,
) -> Dict:
    """Evaluate a model on all test streams and return aggregated metrics."""
    all_rows = []
    for stream_id, df in test_streams:
        sample_rate = infer_sample_rate_hz(df)
        micro_probs, macro_probs = predict_fn(df)
        row = evaluate_micro_probs(micro_probs, macro_probs, df, macro_classes, mm_cfg, sample_rate)
        row["stream_id"] = stream_id
        row["count_diff"] = float(row.get("n_pred", 0.0) - row.get("n_true", 0.0))
        all_rows.append(row)
    # Aggregate
    agg_keys = ["n_pred", "n_true", "tp", "fp", "fn"]
    agg = {k: sum(r.get(k, 0) for r in all_rows) for k in agg_keys}
    p = agg["tp"] / max(1, agg["tp"] + agg["fp"])
    r = agg["tp"] / max(1, agg["tp"] + agg["fn"])
    f1 = 2 * p * r / max(1e-9, p + r)
    avg_keys = [
        "micro_sample_accuracy", "micro_sample_macro_f1",
        "macro_sample_accuracy", "macro_sample_macro_f1",
        "micro_f1_at_10", "micro_f1_at_25", "micro_f1_at_50",
        "macro_f1_at_10", "macro_f1_at_25", "macro_f1_at_50",
        "start_mae_ms", "end_mae_ms", "transition_mae_ms",
    ]
    avgs = {}
    for k in avg_keys:
        vals = [r[k] for r in all_rows if k in r and r[k] is not None and np.isfinite(r[k])]
        avgs[k] = float(np.mean(vals)) if vals else None
    # rep action accuracy
    rep_action_acc = agg["tp"] / max(1, agg["n_pred"])  # placeholder
    # Get actual action accuracy from rows
    total_action_correct = sum(1 for r in all_rows for _ in range(int(r.get("tp", 0))) if r.get("rep_action_accuracy", 0) > 0)
    action_accs = [r.get("rep_action_accuracy", 0) for r in all_rows if r.get("tp", 0) > 0]
    exact_count_streams = sum(1 for r in all_rows if int(r.get("n_pred", 0)) == int(r.get("n_true", 0)))
    over_segmented_streams = sum(1 for r in all_rows if int(r.get("n_pred", 0)) > int(r.get("n_true", 0)))
    under_segmented_streams = sum(1 for r in all_rows if int(r.get("n_pred", 0)) < int(r.get("n_true", 0)))
    zero_tp_streams = sum(1 for r in all_rows if int(r.get("tp", 0)) == 0)
    return {
        "precision": p,
        "recall": r,
        "rep_f1": f1,
        "rep_action_accuracy": float(np.mean(action_accs)) if action_accs else None,
        "n_pred": agg["n_pred"],
        "n_true": agg["n_true"],
        "tp": agg["tp"],
        "fp": agg["fp"],
        "fn": agg["fn"],
        "exact_count_streams": exact_count_streams,
        "over_segmented_streams": over_segmented_streams,
        "under_segmented_streams": under_segmented_streams,
        "zero_tp_streams": zero_tp_streams,
        "stream_count": len(all_rows),
        "stream_rows": all_rows,
        **avgs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare baseline models against DS-MS-TCN")
    parser.add_argument("--config", required=True, help="YAML config (used for data and our proposed model)")
    parser.add_argument("--output", default="artifacts/baseline_comparison", help="Output directory")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = raw.get("data", {}) or {}
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in MicroMacroConfig.__dataclass_fields__})
    train_cfg = TrainConfig(**{k: v for k, v in train_raw.items() if k in TrainConfig.__dataclass_fields__})
    set_seed(train_cfg.seed)

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    actions = _available_actions(
        Path(data_cfg.get("data_dir", "./datasets/raw_data")),
        data_cfg.get("include_actions"),
    )
    macro_classes = [OTHER_LABEL] + [a for a in actions if a != OTHER_LABEL]
    micro_classes = _micro_classes_for_mode(actions, mm_cfg.micro_label_mode)
    semantic_micro_classes = _micro_classes_for_mode(actions, mm_cfg.semantic_micro_label_mode) if mm_cfg.use_dual_micro_head else []

    # Load & split data
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = _load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = _resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    subjects_sorted = sorted(set(subjects))
    configured_test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else subjects_sorted[-1]
    train_all_subjects = _is_all_subjects_mode(configured_test_subject)
    if train_all_subjects:
        test_subject = "__all__"
        train_subjects = list(subjects_sorted)
        train_streams = list(streams)
        test_streams = list(streams)
        evaluation_protocol = "train_all_in_sample"
    else:
        test_subject = configured_test_subject
        train_subjects = [s for s in subjects_sorted if s != test_subject]
        train_streams = _filter_subjects(streams, train_subjects, subject_column)
        test_streams = _filter_subjects(streams, [test_subject], subject_column)
        evaluation_protocol = "subject_holdout"

    stats = compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

    sample_rate = _median_sample_rate(train_streams, float(target_sample_rate))
    slice_len = max(8, int(round(float(mm_cfg.slice_seconds) * sample_rate)))
    stride_len = max(1, int(round(slice_len * (1.0 - float(mm_cfg.overlap)))))

    device = _resolve_device(train_cfg.device)
    print(f"[INFO] protocol={evaluation_protocol} train={len(train_streams)} test={len(test_streams)} device={device}")
    print(f"[INFO] slice_len={slice_len} stride_len={stride_len} sample_rate={sample_rate:.1f}")

    results = {}

    # -----------------------------------------------------------------------
    # Baseline 1: Sliding-window Random Forest
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 1: Sliding-window Random Forest")
    print("=" * 60)
    rf_window = 50  # 0.5s at 100Hz
    rf_stride = 25
    t0 = time.time()
    rf_clf = train_rf_baseline(train_streams, imu_columns, window_size=rf_window, stride=rf_stride)
    rf_train_time = time.time() - t0

    def rf_predict(df):
        probs = predict_rf(rf_clf, df, imu_columns, window_size=rf_window)
        return probs, None

    t0 = time.time()
    rf_results = evaluate_all_streams(rf_predict, test_streams, macro_classes, mm_cfg)
    rf_eval_time = time.time() - t0
    rf_results["train_time_s"] = rf_train_time
    rf_results["eval_time_s"] = rf_eval_time
    rf_results["params"] = "~N/A (sklearn)"
    results["Random Forest"] = rf_results
    print(f"  Rep F1: {rf_results['rep_f1']:.4f}  micro_f1@50: {rf_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Baseline 2: BiLSTM
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 2: Bidirectional LSTM")
    print("=" * 60)
    lstm_model = BiLSTMClassifier(
        input_dim=len(imu_columns), hidden_dim=64, num_classes=len(MICRO_LABELS),
        num_layers=2, dropout=0.3,
    )
    n_params_lstm = sum(p.numel() for p in lstm_model.parameters())
    print(f"  Parameters: {n_params_lstm:,}")
    t0 = time.time()
    train_simple_model(lstm_model, train_streams, imu_columns, slice_len, stride_len,
                       epochs=40, lr=3e-4, device=device, model_name="BiLSTM")
    lstm_train_time = time.time() - t0

    def lstm_predict(df):
        probs = predict_simple_model(lstm_model, df, imu_columns, device, smoothing_window=15)
        return probs, None

    t0 = time.time()
    lstm_results = evaluate_all_streams(lstm_predict, test_streams, macro_classes, mm_cfg)
    lstm_eval_time = time.time() - t0
    lstm_results["train_time_s"] = lstm_train_time
    lstm_results["eval_time_s"] = lstm_eval_time
    lstm_results["params"] = n_params_lstm
    results["BiLSTM"] = lstm_results
    print(f"  Rep F1: {lstm_results['rep_f1']:.4f}  micro_f1@50: {lstm_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Baseline 3: Simple 1D CNN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 3: Simple 1D CNN")
    print("=" * 60)
    cnn_model = SimpleCNN(
        input_dim=len(imu_columns), num_filters=64, num_classes=len(MICRO_LABELS),
        kernel_size=5, num_conv_layers=4, dropout=0.3,
    )
    n_params_cnn = sum(p.numel() for p in cnn_model.parameters())
    print(f"  Parameters: {n_params_cnn:,}")
    t0 = time.time()
    train_simple_model(cnn_model, train_streams, imu_columns, slice_len, stride_len,
                       epochs=40, lr=3e-4, device=device, model_name="1D-CNN")
    cnn_train_time = time.time() - t0

    def cnn_predict(df):
        probs = predict_simple_model(cnn_model, df, imu_columns, device, smoothing_window=15)
        return probs, None

    t0 = time.time()
    cnn_results = evaluate_all_streams(cnn_predict, test_streams, macro_classes, mm_cfg)
    cnn_eval_time = time.time() - t0
    cnn_results["train_time_s"] = cnn_train_time
    cnn_results["eval_time_s"] = cnn_eval_time
    cnn_results["params"] = n_params_cnn
    results["1D CNN"] = cnn_results
    print(f"  Rep F1: {cnn_results['rep_f1']:.4f}  micro_f1@50: {cnn_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Baseline 4: Phase-only causal TCN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 4: Phase-only causal TCN")
    print("=" * 60)
    causal_tcn = PhaseOnlyTCN(
        input_dim=len(imu_columns),
        num_classes=len(MICRO_LABELS),
        num_filters=64,
        num_layers=6,
        kernel_size=3,
        dropout=0.2,
        causal=True,
    )
    n_params_causal_tcn = sum(p.numel() for p in causal_tcn.parameters())
    print(f"  Parameters: {n_params_causal_tcn:,}")
    t0 = time.time()
    train_simple_model(causal_tcn, train_streams, imu_columns, slice_len, stride_len,
                       epochs=40, lr=3e-4, device=device, model_name="PhaseTCN-Causal")
    causal_tcn_train_time = time.time() - t0

    def causal_tcn_predict(df):
        probs = predict_simple_model(causal_tcn, df, imu_columns, device, smoothing_window=15)
        return probs, None

    t0 = time.time()
    causal_tcn_results = evaluate_all_streams(causal_tcn_predict, test_streams, macro_classes, mm_cfg)
    causal_tcn_eval_time = time.time() - t0
    causal_tcn_results["train_time_s"] = causal_tcn_train_time
    causal_tcn_results["eval_time_s"] = causal_tcn_eval_time
    causal_tcn_results["params"] = n_params_causal_tcn
    results["Phase-only causal TCN"] = causal_tcn_results
    print(f"  Rep F1: {causal_tcn_results['rep_f1']:.4f}  micro_f1@50: {causal_tcn_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Baseline 5: Phase-only non-causal TCN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE 5: Phase-only non-causal TCN")
    print("=" * 60)
    noncausal_tcn = PhaseOnlyTCN(
        input_dim=len(imu_columns),
        num_classes=len(MICRO_LABELS),
        num_filters=64,
        num_layers=6,
        kernel_size=3,
        dropout=0.2,
        causal=False,
    )
    n_params_noncausal_tcn = sum(p.numel() for p in noncausal_tcn.parameters())
    print(f"  Parameters: {n_params_noncausal_tcn:,}")
    t0 = time.time()
    train_simple_model(noncausal_tcn, train_streams, imu_columns, slice_len, stride_len,
                       epochs=40, lr=3e-4, device=device, model_name="PhaseTCN-NonCausal")
    noncausal_tcn_train_time = time.time() - t0

    def noncausal_tcn_predict(df):
        probs = predict_simple_model(noncausal_tcn, df, imu_columns, device, smoothing_window=15)
        return probs, None

    t0 = time.time()
    noncausal_tcn_results = evaluate_all_streams(noncausal_tcn_predict, test_streams, macro_classes, mm_cfg)
    noncausal_tcn_eval_time = time.time() - t0
    noncausal_tcn_results["train_time_s"] = noncausal_tcn_train_time
    noncausal_tcn_results["eval_time_s"] = noncausal_tcn_eval_time
    noncausal_tcn_results["params"] = n_params_noncausal_tcn
    results["Phase-only non-causal TCN"] = noncausal_tcn_results
    print(f"  Rep F1: {noncausal_tcn_results['rep_f1']:.4f}  micro_f1@50: {noncausal_tcn_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Our proposed: DS-MS-TCN (dual-head + viterbi)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PROPOSED: DS-MS-TCN (dual-head + viterbi + smoothing)")
    print("=" * 60)
    train_sequences = [df for _, df in train_streams]
    ds = SequenceSliceDataset(
        train_sequences, imu_columns, micro_classes, macro_classes,
        slice_len=slice_len, stride_len=stride_len,
        micro_label_mode=mm_cfg.micro_label_mode,
        semantic_micro_classes=semantic_micro_classes,
        semantic_micro_label_mode=mm_cfg.semantic_micro_label_mode,
        use_gt_micro_probs=False,
    )
    loader = DataLoader(ds, batch_size=int(train_cfg.batch_size), shuffle=True, num_workers=0)
    tcn_model = DSMSTCN(
        DSMSTCNConfig(
            input_channels=len(imu_columns),
            micro_classes=len(micro_classes),
            semantic_micro_classes=len(semantic_micro_classes),
            macro_classes=len(macro_classes),
            num_filters=int(mm_cfg.num_filters),
            num_layers=int(mm_cfg.num_layers),
            kernel_size=int(mm_cfg.kernel_size),
            dropout=float(mm_cfg.dropout),
            causal=bool(mm_cfg.causal),
            num_macro_stages=int(mm_cfg.num_macro_stages),
            use_dual_micro_head=bool(mm_cfg.use_dual_micro_head),
            use_semantic_for_macro=bool(mm_cfg.use_semantic_for_macro),
        )
    )
    n_params_tcn = sum(p.numel() for p in tcn_model.parameters())
    print(f"  Parameters: {n_params_tcn:,}")
    tcn_model.to(device)
    use_amp = bool(train_cfg.amp) and device.type == "cuda"
    optimizer = torch.optim.Adam(tcn_model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay))
    scaler = _make_grad_scaler(use_amp)

    # Weights
    micro_class_weight_tensor = None
    if mm_cfg.micro_class_weights and len(mm_cfg.micro_class_weights) == len(micro_classes):
        micro_class_weight_tensor = torch.tensor(mm_cfg.micro_class_weights, dtype=torch.float32, device=device)
    semantic_class_weight_tensor = None
    if mm_cfg.semantic_micro_class_weights and len(mm_cfg.semantic_micro_class_weights) == len(semantic_micro_classes):
        semantic_class_weight_tensor = torch.tensor(mm_cfg.semantic_micro_class_weights, dtype=torch.float32, device=device)

    t0 = time.time()
    for epoch in range(1, int(train_cfg.epochs) + 1):
        tcn_model.train()
        total_loss, count = 0.0, 0
        for batch in loader:
            x, micro, macro, gt_micro_probs_t, semantic_micro, gt_semantic_probs = batch[:6]
            x = x.to(device)
            micro = micro.to(device)
            macro = macro.to(device)
            semantic_micro = semantic_micro.to(device)
            optimizer.zero_grad()
            with _autocast_context(use_amp):
                out = tcn_model(x)
                losses = ds_ms_tcn_loss(
                    out, micro, macro,
                    alpha=mm_cfg.alpha, beta=mm_cfg.beta, tmse_threshold=mm_cfg.tmse_threshold,
                    include_micro_loss=True, include_macro_loss=True,
                    micro_class_weights=micro_class_weight_tensor,
                    micro_beta=mm_cfg.micro_tmse_weight, micro_tmse_threshold=mm_cfg.micro_tmse_threshold,
                    semantic_target=semantic_micro if semantic_micro_classes else None,
                    semantic_alpha=mm_cfg.semantic_alpha,
                    semantic_class_weights=semantic_class_weight_tensor,
                )
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += losses["loss"].item() * x.size(0)
            count += x.size(0)
        if epoch % 10 == 0 or epoch == int(train_cfg.epochs):
            print(f"  [DS-MS-TCN] epoch={epoch:03d}/{train_cfg.epochs:03d} loss={total_loss / max(1, count):.4f}")
    tcn_train_time = time.time() - t0

    def tcn_predict(df):
        pred = _predict_full_sequence(
            tcn_model, df, imu_columns, device,
            micro_smoothing_window=int(mm_cfg.micro_smoothing_window),
        )
        raw_micro = pred["micro_probs"]
        micro_probs = _collapse_micro_probs_to_phase(raw_micro, micro_classes)
        macro_probs = pred["macro_probs"]
        return micro_probs, macro_probs

    t0 = time.time()
    tcn_results = evaluate_all_streams(tcn_predict, test_streams, macro_classes, mm_cfg)
    tcn_eval_time = time.time() - t0
    tcn_results["train_time_s"] = tcn_train_time
    tcn_results["eval_time_s"] = tcn_eval_time
    tcn_results["params"] = n_params_tcn
    results["DS-MS-TCN (ours)"] = tcn_results
    print(f"  Rep F1: {tcn_results['rep_f1']:.4f}  micro_f1@50: {tcn_results.get('micro_f1_at_50', 0):.4f}")

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (protocol: {}, test_subject: {})".format(evaluation_protocol, test_subject))
    print("=" * 80)

    header_keys = ["start_mae_ms", "end_mae_ms", "transition_mae_ms", "rep_f1", "precision", "recall", "exact_count_streams", "over_segmented_streams", "under_segmented_streams", "micro_f1_at_50"]
    header_labels = ["StartMAE", "EndMAE", "TransMAE", "Rep F1", "Precision", "Recall", "ExactCt", "Over", "Under", "micro_f1@50"]

    # Print table
    col_w = 16
    header = f"{'Model':<25}" + "".join(f"{h:>{col_w}}" for h in header_labels)
    print(header)
    print("-" * len(header))
    for model_name, res in results.items():
        row = f"{model_name:<25}"
        for k in header_keys:
            val = res.get(k)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            elif isinstance(val, float):
                row += f"{val:>{col_w}.4f}"
            elif isinstance(val, int):
                row += f"{val:>{col_w},}"
            else:
                row += f"{str(val):>{col_w}}"
        print(row)

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Convert results for JSON
    json_results = {}
    for model_name, res in results.items():
        json_results[model_name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items() if v is not None}
    with open(out_dir / "comparison_results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n[OK] Results saved to {out_dir / 'comparison_results.json'}")

    # Also save markdown table
    md_lines = [
        "# Baseline Comparison Results\n",
        f"Protocol: `{evaluation_protocol}`\n",
        f"Test subject: `{test_subject}`\n",
        "| Model | StartMAE | EndMAE | TransMAE | Rep F1 | Precision | Recall | ExactCt | Over | Under | micro_f1@50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model_name, res in results.items():
        vals = []
        for k in header_keys:
            v = res.get(k)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, int):
                vals.append(f"{v:,}")
            else:
                vals.append(str(v))
        md_lines.append(f"| {model_name} | " + " | ".join(vals) + " |")
    with open(out_dir / "comparison_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[OK] Markdown table saved to {out_dir / 'comparison_results.md'}")


if __name__ == "__main__":
    main()
