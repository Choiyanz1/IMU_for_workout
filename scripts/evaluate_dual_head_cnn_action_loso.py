"""Evaluate a tiny dual-head CNN for streaming active/action recognition."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import (  # noqa: E402
    ACTIONS,
    ACTIVE_PHASES,
    NON_ACTION,
    WindowMeta,
    evaluate_locks,
    load_non_action_streams,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import stream_action, stream_subject  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, should_exclude, set_seed  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y_active: np.ndarray, y_action: np.ndarray) -> None:
        self.x = x.astype(np.float32)
        self.y_active = y_active.astype(np.int64)
        self.y_action = y_action.astype(np.int64)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y_active[idx], dtype=torch.long),
            torch.tensor(self.y_action[idx], dtype=torch.long),
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0), mode="replicate"))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = self.dropout(F.relu(self.norm(self.conv(x))))
        return out + identity if out.shape == identity.shape else out


class DualHeadActionCNN(nn.Module):
    def __init__(self, in_ch: int = 6, hidden: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, hidden, 7, 1, dropout),
            ConvBlock(hidden, hidden, 5, 2, dropout),
            ConvBlock(hidden, hidden, 5, 4, dropout),
            ConvBlock(hidden, hidden, 5, 8, dropout),
        )
        self.active_head = nn.Linear(hidden, 2)
        self.action_head = nn.Linear(hidden, len(ACTIONS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=2)
        return self.active_head(pooled), self.action_head(pooled)


def normalize_streams(train_streams, eval_streams, imu_columns: Sequence[str]):
    train_values = [df[list(imu_columns)].to_numpy(dtype=np.float32) for _, df in train_streams if len(df)]
    stacked = np.concatenate(train_values, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out = []
    for stream_id, df in eval_streams:
        copied = df.copy()
        values = copied[list(imu_columns)].to_numpy(dtype=np.float32)
        copied.loc[:, list(imu_columns)] = (values - mean) / std
        out.append((stream_id, copied))
    return out


def _starts_for_length(n: int, window_samples: int, stride_samples: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n < window_samples:
        return np.asarray([0], dtype=np.int64)
    starts = list(range(0, n - window_samples + 1, stride_samples))
    tail = n - window_samples
    if starts[-1] != tail:
        starts.append(tail)
    return np.asarray(starts, dtype=np.int64)


def build_raw_windows(streams, imu_columns, window_samples: int, stride_samples: int, active_threshold: float):
    x_rows: list[np.ndarray] = []
    y_active: list[int] = []
    y_action: list[int] = []
    metas: list[WindowMeta] = []
    action_to_idx = {action: idx for idx, action in enumerate(ACTIONS)}

    for stream_id, df in streams:
        if not set(imu_columns).issubset(df.columns):
            continue
        values = df[list(imu_columns)].to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        phase = df["phase"].astype(str).to_numpy() if "phase" in df.columns else np.full(len(df), NON_ACTION, dtype=object)
        active_mask = np.asarray([p in ACTIVE_PHASES for p in phase], dtype=bool)
        action = stream_action(stream_id)
        if action not in action_to_idx:
            action = NON_ACTION
        pad = max(0, window_samples - len(values))
        padded_values = np.pad(values, ((0, pad), (0, 0)), mode="edge") if pad else values
        padded_active = np.pad(active_mask, (0, pad), constant_values=False) if pad else active_mask
        for start in _starts_for_length(len(values), window_samples, stride_samples):
            end = min(int(start) + window_samples, len(values))
            active_fraction = float(np.mean(padded_active[int(start) : int(start) + window_samples]))
            active_label = int(active_fraction >= active_threshold and action in action_to_idx)
            x_rows.append(padded_values[int(start) : int(start) + window_samples].T)
            y_active.append(active_label)
            y_action.append(action_to_idx[action] if active_label else -1)
            metas.append(WindowMeta(stream_id, stream_subject(stream_id), action, int(start), end, active_fraction))

    return (
        np.stack(x_rows).astype(np.float32) if x_rows else np.zeros((0, len(imu_columns), window_samples), dtype=np.float32),
        np.asarray(y_active, dtype=np.int64),
        np.asarray(y_action, dtype=np.int64),
        metas,
    )


def train_model(x_train, y_active_train, y_action_train, args, device: torch.device):
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(x_train))
    n_val = max(1, int(round(len(indices) * args.val_fraction)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_loader = DataLoader(
        WindowDataset(x_train[train_idx], y_active_train[train_idx], y_action_train[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        WindowDataset(x_train[val_idx], y_active_train[val_idx], y_action_train[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DualHeadActionCNN(in_ch=x_train.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    active_counts = np.bincount(y_active_train[train_idx], minlength=2).astype(np.float32)
    active_weights = active_counts.sum() / np.maximum(active_counts, 1.0)
    active_weights[1] *= float(args.active_positive_weight_scale)
    active_criterion = nn.CrossEntropyLoss(weight=torch.tensor(active_weights, dtype=torch.float32, device=device))
    action_criterion = nn.CrossEntropyLoss()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for x, y_active, y_action in train_loader:
            x = x.to(device)
            y_active = y_active.to(device)
            y_action = y_action.to(device)
            optimizer.zero_grad(set_to_none=True)
            active_logits, action_logits = model(x)
            active_loss = active_criterion(active_logits, y_active)
            valid_action = y_action >= 0
            if torch.any(valid_action):
                action_loss = action_criterion(action_logits[valid_action], y_action[valid_action])
            else:
                action_loss = torch.zeros((), device=device)
            loss = active_loss + args.action_loss_weight * action_loss
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y_active, y_action in val_loader:
                x = x.to(device)
                y_active = y_active.to(device)
                y_action = y_action.to(device)
                active_logits, action_logits = model(x)
                active_loss = active_criterion(active_logits, y_active)
                valid_action = y_action >= 0
                action_loss = action_criterion(action_logits[valid_action], y_action[valid_action]) if torch.any(valid_action) else torch.zeros((), device=device)
                val_loss += float((active_loss + args.action_loss_weight * action_loss).cpu())
                val_batches += 1
        avg_val = val_loss / max(1, val_batches)
        print(f"    epoch {epoch}/{args.epochs}: train={train_loss / max(1, batches):.4f} val={avg_val:.4f}", flush=True)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_model(model, x: np.ndarray, args, device: torch.device):
    model.eval()
    loader = DataLoader(WindowDataset(x, np.zeros(len(x)), np.full(len(x), -1)), batch_size=args.batch_size, shuffle=False, num_workers=0)
    active_probs = []
    action_probs = []
    active_pred = []
    action_pred = []
    for xb, _, _ in loader:
        xb = xb.to(device)
        active_logits, action_logits = model(xb)
        ap = torch.softmax(active_logits, dim=1).cpu().numpy()
        xp = torch.softmax(action_logits, dim=1).cpu().numpy()
        active_probs.append(ap[:, 1])
        action_probs.append(xp)
        active_pred.append(np.argmax(ap, axis=1))
        action_pred.append(np.argmax(xp, axis=1))
    return np.concatenate(active_pred), np.concatenate(action_pred), np.concatenate(active_probs), np.concatenate(action_probs)


def binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_fold(test_subject, train_streams, test_streams, imu_columns, args, device):
    train_norm = normalize_streams(train_streams, train_streams, imu_columns)
    test_norm = normalize_streams(train_streams, test_streams, imu_columns)
    x_train, y_active_train, y_action_train, _ = build_raw_windows(train_norm, imu_columns, args.window_samples, args.stride_samples, args.window_active_threshold)
    x_test, y_active_test, y_action_test, metas = build_raw_windows(test_norm, imu_columns, args.window_samples, args.stride_samples, args.window_active_threshold)
    model = train_model(x_train, y_active_train, y_action_train, args, device)
    active_pred, action_pred_idx, active_probs, action_probs = predict_model(model, x_test, args, device)
    active_mask = y_active_test == 1
    action_labels = np.asarray(ACTIONS, dtype=object)
    y_action_label = np.asarray([ACTIONS[idx] if idx >= 0 else NON_ACTION for idx in y_action_test], dtype=object)
    pred_action_label = action_labels[action_pred_idx]
    gated_pred = np.where(active_probs >= args.eval_active_threshold, pred_action_label, NON_ACTION)
    lock_eval = evaluate_locks(metas, action_probs, active_probs, args)
    return {
        "test_subject": test_subject,
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        "train_active_windows": int(np.sum(y_active_train == 1)),
        "train_non_action_windows": int(np.sum(y_active_train == 0)),
        "test_active_windows": int(np.sum(y_active_test == 1)),
        "test_non_action_windows": int(np.sum(y_active_test == 0)),
        "active_head": binary_metrics(y_active_test, active_pred),
        "action_head_on_true_active": {
            "accuracy": float(accuracy_score(y_action_label[active_mask], pred_action_label[active_mask])) if np.any(active_mask) else 0.0,
            "macro_f1": float(f1_score(y_action_label[active_mask], pred_action_label[active_mask], labels=ACTIONS, average="macro", zero_division=0)) if np.any(active_mask) else 0.0,
        },
        "gated_9class": {
            "accuracy": float(accuracy_score(y_action_label, gated_pred)),
            "macro_f1": float(f1_score(y_action_label, gated_pred, labels=[*ACTIONS, NON_ACTION], average="macro", zero_division=0)),
        },
        "lock": lock_eval["summary"],
        "lock_rows": lock_eval["rows"],
    }


def aggregate_folds(folds):
    def mean_path(*keys):
        vals = []
        for fold in folds:
            item = fold
            for key in keys:
                item = item[key]
            if item is not None:
                vals.append(float(item))
        return float(np.mean(vals)) if vals else None

    return {
        "folds": len(folds),
        "active_head": {k: mean_path("active_head", k) for k in ["accuracy", "precision", "recall", "f1", "macro_f1"]},
        "action_head_on_true_active": {k: mean_path("action_head_on_true_active", k) for k in ["accuracy", "macro_f1"]},
        "gated_9class": {k: mean_path("gated_9class", k) for k in ["accuracy", "macro_f1"]},
        "lock": {k: mean_path("lock", k) for k in ["action_lock_rate", "action_lock_accuracy", "median_lock_time_s", "non_action_false_lock_rate"]},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dual-head tiny CNN action recognizer with LOSO splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/dual_head_cnn_loso/summary.json")
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=100)
    parser.add_argument("--window-active-threshold", type=float, default=0.5)
    parser.add_argument("--eval-active-threshold", type=float, default=0.5)
    parser.add_argument("--lock-active-threshold", type=float, default=0.55)
    parser.add_argument("--lock-threshold", type=float, default=0.55)
    parser.add_argument("--lock-margin", type=float, default=0.10)
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--min-lock-windows", type=int, default=3)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--active-positive-weight-scale", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rest-fragments", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    imu_columns = list(raw_cfg.get("feature", {}).get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    set_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in set_streams if not should_exclude(sid)]
    rest_streams = [] if args.no_rest_fragments else load_non_action_streams(raw_cfg)
    streams = [*set_streams, *rest_streams]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects
    print(f"set_streams={len(set_streams)} rest_streams={len(rest_streams)} subjects={subjects} device={device}", flush=True)
    folds = []
    for fold_idx, test_subject in enumerate(eval_subjects, start=1):
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject}", flush=True)
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        fold = evaluate_fold(test_subject, train_streams, test_streams, imu_columns, args, device)
        folds.append(fold)
        print(
            "  active_f1={:.4f} action_acc={:.4f} gated_macro={:.4f} lock_acc={:.4f} lock_rate={:.4f}".format(
                fold["active_head"]["f1"],
                fold["action_head_on_true_active"]["accuracy"],
                fold["gated_9class"]["macro_f1"],
                fold["lock"]["action_lock_accuracy"],
                fold["lock"]["action_lock_rate"],
            ),
            flush=True,
        )
    output = {
        "settings": {
            "model": "dual_head_tiny_causal_cnn_action_probe",
            "input_columns": imu_columns,
            "actions": ACTIONS,
            "non_action_label": NON_ACTION,
            "window_samples": args.window_samples,
            "stride_samples": args.stride_samples,
            "sample_rate_hz": args.sample_rate_hz,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "set_streams": len(set_streams),
            "rest_streams": len(rest_streams),
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "overall": aggregate_folds(folds),
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nOVERALL")
    print(json.dumps(output["overall"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
