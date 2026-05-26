"""BiLSTM Baseline (Non-Causal) with 7-fold LOSO.

Bidirectional LSTM for per-time-step micro-label classification.
Non-causal: can look at future context (upper bound baseline).
Usage:
    # Smoke test (3 subjects)
    python scripts/evaluate_bilstm_loso.py --config config.yaml \
        --output artifacts/bilstm_smoke --subjects haoyu,kevin,yoru

    # Full 7-fold
    python scripts/evaluate_bilstm_loso.py --config config.yaml \
        --output artifacts/bilstm_7fold
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")

# ---------------------------------------------------------------------------
# BiLSTM Model
# ---------------------------------------------------------------------------

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for temporal classification."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        h, _ = self.lstm(x)
        logits = self.fc(h)  # [B, T, num_classes]
        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamSliceDataset(Dataset):
    def __init__(self, streams, imu_columns, slice_len, stride_len):
        self.items = []
        for _, df in streams:
            x = df[list(imu_columns)].to_numpy(dtype=np.float32)
            labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
            label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/len(train_loader):.4f}")


def predict_stream(model: nn.Module, df: pd.DataFrame, imu_columns: Sequence[str], device: torch.device, slice_len: int = 256):
    """Predict per-time-step micro-label probabilities for a single stream."""
    model.eval()
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    probs = np.zeros((n, len(cb.MICRO_LABELS)), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, slice_len // 2):
            end = min(start + slice_len, n)
            xi = x[start:end]
            if len(xi) < slice_len:
                pad = slice_len - len(xi)
                xi = np.pad(xi, ((0, pad), (0, 0)), mode="constant")
            x_t = torch.from_numpy(xi).unsqueeze(0).to(device)  # [1, T, C]
            logits = model(x_t)  # [1, T, num_classes]
            logits = logits[0, :end - start, :].cpu().numpy()
            local_probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

            for i in range(end - start):
                idx = start + i
                if idx < n:
                    if probs[idx].sum() == 0:
                        probs[idx] = local_probs[i]
                    else:
                        probs[idx] = (probs[idx] + local_probs[i]) / 2.0

    return probs


# ---------------------------------------------------------------------------
# LOSO Evaluation
# ---------------------------------------------------------------------------

def run_loso(
    config_path: Path,
    output_dir: Path,
    subjects: List[str] | None = None,
    slice_len: int = 256,
    train_stride: int = 64,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    smoothing_window: int = 15,
) -> Dict:
    """Run full LOSO evaluation."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))

    print(f"[INFO] Loading streams...")
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)

    if subjects is None:
        subjects = sorted(set(all_subjects))

    print(f"[INFO] BiLSTM | subjects={subjects} | actions={available_actions}")
    print(f"[INFO] slice_len={slice_len}, epochs={epochs}, lr={lr}, hidden={hidden_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for test_subject in subjects:
        fold_file = output_dir / f"fold_{test_subject}.json"
        if fold_file.exists():
            print(f"[Resume] Loading {fold_file}")
            with open(fold_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        train_streams = [(sid, df) for sid, df in all_streams
                        if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams
                       if sid.split("/")[0] == test_subject]

        if not test_streams:
            print(f"[Skip] No test streams for {test_subject}")
            continue

        print(f"\n[Fold] test={test_subject} train={len(train_streams)} test={len(test_streams)}")

        t0 = time.time()
        train_ds = StreamSliceDataset(train_streams, imu_columns, slice_len, train_stride)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        print(f"  Dataset: {len(train_ds)} slices")

        model = BiLSTMModel(
            input_dim=len(imu_columns),
            num_classes=len(cb.MICRO_LABELS),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        train_model(model, train_loader, epochs=epochs, lr=lr, weight_decay=weight_decay, device=device)
        train_time = time.time() - t0

        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_streams, start=1):
            probs = predict_stream(model, df, imu_columns, device, slice_len=slice_len)
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 10 == 0 or stream_idx == len(test_streams):
                print(f"  Predicted {stream_idx}/{len(test_streams)} test streams", flush=True)

        # Apply smoothing
        smoothed = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smoothing_window) > 1:
                smooth = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smoothing_window) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    count = i - start + 1
                    smooth[i] = total / float(count)
                cur = smooth
            smoothed.append((stream_id, df, cur))

        # Evaluate
        actions = cb._available_actions(
            Path(data_cfg.get("data_dir", "./datasets/raw_data")),
            data_cfg.get("include_actions"),
        )
        macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]

        def predict_fn_from_cache(df, _cache_iter=iter(smoothed)):
            sid, _, cached_probs = next(_cache_iter)
            return cached_probs, None

        results = cb.evaluate_all_streams(
            predict_fn_from_cache, test_streams, macro_classes, mm_cfg
        )
        results["model_name"] = "BiLSTM (7-fold LOSO)"
        results["evaluation_protocol"] = "loso"
        results["test_subject"] = test_subject
        results["train_time_s"] = train_time
        results["config"] = {
            "slice_len": int(slice_len), "train_stride": int(train_stride),
            "smoothing_window": int(smoothing_window), "epochs": int(epochs),
            "lr": float(lr), "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers), "dropout": float(dropout),
        }

        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved: {fold_file} | Rep F1={results.get('rep_f1', 0):.4f}")
        all_results.append(results)

    if not all_results:
        return {"error": "No results"}

    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    summary = {
        "model": "BiLSTM (7-fold LOSO)",
        "n_folds": len(all_results),
        "subjects": subjects,
        "overall": {
            "precision": p, "recall": r, "rep_f1": f1,
            "n_true": sum(r["n_true"] for r in all_results),
            "n_pred": sum(r["n_pred"] for r in all_results),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"BILSTM GRAND SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary["overall"], indent=2))
    print(f"\n[OK] Results saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", default="",
                        help="Comma-separated subjects (default: all). Use subset for smoke test.")
    parser.add_argument("--slice-len", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--smoothing-window", type=int, default=15)
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None

    run_loso(
        Path(args.config),
        Path(args.output),
        subjects=subjects,
        slice_len=args.slice_len,
        train_stride=args.train_stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()
