from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_compare_baselines_module():
    path = ROOT / "scripts" / "compare_baselines.py"
    spec = importlib.util.spec_from_file_location("compare_baselines_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load compare_baselines module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_compare_baselines_module()

from models.ds_ms_tcn import SingleStageFeatures


def _make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def _autocast_context(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


class BoundarySliceDataset(Dataset):
    def __init__(self, streams, imu_columns: Sequence[str], slice_len: int, stride_len: int, boundary_radius: int = 2):
        self.items = []
        self.boundary_radius = max(0, int(boundary_radius))
        for _, df in streams:
            x = df[list(imu_columns)].to_numpy(dtype=np.float32)
            labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
            y = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
            for start in range(0, max(1, len(x) - slice_len + 1), stride_len):
                end = min(start + slice_len, len(x))
                xi = x[start:end]
                yi = y[start:end]
                bi = self._build_boundary_targets(yi)
                if len(xi) < slice_len:
                    pad = slice_len - len(xi)
                    xi = np.pad(xi, ((0, pad), (0, 0)), mode="constant")
                    yi = np.pad(yi, (0, pad), mode="constant", constant_values=-100)
                    bi = np.pad(bi, ((0, pad), (0, 0)), mode="constant")
                self.items.append((xi, yi, bi))

    def _mark(self, arr: np.ndarray, idx: int, ch: int):
        if idx < 0 or idx >= len(arr):
            return
        left = max(0, idx - self.boundary_radius)
        right = min(len(arr), idx + self.boundary_radius + 1)
        arr[left:right, ch] = 1.0

    def _build_boundary_targets(self, y: np.ndarray) -> np.ndarray:
        out = np.zeros((len(y), 3), dtype=np.float32)
        labels = [cb.MICRO_LABELS[int(v)] for v in y]
        runs = cb.labels_to_runs(labels, positive_labels=(cb.CONCENTRIC_LABEL, cb.ECCENTRIC_LABEL), min_length=1)
        reps, _ = cb.pair_concentric_eccentric_reps(runs, micro_source="gt", max_gap_samples=0)
        for rep in reps:
            self._mark(out, int(rep.start_idx), 0)
            self._mark(out, int(rep.transition_idx), 1)
            self._mark(out, min(len(out) - 1, int(rep.end_idx)), 2)
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, b = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(b)


class BoundaryAwarePhaseTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_filters: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.features = SingleStageFeatures(
            input_channels=input_dim,
            num_filters=num_filters,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
        )
        self.phase_head = nn.Conv1d(num_filters, num_classes, kernel_size=1)
        self.boundary_head = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x)  # [B,T,F]
        feat_t = feat.transpose(1, 2)
        phase_logits = self.phase_head(feat_t).transpose(1, 2)
        boundary_logits = self.boundary_head(feat_t).transpose(1, 2)
        return phase_logits, boundary_logits


def _phase_class_weights(train_streams, imu_columns: Sequence[str]) -> torch.Tensor:
    counts = np.zeros(len(cb.MICRO_LABELS), dtype=np.float64)
    for _, df in train_streams:
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        counts += np.bincount(idx, minlength=len(cb.MICRO_LABELS)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model: nn.Module, train_streams, imu_columns: Sequence[str], slice_len: int, stride_len: int, epochs: int, lr: float, device: torch.device, boundary_loss_weight: float, boundary_focus: float, boundary_radius: int):
    ds = BoundarySliceDataset(train_streams, imu_columns, slice_len, stride_len, boundary_radius=boundary_radius)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    model.to(device)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    class_weights = _phase_class_weights(train_streams, imu_columns).to(device)
    pos_weight = torch.tensor([10.0, 6.0, 10.0], dtype=torch.float32, device=device)

    print(f"  [BoundaryTCN] Training on {len(ds)} slices, {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for x, y, b in loader:
            x = x.to(device)
            y = y.to(device)
            b = b.to(device)
            optimizer.zero_grad()
            with _autocast_context(use_amp):
                phase_logits, boundary_logits = model(x)
                ce = F.cross_entropy(
                    phase_logits.reshape(-1, phase_logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                    weight=class_weights,
                ).reshape(y.shape)
                valid = (y != -100).float()
                boundary_any = (b.sum(dim=-1) > 0).float()
                weights = 1.0 + boundary_focus * boundary_any
                phase_loss = (ce * weights * valid).sum() / torch.clamp((weights * valid).sum(), min=1.0)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits,
                    b,
                    pos_weight=pos_weight,
                    reduction="none",
                )
                boundary_loss = (boundary_loss * valid.unsqueeze(-1)).sum() / torch.clamp(valid.sum() * b.shape[-1], min=1.0)
                loss = phase_loss + boundary_loss_weight * boundary_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * x.size(0)
            count += x.size(0)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [BoundaryTCN] epoch={epoch:03d}/{epochs:03d} loss={total_loss / max(1, count):.4f}")


def _segment_causal_smooth(probs: np.ndarray, boundary_prob: np.ndarray, smoothing_window: int, boundary_threshold: float) -> np.ndarray:
    if smoothing_window <= 1:
        return probs
    smoothed = np.zeros_like(probs)
    boundary_strength = np.max(boundary_prob, axis=1)
    split_points = [i for i, v in enumerate(boundary_strength) if v >= boundary_threshold]
    segment_starts = [0]
    for idx in split_points:
        if idx > segment_starts[-1]:
            segment_starts.append(idx)
    segment_starts.append(len(probs))
    for si in range(len(segment_starts) - 1):
        start = segment_starts[si]
        end = segment_starts[si + 1]
        chunk = probs[start:end]
        if len(chunk) == 0:
            continue
        csum = np.cumsum(chunk, axis=0)
        for i in range(len(chunk)):
            local_start = max(0, i - smoothing_window + 1)
            total = csum[i] - (csum[local_start - 1] if local_start > 0 else 0.0)
            smoothed[start + i] = total / float(i - local_start + 1)
    return smoothed


def predict_model(model: nn.Module, df, imu_columns: Sequence[str], device: torch.device, smoothing_window: int = 15, boundary_threshold: float = 0.5):
    model.eval()
    x = torch.from_numpy(df[list(imu_columns)].to_numpy(dtype=np.float32))[None].to(device)
    with torch.no_grad():
        phase_logits, boundary_logits = model(x)
    phase_probs = torch.softmax(phase_logits, dim=-1).cpu().numpy()[0]
    boundary_prob = torch.sigmoid(boundary_logits).cpu().numpy()[0]
    phase_probs = _segment_causal_smooth(phase_probs, boundary_prob, smoothing_window, boundary_threshold)
    return phase_probs, None


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a boundary-aware phase TCN for rep cutting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/boundary_tcn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-focus", type=float, default=2.0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = raw.get("data", {}) or {}
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    train_cfg = cb.TrainConfig(**{k: v for k, v in train_raw.items() if k in cb.TrainConfig.__dataclass_fields__})
    cb.set_seed(train_cfg.seed)

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    actions = cb._available_actions(Path(data_cfg.get("data_dir", "./datasets/raw_data")), data_cfg.get("include_actions"))
    macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    subjects_sorted = sorted(set(subjects))
    configured_test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else subjects_sorted[-1]
    train_all_subjects = cb._is_all_subjects_mode(configured_test_subject)
    if train_all_subjects:
        test_subject = "__all__"
        train_streams = list(streams)
        test_streams = list(streams)
        evaluation_protocol = "train_all_in_sample"
    else:
        test_subject = configured_test_subject
        train_subjects = [s for s in subjects_sorted if s != test_subject]
        train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
        test_streams = cb._filter_subjects(streams, [test_subject], subject_column)
        evaluation_protocol = "subject_holdout"

    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    sample_rate = cb._median_sample_rate(train_streams, float(target_sample_rate))
    slice_len = max(8, int(round(float(mm_cfg.slice_seconds) * sample_rate)))
    stride_len = max(1, int(round(slice_len * (1.0 - float(mm_cfg.overlap)))))
    device = cb._resolve_device(train_cfg.device)

    print(f"[INFO] protocol={evaluation_protocol} test_subject={test_subject} train={len(train_streams)} test={len(test_streams)} device={device}")
    print(f"[INFO] slice_len={slice_len} stride_len={stride_len} sample_rate={sample_rate:.1f}")

    model = BoundaryAwarePhaseTCN(
        input_dim=len(imu_columns),
        num_classes=len(cb.MICRO_LABELS),
        num_filters=int(mm_cfg.num_filters),
        num_layers=int(mm_cfg.num_layers),
        kernel_size=int(mm_cfg.kernel_size),
        dropout=float(mm_cfg.dropout),
        causal=bool(args.causal),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] model=BoundaryAwarePhaseTCN params={n_params:,} causal={bool(args.causal)}")

    t0 = time.time()
    train_model(
        model,
        train_streams,
        imu_columns,
        slice_len,
        stride_len,
        epochs=int(args.epochs),
        lr=float(args.lr),
        device=device,
        boundary_loss_weight=float(args.boundary_loss_weight),
        boundary_focus=float(args.boundary_focus),
        boundary_radius=int(args.boundary_radius),
    )
    train_time = time.time() - t0

    def predict_fn(df):
        return predict_model(
            model,
            df,
            imu_columns,
            device,
            smoothing_window=int(args.smoothing_window),
            boundary_threshold=float(args.boundary_threshold),
        )

    t0 = time.time()
    results = cb.evaluate_all_streams(predict_fn, test_streams, macro_classes, mm_cfg)
    eval_time = time.time() - t0
    results["train_time_s"] = train_time
    results["eval_time_s"] = eval_time
    results["params"] = n_params
    results["model_name"] = "Boundary-Aware Phase TCN"
    results["evaluation_protocol"] = evaluation_protocol
    results["test_subject"] = test_subject
    results["config"] = {
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "boundary_loss_weight": float(args.boundary_loss_weight),
        "boundary_focus": float(args.boundary_focus),
        "boundary_radius": int(args.boundary_radius),
        "smoothing_window": int(args.smoothing_window),
        "boundary_threshold": float(args.boundary_threshold),
        "causal": bool(args.causal),
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    md_lines = [
        "# Boundary-Aware Phase TCN Results\n",
        f"Protocol: `{evaluation_protocol}`\n",
        f"Test subject: `{test_subject}`\n",
        f"Rep F1: `{results.get('rep_f1'):.4f}`\n",
        f"Precision: `{results.get('precision'):.4f}`\n",
        f"Recall: `{results.get('recall'):.4f}`\n",
        f"Start MAE: `{results.get('start_mae_ms'):.4f}`\n",
        f"End MAE: `{results.get('end_mae_ms'):.4f}`\n",
        f"Transition MAE: `{results.get('transition_mae_ms'):.4f}`\n",
        f"micro_f1@50: `{results.get('micro_f1_at_50'):.4f}`\n",
    ]
    (out_dir / "results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({k: results[k] for k in ["rep_f1", "precision", "recall", "start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_50"]}, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
