"""Train a causal CNN active segmenter on set+rest timelines.

This tests whether a temporal model trained on full timelines can solve the
active/rest gate better than RF window classifiers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import load_non_action_streams  # noqa: E402
from scripts.evaluate_periodic_active_gate_loso import (  # noqa: E402
    active_segments,
    append_rest_tail,
    predict_active_prob as predict_rf_active_prob,
    state_machine,
    train_gate,
)
from scripts.new_c_pipeline.compare_phase_models import PhaseCompareConfig  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import stream_subject  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import CausalCNN_PhaseOnly, EXCLUDED_SESSIONS, set_seed, should_exclude  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


ACTIVE_PHASES = {"concentric", "eccentric"}


def active_labels(df: pd.DataFrame) -> np.ndarray:
    if "phase" not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    phases = df["phase"].astype(str).to_numpy()
    return np.asarray([1 if phase in ACTIVE_PHASES else 0 for phase in phases], dtype=np.int64)


def compute_norm(streams, imu_columns):
    values = [df[list(imu_columns)].to_numpy(dtype=np.float32) for _sid, df in streams if len(df)]
    stacked = np.concatenate(values, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def slice_starts(n: int, length: int, stride: int) -> list[int]:
    if n <= 0:
        return []
    if n <= length:
        return [0]
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return starts


class ActiveTimelineDataset(Dataset):
    def __init__(self, streams, imu_columns, mean, std, seq_len: int, stride: int):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _sid, df in streams:
            if not set(imu_columns).issubset(df.columns) or len(df) == 0:
                continue
            x = ((df[list(imu_columns)].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
            y = active_labels(df)
            pad = max(0, seq_len - len(x))
            if pad:
                x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
                y = np.pad(y, (0, pad), constant_values=0)
            for start in slice_starts(len(df), seq_len, stride):
                self.samples.append((x[start : start + seq_len], y[start : start + seq_len].astype(np.int64)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return torch.from_numpy(x.T).float(), torch.from_numpy(y).long()


def train_model(train_streams, imu_columns, args, device):
    mean, std = compute_norm(train_streams, imu_columns)
    ds = ActiveTimelineDataset(train_streams, imu_columns, mean, std, args.seq_len, args.train_stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = CausalCNN_PhaseOnly(6, args.hidden, 2, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = 0
    total = 0
    for _x, y in loader:
        pos += int((y == 1).sum())
        total += int(y.numel())
    neg = max(1, total - pos)
    pos_weight = min(float(neg / max(1, pos)), args.max_pos_weight)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"    active epoch {epoch + 1}/{args.epochs}: loss={np.mean(losses):.4f}", flush=True)
    return model, mean, std, {"train_slices": len(ds), "pos_rate": float(pos / max(1, total)), "pos_weight": pos_weight}


def predict_active_prob(model, df, imu_columns, mean, std, args, device) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    x = ((df[list(imu_columns)].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    prob_accum = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    pad = max(0, args.seq_len - n)
    padded = np.pad(x, ((0, pad), (0, 0)), mode="edge") if pad else x
    model.eval()
    with torch.no_grad():
        for start in slice_starts(n, args.seq_len, args.eval_stride):
            window = padded[start : start + args.seq_len]
            logits = model(torch.from_numpy(window.T).float().unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()[0, 1]
            end = min(start + args.seq_len, n)
            valid_len = end - start
            prob_accum[start:end] += probs[:valid_len]
            counts[start:end] += 1.0
    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts


def mask_metrics(gt: np.ndarray, pred: np.ndarray, sample_rate_hz: float):
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred.astype(np.int64), average="binary", zero_division=0)
    false_active = np.logical_and(gt == 0, pred)
    missed = np.logical_and(gt == 1, ~pred)
    return {
        "samples": int(len(gt)),
        "active_samples": int(np.sum(gt == 1)),
        "pred_active_samples": int(np.sum(pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(gt, pred.astype(np.int64), average="macro", zero_division=0)),
        "false_active_sec": float(np.sum(false_active) / sample_rate_hz),
        "missed_active_sec": float(np.sum(missed) / sample_rate_hz),
    }


def verify_active_segments(cnn_mask: np.ndarray, verifier_prob: np.ndarray, args) -> np.ndarray:
    if args.hybrid_verifier_mode == "none" or len(cnn_mask) == 0:
        return cnn_mask
    out = np.zeros_like(cnn_mask, dtype=bool)
    threshold = float(args.hybrid_min_prob)
    min_fraction = float(args.hybrid_min_fraction)
    min_mean = float(args.hybrid_min_mean)
    for start, end in active_segments(cnn_mask):
        segment_prob = verifier_prob[start:end]
        if len(segment_prob) == 0:
            continue
        fraction = float(np.mean(segment_prob >= threshold))
        mean_prob = float(np.mean(segment_prob))
        if fraction >= min_fraction and mean_prob >= min_mean:
            out[start:end] = True
    return out


def aggregate(rows, sample_rate_hz: float):
    if not rows:
        return {}
    total_samples = int(sum(row["samples"] for row in rows))
    false_sec = float(sum(row["false_active_sec"] for row in rows))
    missed_sec = float(sum(row["missed_active_sec"] for row in rows))
    return {
        "streams": len(rows),
        "duration_min": float(total_samples / sample_rate_hz / 60.0),
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "false_active_sec": false_sec,
        "missed_active_sec": missed_sec,
        "false_active_per_min": float(false_sec / max(total_samples / sample_rate_hz / 60.0, 1e-8)),
        "missed_active_per_min": float(missed_sec / max(total_samples / sample_rate_hz / 60.0, 1e-8)),
    }


def eval_stream(stream_id, df, model, mean, std, cfg, args, device, verifier=None):
    prob = predict_active_prob(model, df, cfg.imu_columns, mean, std, args, device)
    pred = state_machine(prob, args)
    verifier_stats = {}
    if verifier is not None:
        clf, scaler, mode = verifier
        verifier_prob = predict_rf_active_prob(df, cfg.imu_columns, clf, scaler, args, mode)
        pred = verify_active_segments(pred, verifier_prob, args)
        verifier_stats = {
            "verifier_mean_probability": float(np.mean(verifier_prob)) if len(verifier_prob) else 0.0,
            "verifier_max_probability": float(np.max(verifier_prob)) if len(verifier_prob) else 0.0,
        }
    gt = active_labels(df)
    row = mask_metrics(gt, pred, args.sample_rate_hz)
    row.update({"stream_id": stream_id, "segments": len(active_segments(pred)), "max_prob": float(np.max(prob)) if len(prob) else 0.0})
    row.update(verifier_stats)
    return row, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN active segmenter on full timelines.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/active_cnn_full_timeline_loso/summary.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--train-stride", type=int, default=150)
    parser.add_argument("--eval-stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=4.0)
    parser.add_argument("--hybrid-verifier-mode", choices=["none", "basic", "periodic"], default="none")
    parser.add_argument("--hybrid-min-prob", type=float, default=0.65)
    parser.add_argument("--hybrid-min-fraction", type=float, default=0.25)
    parser.add_argument("--hybrid-min-mean", type=float, default=0.45)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=50)
    parser.add_argument("--window-active-fraction", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--label-mode", choices=["binary", "tri_motion"], default="binary")
    parser.add_argument("--transition-energy-quantile", type=float, default=0.7)
    parser.add_argument("--train-rest-tail-seconds", type=float, default=20.0)
    parser.add_argument("--rest-tail-seconds", type=float, default=20.0)
    parser.add_argument("--enter-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=0.45)
    parser.add_argument("--enter-hold-samples", type=int, default=50)
    parser.add_argument("--exit-hold-samples", type=int, default=100)
    parser.add_argument("--min-active-samples", type=int, default=200)
    parser.add_argument("--bridge-gap-samples", type=int, default=50)
    parser.add_argument("--cooldown-samples", type=int, default=0)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-rest-streams-per-fold", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = PhaseCompareConfig()
    all_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    subjects = sorted({stream_subject(sid) for sid, _df in set_streams})
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects
    data_dir = Path(raw_cfg.get("data", {}).get("data_dir", "datasets/raw_data"))
    print(f"sets={len(set_streams)} rest={len(rest_streams)} subjects={subjects} device={device}", flush=True)

    set_rows_all = []
    rest_rows_all = []
    appended_rows_all = []
    folds = []
    for fold_idx, test_subject in enumerate(eval_subjects, start=1):
        train_set = [(sid, df) for sid, df in set_streams if stream_subject(sid) != test_subject]
        test_set = [(sid, df) for sid, df in set_streams if stream_subject(sid) == test_subject]
        train_rest = [(sid, df) for sid, df in rest_streams if stream_subject(sid) != test_subject]
        test_rest = [(sid, df) for sid, df in rest_streams if stream_subject(sid) == test_subject]
        if args.max_rest_streams_per_fold and args.max_rest_streams_per_fold > 0:
            test_rest = test_rest[: args.max_rest_streams_per_fold]

        train_streams = [*train_rest]
        for stream_id, df in train_set:
            combined, _set_len, rest_len = append_rest_tail(stream_id, df, data_dir, args.train_rest_tail_seconds, args.sample_rate_hz, cfg.imu_columns)
            train_streams.append((f"{stream_id}+train_tail" if rest_len else stream_id, combined))
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject} train_streams={len(train_streams)}", flush=True)
        model, mean, std, train_info = train_model(train_streams, cfg.imu_columns, args, device)
        verifier = None
        if args.hybrid_verifier_mode != "none":
            clf, scaler, verifier_info = train_gate(train_streams, cfg.imu_columns, args, args.hybrid_verifier_mode)
            train_info["hybrid_verifier"] = verifier_info
            verifier = (clf, scaler, args.hybrid_verifier_mode)

        fold_set = []
        fold_rest = []
        fold_app = []
        for stream_id, df in test_set:
            row, _pred = eval_stream(stream_id, df, model, mean, std, cfg, args, device, verifier)
            fold_set.append(row)
            set_rows_all.append(row)
            combined, set_len, rest_len = append_rest_tail(stream_id, df, data_dir, args.rest_tail_seconds, args.sample_rate_hz, cfg.imu_columns)
            if rest_len > 0:
                app_row, app_pred = eval_stream(f"{stream_id}+rest_tail", combined, model, mean, std, cfg, args, device, verifier)
                rest_pred = app_pred[set_len:]
                app_row.update(
                    {
                        "stream_id": stream_id,
                        "set_samples": int(set_len),
                        "rest_samples": int(rest_len),
                        "rest_pred_active_samples": int(np.sum(rest_pred)),
                        "rest_tail_active_rate": float(np.mean(rest_pred)) if len(rest_pred) else 0.0,
                        "rest_tail_segments": len(active_segments(rest_pred)),
                    }
                )
                fold_app.append(app_row)
                appended_rows_all.append(app_row)
        for stream_id, df in test_rest:
            row, pred = eval_stream(stream_id, df, model, mean, std, cfg, args, device, verifier)
            row.update({"false_active_segments": len(active_segments(pred))})
            fold_rest.append(row)
            rest_rows_all.append(row)
        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "train": train_info,
            "set_summary": aggregate(fold_set, args.sample_rate_hz),
            "rest_summary": aggregate(fold_rest, args.sample_rate_hz),
            "appended_summary": aggregate(fold_app, args.sample_rate_hz),
            "rest_false_active_segments": int(sum(row.get("false_active_segments", 0) for row in fold_rest)),
            "appended_rest_tail_segments": int(sum(row.get("rest_tail_segments", 0) for row in fold_app)),
        }
        folds.append(fold_summary)
        print(
            f"  setF1={fold_summary['set_summary'].get('mean_f1', 0):.3f} restFA/min={fold_summary['rest_summary'].get('false_active_per_min', 0):.2f} appRestRate={np.mean([r['rest_tail_active_rate'] for r in fold_app]) if fold_app else 0:.3f}",
            flush=True,
        )

    output = {
        "settings": vars(args),
        "excluded_sessions": EXCLUDED_SESSIONS,
        "set_total": aggregate(set_rows_all, args.sample_rate_hz),
        "rest_total": aggregate(rest_rows_all, args.sample_rate_hz),
        "appended_total": aggregate(appended_rows_all, args.sample_rate_hz),
        "rest_false_active_segments": int(sum(row.get("false_active_segments", 0) for row in rest_rows_all)),
        "appended_rest_tail_segments": int(sum(row.get("rest_tail_segments", 0) for row in appended_rows_all)),
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        f"TOTAL setF1={output['set_total'].get('mean_f1', 0):.3f} restFA/min={output['rest_total'].get('false_active_per_min', 0):.2f}",
        flush=True,
    )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
