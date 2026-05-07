"""DS-MS-TCN micro/macro recognition and rep segmentation.

Usage:
  python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn
  python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source dtw
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from models.ds_ms_tcn import DSMSTCN, DSMSTCNConfig, ds_ms_tcn_loss
from evaluation.reporting import primary_metric_table, write_run_manifest
from preprocessing.dtw_micro_adapter import (
    DTWMicroConfig,
    detect_dtw_micro_runs,
    dtw_runs_to_micro_scores,
    fit_dtw_micro_templates,
)
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    OTHER_LABEL,
    aggregate_action_for_reps,
    diagnostics_to_rows,
    edit_score,
    labels_to_runs,
    macro_labels_from_action,
    match_segments,
    micro_labels_from_phase,
    pair_concentric_eccentric_reps,
    rep_metrics,
    reps_to_rows,
    sample_classification_metrics,
    segment_iou_f1,
    truth_reps_from_labels,
    write_action_prediction_svg,
    write_micro_macro_svg,
)
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz
from preprocessing.window_pipeline import ZScoreStats, apply_zscore, compute_train_stats, set_seed


@dataclass
class MicroMacroConfig:
    slice_seconds: float = 40.0
    overlap: float = 0.50
    num_filters: int = 64
    num_layers: int = 9
    kernel_size: int = 3
    dropout: float = 0.2
    causal: bool = True
    alpha: float = 1.0
    beta: float = 0.15
    tmse_threshold: float = 4.0
    max_phase_gap_samples: int = 0
    min_phase_samples: int = 3
    plot_max_streams: int = 24
    train_on_modes: List[str] = field(default_factory=lambda: ["sets", "whole"])


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 30
    lr: float = 0.0001
    weight_decay: float = 0.00001
    test_subject: str | None = None
    num_workers: int | str = "auto"
    device: str = "auto"
    pin_memory: bool | str = "auto"
    amp: bool = True


def _resolve_device(device_setting: str) -> torch.device:
    requested = str(device_setting).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("[WARN] MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_num_workers(num_workers: int | str, device: torch.device) -> int:
    if str(num_workers).lower() != "auto":
        return max(0, int(num_workers))
    cpu_count = os.cpu_count() or 1
    if device.type == "cpu":
        return max(0, min(2, cpu_count - 1))
    return max(0, min(4, cpu_count - 1))


def _resolve_pin_memory(pin_memory: bool | str, device: torch.device) -> bool:
    if isinstance(pin_memory, str) and pin_memory.lower() == "auto":
        return device.type == "cuda"
    return bool(pin_memory)


class SequenceSliceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[pd.DataFrame],
        imu_columns: Sequence[str],
        macro_classes: Sequence[str],
        slice_len: int,
        stride_len: int,
        use_gt_micro_probs: bool = False,
    ) -> None:
        self.items: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.imu_columns = list(imu_columns)
        self.macro_to_idx = {str(c): i for i, c in enumerate(macro_classes)}
        self.micro_to_idx = {label: i for i, label in enumerate(MICRO_LABELS)}
        self.slice_len = int(slice_len)
        self.use_gt_micro_probs = bool(use_gt_micro_probs)

        for seq in sequences:
            if "phase" not in seq.columns or "action_type" not in seq.columns:
                continue
            x = seq[self.imu_columns].to_numpy(dtype=np.float32)
            micro_labels = micro_labels_from_phase(seq["phase"].to_numpy())
            macro_labels = macro_labels_from_action(seq["action_type"].astype(str).to_numpy(), micro_labels)
            micro_idx = np.asarray([self.micro_to_idx[str(v)] for v in micro_labels], dtype=np.int64)
            macro_idx = np.asarray([self.macro_to_idx.get(str(v), self.macro_to_idx[OTHER_LABEL]) for v in macro_labels], dtype=np.int64)
            n = len(seq)
            starts = list(range(0, max(1, n - self.slice_len + 1), max(1, int(stride_len))))
            if not starts or starts[-1] + self.slice_len < n:
                starts.append(max(0, n - self.slice_len))
            for start in sorted(set(starts)):
                end = min(n, start + self.slice_len)
                self.items.append(self._make_item(x[start:end], micro_idx[start:end], macro_idx[start:end]))

    def _make_item(self, x: np.ndarray, micro: np.ndarray, macro: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid = len(x)
        if valid < self.slice_len:
            pad = self.slice_len - valid
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            micro = np.pad(micro, (0, pad), mode="constant", constant_values=-100)
            macro = np.pad(macro, (0, pad), mode="constant", constant_values=-100)
        gt_micro_probs = np.zeros((self.slice_len, len(MICRO_LABELS)), dtype=np.float32)
        for i, idx in enumerate(micro):
            if idx >= 0:
                gt_micro_probs[i, int(idx)] = 1.0
            else:
                gt_micro_probs[i, MICRO_LABELS.index(OTHER_LABEL)] = 1.0
        return x.astype(np.float32), micro.astype(np.int64), macro.astype(np.int64), gt_micro_probs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        x, micro, macro, gt_micro_probs = self.items[idx]
        return (
            torch.from_numpy(x),
            torch.from_numpy(micro),
            torch.from_numpy(macro),
            torch.from_numpy(gt_micro_probs),
        )


def _matches_any(path: Path, base_dir: Path, patterns: Sequence[str]) -> bool:
    try:
        parts = path.relative_to(base_dir).parts
    except ValueError:
        parts = path.parts
    return any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in patterns)


def _natural_key(path: Path) -> List[int | str]:
    import re
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", path.stem)]


def _subject_dirs(data_dir: Path) -> List[Path]:
    return [p for p in sorted(data_dir.iterdir()) if p.is_dir()]


def _load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _available_actions(data_dir: Path, include_actions: Sequence[str] | None) -> List[str]:
    if include_actions:
        return [str(a) for a in include_actions]
    actions = sorted({p.name for subj in _subject_dirs(data_dir) for p in subj.iterdir() if p.is_dir() and "rest" not in p.name})
    return actions


def _load_set_sequences(
    data_dir: Path,
    subject: str,
    action: str,
    exclude_patterns: Sequence[str],
) -> List[Tuple[str, pd.DataFrame]]:
    action_dir = data_dir / subject / action
    if not action_dir.exists():
        return []
    streams: List[Tuple[str, pd.DataFrame]] = []
    for set_dir in sorted(action_dir.iterdir()):
        if not set_dir.is_dir() or not set_dir.name.startswith("set"):
            continue
        if _matches_any(set_dir, data_dir, exclude_patterns):
            continue
        frames = []
        for csv_path in sorted(set_dir.glob("*.csv"), key=_natural_key):
            if _matches_any(csv_path, data_dir, exclude_patterns):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if "phase" not in df.columns:
                continue
            df = df.copy()
            if "action_type" not in df.columns:
                df["action_type"] = action
            if "subject_id" not in df.columns:
                df["subject_id"] = subject
            df["_split_subject"] = subject
            df["_source_file"] = csv_path.name
            frames.append(df)
        if frames:
            streams.append((f"{subject}/{action}/{set_dir.name}", pd.concat(frames, ignore_index=True)))
    return streams


def _load_whole_sequences(
    data_dir: Path,
    subject: str,
    include_actions: Sequence[str],
) -> List[Tuple[str, pd.DataFrame]]:
    streams = []
    allowed = set(str(a) for a in include_actions)
    for csv_path in sorted((data_dir / subject).glob("*whole_session*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if not {"phase", "action_type"}.issubset(df.columns):
            continue
        df = df.copy()
        if "subject_id" not in df.columns:
            df["subject_id"] = subject
        df["_split_subject"] = subject
        df.loc[~df["action_type"].astype(str).isin(allowed), "action_type"] = OTHER_LABEL
        streams.append((f"{subject}/{csv_path.stem}", df))
    return streams


def _load_streams(raw_cfg: Dict, modes: Sequence[str]) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str], List[str]]:
    data_cfg = raw_cfg.get("data", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])
    actions = _available_actions(data_dir, data_cfg.get("include_actions"))
    subjects = [p.name for p in _subject_dirs(data_dir)]
    streams: List[Tuple[str, pd.DataFrame]] = []
    for subject in subjects:
        if "sets" in modes:
            for action in actions:
                streams.extend(_load_set_sequences(data_dir, subject, action, exclude_patterns))
        if "whole" in modes:
            streams.extend(_load_whole_sequences(data_dir, subject, actions))
    return streams, subjects, actions


def _filter_subjects(streams: Sequence[Tuple[str, pd.DataFrame]], subjects: Sequence[str], subject_column: str) -> List[Tuple[str, pd.DataFrame]]:
    allowed = set(str(s) for s in subjects)
    out = []
    for stream_id, df in streams:
        split_subject = str(df.iloc[0]["_split_subject"]) if "_split_subject" in df.columns else str(df.iloc[0][subject_column])
        if split_subject in allowed:
            out.append((stream_id, df))
    return out


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-8)


def _predict_full_sequence(
    model: DSMSTCN,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    device: torch.device,
    external_micro: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    x = torch.from_numpy(df[list(imu_columns)].to_numpy(dtype=np.float32))[None, :, :].to(device)
    ext = torch.from_numpy(external_micro.astype(np.float32))[None, :, :].to(device) if external_micro is not None else None
    with torch.no_grad():
        out = model(x, external_micro_probs=ext)
    return {
        "micro_probs": out["micro_probs"].detach().cpu().numpy()[0],
        "macro4_probs": torch.softmax(out["macro4_logits"], dim=-1).detach().cpu().numpy()[0],
    }


def _median_sample_rate(streams: Sequence[Tuple[str, pd.DataFrame]], fallback_hz: float) -> float:
    rates = []
    for _, df in streams:
        try:
            rate = float(infer_sample_rate_hz(df))
        except Exception:
            continue
        if np.isfinite(rate) and rate > 0:
            rates.append(rate)
    return float(np.median(rates)) if rates else float(fallback_hz)


def _macro_runs_from_probs(probs: np.ndarray, macro_classes: Sequence[str], min_length: int = 1):
    labels = [macro_classes[int(i)] for i in np.argmax(probs, axis=1)]
    return labels_to_runs(labels, positive_labels=None, probabilities=None, min_length=min_length)


def _classification_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
    label_list = [str(x) for x in labels]
    matrix = pd.DataFrame(0, index=label_list, columns=label_list, dtype=int)
    for true, pred in zip(y_true, y_pred):
        true_s = str(true) if str(true) in matrix.index else OTHER_LABEL
        pred_s = str(pred) if str(pred) in matrix.columns else OTHER_LABEL
        matrix.loc[true_s, pred_s] += 1
    acc = float(np.trace(matrix.to_numpy()) / max(1, matrix.to_numpy().sum()))
    return {"accuracy": acc, "confusion_matrix": matrix}


def _train_model(
    model: DSMSTCN,
    loader: DataLoader,
    cfg: TrainConfig,
    mm_cfg: MicroMacroConfig,
    use_gt_micro_probs: bool,
) -> None:
    device = torch.device(cfg.device)
    model.to(device)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for x, micro, macro, gt_micro_probs in loader:
            non_blocking = device.type == "cuda"
            x = x.to(device, non_blocking=non_blocking)
            micro = micro.to(device, non_blocking=non_blocking)
            macro = macro.to(device, non_blocking=non_blocking)
            gt_micro_probs = gt_micro_probs.to(device, non_blocking=non_blocking)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, external_micro_probs=gt_micro_probs if use_gt_micro_probs else None)
                losses = ds_ms_tcn_loss(
                    out,
                    micro,
                    macro,
                    alpha=mm_cfg.alpha,
                    beta=mm_cfg.beta,
                    tmse_threshold=mm_cfg.tmse_threshold,
                    include_micro_loss=not use_gt_micro_probs,
                )
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(losses["loss"].detach().cpu()) * len(x)
            count += len(x)
        print(f"[INFO] epoch={epoch:03d} loss={total / max(1, count):.4f}")


def _evaluate_streams(
    model: DSMSTCN,
    streams: Sequence[Tuple[str, pd.DataFrame]],
    train_sequences_for_dtw: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    macro_classes: Sequence[str],
    micro_source: str,
    mm_cfg: MicroMacroConfig,
    dtw_cfg: DTWMicroConfig,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, object]:
    dtw_templates = {}
    if micro_source == "dtw":
        dtw_templates = fit_dtw_micro_templates(train_sequences_for_dtw, imu_columns, dtw_cfg)
        print(f"[INFO] DTW templates: {sorted(dtw_templates)}")

    pred_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    true_actions: List[str] = []
    pred_actions: List[str] = []
    plot_count = 0

    for stream_idx, (stream_id, df) in enumerate(streams, start=1):
        sample_rate = infer_sample_rate_hz(df)
        if micro_source == "dtw":
            print(
                f"[INFO] DTW eval {stream_idx}/{len(streams)} stream={stream_id} "
                f"samples={len(df)} stride={dtw_cfg.detection_stride} "
                f"duration_stride={dtw_cfg.duration_stride or dtw_cfg.detection_stride} "
                f"downsample={dtw_cfg.dtw_downsample_factor} "
                f"max_windows_per_label={dtw_cfg.max_windows_per_label}",
                flush=True,
            )
            dtw_runs, dtw_stats = detect_dtw_micro_runs(
                df,
                dtw_templates,
                imu_columns,
                dtw_cfg,
                return_stats=True,
            )
            print(
                f"[INFO] DTW eval {stream_idx}/{len(streams)} stream={stream_id} "
                f"windows={dtw_stats.get('windows_scored', 0)} "
                f"candidates={dtw_stats.get('candidates', 0)} kept={dtw_stats.get('kept', 0)}",
                flush=True,
            )
            external = dtw_runs_to_micro_scores(len(df), dtw_runs)
            pred = _predict_full_sequence(model, df, imu_columns, device, external_micro=external)
        else:
            print(f"[INFO] TCN eval {stream_idx}/{len(streams)} stream={stream_id} samples={len(df)}", flush=True)
            pred = _predict_full_sequence(model, df, imu_columns, device)
        micro_probs = pred["micro_probs"]
        macro_probs = pred["macro4_probs"]
        pred_micro_labels = [MICRO_LABELS[int(i)] for i in np.argmax(micro_probs, axis=1)]
        gt_micro_labels = micro_labels_from_phase(df["phase"].to_numpy())
        gt_macro_labels = macro_labels_from_action(
            df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else [OTHER_LABEL] * len(df),
            gt_micro_labels,
        )
        pred_macro_labels = [macro_classes[int(i)] for i in np.argmax(macro_probs, axis=1)]
        pred_micro_runs = labels_to_runs(
            pred_micro_labels,
            positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
            probabilities=micro_probs,
            min_length=mm_cfg.min_phase_samples,
        )
        pred_reps, diagnostics = pair_concentric_eccentric_reps(
            pred_micro_runs,
            micro_source=micro_source,
            max_gap_samples=mm_cfg.max_phase_gap_samples,
        )
        pred_reps = aggregate_action_for_reps(pred_reps, macro_probs, macro_classes)
        truth_reps = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        gt_micro_runs = labels_to_runs(
            gt_micro_labels,
            positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
            min_length=mm_cfg.min_phase_samples,
        )
        gt_macro_runs = labels_to_runs(
            gt_macro_labels,
            positive_labels=[label for label in macro_classes if label != OTHER_LABEL],
            min_length=mm_cfg.min_phase_samples,
        )
        pred_macro_runs = _macro_runs_from_probs(macro_probs, macro_classes, min_length=mm_cfg.min_phase_samples)
        metrics = rep_metrics(pred_reps, truth_reps, sample_rate_hz=sample_rate)
        micro_sample = sample_classification_metrics(gt_micro_labels, pred_micro_labels, MICRO_LABELS)
        macro_sample = sample_classification_metrics(gt_macro_labels, pred_macro_labels, macro_classes)
        micro_iou = segment_iou_f1(gt_micro_runs, pred_micro_runs)
        macro_iou = segment_iou_f1(gt_macro_runs, pred_macro_runs)
        metric_rows.append({
            "stream_id": stream_id,
            "micro_source": micro_source,
            "sample_rate_hz": sample_rate,
            **metrics,
            "micro_sample_accuracy": micro_sample["accuracy"],
            "micro_sample_macro_f1": micro_sample["macro_f1"],
            "macro_sample_accuracy": macro_sample["accuracy"],
            "macro_sample_macro_f1": macro_sample["macro_f1"],
            "micro_edit": edit_score(gt_micro_runs, pred_micro_runs),
            "macro_edit": edit_score(gt_macro_runs, pred_macro_runs),
            "micro_f1_at_10": micro_iou["f1_at_10"],
            "micro_f1_at_25": micro_iou["f1_at_25"],
            "micro_f1_at_50": micro_iou["f1_at_50"],
            "macro_f1_at_10": macro_iou["f1_at_10"],
            "macro_f1_at_25": macro_iou["f1_at_25"],
            "macro_f1_at_50": macro_iou["f1_at_50"],
        })
        pred_rows.extend(reps_to_rows(stream_id, pred_reps))
        diag_rows.extend(diagnostics_to_rows(stream_id, diagnostics))

        for pi, ti, _ in match_segments(
            [(r.start_idx, r.end_idx) for r in pred_reps],
            [(r.start_idx, r.end_idx) for r in truth_reps],
        ):
            true_actions.append(truth_reps[ti].pred_action_type)
            pred_actions.append(pred_reps[pi].pred_action_type)

        if plot_count < int(mm_cfg.plot_max_streams):
            plot_name = stream_id.replace("/", "_") + ".svg"
            rep_plot_path = output_dir / "plots" / "rep" / micro_source / plot_name
            action_plot_path = output_dir / "plots" / "action" / micro_source / plot_name
            write_micro_macro_svg(
                rep_plot_path,
                stream_id,
                df,
                gt_micro_runs,
                pred_micro_runs,
                truth_reps,
                pred_reps,
                pred_macro_runs,
                sample_rate,
            )
            write_action_prediction_svg(
                action_plot_path,
                stream_id,
                df,
                gt_macro_runs,
                pred_macro_runs,
                pred_reps,
                sample_rate,
            )
            plot_count += 1

    pred_df = pd.DataFrame(pred_rows)
    diag_df = pd.DataFrame(diag_rows)
    metrics_df = pd.DataFrame(metric_rows)
    pred_df.to_csv(output_dir / "detections" / f"rep_detections_{micro_source}.csv", index=False)
    diag_df.to_csv(output_dir / "detections" / f"pairing_diagnostics_{micro_source}.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics" / f"stream_metrics_{micro_source}.csv", index=False)

    action_summary = _classification_counts(true_actions, pred_actions, macro_classes + ["uncertain"])
    action_summary["confusion_matrix"].to_csv(output_dir / "metrics" / f"rep_action_confusion_{micro_source}.csv")

    overall = {}
    if not metrics_df.empty:
        for key in ("n_pred", "n_true", "tp", "fp", "fn"):
            overall[key] = float(metrics_df[key].sum())
        tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        overall.update({
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
            "start_mae_ms": float(metrics_df["start_mae_ms"].dropna().mean()) if "start_mae_ms" in metrics_df else float("nan"),
            "end_mae_ms": float(metrics_df["end_mae_ms"].dropna().mean()) if "end_mae_ms" in metrics_df else float("nan"),
            "transition_mae_ms": float(metrics_df["transition_mae_ms"].dropna().mean()) if "transition_mae_ms" in metrics_df else float("nan"),
            "rep_action_accuracy": float(action_summary["accuracy"]),
        })
        for key in (
            "micro_sample_accuracy",
            "micro_sample_macro_f1",
            "macro_sample_accuracy",
            "macro_sample_macro_f1",
            "micro_edit",
            "macro_edit",
            "micro_f1_at_10",
            "micro_f1_at_25",
            "micro_f1_at_50",
            "macro_f1_at_10",
            "macro_f1_at_25",
            "macro_f1_at_50",
        ):
            if key in metrics_df:
                overall[key] = float(metrics_df[key].dropna().mean())
    return {"overall": overall, "plot_count": plot_count}


def _fmt_metric(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _write_report_md(
    output_dir: Path,
    summary: Dict[str, object],
    config_path: Path,
    train_cfg: TrainConfig,
    mm_cfg: MicroMacroConfig,
    dtw_cfg: DTWMicroConfig,
) -> None:
    overall = dict(summary.get("overall", {}) or {})
    primary_keys = [
        "precision",
        "recall",
        "f1",
        "start_mae_ms",
        "end_mae_ms",
        "transition_mae_ms",
        "rep_action_accuracy",
        "micro_sample_macro_f1",
        "micro_f1_at_10",
        "micro_f1_at_25",
        "micro_f1_at_50",
        "macro_sample_macro_f1",
        "macro_f1_at_10",
        "macro_f1_at_25",
        "macro_f1_at_50",
        "micro_edit",
        "macro_edit",
    ]
    metric_rows = ["| Metric | Value |", "|---|---:|"]
    for key in primary_keys:
        if key in overall:
            metric_rows.append(f"| `{key}` | {_fmt_metric(overall[key])} |")

    artifact_rows = [
        "| Artifact | Path |",
        "|---|---|",
        f"| Summary JSON | `{(output_dir / 'metrics' / 'summary.json').as_posix()}` |",
        f"| Stream metrics | `{(output_dir / 'metrics' / ('stream_metrics_' + str(summary.get('micro_source')) + '.csv')).as_posix()}` |",
        f"| Rep detections | `{(output_dir / 'detections' / ('rep_detections_' + str(summary.get('micro_source')) + '.csv')).as_posix()}` |",
        f"| Pairing diagnostics | `{(output_dir / 'detections' / ('pairing_diagnostics_' + str(summary.get('micro_source')) + '.csv')).as_posix()}` |",
        f"| Rep plots | `{(output_dir / 'plots' / 'rep' / str(summary.get('micro_source'))).as_posix()}` |",
        f"| Action plots | `{(output_dir / 'plots' / 'action' / str(summary.get('micro_source'))).as_posix()}` |",
        f"| Model checkpoint | `{(output_dir / 'models' / 'ds_ms_tcn.pt').as_posix()}` |",
    ]

    config_rows = [
        "| Setting | Value |",
        "|---|---|",
        f"| Config | `{config_path.as_posix()}` |",
        f"| Micro source | `{summary.get('micro_source')}` |",
        f"| Modes | `{summary.get('modes')}` |",
        f"| Train subjects | `{summary.get('verified_train_split_subjects')}` |",
        f"| Test subjects | `{summary.get('verified_test_split_subjects')}` |",
        f"| Device | `{train_cfg.device}` |",
        f"| Batch size | `{train_cfg.batch_size}` |",
        f"| Epochs | `{train_cfg.epochs}` |",
        f"| Slice seconds | `{mm_cfg.slice_seconds}` |",
        f"| Overlap | `{mm_cfg.overlap}` |",
        f"| TCN filters/layers | `{mm_cfg.num_filters}/{mm_cfg.num_layers}` |",
        f"| Causal convolution | `{mm_cfg.causal}` |",
        f"| DTW stride/duration/downsample/max windows | `{dtw_cfg.detection_stride}/{dtw_cfg.duration_stride or dtw_cfg.detection_stride}/{dtw_cfg.dtw_downsample_factor}/{dtw_cfg.max_windows_per_label}` |",
    ]

    text = "\n".join(
        [
            f"# Micro/Macro Recognition Report ({summary.get('micro_source')})",
            "",
            "## Key Metrics",
            "",
            "\n".join(metric_rows),
            "",
            "## Experiment",
            "",
            "\n".join(config_rows),
            "",
            "## Artifacts",
            "",
            "\n".join(artifact_rows),
            "",
            "## Notes",
            "",
            "- Rep segmentation `f1` uses IoU >= 0.50 matching at rep level.",
            "- Micro/macro `f1_at_10/25/50` are segment IoU-F1 metrics.",
            "- Action plots are separated from rep segmentation plots so the color-coded action prediction is easier to inspect.",
            "- Run `python -m evaluation.streaming_micro_macro` on a trained TCN run to generate online replay CSV/SVG/HTML outputs.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def run(
    config_path: Path,
    micro_source: str | None,
    mode: str,
    no_timestamp: bool,
    dry_run: bool,
    _run_stamp: str | None = None,
    output_dir_override: Path | None = None,
) -> None:
    raw = _load_config(config_path)
    if output_dir_override is not None:
        raw.setdefault("io", {})["micro_macro_output_dir"] = str(output_dir_override)
    feature_cfg = raw.get("feature", {}) or {}
    train_cfg = TrainConfig(**(raw.get("train", {}) or {}))
    mm_raw = dict(raw.get("micro_macro", {}) or {})
    dtw_raw = dict(mm_raw.pop("dtw", {}) or {})
    configured_micro_source = str(mm_raw.pop("micro_source", "both"))
    resolved_micro_source = str(micro_source or configured_micro_source)
    if resolved_micro_source == "both":
        stamp = _run_stamp or ("latest" if no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S"))
        for source in ("tcn", "dtw"):
            run(config_path, source, mode, no_timestamp, dry_run, _run_stamp=stamp, output_dir_override=output_dir_override)
        return
    if resolved_micro_source not in {"tcn", "dtw"}:
        raise ValueError(f"Unsupported micro_source: {resolved_micro_source}")
    mm_cfg = MicroMacroConfig(**mm_raw)
    dtw_cfg = DTWMicroConfig(**dtw_raw)
    set_seed(int(train_cfg.seed))
    device = _resolve_device(train_cfg.device)
    train_cfg.device = str(device)
    resolved_num_workers = _resolve_num_workers(train_cfg.num_workers, device)
    resolved_pin_memory = _resolve_pin_memory(train_cfg.pin_memory, device)

    modes = list(mm_cfg.train_on_modes)
    if mode != "both":
        modes = [mode]
    streams, subjects, actions = _load_streams(raw, modes)
    imu_columns = tuple(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    macro_classes = [OTHER_LABEL] + [a for a in actions if a != OTHER_LABEL]
    if not streams:
        raise RuntimeError("No streams found for micro/macro recognition")

    base_out = Path((raw.get("io", {}) or {}).get("micro_macro_output_dir", "./artifacts/micro_macro_recognition"))
    run_stamp = _run_stamp or ("latest" if no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = base_out / run_stamp / resolved_micro_source
    for sub in ("models", "metrics", "detections", "plots", "metadata"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    subjects_sorted = sorted(set(subjects))
    test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else subjects_sorted[-1]
    if test_subject not in subjects_sorted:
        raise ValueError(f"Configured test_subject={test_subject!r} not found in subjects: {subjects_sorted}")
    train_subjects = [s for s in subjects_sorted if s != test_subject]
    train_streams = _filter_subjects(streams, train_subjects, subject_column)
    test_streams = _filter_subjects(streams, [test_subject], subject_column)
    train_sequences = [df for _, df in train_streams]
    test_sequences = [df for _, df in test_streams]
    train_split_subjects = {str(df.iloc[0]["_split_subject"]) for _, df in train_streams if "_split_subject" in df.columns and len(df)}
    test_split_subjects = {str(df.iloc[0]["_split_subject"]) for _, df in test_streams if "_split_subject" in df.columns and len(df)}
    overlap_subjects = sorted(train_split_subjects & test_split_subjects)
    if overlap_subjects:
        raise RuntimeError(f"Subject-wise split leakage detected: {overlap_subjects}")
    print(f"[INFO] micro_source={resolved_micro_source} modes={modes} train_subjects={train_subjects} test_subject={test_subject}")
    print(f"[INFO] streams train={len(train_streams)} test={len(test_streams)} actions={macro_classes}")
    print(f"[INFO] subject-wise split verified: train={sorted(train_split_subjects)} test={sorted(test_split_subjects)}")
    print(
        "[INFO] resources "
        f"device={device} amp={bool(train_cfg.amp) and device.type == 'cuda'} "
        f"num_workers={resolved_num_workers} pin_memory={resolved_pin_memory}"
    )

    stats = compute_train_stats(train_sequences, imu_columns)
    stats.save(output_dir / "metadata" / "zscore_stats.json")
    train_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    train_sequences = [df for _, df in train_streams]

    fallback_sample_rate = float((raw.get("window", {}) or {}).get("sample_rate_hz", 50))
    sample_rate = _median_sample_rate(train_streams, fallback_sample_rate)
    slice_len = max(8, int(round(float(mm_cfg.slice_seconds) * sample_rate)))
    stride_len = max(1, int(round(slice_len * (1.0 - float(mm_cfg.overlap)))))
    single_rf = 1 + (int(mm_cfg.kernel_size) - 1) * sum(2 ** i for i in range(int(mm_cfg.num_layers)))
    total_rf = 1 + 4 * (single_rf - 1)
    if bool(mm_cfg.causal) and slice_len < total_rf:
        print(
            f"[WARN] slice_len={slice_len} is shorter than estimated causal total RF={total_rf}; "
            "increase slice_seconds or reduce num_layers for full context.",
            flush=True,
        )
    print(
        f"[INFO] train sample_rate_hz={sample_rate:.3f} slice_len={slice_len} "
        f"stride_len={stride_len} single_stage_rf={single_rf} total_causal_rf={total_rf}",
        flush=True,
    )
    use_gt_micro_probs = resolved_micro_source == "dtw"
    ds = SequenceSliceDataset(
        train_sequences,
        imu_columns,
        macro_classes,
        slice_len=slice_len,
        stride_len=stride_len,
        use_gt_micro_probs=use_gt_micro_probs,
    )
    if dry_run:
        print(f"[DRY RUN] slices={len(ds)} slice_len={slice_len} stride_len={stride_len}")
        return
    loader_kwargs = {
        "batch_size": int(train_cfg.batch_size),
        "shuffle": True,
        "num_workers": resolved_num_workers,
        "pin_memory": resolved_pin_memory,
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ds, **loader_kwargs)
    model = DSMSTCN(
        DSMSTCNConfig(
            input_channels=len(imu_columns),
            micro_classes=len(MICRO_LABELS),
            macro_classes=len(macro_classes),
            num_filters=int(mm_cfg.num_filters),
            num_layers=int(mm_cfg.num_layers),
            kernel_size=int(mm_cfg.kernel_size),
            dropout=float(mm_cfg.dropout),
            causal=bool(mm_cfg.causal),
        )
    )
    _train_model(model, loader, train_cfg, mm_cfg, use_gt_micro_probs=use_gt_micro_probs)
    torch.save(
        {
            "model_state": model.state_dict(),
            "macro_classes": macro_classes,
            "micro_classes": list(MICRO_LABELS),
            "imu_columns": list(imu_columns),
            "config": asdict(mm_cfg),
        },
        output_dir / "models" / "ds_ms_tcn.pt",
    )

    summary = _evaluate_streams(
        model=model,
        streams=test_streams,
        train_sequences_for_dtw=train_sequences,
        imu_columns=imu_columns,
        macro_classes=macro_classes,
        micro_source=resolved_micro_source,
        mm_cfg=mm_cfg,
        dtw_cfg=dtw_cfg,
        output_dir=output_dir,
        device=device,
    )
    summary.update(
        {
            "micro_source": resolved_micro_source,
            "task": "micro_macro_recognition",
            "model_name": f"ds_ms_tcn_{resolved_micro_source}",
            "configured_micro_source": configured_micro_source,
            "resolved_micro_source": resolved_micro_source,
            "modes": modes,
            "train_subjects": train_subjects,
            "test_subject": test_subject,
            "verified_train_split_subjects": sorted(train_split_subjects),
            "verified_test_split_subjects": sorted(test_split_subjects),
            "subject_split_overlap": overlap_subjects,
            "macro_classes": macro_classes,
            "micro_classes": list(MICRO_LABELS),
            "sample_rate_hz_for_training_slices": sample_rate,
            "slice_len_samples": slice_len,
            "stride_len_samples": stride_len,
            "primary_metrics": primary_metric_table(summary.get("overall", {}) or {}),
        }
    )
    (output_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(
        output_dir,
        task="micro_macro_recognition",
        model_name=f"ds_ms_tcn_{resolved_micro_source}",
        config_path=config_path,
        extras={
            "micro_source": resolved_micro_source,
            "modes": modes,
            "train_subjects": train_subjects,
            "test_subject": test_subject,
            "sample_rate_hz_for_training_slices": sample_rate,
            "slice_len_samples": slice_len,
            "stride_len_samples": stride_len,
        },
    )
    _write_report_md(output_dir, summary, config_path, train_cfg, mm_cfg, dtw_cfg)
    shutil.copy2(config_path, output_dir / "metadata" / "config_snapshot.yaml")
    print(json.dumps(summary["overall"], indent=2))
    print(f"[OK] Wrote outputs to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DS-MS-TCN micro/macro rep segmentation and action recognition")
    parser.add_argument("--config", type=Path, default=Path("configs/micro_macro_recognition.yaml"))
    parser.add_argument("--micro-source", choices=["tcn", "dtw", "both"], default=None, help="Override micro_macro.micro_source from config")
    parser.add_argument("--mode", choices=["sets", "whole", "both"], default="both")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override io.micro_macro_output_dir")
    parser.add_argument("--run-stamp", default=None, help="Run folder name under the output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        args.micro_source,
        args.mode,
        args.no_timestamp,
        args.dry_run,
        _run_stamp=args.run_stamp,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
