"""DS-MS-TCN micro/macro recognition and rep segmentation.

Usage:
  python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source tcn
  python -m train.micro_macro_recognition --config configs/micro_macro_recognition.yaml --micro-source dtw
"""
from __future__ import annotations

import argparse
import fnmatch
import json
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
    labels_to_runs,
    macro_labels_from_action,
    match_segments,
    micro_labels_from_phase,
    pair_concentric_eccentric_reps,
    rep_metrics,
    reps_to_rows,
    truth_reps_from_labels,
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
    num_workers: int = 0
    device: str = "cpu"


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
        if subject_column in df.columns and str(df.iloc[0][subject_column]) in allowed:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for x, micro, macro, gt_micro_probs in loader:
            x = x.to(device)
            micro = micro.to(device)
            macro = macro.to(device)
            gt_micro_probs = gt_micro_probs.to(device)
            optimizer.zero_grad()
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
            losses["loss"].backward()
            optimizer.step()
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

    for stream_id, df in streams:
        sample_rate = infer_sample_rate_hz(df)
        if micro_source == "dtw":
            dtw_runs = detect_dtw_micro_runs(df, dtw_templates, imu_columns, dtw_cfg)
            external = dtw_runs_to_micro_scores(len(df), dtw_runs)
            pred = _predict_full_sequence(model, df, imu_columns, device, external_micro=external)
        else:
            pred = _predict_full_sequence(model, df, imu_columns, device)
        micro_probs = pred["micro_probs"]
        macro_probs = pred["macro4_probs"]
        pred_micro_labels = [MICRO_LABELS[int(i)] for i in np.argmax(micro_probs, axis=1)]
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
        metrics = rep_metrics(pred_reps, truth_reps, sample_rate_hz=sample_rate)
        metric_rows.append({"stream_id": stream_id, "micro_source": micro_source, "sample_rate_hz": sample_rate, **metrics})
        pred_rows.extend(reps_to_rows(stream_id, pred_reps))
        diag_rows.extend(diagnostics_to_rows(stream_id, diagnostics))

        for pi, ti, _ in match_segments(
            [(r.start_idx, r.end_idx) for r in pred_reps],
            [(r.start_idx, r.end_idx) for r in truth_reps],
        ):
            true_actions.append(truth_reps[ti].pred_action_type)
            pred_actions.append(pred_reps[pi].pred_action_type)

        if plot_count < int(mm_cfg.plot_max_streams):
            gt_micro = micro_labels_from_phase(df["phase"].to_numpy())
            gt_runs = labels_to_runs(gt_micro, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=mm_cfg.min_phase_samples)
            macro_runs = _macro_runs_from_probs(macro_probs, macro_classes, min_length=mm_cfg.min_phase_samples)
            plot_path = output_dir / "plots" / micro_source / (stream_id.replace("/", "_") + ".svg")
            write_micro_macro_svg(plot_path, stream_id, df, gt_runs, pred_micro_runs, truth_reps, pred_reps, macro_runs, sample_rate)
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
    return {"overall": overall, "plot_count": plot_count}


def run(
    config_path: Path,
    micro_source: str | None,
    mode: str,
    no_timestamp: bool,
    dry_run: bool,
    _run_stamp: str | None = None,
) -> None:
    raw = _load_config(config_path)
    feature_cfg = raw.get("feature", {}) or {}
    train_cfg = TrainConfig(**(raw.get("train", {}) or {}))
    mm_raw = dict(raw.get("micro_macro", {}) or {})
    dtw_raw = dict(mm_raw.pop("dtw", {}) or {})
    configured_micro_source = str(mm_raw.pop("micro_source", "both"))
    resolved_micro_source = str(micro_source or configured_micro_source)
    if resolved_micro_source == "both":
        stamp = _run_stamp or ("latest" if no_timestamp else datetime.now().strftime("%Y%m%d_%H%M%S"))
        for source in ("tcn", "dtw"):
            run(config_path, source, mode, no_timestamp, dry_run, _run_stamp=stamp)
        return
    if resolved_micro_source not in {"tcn", "dtw"}:
        raise ValueError(f"Unsupported micro_source: {resolved_micro_source}")
    mm_cfg = MicroMacroConfig(**mm_raw)
    dtw_cfg = DTWMicroConfig(**dtw_raw)
    set_seed(int(train_cfg.seed))

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
    test_subject = subjects_sorted[-1]
    train_subjects = [s for s in subjects_sorted if s != test_subject]
    train_streams = _filter_subjects(streams, train_subjects, subject_column)
    test_streams = _filter_subjects(streams, [test_subject], subject_column)
    train_sequences = [df for _, df in train_streams]
    test_sequences = [df for _, df in test_streams]
    print(f"[INFO] micro_source={resolved_micro_source} modes={modes} train_subjects={train_subjects} test_subject={test_subject}")
    print(f"[INFO] streams train={len(train_streams)} test={len(test_streams)} actions={macro_classes}")

    stats = compute_train_stats(train_sequences, imu_columns)
    stats.save(output_dir / "metadata" / "zscore_stats.json")
    train_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    train_sequences = [df for _, df in train_streams]

    sample_rate = int((raw.get("window", {}) or {}).get("sample_rate_hz", 50))
    slice_len = max(8, int(round(float(mm_cfg.slice_seconds) * sample_rate)))
    stride_len = max(1, int(round(slice_len * (1.0 - float(mm_cfg.overlap)))))
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
    loader = DataLoader(ds, batch_size=int(train_cfg.batch_size), shuffle=True, num_workers=int(train_cfg.num_workers))
    model = DSMSTCN(
        DSMSTCNConfig(
            input_channels=len(imu_columns),
            micro_classes=len(MICRO_LABELS),
            macro_classes=len(macro_classes),
            num_filters=int(mm_cfg.num_filters),
            num_layers=int(mm_cfg.num_layers),
            kernel_size=int(mm_cfg.kernel_size),
            dropout=float(mm_cfg.dropout),
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
        device=torch.device(train_cfg.device),
    )
    summary.update(
        {
            "micro_source": resolved_micro_source,
            "configured_micro_source": configured_micro_source,
            "resolved_micro_source": resolved_micro_source,
            "modes": modes,
            "train_subjects": train_subjects,
            "test_subject": test_subject,
            "macro_classes": macro_classes,
            "micro_classes": list(MICRO_LABELS),
        }
    )
    (output_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config, args.micro_source, args.mode, args.no_timestamp, args.dry_run)


if __name__ == "__main__":
    main()
