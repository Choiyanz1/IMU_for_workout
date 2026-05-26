"""Same-dataset fixed baselines for rep-structured workout evaluation.

Runs deployable/common baselines under the current 9-fold LOSO protocol and
reports the four core project metrics:

- Count MAE
- Rep IoU-F1@50
- Phase IoU-F1@50
- C/E ratio MAE

Boundary-only baselines such as peak detection do not naturally emit C/E phases.
For those, this script uses train-fold per-action phase-order and phase-ratio
priors to create a simple phase trace from predicted rep boundaries. This makes
the four metrics computable, while keeping the baseline intentionally simple.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import (  # noqa: E402
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    RepDetection,
    SegmentRun,
    labels_to_runs,
    pair_concentric_eccentric_reps,
)
from scripts.evaluate_peak_baseline import (  # noqa: E402
    compute_6axis_mag,
    compute_acc_mag,
    compute_gyro_mag,
    detect_peaks,
    peaks_to_reps,
    smooth_signal,
)
from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    extract_active_segments,
    predict_active,
    predict_rf_phase,
    smooth_phase_probs,
    train_active_detector,
    train_rf_phase,
)
from scripts.new_c_pipeline.master_eval import (  # noqa: E402
    compute_ce_ratio_metrics,
    compute_rep_ce_ratios,
    evaluate_phase_segments,
    extract_phase_segments,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import aggregate_rich  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    PhaseDataset,
    extract_active_segments_data,
    normalize,
    predict_fast,
    set_seed,
    should_exclude,
    train_fast,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


class CausalTCNLite(nn.Module):
    """Small causal TCN baseline distinct from the current CNN encoder."""

    def __init__(self, in_ch: int = 6, hidden: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList()
        current = in_ch
        for dilation in (1, 2, 4, 8):
            self.blocks.append(CausalTCNBlock(current, hidden, dilation=dilation, dropout=dropout))
            current = hidden
        self.head = nn.Conv1d(hidden, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class CausalTCNBlock(nn.Module):
    def __init__(self, in_ch: int, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(in_ch, hidden, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, hidden, kernel_size=1) if in_ch != hidden else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        y = self.conv1(F.pad(x, (self.pad, 0), mode="reflect"))
        y = self.dropout(F.relu(self.norm1(y)))
        y = self.conv2(F.pad(y, (self.pad, 0), mode="reflect"))
        y = self.dropout(F.relu(self.norm2(y)))
        return y + residual


class BiLSTMPhase(nn.Module):
    """Bidirectional LSTM sequence-labeling baseline for wearable HAR comparison."""

    def __init__(self, in_ch: int = 6, hidden: int = 64, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        y, _ = self.lstm(seq)
        logits = self.head(y)
        return logits.transpose(1, 2)


def stream_subject(stream_id: str) -> str:
    return stream_id.split("/")[0]


def stream_action(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def rep_iou(pred: RepDetection, gt: RepDetection) -> float:
    inter = max(0, min(pred.end_idx, gt.end_idx) - max(pred.start_idx, gt.start_idx))
    union = max(pred.end_idx, gt.end_idx) - min(pred.start_idx, gt.start_idx)
    return inter / union if union > 0 else 0.0


def evaluate_reps_iou50(pred_reps: Sequence[RepDetection], gt_reps: Sequence[RepDetection]) -> Dict[str, float]:
    tp = 0
    matched_gt = set()
    for pred in pred_reps:
        best_iou = 0.0
        best_gt = None
        for gi, gt in enumerate(gt_reps):
            if gi in matched_gt:
                continue
            iou = rep_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        if best_iou >= 0.50 and best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_reps) - tp
    fn = len(gt_reps) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "pred_count": len(pred_reps),
        "gt_count": len(gt_reps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_count": 1 if len(pred_reps) == len(gt_reps) else 0,
        "over": 1 if len(pred_reps) > len(gt_reps) else 0,
        "under": 1 if len(pred_reps) < len(gt_reps) else 0,
    }


def phase_labels_to_reps(phase_arr: np.ndarray, min_phase: int = 3, max_gap: int = 3) -> List[RepDetection]:
    runs = labels_to_runs(phase_arr, positive_labels={ECCENTRIC_LABEL, CONCENTRIC_LABEL}, min_length=min_phase)
    if not runs:
        return []
    merged: List[SegmentRun] = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(
                label=run.label,
                start_idx=merged[-1].start_idx,
                end_idx=run.end_idx,
                confidence=(merged[-1].confidence + run.confidence) / 2.0,
            )
        else:
            merged.append(run)
    reps, _ = pair_concentric_eccentric_reps(merged, micro_source="baseline_phase", max_gap_samples=max_gap)
    return reps


def evaluate_prediction(stream_id: str, df: pd.DataFrame, pred_phase_arr: np.ndarray, pred_reps: Sequence[RepDetection]) -> Dict:
    gt_phases = df["phase"].to_numpy()
    gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
    rep_m = evaluate_reps_iou50(pred_reps, gt_reps)

    valid = np.isin(gt_phases, [CONCENTRIC_LABEL, ECCENTRIC_LABEL])
    if valid.any():
        y_true = np.array([1 if p == CONCENTRIC_LABEL else 0 for p in gt_phases[valid]], dtype=np.int64)
        y_pred = np.array([1 if p == CONCENTRIC_LABEL else 0 for p in pred_phase_arr[valid]], dtype=np.int64)
        phase_accuracy = float(accuracy_score(y_true, y_pred))
        phase_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        phase_accuracy = 0.0
        phase_macro_f1 = 0.0

    gt_c = extract_phase_segments(gt_phases, CONCENTRIC_LABEL)
    gt_e = extract_phase_segments(gt_phases, ECCENTRIC_LABEL)
    pred_c = extract_phase_segments(pred_phase_arr, CONCENTRIC_LABEL)
    pred_e = extract_phase_segments(pred_phase_arr, ECCENTRIC_LABEL)
    c_seg = evaluate_phase_segments(pred_c, gt_c)
    e_seg = evaluate_phase_segments(pred_e, gt_e)

    pred_ratios = compute_rep_ce_ratios(pred_reps, pred_phase_arr)
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)

    return {
        "stream_id": stream_id,
        "subject": stream_subject(stream_id),
        "action": stream_action(stream_id),
        "pred_count": rep_m["pred_count"],
        "gt_count": rep_m["gt_count"],
        "count_error": abs(rep_m["pred_count"] - rep_m["gt_count"]),
        **{k: v for k, v in rep_m.items() if k not in ["pred_count", "gt_count"]},
        "phase_accuracy": phase_accuracy,
        "phase_macro_f1": phase_macro_f1,
        "transition_mae_ms": None,
        "concentric_seg_f1": c_seg["f1"],
        "eccentric_seg_f1": e_seg["f1"],
        **ce_metrics,
    }


def aggregate_by_key(results: Sequence[Dict], key_name: str) -> Dict[str, Dict]:
    grouped = defaultdict(list)
    for result in results:
        grouped[str(result.get(key_name, "unknown"))].append(result)
    return {key: aggregate_rich(items) for key, items in sorted(grouped.items())}


def estimate_peak_duration_prior(train_streams: Sequence[Tuple[str, pd.DataFrame]], action: str) -> Dict[str, float]:
    durations = []
    for sid, df in train_streams:
        if stream_action(sid) != action or "phase" not in df.columns:
            continue
        for rep in truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=3):
            durations.append(max(1, int(rep.end_idx) - int(rep.start_idx)))
    if not durations:
        return {"median": 200.0, "min": 100.0, "max": 400.0, "sample_rate": 100.0}
    arr = np.asarray(durations, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "min": float(np.percentile(arr, 10)),
        "max": float(np.percentile(arr, 90)),
        "sample_rate": 100.0,
    }


def estimate_phase_prior(train_streams: Sequence[Tuple[str, pd.DataFrame]], action: str) -> Dict[str, float | str]:
    c_fracs = []
    first_labels = []
    for sid, df in train_streams:
        if stream_action(sid) != action or "phase" not in df.columns:
            continue
        phases = df["phase"].to_numpy()
        for rep in truth_reps_from_labels(phases, min_phase_samples=3):
            seg = phases[rep.start_idx:rep.end_idx]
            active = [str(p) for p in seg if str(p) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL}]
            if not active:
                continue
            c_count = sum(1 for p in active if p == CONCENTRIC_LABEL)
            c_fracs.append(c_count / len(active))
            first_labels.append(active[0])
    if not c_fracs:
        return {"c_frac": 0.5, "first_label": ECCENTRIC_LABEL}
    first_label = Counter(first_labels).most_common(1)[0][0]
    return {"c_frac": float(np.median(c_fracs)), "first_label": first_label}


def rep_tuples_to_detections(reps: Iterable[Tuple[int, int]], phase_prior: Dict[str, float | str]) -> List[RepDetection]:
    detections = []
    c_frac = float(phase_prior["c_frac"])
    first_label = str(phase_prior["first_label"])
    for start, end in reps:
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        dur = end - start
        c_len = int(round(dur * c_frac))
        c_len = min(max(c_len, 1), dur - 1) if dur > 1 else 1
        if first_label == CONCENTRIC_LABEL:
            transition = start + c_len
        else:
            transition = end - c_len
        transition = int(min(max(transition, start + 1), end - 1)) if end - start > 1 else start
        detections.append(
            RepDetection(
                start_idx=start,
                transition_idx=transition,
                end_idx=end,
                micro_source="peak_phase_prior",
                micro_confidence=1.0,
                pred_action_type="unknown",
            )
        )
    return detections


def reps_to_phase_arr(n: int, reps: Sequence[RepDetection], phase_prior: Dict[str, float | str]) -> np.ndarray:
    phase_arr = np.array(["other"] * n, dtype=object)
    first_label = str(phase_prior["first_label"])
    for rep in reps:
        start = max(0, int(rep.start_idx))
        transition = min(n, int(rep.transition_idx))
        end = min(n, int(rep.end_idx))
        if start >= end:
            continue
        if first_label == CONCENTRIC_LABEL:
            phase_arr[start:transition] = CONCENTRIC_LABEL
            phase_arr[transition:end] = ECCENTRIC_LABEL
        else:
            phase_arr[start:transition] = ECCENTRIC_LABEL
            phase_arr[transition:end] = CONCENTRIC_LABEL
    return phase_arr


def run_peak_fold(
    test_streams: Sequence[Tuple[str, pd.DataFrame]],
    train_streams: Sequence[Tuple[str, pd.DataFrame]],
    mag_mode: str,
) -> List[Dict]:
    duration_priors = {action: estimate_peak_duration_prior(train_streams, action) for action in sorted({stream_action(sid) for sid, _ in train_streams})}
    phase_priors = {action: estimate_phase_prior(train_streams, action) for action in sorted({stream_action(sid) for sid, _ in train_streams})}
    results = []
    for stream_id, df in test_streams:
        action = stream_action(stream_id)
        if mag_mode == "acc":
            mag = compute_acc_mag(df)
        elif mag_mode == "gyro":
            mag = compute_gyro_mag(df)
        elif mag_mode == "6axis":
            mag = compute_6axis_mag(df)
        else:
            raise ValueError(f"Unknown mag_mode: {mag_mode}")
        smoothed = smooth_signal(mag, window=9)
        duration_prior = duration_priors.get(action, {"median": 200.0, "min": 100.0, "max": 400.0, "sample_rate": 100.0})
        phase_prior = phase_priors.get(action, {"c_frac": 0.5, "first_label": ECCENTRIC_LABEL})
        peaks = detect_peaks(smoothed, duration_prior)
        rep_tuples = peaks_to_reps(peaks, smoothed, duration_prior)
        pred_reps = rep_tuples_to_detections(rep_tuples, phase_prior)
        pred_phase_arr = reps_to_phase_arr(len(df), pred_reps, phase_prior)
        results.append(evaluate_prediction(stream_id, df, pred_phase_arr, pred_reps))
    return results


def run_rf_phase_fold(test_streams: Sequence[Tuple[str, pd.DataFrame]], train_streams: Sequence[Tuple[str, pd.DataFrame]], cfg: PhaseCompareConfig) -> List[Dict]:
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, cfg)
    results = []
    for stream_id, df in test_streams:
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, cfg)
        phase_probs = smooth_phase_probs(phase_probs, cfg.smoothing_window)
        hard = np.argmax(phase_probs, axis=1)
        pred_phase_arr = np.array(["other"] * len(df), dtype=object)
        for start, end in active_segments:
            labels = hard[int(start):int(end)]
            pred_phase_arr[int(start):int(end)] = np.where(labels == 1, CONCENTRIC_LABEL, ECCENTRIC_LABEL)
        pred_reps = phase_labels_to_reps(pred_phase_arr, cfg.min_phase_samples, cfg.max_phase_gap_samples)
        results.append(evaluate_prediction(stream_id, df, pred_phase_arr, pred_reps))
    return results


def make_deep_model(model_name: str, in_ch: int, hidden: int) -> nn.Module:
    if model_name == "tcn_lite":
        return CausalTCNLite(in_ch=in_ch, hidden=hidden)
    if model_name == "bilstm":
        return BiLSTMPhase(in_ch=in_ch, hidden=hidden)
    raise ValueError(f"Unsupported deep model: {model_name}")


def train_deep_phase_model(
    train_streams: Sequence[Tuple[str, pd.DataFrame]],
    cfg: PhaseCompareConfig,
    model_name: str,
    epochs: int,
    hidden: int,
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    segments, labels = extract_active_segments_data(train_streams, cfg.imu_columns)
    if not segments:
        raise RuntimeError(f"No active segments for {model_name}")
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]
    train_ds = PhaseDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    model = make_deep_model(model_name, in_ch=len(cfg.imu_columns), hidden=hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, mean, std


def run_deep_phase_fold(
    test_streams: Sequence[Tuple[str, pd.DataFrame]],
    train_streams: Sequence[Tuple[str, pd.DataFrame]],
    cfg: PhaseCompareConfig,
    model_name: str,
    epochs: int,
    hidden: int,
) -> List[Dict]:
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    model, mean, std = train_deep_phase_model(train_streams, cfg, model_name, epochs, hidden)
    results = []
    for stream_id, df in test_streams:
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
        hard = np.argmax(phase_probs, axis=1)
        pred_phase_arr = np.array([ECCENTRIC_LABEL if label == 0 else CONCENTRIC_LABEL for label in hard], dtype=object)
        pred_reps = phase_labels_to_reps(pred_phase_arr, cfg.min_phase_samples, cfg.max_phase_gap_samples)
        results.append(evaluate_prediction(stream_id, df, pred_phase_arr, pred_reps))
    return results


def summarize_model(name: str, model_results: Sequence[Dict], metadata: Dict) -> Dict:
    return {
        "metadata": metadata,
        "overall": aggregate_rich(model_results),
        "per_action": aggregate_by_key(model_results, "action"),
        "per_subject": aggregate_by_key(model_results, "subject"),
        "streams": list(model_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/fixed_baseline_comparison/same_dataset_9fold_20260519.json")
    parser.add_argument(
        "--models",
        default="peak_acc,peak_6axis,rf_phase",
        help="Comma-separated: peak_acc, peak_gyro, peak_6axis, rf_phase, tcn_lite, bilstm",
    )
    parser.add_argument("--deep-epochs", type=int, default=20)
    parser.add_argument("--deep-hidden", type=int, default=64)
    args = parser.parse_args()

    set_seed(42)

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in streams if not should_exclude(sid)]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = PhaseCompareConfig()

    print(f"Subjects: {subjects}")
    print(f"Streams: {len(streams)}")
    print(f"Models: {models}")

    by_model: Dict[str, List[Dict]] = {m: [] for m in models}
    folds = []
    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}\nFold {fold_idx}/{len(subjects)} test={test_subject}\n{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        fold_entry = {"fold": fold_idx, "test_subject": test_subject, "models": {}}

        for model_name in models:
            print(f"  Running {model_name}...")
            if model_name.startswith("peak_"):
                mag_mode = model_name.replace("peak_", "")
                results = run_peak_fold(test_streams, train_streams, mag_mode)
                metadata = {
                    "type": "boundary_peak_detection_with_train_fold_phase_prior",
                    "mag_mode": mag_mode,
                    "phase_prior": "per-action train-fold first phase + median C fraction",
                }
            elif model_name == "rf_phase":
                results = run_rf_phase_fold(test_streams, train_streams, cfg)
                metadata = {
                    "type": "per-action active detector + per-action RF C/E phase classifier",
                    "window": cfg.phase_window_size,
                    "stride": cfg.phase_stride,
                    "smoothing_window": cfg.smoothing_window,
                }
            elif model_name in {"tcn_lite", "bilstm"}:
                results = run_deep_phase_fold(test_streams, train_streams, cfg, model_name, args.deep_epochs, args.deep_hidden)
                metadata = {
                    "type": "deep_sequence_phase_model_with_shared_active_detector",
                    "model": model_name,
                    "epochs": args.deep_epochs,
                    "hidden": args.deep_hidden,
                    "smoothing": "MA25 + Viterbi penalty=0.3 via predict_fast",
                }
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            by_model[model_name].extend(results)
            fold_summary = aggregate_rich(results)
            fold_entry["models"][model_name] = fold_summary
            print(
                f"    RepF1={fold_summary['rep_f1']:.4f} Exact={fold_summary['exact_count_acc']:.3f} "
                f"MAE={fold_summary['mean_abs_count_error']:.3f} PhaseIoU={fold_summary['phase_seg_iou_f1_50_avg']:.4f} "
                f"CE_MAE={fold_summary['ce_ratio_mae']:.4f}"
            )
        folds.append(fold_entry)

    output = {
        "settings": {
            "protocol": "9-fold LOSO on current sets with excluded light-weight sessions",
            "excluded_sessions": EXCLUDED_SESSIONS,
            "models": models,
            "core_metrics": ["mean_abs_count_error", "rep_f1", "phase_seg_iou_f1_50_avg", "ce_ratio_mae"],
        },
        "models": {
            name: summarize_model(
                name,
                results,
                metadata={"fixed_baseline": True, "source": "fixed_same_dataset_baselines_9fold.py"},
            )
            for name, results in by_model.items()
        },
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    for name, payload in output["models"].items():
        print(f"\n{name}")
        print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
