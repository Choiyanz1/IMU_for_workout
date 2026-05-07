from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple, overload

import numpy as np
import pandas as pd

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    MICRO_LABELS,
    OTHER_LABEL,
    SegmentRun,
    labels_to_runs,
    micro_labels_from_phase,
)
from preprocessing.sdtw_rep_segmentation import dtw_path_cost, normalize_matrix, smooth_matrix


@dataclass
class DTWMicroConfig:
    smoothing_window: int = 7
    duration_min_ratio: float = 0.55
    duration_max_ratio: float = 1.75
    threshold_percentile: float = 75.0
    threshold_margin: float = 0.10
    detection_stride: int = 3
    duration_stride: int = 0
    dtw_downsample_factor: int = 1
    max_windows_per_label: int = 0
    max_overlap_iou: float = 0.20


@dataclass
class DTWMicroTemplate:
    label: str
    signal_name: str
    query: np.ndarray
    median_duration: int
    min_duration: int
    max_duration: int
    cost_threshold: float


def _motion_matrix(df: pd.DataFrame, imu_columns: Sequence[str], smoothing_window: int) -> Tuple[np.ndarray, List[str]]:
    available = [col for col in imu_columns if col in df.columns]
    if not available:
        raise ValueError("No IMU columns available for DTW micro adapter")
    base = df[available].to_numpy(dtype=np.float64)
    names = list(available)
    extras = []
    extra_names = []
    acc_idx = [i for i, c in enumerate(available) if c.startswith("a")]
    gyro_idx = [i for i, c in enumerate(available) if c.startswith("g")]
    if len(acc_idx) >= 2:
        extras.append(np.sqrt(np.sum(base[:, acc_idx] ** 2, axis=1, keepdims=True)))
        extra_names.append("acc_mag")
    if len(gyro_idx) >= 2:
        extras.append(np.sqrt(np.sum(base[:, gyro_idx] ** 2, axis=1, keepdims=True)))
        extra_names.append("gyro_mag")
    matrix = np.concatenate([base] + extras, axis=1) if extras else base
    matrix = normalize_matrix(smooth_matrix(matrix, smoothing_window))
    return matrix, names + extra_names


def _best_signal(segment: pd.DataFrame, imu_columns: Sequence[str], cfg: DTWMicroConfig) -> Tuple[str, np.ndarray]:
    matrix, names = _motion_matrix(segment, imu_columns, cfg.smoothing_window)
    change = np.sum(np.abs(np.diff(matrix, axis=0)), axis=0) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    idx = int(np.argmax(change))
    return names[idx], matrix[:, idx]


def _segment_iou(a: SegmentRun, b: SegmentRun) -> float:
    left = max(a.start_idx, b.start_idx)
    right = min(a.end_idx, b.end_idx)
    inter = max(0, right - left)
    union = max(a.end_idx, b.end_idx) - min(a.start_idx, b.start_idx)
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(runs: Sequence[SegmentRun], max_iou: float) -> List[SegmentRun]:
    selected: List[SegmentRun] = []
    for run in sorted(runs, key=lambda r: r.confidence, reverse=True):
        if all(_segment_iou(run, kept) <= max_iou for kept in selected):
            selected.append(run)
    return sorted(selected, key=lambda r: r.start_idx)


def fit_dtw_micro_templates(
    sequences: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    cfg: DTWMicroConfig | None = None,
) -> Dict[str, DTWMicroTemplate]:
    if cfg is None:
        cfg = DTWMicroConfig()
    templates: Dict[str, DTWMicroTemplate] = {}
    for label in (CONCENTRIC_LABEL, ECCENTRIC_LABEL):
        segments: List[pd.DataFrame] = []
        for seq in sequences:
            if "phase" not in seq.columns:
                continue
            micro = micro_labels_from_phase(seq["phase"].to_numpy())
            runs = labels_to_runs(micro, positive_labels=(label,), min_length=3)
            for run in runs:
                segments.append(seq.iloc[run.start_idx:run.end_idx].copy().reset_index(drop=True))
        if not segments:
            continue
        durations = np.asarray([len(seg) for seg in segments], dtype=np.float64)
        median_duration = int(round(float(np.median(durations))))
        exemplar = sorted(segments, key=lambda seg: abs(len(seg) - median_duration))[0]
        signal_name, query = _best_signal(exemplar, imu_columns, cfg)
        costs: List[float] = []
        for seg in segments:
            matrix, names = _motion_matrix(seg, imu_columns, cfg.smoothing_window)
            if signal_name not in names:
                continue
            cost, _ = dtw_path_cost(query, matrix[:, names.index(signal_name)])
            if np.isfinite(cost):
                costs.append(float(cost))
        threshold = (
            float(np.percentile(costs, cfg.threshold_percentile)) + float(cfg.threshold_margin)
            if costs else 1.0 + float(cfg.threshold_margin)
        )
        min_duration = max(2, int(round(median_duration * float(cfg.duration_min_ratio))))
        max_duration = max(min_duration + 1, int(round(median_duration * float(cfg.duration_max_ratio))))
        templates[label] = DTWMicroTemplate(
            label=label,
            signal_name=signal_name,
            query=query,
            median_duration=median_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            cost_threshold=threshold,
        )
    return templates


@overload
def detect_dtw_micro_runs(
    df: pd.DataFrame,
    templates: Dict[str, DTWMicroTemplate],
    imu_columns: Sequence[str],
    cfg: DTWMicroConfig | None = None,
    return_stats: Literal[False] = False,
) -> List[SegmentRun]: ...


@overload
def detect_dtw_micro_runs(
    df: pd.DataFrame,
    templates: Dict[str, DTWMicroTemplate],
    imu_columns: Sequence[str],
    cfg: DTWMicroConfig | None = None,
    return_stats: Literal[True] = True,
) -> Tuple[List[SegmentRun], Dict[str, int]]: ...


def detect_dtw_micro_runs(
    df: pd.DataFrame,
    templates: Dict[str, DTWMicroTemplate],
    imu_columns: Sequence[str],
    cfg: DTWMicroConfig | None = None,
    return_stats: bool = False,
) -> List[SegmentRun]:
    if cfg is None:
        cfg = DTWMicroConfig()
    if not templates:
        return ([], {"windows_scored": 0, "candidates": 0}) if return_stats else []
    matrix, names = _motion_matrix(df, imu_columns, cfg.smoothing_window)
    candidates: List[SegmentRun] = []
    n = len(df)
    stride = max(1, int(cfg.detection_stride))
    duration_stride = max(1, int(cfg.duration_stride or cfg.detection_stride))
    downsample = max(1, int(cfg.dtw_downsample_factor))
    windows_scored = 0
    for label, template in templates.items():
        if template.signal_name not in names:
            continue
        signal = matrix[:, names.index(template.signal_name)]
        query = template.query[::downsample] if downsample > 1 else template.query
        durations = list(range(int(template.min_duration), int(template.max_duration) + 1, duration_stride))
        starts = list(range(0, max(0, n - template.min_duration + 1), stride))
        max_windows = int(cfg.max_windows_per_label)
        if max_windows > 0 and len(starts) * max(1, len(durations)) > max_windows:
            keep_starts = max(1, max_windows // max(1, len(durations)))
            idx = np.linspace(0, len(starts) - 1, num=min(keep_starts, len(starts)), dtype=int)
            starts = [starts[int(i)] for i in np.unique(idx)]
        for start in starts:
            for duration in durations:
                end = start + duration
                if end > n:
                    break
                windows_scored += 1
                window = signal[start:end]
                if downsample > 1:
                    window = window[::downsample]
                cost, _ = dtw_path_cost(query, window)
                if not np.isfinite(cost) or cost > template.cost_threshold:
                    continue
                confidence = max(1e-4, 1.0 - float(cost) / max(template.cost_threshold, 1e-8))
                candidates.append(SegmentRun(label=label, start_idx=start, end_idx=end, confidence=confidence))
    kept = _nms(candidates, cfg.max_overlap_iou)
    if return_stats:
        return kept, {"windows_scored": windows_scored, "candidates": len(candidates), "kept": len(kept)}
    return kept


def dtw_runs_to_micro_scores(
    n_samples: int,
    runs: Sequence[SegmentRun],
    min_other_score: float = 0.55,
) -> np.ndarray:
    scores = np.zeros((int(n_samples), len(MICRO_LABELS)), dtype=np.float32)
    scores[:, MICRO_LABELS.index(OTHER_LABEL)] = float(min_other_score)
    for run in runs:
        if run.label not in MICRO_LABELS:
            continue
        idx = MICRO_LABELS.index(run.label)
        start = max(0, int(run.start_idx))
        end = min(int(n_samples), int(run.end_idx))
        if end <= start:
            continue
        conf = float(max(run.confidence, 1e-4))
        scores[start:end, idx] = np.maximum(scores[start:end, idx], conf)
        scores[start:end, MICRO_LABELS.index(OTHER_LABEL)] = np.minimum(
            scores[start:end, MICRO_LABELS.index(OTHER_LABEL)],
            max(0.05, 1.0 - conf),
        )
    sums = scores.sum(axis=1, keepdims=True)
    sums[sums < 1e-8] = 1.0
    return scores / sums
