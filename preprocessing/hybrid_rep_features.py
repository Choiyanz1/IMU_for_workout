"""Per-candidate feature extraction for hybrid SDTW + classifier rep segmentation.

A "candidate" is an SDTW-proposed segment (start_idx, end_idx, cost, ...).
The features computed here describe the underlying IMU window so a tabular
classifier (e.g. AutoGluon) can decide whether the candidate is a true rep
or a false positive (e.g. rest noise, partial movement).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocessing.sdtw_rep_segmentation import SegmentDetection


def _segment_array(
    stream_df: pd.DataFrame,
    candidate: SegmentDetection,
    columns: Sequence[str],
) -> np.ndarray:
    if not columns:
        return np.empty((0, 0), dtype=np.float64)
    end = min(int(candidate.end_idx), len(stream_df))
    start = max(0, int(candidate.start_idx))
    if end <= start:
        return np.empty((0, len(columns)), dtype=np.float64)
    return stream_df.iloc[start:end][list(columns)].to_numpy(dtype=np.float64)


def _per_channel_stats(arr: np.ndarray, columns: Sequence[str], prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if arr.size == 0:
        return feats
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    vmin = arr.min(axis=0)
    vmax = arr.max(axis=0)
    rng = vmax - vmin
    rms = np.sqrt(np.mean(arr ** 2, axis=0))
    centered = arr - mean[None, :]
    var = np.sum(centered ** 2, axis=0)
    var = np.where(var < 1e-12, 1.0, var)
    if len(arr) > 1:
        autocorr1 = np.sum(centered[:-1, :] * centered[1:, :], axis=0) / var
        signs = np.sign(centered)
        zcr = np.abs(np.diff(signs, axis=0)).sum(axis=0) / (len(arr) - 1) / 2.0
    else:
        autocorr1 = np.zeros_like(mean)
        zcr = np.zeros_like(mean)
    for ci, name in enumerate(columns):
        feats[f"{prefix}_{name}_mean"] = float(mean[ci])
        feats[f"{prefix}_{name}_std"] = float(std[ci])
        feats[f"{prefix}_{name}_min"] = float(vmin[ci])
        feats[f"{prefix}_{name}_max"] = float(vmax[ci])
        feats[f"{prefix}_{name}_range"] = float(rng[ci])
        feats[f"{prefix}_{name}_rms"] = float(rms[ci])
        feats[f"{prefix}_{name}_autocorr1"] = float(autocorr1[ci])
        feats[f"{prefix}_{name}_zcr"] = float(zcr[ci])
    return feats


def _magnitude_stats(arr: np.ndarray, idx: Sequence[int], prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    if arr.size == 0 or len(idx) < 2:
        return feats
    mag = np.sqrt(np.sum(arr[:, list(idx)] ** 2, axis=1))
    feats[f"{prefix}_mean"] = float(mag.mean())
    feats[f"{prefix}_std"] = float(mag.std())
    feats[f"{prefix}_min"] = float(mag.min())
    feats[f"{prefix}_max"] = float(mag.max())
    feats[f"{prefix}_range"] = float(mag.max() - mag.min())
    feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(mag ** 2)))
    return feats


def _edge_window(
    stream_df: pd.DataFrame,
    centre_idx: int,
    columns: Sequence[str],
    half_window: int,
) -> np.ndarray:
    """Return an array slice [centre - half, centre + half] clipped to stream."""
    if not columns or half_window <= 0:
        return np.empty((0, len(columns)), dtype=np.float64)
    n = len(stream_df)
    lo = max(0, int(centre_idx) - int(half_window))
    hi = min(n, int(centre_idx) + int(half_window) + 1)
    if hi <= lo:
        return np.empty((0, len(columns)), dtype=np.float64)
    return stream_df.iloc[lo:hi][list(columns)].to_numpy(dtype=np.float64)


def _edge_gradient(
    stream_df: pd.DataFrame,
    centre_idx: int,
    columns: Sequence[str],
    half_window: int,
) -> Dict[str, float]:
    """First difference around the boundary point, summarised per channel."""
    feats: Dict[str, float] = {}
    arr = _edge_window(stream_df, centre_idx, columns, half_window)
    if arr.shape[0] < 2:
        return feats
    diffs = np.diff(arr, axis=0)
    abs_diffs = np.abs(diffs)
    mean = abs_diffs.mean(axis=0)
    peak = abs_diffs.max(axis=0)
    direction = diffs.mean(axis=0)
    for ci, name in enumerate(columns):
        feats[f"edge_grad_{name}_mean_abs"] = float(mean[ci])
        feats[f"edge_grad_{name}_peak_abs"] = float(peak[ci])
        feats[f"edge_grad_{name}_direction"] = float(direction[ci])
    return feats


def compute_candidate_features(
    stream_df: pd.DataFrame,
    candidate: SegmentDetection,
    imu_columns: Sequence[str],
    sample_rate_hz: float,
    edge_window_samples: int = 0,
) -> Dict[str, float]:
    """Build a flat feature dict for a single SDTW candidate.

    Includes:
    - SDTW cost / normalized cost
    - Duration (samples + seconds) and stream position
    - Per-channel statistics over the candidate window
    - Accelerometer / gyroscope / magnetometer magnitude statistics
    - Optional edge-window features around the predicted start / end. These
      describe local IMU dynamics on each boundary and are what the boundary
      refiner regresses on. Set ``edge_window_samples > 0`` to include them.
    """
    available = [col for col in imu_columns if col in stream_df.columns]
    seg = _segment_array(stream_df, candidate, available)

    duration = int(max(0, candidate.end_idx - candidate.start_idx))
    stream_len = max(1, len(stream_df))
    feats: Dict[str, float] = {
        "sdtw_cost": float(candidate.cost),
        "sdtw_normalized_cost": float(candidate.normalized_cost),
        "duration_samples": float(duration),
        "duration_seconds": float(duration) / max(float(sample_rate_hz), 1e-6),
        "stream_position": float(candidate.start_idx) / float(stream_len),
        "stream_length_samples": float(stream_len),
    }

    feats.update(_per_channel_stats(seg, available, "seg"))

    acc_idx = [i for i, c in enumerate(available) if c.startswith("a")]
    gyro_idx = [i for i, c in enumerate(available) if c.startswith("g")]
    mag_idx = [i for i, c in enumerate(available) if c.startswith("m")]
    feats.update(_magnitude_stats(seg, acc_idx, "acc_mag"))
    feats.update(_magnitude_stats(seg, gyro_idx, "gyro_mag"))
    feats.update(_magnitude_stats(seg, mag_idx, "mag_mag"))

    if edge_window_samples and edge_window_samples > 0:
        for tag, centre in (("start", candidate.start_idx), ("end", candidate.end_idx)):
            edge_arr = _edge_window(stream_df, centre, available, edge_window_samples)
            feats.update(_per_channel_stats(edge_arr, available, f"edge_{tag}"))
            feats.update(_magnitude_stats(edge_arr, acc_idx, f"edge_{tag}_acc_mag"))
            feats.update(_magnitude_stats(edge_arr, gyro_idx, f"edge_{tag}_gyro_mag"))
            grad = _edge_gradient(stream_df, centre, available, edge_window_samples)
            for k, v in grad.items():
                feats[k.replace("edge_grad_", f"edge_{tag}_grad_")] = v

    return feats


def candidate_iou(candidate: SegmentDetection, truth_start: int, truth_end: int) -> float:
    left = max(int(candidate.start_idx), int(truth_start))
    right = min(int(candidate.end_idx), int(truth_end))
    inter = max(0, right - left)
    union = max(int(candidate.end_idx), int(truth_end)) - min(int(candidate.start_idx), int(truth_start))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def label_candidates_by_iou(
    candidates: Sequence[SegmentDetection],
    truth_segments: Sequence[Tuple[int, int]],
    iou_threshold: float,
) -> List[int]:
    """Label each candidate 1 if best-IoU vs any ground-truth segment >= threshold."""
    labels: List[int] = []
    for cand in candidates:
        best = 0.0
        for start, end in truth_segments:
            iou = candidate_iou(cand, start, end)
            if iou > best:
                best = iou
        labels.append(1 if best >= iou_threshold else 0)
    return labels


def best_iou_per_candidate(
    candidates: Sequence[SegmentDetection],
    truth_segments: Sequence[Tuple[int, int]],
) -> List[float]:
    out: List[float] = []
    for cand in candidates:
        best = 0.0
        for start, end in truth_segments:
            iou = candidate_iou(cand, start, end)
            if iou > best:
                best = iou
        out.append(best)
    return out


def best_truth_match(
    candidate: SegmentDetection,
    truth_segments: Sequence[Tuple[int, int]],
) -> Tuple[float, Tuple[int, int] | None]:
    """Return (best_iou, matched_truth) where matched_truth is None if there
    is no overlap. Used to derive boundary regression targets."""
    best_iou = 0.0
    best_seg: Tuple[int, int] | None = None
    for start, end in truth_segments:
        iou = candidate_iou(candidate, start, end)
        if iou > best_iou:
            best_iou = iou
            best_seg = (int(start), int(end))
    return best_iou, best_seg
