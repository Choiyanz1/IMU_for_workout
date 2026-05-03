from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import random

import numpy as np
import pandas as pd
try:
    import torch
except Exception:  # pragma: no cover - lightweight preprocessing-only runtime
    torch = None


@dataclass
class WindowConfig:
    sample_rate_hz: int = 50
    window_seconds: float = 1.5
    stride_seconds: float = 0.5

    @property
    def window_size(self) -> int:
        return int(self.sample_rate_hz * self.window_seconds)

    @property
    def stride_size(self) -> int:
        return int(self.sample_rate_hz * self.stride_seconds)


@dataclass
class ZScoreStats:
    mean: np.ndarray
    std: np.ndarray

    def save(self, path: Path) -> None:
        payload = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ZScoreStats":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


class LabelEncoder:
    def __init__(self) -> None:
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: Dict[int, str] = {}

    def fit(self, labels: Sequence[str]) -> None:
        classes = sorted(set(str(x) for x in labels))
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.index_to_class = {i: c for c, i in self.class_to_index.items()}

    def transform(self, labels: Sequence[str]) -> np.ndarray:
        return np.asarray([self.class_to_index[str(x)] for x in labels], dtype=np.int64)

    def inverse_transform(self, indices: Sequence[int]) -> List[str]:
        return [self.index_to_class[int(i)] for i in indices]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.class_to_index, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _ensure_datetime_or_numeric_time(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.number):
        return series.to_numpy(dtype=np.float64)

    dt = pd.to_datetime(series, errors="coerce")
    if dt.isna().all():
        raise ValueError("time column could not be parsed as numeric or datetime")
    return (dt.astype("int64") / 1e9).to_numpy(dtype=np.float64)


def _auto_detect_time_unit(time_values: np.ndarray) -> float:
    """Detect whether timestamps are in seconds, milliseconds, or microseconds.

    Returns a divisor to convert to seconds.
    Heuristic: compute median inter-sample interval and check which unit
    gives a plausible sensor rate (1-1000 Hz).
    """
    if len(time_values) < 2:
        return 1.0
    diffs = np.diff(time_values)
    median_diff = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0

    # Try each unit and see which gives a plausible sample rate
    for divisor, unit_name in [(1.0, "seconds"), (1e3, "milliseconds"), (1e6, "microseconds")]:
        rate = divisor / median_diff
        if 1.0 <= rate <= 2000.0:
            return divisor
    return 1.0


def resample_sequence(
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    time_column: str,
    target_rate_hz: int,
) -> pd.DataFrame:
    time_values = _ensure_datetime_or_numeric_time(df[time_column])
    features = df.loc[:, imu_columns].to_numpy(dtype=np.float32)

    if len(time_values) < 2:
        return df.copy().reset_index(drop=True)

    # Auto-detect time unit and convert to seconds
    divisor = _auto_detect_time_unit(time_values)
    if divisor != 1.0:
        time_values = time_values / divisor

    start_t, end_t = time_values[0], time_values[-1]
    step = 1.0 / float(target_rate_hz)
    new_t = np.arange(start_t, end_t + 1e-9, step, dtype=np.float64)

    if len(new_t) < 2:
        return df.copy().reset_index(drop=True)

    resampled = np.zeros((len(new_t), features.shape[1]), dtype=np.float32)
    for i in range(features.shape[1]):
        resampled[:, i] = np.interp(new_t, time_values, features[:, i]).astype(np.float32)

    out = pd.DataFrame(resampled, columns=imu_columns)
    out[time_column] = new_t

    # For non-IMU, non-time columns: use nearest-neighbor mapping when the
    # column has multiple unique values (e.g. phase), otherwise broadcast.
    nearest_idx = np.searchsorted(time_values, new_t, side="left").clip(0, len(time_values) - 1)
    # Refine: pick the closer of left and left-1
    left = np.maximum(nearest_idx - 1, 0)
    right = nearest_idx
    dist_left = np.abs(new_t - time_values[left])
    dist_right = np.abs(new_t - time_values[right])
    nearest_idx = np.where(dist_left <= dist_right, left, right)

    for col in df.columns:
        if col in imu_columns or col == time_column:
            continue
        col_values = df[col].to_numpy()
        if len(set(col_values[:20])) > 1 or len(set(col_values)) > 1:
            # Column varies within the sequence → nearest-neighbor map
            out[col] = col_values[nearest_idx]
        else:
            out[col] = col_values[0]

    return out.reset_index(drop=True)


def compute_train_stats(sequences: Sequence[pd.DataFrame], imu_columns: Sequence[str]) -> ZScoreStats:
    stacked = np.concatenate(
        [seq.loc[:, imu_columns].to_numpy(dtype=np.float32) for seq in sequences if not seq.empty],
        axis=0,
    )
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ZScoreStats(mean=mean, std=std)


def apply_zscore(df: pd.DataFrame, imu_columns: Sequence[str], stats: ZScoreStats) -> pd.DataFrame:
    out = df.copy()
    x = out.loc[:, imu_columns].to_numpy(dtype=np.float32)
    x = (x - stats.mean) / stats.std
    out.loc[:, imu_columns] = x
    return out


def extract_windows(
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    label_column: str,
    subject_column: str,
    window_cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    x = df.loc[:, imu_columns].to_numpy(dtype=np.float32)
    labels = df[label_column].astype(str).to_numpy()

    ws = window_cfg.window_size
    ss = window_cfg.stride_size

    if len(df) < ws:
        return np.empty((0, ws, len(imu_columns)), dtype=np.float32), np.empty((0,), dtype=object), []

    windows: List[np.ndarray] = []
    y: List[str] = []
    meta: List[Dict[str, str]] = []

    subject_value = str(df.iloc[0][subject_column])

    for start in range(0, len(df) - ws + 1, ss):
        end = start + ws
        seg_x = x[start:end]
        seg_y = labels[start:end]

        values, counts = np.unique(seg_y, return_counts=True)
        label = str(values[np.argmax(counts)])

        windows.append(seg_x)
        y.append(label)
        meta.append(
            {
                "subject_id": subject_value,
                "start_idx": str(start),
                "end_idx": str(end),
            }
        )

    return np.stack(windows), np.asarray(y, dtype=object), meta


def split_subjects(
    subjects: Sequence[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    unique_subjects = sorted(set(str(s) for s in subjects))
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 1

    n_train = min(n_train, n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(n_val, n - n_train - 1) if n >= 3 else 0

    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train : n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val :]

    if not test_subjects and val_subjects:
        test_subjects = [val_subjects.pop()]
    if not val_subjects and len(train_subjects) > 1:
        val_subjects = [train_subjects.pop()]

    return train_subjects, val_subjects, test_subjects


def build_window_dataset(
    sequences: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    label_column: str,
    subject_column: str,
    window_cfg: WindowConfig,
    label_encoder: Optional[LabelEncoder] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]], LabelEncoder]:
    cached_windows: List[np.ndarray] = []
    cached_labels: List[np.ndarray] = []
    cached_meta: List[List[Dict[str, str]]] = []
    raw_labels: List[str] = []

    for seq in sequences:
        w, y, m = extract_windows(seq, imu_columns, label_column, subject_column, window_cfg)
        if len(w) == 0:
            continue
        cached_windows.append(w)
        cached_labels.append(y)
        cached_meta.append(m)
        raw_labels.extend(y.tolist())

    if not cached_windows:
        raise RuntimeError("No windows extracted. Check sample rate, window length, and CSV quality.")

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(raw_labels)

    all_windows = np.concatenate(cached_windows, axis=0)
    all_labels = np.concatenate([label_encoder.transform(y.tolist()) for y in cached_labels], axis=0)

    all_meta: List[Dict[str, str]] = []
    for m in cached_meta:
        all_meta.extend(m)

    return all_windows, all_labels, all_meta, label_encoder
