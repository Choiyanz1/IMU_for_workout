"""Run the exported full automatic workout pipeline on IMU samples.

This is a deployment-facing replay/stream skeleton. It uses causal trailing
windows and bounded fixed-lag decoding, but this script currently emits the
final JSON summary after the input stream ends. The `FullAutoWorkoutPipeline`
class is structured so it can be moved into an incremental Luckfox loop later.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW_COL_IMU_START = 14
IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")
ACTIONS = [
    "db_bench_press",
    "db_biceps_curl",
    "db_rdl",
    "db_shoulder_press",
    "db_squat",
    "db_triceps_curl",
    "db_weighted_crunch",
    "one_arm_db_row",
]
TOP5_ACTIONS = ["db_rdl", "db_shoulder_press", "db_bench_press", "one_arm_db_row", "db_weighted_crunch"]


@dataclass
class PipelineResult:
    samples: int
    active_samples: int
    top_action: str | None
    top_confidence: float
    action_posterior: list[float] | None
    raw_count_before_soft_merge: int
    count: int
    reps: list[dict[str, int | float | str]]


@dataclass
class RepEvent:
    start_idx: int
    transition_idx: int
    end_idx: int
    source: str = "online_phase"
    confidence: float = 1.0


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)


def trailing_window(values: np.ndarray, end_exclusive: int, size: int) -> np.ndarray:
    start = max(0, int(end_exclusive) - int(size))
    window = values[start:end_exclusive]
    if len(window) == 0:
        window = values[:1]
    if len(window) < size:
        window = np.pad(window, ((size - len(window), 0), (0, 0)), mode="edge")
    return window.astype(np.float32, copy=False)


def window_ends(n: int, window_samples: int, stride_samples: int) -> list[int]:
    if n <= 0:
        return []
    return [end for end in sorted(set([1, n, *range(int(stride_samples), n + 1, int(stride_samples))])) if end > 0]


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    def stats(values: np.ndarray) -> np.ndarray:
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        vmin = np.min(values, axis=1)
        vmax = np.max(values, axis=1)
        median = np.median(values, axis=1)
        q25 = np.quantile(values, 0.25, axis=1)
        q75 = np.quantile(values, 0.75, axis=1)
        iqr = q75 - q25
        rms = np.sqrt(np.mean(values**2, axis=1))
        variation = np.sum(np.abs(np.diff(values, axis=1)), axis=1)
        return np.concatenate([mean, std, vmin, vmax, median, q25, q75, iqr, rms, variation], axis=1)

    features = [stats(arr)]
    diff = np.diff(arr, axis=1, prepend=arr[:, :1, :])
    features.append(stats(diff))
    for values in (arr, diff):
        acc_norm = np.linalg.norm(values[:, :, :3], axis=2)[:, :, None]
        gyro_norm = np.linalg.norm(values[:, :, 3:6], axis=2)[:, :, None]
        features.append(stats(acc_norm))
        features.append(stats(gyro_norm))
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def extract_active_base_features_batch(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    vmin = np.min(arr, axis=1)
    vmax = np.max(arr, axis=1)
    median = np.median(arr, axis=1)
    q25 = np.quantile(arr, 0.25, axis=1)
    q75 = np.quantile(arr, 0.75, axis=1)
    total_variation = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    mag = np.sqrt(np.sum(arr**2, axis=2))
    mag_stats = np.stack([np.mean(mag, axis=1), np.std(mag, axis=1), np.max(mag, axis=1)], axis=1)
    per_channel = np.stack([mean, std, vmin, vmax, median, q25, q75, total_variation], axis=-1).reshape(arr.shape[0], -1)
    return np.concatenate([per_channel, mag_stats], axis=1).astype(np.float32, copy=False)


def normalized_autocorr_max(signal: np.ndarray, min_lag: int, max_lag: int, lag_step: int = 5) -> tuple[float, int]:
    x = np.asarray(signal, dtype=np.float32)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-8 or len(x) < min_lag + 2:
        return 0.0, 0
    max_lag = min(max_lag, len(x) - 1)
    best = 0.0
    best_lag = 0
    for lag in range(max(1, min_lag), max_lag + 1, max(1, int(lag_step))):
        score = float(np.dot(x[:-lag], x[lag:]) / denom)
        if score > best:
            best = score
            best_lag = lag
    return best, best_lag


def spectral_features(signal: np.ndarray, sample_rate_hz: float, band: tuple[float, float]) -> tuple[float, float, float]:
    x = np.asarray(signal, dtype=np.float32)
    x = x - float(np.mean(x))
    if len(x) < 4:
        return 0.0, 0.0, 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    power[0] = 0.0
    total = float(np.sum(power))
    if total <= 1e-8:
        return 0.0, 0.0, 0.0
    lo, hi = band
    band_mask = (freqs >= lo) & (freqs <= hi)
    band_power = float(np.sum(power[band_mask]) / total)
    dom_idx = int(np.argmax(power))
    dom_freq = float(freqs[dom_idx])
    probs = power / total
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))) / np.log(len(probs)))
    return band_power, dom_freq, entropy


def periodic_features_batch(windows: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    rows = []
    min_lag = int(round(sample_rate_hz * 0.25))
    max_lag = int(round(sample_rate_hz * 2.0))
    for window in arr:
        acc_mag = np.linalg.norm(window[:, :3], axis=1)
        gyro_mag = np.linalg.norm(window[:, 3:6], axis=1)
        jerk = np.diff(window, axis=0, prepend=window[:1])
        acc_jerk_mag = np.linalg.norm(jerk[:, :3], axis=1)
        gyro_jerk_mag = np.linalg.norm(jerk[:, 3:6], axis=1)
        feats: list[float] = []
        for signal in (acc_mag, gyro_mag, acc_jerk_mag, gyro_jerk_mag):
            ac, lag = normalized_autocorr_max(signal, min_lag, max_lag, lag_step=5)
            band_power, dom_freq, entropy = spectral_features(signal, sample_rate_hz, (0.25, 3.0))
            centered = signal - float(np.mean(signal))
            zero_cross = float(np.mean(centered[:-1] * centered[1:] < 0)) if len(centered) > 1 else 0.0
            feats.extend(
                [
                    float(np.mean(signal)),
                    float(np.std(signal)),
                    float(np.percentile(signal, 95) - np.percentile(signal, 5)),
                    float(np.mean(signal**2)),
                    ac,
                    float(lag / sample_rate_hz) if lag else 0.0,
                    band_power,
                    dom_freq,
                    entropy,
                    zero_cross,
                ]
            )
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


def extract_gate_features(windows: np.ndarray, mode: str, sample_rate_hz: float) -> np.ndarray:
    base = extract_active_base_features_batch(windows)
    if mode == "basic":
        return base
    return np.concatenate([base, periodic_features_batch(windows, sample_rate_hz)], axis=1).astype(np.float32, copy=False)


def _parse_raw_zig_line(line: str) -> tuple[float, float, float, float, float, float] | None:
    cols = [c.strip() for c in line.split(",")]
    if len(cols) < RAW_COL_IMU_START + 6 or (len(cols) > 1 and cols[1] != "IMU"):
        return None
    try:
        return tuple(float(cols[RAW_COL_IMU_START + i]) for i in range(6))  # type: ignore[return-value]
    except ValueError:
        return None


def _iter_samples_from_lines(lines: Iterable[str]) -> Iterable[tuple[float, float, float, float, float, float]]:
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("serial_num"):
            continue
        parsed = _parse_raw_zig_line(raw)
        if parsed is not None:
            yield parsed


def smooth_ma(phase_probs: np.ndarray, window: int) -> np.ndarray:
    n = len(phase_probs)
    out = np.copy(phase_probs)
    if window <= 1:
        return out
    for c in range(phase_probs.shape[1]):
        cumsum = np.cumsum(phase_probs[:, c])
        for i in range(n):
            start = max(0, i - window + 1)
            total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
            out[i, c] = total / (i - start + 1)
    return out


def clean_active_mask(mask: np.ndarray, min_active_samples: int, bridge_gap_samples: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    n = len(out)
    bridge = max(0, int(bridge_gap_samples))
    if bridge:
        i = 0
        while i < n:
            if out[i]:
                i += 1
                continue
            start = i
            while i < n and not out[i]:
                i += 1
            if start > 0 and i < n and i - start <= bridge:
                out[start:i] = True
    min_len = max(0, int(min_active_samples))
    if min_len > 1:
        i = 0
        while i < n:
            if not out[i]:
                i += 1
                continue
            start = i
            while i < n and out[i]:
                i += 1
            if i - start < min_len:
                out[start:i] = False
    return out


def active_state_machine(prob: np.ndarray, args: SimpleNamespace) -> np.ndarray:
    n = len(prob)
    state = False
    enter_count = 0
    exit_count = 0
    mask = np.zeros(n, dtype=bool)
    enter_hold = max(1, int(args.enter_hold_samples))
    exit_hold = max(1, int(args.exit_hold_samples))
    cooldown_until = -1
    for i, p in enumerate(prob):
        if not state:
            if i < cooldown_until:
                enter_count = 0
                continue
            if p >= args.enter_threshold:
                enter_count += 1
                if enter_count >= enter_hold:
                    state = True
                    start = max(0, i - enter_hold + 1)
                    mask[start : i + 1] = True
                    exit_count = 0
            else:
                enter_count = 0
        else:
            if p < args.exit_threshold:
                exit_count += 1
                if exit_count >= exit_hold:
                    end = max(0, i - exit_hold + 1)
                    mask[end : i + 1] = False
                    state = False
                    cooldown_until = i + max(0, int(args.cooldown_samples))
                    enter_count = 0
                    exit_count = 0
                else:
                    mask[i] = True
            else:
                exit_count = 0
                mask[i] = True
    return clean_active_mask(mask, args.min_active_samples, args.bridge_gap_samples)


def fixed_lag_viterbi_decode(phase_probs: np.ndarray, penalty: float, lag_samples: int) -> np.ndarray:
    n = len(phase_probs)
    if n == 0:
        return phase_probs.copy()
    lag = max(0, int(lag_samples))
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2), dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    dp[0] = log_probs[0]
    for t in range(1, n):
        for state in range(2):
            stay = dp[t - 1, state]
            switch = dp[t - 1, 1 - state] - penalty
            if stay >= switch:
                dp[t, state] = log_probs[t, state] + stay
                back[t, state] = state
            else:
                dp[t, state] = log_probs[t, state] + switch
                back[t, state] = 1 - state
        finalize_idx = t - lag
        if finalize_idx >= 0:
            state = int(np.argmax(dp[t]))
            for k in range(t, finalize_idx, -1):
                state = int(back[k, state])
            labels[finalize_idx] = state
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    labels[labels < 0] = path[labels < 0]
    result = np.zeros((n, 2), dtype=np.float32)
    result[labels == 0, 0] = 1.0
    result[labels == 1, 1] = 1.0
    return result


class OnlineRepParser:
    def __init__(self, min_phase_samples: int = 3, max_gap_samples: int = 3) -> None:
        self.min_phase_samples = int(min_phase_samples)
        self.max_gap_samples = int(max_gap_samples)
        self.current_label: str | None = None
        self.current_start: int | None = None
        self.last_concentric: tuple[int, int] | None = None

    def update(self, idx: int, label: str | None) -> list[RepEvent]:
        emitted: list[RepEvent] = []
        if label is None:
            emitted.extend(self._close_run(idx))
            self.current_label = None
            self.current_start = None
            return emitted
        if self.current_label is None:
            self.current_label = label
            self.current_start = idx
            return emitted
        if label != self.current_label:
            emitted.extend(self._close_run(idx))
            self.current_label = label
            self.current_start = idx
        return emitted

    def finish(self, n_samples: int) -> list[RepEvent]:
        return self._close_run(n_samples)

    def _close_run(self, end_idx: int) -> list[RepEvent]:
        if self.current_label is None or self.current_start is None:
            return []
        start = int(self.current_start)
        end = int(end_idx)
        label = self.current_label
        self.current_label = None
        self.current_start = None
        if end - start < self.min_phase_samples:
            return []
        if label == "concentric":
            self.last_concentric = (start, end)
            return []
        if label == "eccentric" and self.last_concentric is not None:
            c_start, c_end = self.last_concentric
            if start - c_end <= self.max_gap_samples:
                self.last_concentric = None
                return [RepEvent(start_idx=int(c_start), transition_idx=int(start), end_idx=int(end))]
        return []


def parse_reps_masked(hard_labels: np.ndarray, active_mask: np.ndarray) -> list[RepEvent]:
    parser = OnlineRepParser()
    reps: list[RepEvent] = []
    for idx, label_idx in enumerate(hard_labels):
        label = None
        if bool(active_mask[idx]):
            label = "eccentric" if int(label_idx) == 0 else "concentric"
        reps.extend(parser.update(idx, label))
    reps.extend(parser.finish(len(hard_labels)))
    return reps


def filter_event_confirmed_reps(reps: list[RepEvent], min_reps: int, max_gap_samples: int) -> list[RepEvent]:
    if min_reps <= 0 or not reps:
        return reps
    ordered = sorted(reps, key=lambda rep: int(rep.start_idx))
    groups: list[list[RepEvent]] = []
    current = [ordered[0]]
    for rep in ordered[1:]:
        gap = int(rep.start_idx) - int(current[-1].end_idx)
        if gap <= int(max_gap_samples):
            current.append(rep)
        else:
            groups.append(current)
            current = [rep]
    groups.append(current)
    kept: list[RepEvent] = []
    for group in groups:
        if len(group) >= int(min_reps):
            kept.extend(group)
    return kept


class OnlineSoftMerger:
    def __init__(self, max_gap_samples: int) -> None:
        self.max_gap_samples = int(max_gap_samples)
        self.pending: RepEvent | None = None
        self.final_reps: list[RepEvent] = []

    def add(self, rep: RepEvent, threshold: float | None) -> None:
        if threshold is None or threshold <= 0:
            self._flush_pending()
            self.final_reps.append(rep)
            return
        duration = int(rep.end_idx) - int(rep.start_idx)
        if self.pending is not None:
            gap = int(rep.start_idx) - int(self.pending.end_idx)
            if gap <= self.max_gap_samples:
                self.final_reps.append(self._merge(self.pending, rep))
                self.pending = None
                return
            self._flush_pending()
        if duration < threshold:
            self.pending = rep
        else:
            self.final_reps.append(rep)

    def finish(self) -> list[RepEvent]:
        self._flush_pending()
        return self.final_reps

    def _flush_pending(self) -> None:
        if self.pending is not None:
            self.final_reps.append(self.pending)
            self.pending = None

    @staticmethod
    def _merge(a: RepEvent, b: RepEvent) -> RepEvent:
        return RepEvent(start_idx=int(a.start_idx), transition_idx=int(a.transition_idx), end_idx=int(b.end_idx), source="online_soft_merge")


class StatefulSoftMerger:
    def __init__(self, max_gap_samples: int) -> None:
        self.max_gap_samples = int(max_gap_samples)
        self.pending: RepEvent | None = None

    def add(self, rep: RepEvent, threshold: float | None) -> list[RepEvent]:
        emitted: list[RepEvent] = []
        if threshold is None or threshold <= 0:
            emitted.extend(self.flush())
            emitted.append(rep)
            return emitted
        duration = int(rep.end_idx) - int(rep.start_idx)
        if self.pending is not None:
            gap = int(rep.start_idx) - int(self.pending.end_idx)
            if gap <= self.max_gap_samples:
                emitted.append(OnlineSoftMerger._merge(self.pending, rep))
                self.pending = None
                return emitted
            emitted.extend(self.flush())
        if duration < threshold:
            self.pending = rep
        else:
            emitted.append(rep)
        return emitted

    def flush(self) -> list[RepEvent]:
        if self.pending is None:
            return []
        rep = self.pending
        self.pending = None
        return [rep]


def threshold_for_action(duration_priors: dict, action: str, percentile: int = 5) -> float:
    key = str(percentile)
    if action in duration_priors and key in duration_priors[action]:
        return float(duration_priors[action][key])
    return float(duration_priors.get("__global__", {}).get(key, 0.0))


def soft_threshold_from_context(posterior: np.ndarray | None, duration_priors: dict, args: SimpleNamespace) -> float | None:
    if posterior is None:
        return None
    order = np.argsort(posterior)[::-1]
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    top5_indices = [ACTIONS.index(action) for action in TOP5_ACTIONS if action in ACTIONS]
    top5_mass = float(np.sum(posterior[top5_indices]))
    top_conf = float(posterior[top])
    margin = float(posterior[top] - posterior[second])
    if top5_mass < args.soft_top5_mass_threshold or top_conf < args.soft_action_confidence_threshold or margin < args.soft_margin_threshold:
        return None
    weights = np.asarray([posterior[ACTIONS.index(action)] for action in TOP5_ACTIONS], dtype=np.float32)
    thresholds = np.asarray([threshold_for_action(duration_priors, action, 5) for action in TOP5_ACTIONS], dtype=np.float32)
    return float(np.dot(weights, thresholds) / max(float(weights.sum()), 1e-8))


def apply_online_soft_merge(reps: list[RepEvent], posterior_by_sample, duration_priors: dict, args: SimpleNamespace, scale: float) -> list[RepEvent]:
    merger = OnlineSoftMerger(args.max_gap_samples)
    n = len(posterior_by_sample)
    for rep in reps:
        idx = min(max(int(rep.end_idx) + int(args.fixed_lag_samples), 0), max(0, n - 1))
        threshold = soft_threshold_from_context(posterior_by_sample[idx], duration_priors, args)
        if threshold is not None:
            threshold *= float(scale)
        merger.add(rep, threshold)
    return merger.finish()


def probs_for_actions(action_rf, raw_probs: np.ndarray) -> np.ndarray:
    out = np.zeros((len(raw_probs), len(ACTIONS)), dtype=np.float32)
    class_to_col = {str(cls): idx for idx, cls in enumerate(action_rf.classes_)}
    for j, action in enumerate(ACTIONS):
        col = class_to_col.get(action)
        if col is not None:
            out[:, j] = raw_probs[:, col]
    return out


def load_samples(path: str) -> np.ndarray:
    if path == "-":
        return np.asarray(list(_iter_samples_from_lines(sys.stdin)), dtype=np.float32)
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if all(col in first for col in IMU_COLUMNS):
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                try:
                    rows.append(tuple(float(row[col]) for col in IMU_COLUMNS))
                except (TypeError, ValueError, KeyError):
                    continue
            return np.asarray(rows, dtype=np.float32)
        return np.asarray(list(_iter_samples_from_lines(f)), dtype=np.float32)


def iter_samples(path: str) -> Iterable[tuple[float, float, float, float, float, float]]:
    if path == "-":
        yield from _iter_samples_from_lines(sys.stdin)
        return
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if all(col in first for col in IMU_COLUMNS):
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    yield tuple(float(row[col]) for col in IMU_COLUMNS)  # type: ignore[misc]
                except (TypeError, ValueError, KeyError):
                    continue
        else:
            yield from _iter_samples_from_lines(f)


class TorchPhaseRunner:
    def __init__(self, artifact_dir: Path, device: str) -> None:
        import torch

        from scripts.new_c_pipeline.test_pca_input import CausalCNN_PhaseOnly

        checkpoint = torch.load(artifact_dir / "phase_model.pt", map_location=device)
        self.device = torch.device(device)
        self.model = CausalCNN_PhaseOnly(
            checkpoint.get("input_channels", 6),
            checkpoint.get("hidden", 64),
            checkpoint.get("num_classes", 2),
            checkpoint.get("dropout", 0.2),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.torch = torch

    def predict_tail(self, window: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(self.device)
            logits = self.model(x).cpu().numpy()[0, :, -1]
        return _softmax(logits[None, :], axis=1)[0].astype(np.float32)


class OnnxPhaseRunner:
    def __init__(self, artifact_dir: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(artifact_dir / "phase_model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_tail(self, window: np.ndarray) -> np.ndarray:
        x = window.astype(np.float32).T[None, :, :]
        logits = self.session.run(None, {self.input_name: x})[0][0, :, -1]
        return _softmax(logits[None, :], axis=1)[0].astype(np.float32)


class RknnPhaseRunner:
    def __init__(self, artifact_dir: Path) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ModuleNotFoundError:
            from rknn.api import RKNN as RKNNLite

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(artifact_dir / "phase_model.rknn"))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def predict_tail(self, window: np.ndarray) -> np.ndarray:
        x = window.astype(np.float32).T[None, :, :]
        logits = self.rknn.inference(inputs=[x])[0][0, :, -1]
        return _softmax(logits[None, :], axis=1)[0].astype(np.float32)


class JsonScaler:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.mean = np.asarray(data["mean"], dtype=np.float32)
        self.scale = np.asarray(data["scale"], dtype=np.float32)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.scale


class JsonRandomForest:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.classes_ = np.asarray(data["classes"], dtype=object)
        self.n_features_in_ = int(data["n_features_in"])
        self.trees = []
        for tree in data["trees"]:
            self.trees.append(
                {
                    "children_left": np.asarray(tree["children_left"], dtype=np.int32),
                    "children_right": np.asarray(tree["children_right"], dtype=np.int32),
                    "feature": np.asarray(tree["feature"], dtype=np.int32),
                    "threshold": np.asarray(tree["threshold"], dtype=np.float32),
                    "proba": np.asarray(tree["proba"], dtype=np.float32),
                }
            )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise RuntimeError(f"Expected 2D features, got {x.shape}")
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float32)
        for i, row in enumerate(x):
            proba = np.zeros((len(self.classes_),), dtype=np.float32)
            for tree in self.trees:
                node = 0
                while tree["children_left"][node] != -1:
                    feature = int(tree["feature"][node])
                    threshold = float(tree["threshold"][node])
                    if row[feature] <= threshold:
                        node = int(tree["children_left"][node])
                    else:
                        node = int(tree["children_right"][node])
                proba += tree["proba"][node]
            out[i] = proba / max(1, len(self.trees))
        return out


def load_rf_models(artifact_dir: Path, rf_runtime: str):
    if rf_runtime == "auto":
        rf_runtime = "json" if (artifact_dir / "active_gate_rf.json").exists() else "sklearn"
    if rf_runtime == "json":
        return (
            JsonRandomForest(artifact_dir / "active_gate_rf.json"),
            JsonScaler(artifact_dir / "active_gate_scaler.json"),
            JsonRandomForest(artifact_dir / "action_active_rf.json"),
            JsonRandomForest(artifact_dir / "action_rf.json"),
        )
    import joblib

    return (
        joblib.load(artifact_dir / "active_gate_rf.joblib"),
        joblib.load(artifact_dir / "active_gate_scaler.joblib"),
        joblib.load(artifact_dir / "action_active_rf.joblib"),
        joblib.load(artifact_dir / "action_rf.joblib"),
    )


class FullAutoWorkoutPipeline:
    def __init__(self, artifact_dir: Path, runtime: str = "torch", device: str = "cpu", rf_runtime: str = "auto") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((self.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.config = json.loads((self.artifact_dir / "pipeline_config.json").read_text(encoding="utf-8"))
        self.norm = json.loads((self.artifact_dir / "normalization.json").read_text(encoding="utf-8"))
        self.imu_columns = list(self.norm.get("imu_columns", IMU_COLUMNS))
        self.phase_mean = np.asarray(self.norm["phase_mean"], dtype=np.float32)
        self.phase_std = np.where(np.asarray(self.norm["phase_std"], dtype=np.float32) < 1e-8, 1.0, np.asarray(self.norm["phase_std"], dtype=np.float32))
        self.action_mean = np.asarray(self.norm["action_mean"], dtype=np.float32)
        self.action_std = np.where(np.asarray(self.norm["action_std"], dtype=np.float32) < 1e-8, 1.0, np.asarray(self.norm["action_std"], dtype=np.float32))
        self.active_clf, self.active_scaler, self.action_active_rf, self.action_rf = load_rf_models(self.artifact_dir, rf_runtime)
        if runtime == "torch":
            self.phase_runner = TorchPhaseRunner(self.artifact_dir, device)
        elif runtime == "onnx":
            self.phase_runner = OnnxPhaseRunner(self.artifact_dir)
        else:
            self.phase_runner = RknnPhaseRunner(self.artifact_dir)

    def _active_args(self) -> SimpleNamespace:
        cfg = self.config["active_gate"]
        return SimpleNamespace(
            window_samples=int(cfg["window_samples"]),
            stride_samples=int(cfg["stride_samples"]),
            sample_rate_hz=float(self.config["sample_rate_hz"]),
            enter_threshold=float(cfg["enter_threshold"]),
            exit_threshold=float(cfg["exit_threshold"]),
            enter_hold_samples=int(cfg["enter_hold_samples"]),
            exit_hold_samples=int(cfg["exit_hold_samples"]),
            min_active_samples=int(cfg["min_active_samples"]),
            bridge_gap_samples=int(cfg["bridge_gap_samples"]),
            cooldown_samples=0,
        )

    def _decoder_args(self) -> SimpleNamespace:
        phase = self.config["phase_decoder"]
        soft = self.config["soft_top5"]
        event = self.config["event_confirmation"]
        return SimpleNamespace(
            fixed_lag_samples=int(phase["fixed_lag_samples"]),
            max_gap_samples=int(soft["max_gap_samples"]),
            min_confirmed_reps=0,
            confirmed_set_gap_samples=300,
            soft_top5_mass_threshold=float(soft["top5_mass_threshold"]),
            soft_action_confidence_threshold=float(soft["action_confidence_threshold"]),
            soft_margin_threshold=float(soft["margin_threshold"]),
            event_confirm_min_reps=int(event["min_reps"]) if event.get("enabled", True) else 0,
            event_confirm_gap_samples=int(event["gap_samples"]),
            event_confirm_split_inactive_samples=0,
            event_confirm_post_event_cooldown_samples=0,
            event_confirm_action_active_threshold=0.5,
            event_confirm_action_active_min_fraction=0.0,
            event_confirm_action_active_min_mean=0.0,
            event_confirm_action_active_min_max=0.0,
            event_confirm_top_confidence_threshold=0.0,
            event_confirm_margin_threshold=0.0,
            event_confirm_top5_mass_threshold=0.0,
            event_confirm_use_rep_evidence=False,
            event_confirm_drop_low_evidence_reps=False,
            event_confirm_rep_action_active_min_max=0.0,
            event_confirm_rep_min_duration_samples=0,
            event_confirm_rep_max_duration_samples=0,
        )

    def _predict_active(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        active_cfg = self.config["active_gate"]
        args = self._active_args()
        n = len(samples)
        if n == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
        ends = window_ends(n, args.window_samples, args.stride_samples)
        windows = np.stack([trailing_window(samples, end, args.window_samples) for end in ends]).astype(np.float32)
        features = extract_gate_features(windows, str(active_cfg.get("feature_mode", "periodic")), args.sample_rate_hz)
        probs = self.active_clf.predict_proba(self.active_scaler.transform(features))
        class_to_col = {int(cls): idx for idx, cls in enumerate(self.active_clf.classes_)}
        active_col = class_to_col.get(1, 0)
        prob = np.zeros(n, dtype=np.float32)
        prev = 0
        for i, end in enumerate(ends):
            prob[prev:int(end)] = float(probs[i, active_col])
            prev = int(end)
        if prev < n:
            prob[prev:] = prob[prev - 1] if prev > 0 else 0.0
        mask = active_state_machine(prob, args)
        return prob.astype(np.float32), mask.astype(bool)

    def _predict_action_posterior(self, samples: np.ndarray) -> tuple[list[np.ndarray | None], np.ndarray | None, np.ndarray]:
        n = len(samples)
        cfg = self.config["action_branch"]
        window = int(cfg["window_samples"])
        stride = int(cfg["stride_samples"])
        ends = list(range(window, n + 1, stride))
        if n >= window and (not ends or ends[-1] != n):
            ends.append(n)
        posterior_by_sample: list[np.ndarray | None] = [None] * n
        action_active_by_sample = np.zeros(n, dtype=np.float32)
        if not ends:
            return posterior_by_sample, None, action_active_by_sample
        windows = np.stack([trailing_window(samples, end, window) for end in ends]).astype(np.float32)
        windows = (windows - self.action_mean) / self.action_std
        features = extract_features_batch(windows)
        active_raw = self.action_active_rf.predict_proba(features)
        active_col = {int(cls): idx for idx, cls in enumerate(self.action_active_rf.classes_)}.get(1, 0)
        active_probs = np.asarray([float(row[active_col]) for row in active_raw], dtype=np.float32)
        action_rows = probs_for_actions(self.action_rf, self.action_rf.predict_proba(features))
        running = np.zeros((len(ACTIONS),), dtype=np.float64)
        weight = 0.0
        posterior: np.ndarray | None = None
        prev = 0
        for i, end in enumerate(ends):
            w = max(float(active_probs[i]), 1e-6)
            running += np.asarray(action_rows[i], dtype=np.float64) * w
            weight += w
            posterior = (running / max(weight, 1e-8)).astype(np.float32)
            posterior_by_sample[prev:int(end)] = [posterior.copy()] * (int(end) - prev)
            action_active_by_sample[prev:int(end)] = float(active_probs[i])
            prev = int(end)
        if prev < n:
            posterior_by_sample[prev:] = [posterior.copy() if posterior is not None else None] * (n - prev)
            action_active_by_sample[prev:] = action_active_by_sample[prev - 1] if prev > 0 else 0.0
        return posterior_by_sample, posterior, action_active_by_sample

    def _predict_phase(self, samples: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        n = len(samples)
        phase_cfg = self.config["phase_decoder"]
        phase_raw = np.ones((n, 2), dtype=np.float32) * 0.5
        if n == 0:
            return phase_raw
        x = (samples - self.phase_mean) / self.phase_std
        step = int(phase_cfg["step_samples"])
        win = int(phase_cfg["window_samples"])
        ends = sorted(set([1, n, *range(step, n + 1, step)]))
        prev_t = -1
        for end in ends:
            t = int(end) - 1
            if bool(active_mask[t]):
                probs = self.phase_runner.predict_tail(trailing_window(x, end, win))
            else:
                probs = np.asarray([0.5, 0.5], dtype=np.float32)
            phase_raw[max(0, prev_t + 1) : t + 1] = probs
            prev_t = t
        smoothed = smooth_ma(phase_raw, int(phase_cfg["smoothing_window"]))
        fixed = fixed_lag_viterbi_decode(smoothed, float(phase_cfg["viterbi_penalty"]), int(phase_cfg["fixed_lag_samples"]))
        if bool(phase_cfg.get("fixed_lag_active_mask", True)):
            fixed[~active_mask] = 0.5
        return fixed

    def run(self, samples: np.ndarray) -> PipelineResult:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 6:
            raise RuntimeError(f"Expected [N, 6] samples, got {samples.shape}")
        active_prob, active_state = self._predict_active(samples)
        active_state = clean_active_mask(
            active_state,
            int(self.config["active_gate"].get("min_active_samples", 0)),
            int(self.config["active_gate"].get("bridge_gap_samples", 0)),
        )
        posterior_by_sample, posterior, action_active = self._predict_action_posterior(samples)
        fixed_probs = self._predict_phase(samples, active_state)
        reps = parse_reps_masked(np.argmax(fixed_probs, axis=1), active_state)

        args = self._decoder_args()
        # Event confirmation is applied before soft merge, matching the active setting used in research replay.
        reps = filter_event_confirmed_reps(reps, args.event_confirm_min_reps, args.event_confirm_gap_samples)
        soft_cfg = self.config["soft_top5"]
        merged = apply_online_soft_merge(
            reps,
            posterior_by_sample,
            soft_cfg.get("duration_priors", {}),
            args,
            float(soft_cfg.get("threshold_scale", 0.8)),
        )
        sample_rate = float(self.config.get("sample_rate_hz", 100.0))
        top_action = None
        top_conf = 0.0
        posterior_list = None
        if posterior is not None:
            order = np.argsort(posterior)[::-1]
            top_idx = int(order[0])
            top_action = ACTIONS[top_idx]
            top_conf = float(posterior[top_idx])
            posterior_list = [float(v) for v in posterior.tolist()]
        return PipelineResult(
            samples=int(len(samples)),
            active_samples=int(np.sum(active_state)),
            top_action=top_action,
            top_confidence=top_conf,
            action_posterior=posterior_list,
            raw_count_before_soft_merge=int(len(reps)),
            count=int(len(merged)),
            reps=[self._rep_to_dict(rep, sample_rate, top_action) for rep in merged],
        )

    @staticmethod
    def _rep_to_dict(rep, sample_rate: float, action: str | None) -> dict[str, int | float | str]:
        return {
            "action": action or "unknown",
            "start_idx": int(rep.start_idx),
            "transition_idx": int(rep.transition_idx),
            "end_idx": int(rep.end_idx),
            "start_sec": round(int(rep.start_idx) / sample_rate, 3),
            "transition_sec": round(int(rep.transition_idx) / sample_rate, 3),
            "end_sec": round(int(rep.end_idx) / sample_rate, 3),
        }


class StatefulLivePipeline:
    def __init__(self, pipeline: FullAutoWorkoutPipeline) -> None:
        self.pipeline = pipeline
        self.samples: list[tuple[float, float, float, float, float, float]] = []
        self.sample_rate = float(pipeline.config.get("sample_rate_hz", 100.0))
        self.active_cfg = pipeline.config["active_gate"]
        self.action_cfg = pipeline.config["action_branch"]
        self.phase_cfg = pipeline.config["phase_decoder"]
        self.soft_cfg = pipeline.config["soft_top5"]
        self.args = pipeline._decoder_args()
        self.active_state = False
        self.active_enter_count = 0
        self.active_exit_count = 0
        self.active_mask: list[bool] = []
        self.active_prob = 0.0
        self.last_active_update = 0
        self.posterior: np.ndarray | None = None
        self.action_running = np.zeros((len(ACTIONS),), dtype=np.float64)
        self.action_weight = 0.0
        self.action_active_prob = 0.0
        self.posterior_by_sample: list[np.ndarray | None] = []
        self.raw_phase_probs: list[np.ndarray] = []
        self.dp: list[np.ndarray] = []
        self.back: list[np.ndarray] = []
        self.finalized_until = -1
        self.parser = OnlineRepParser()
        self.event_group: list[RepEvent] = []
        self.event_confirmed = False
        self.soft_merger = StatefulSoftMerger(int(self.soft_cfg.get("max_gap_samples", 50)))
        self.emitted_count = 0

    def process_sample(self, sample: tuple[float, float, float, float, float, float]) -> list[dict[str, object]]:
        self.samples.append(sample)
        n = len(self.samples)
        self.posterior_by_sample.append(None if self.posterior is None else self.posterior.copy())
        self._maybe_update_active(n)
        self._maybe_update_action(n)
        events: list[dict[str, object]] = []
        if n == 1 or n % int(self.phase_cfg["step_samples"]) == 0:
            events.extend(self._update_phase_until(n))
        return events

    def finish(self) -> list[dict[str, object]]:
        events = self._update_phase_until(len(self.samples), force=True)
        for rep in self.parser.finish(len(self.samples)):
            events.extend(self._handle_raw_rep(rep))
        for rep in self._flush_event_group(final=True):
            events.extend(self._emit_confirmed_rep(rep))
        for rep in self.soft_merger.flush():
            events.append(self._rep_event(rep))
        events.append(
            {
                "type": "summary",
                "samples": len(self.samples),
                "active_samples": int(sum(self.active_mask)),
                "top_action": self.top_action(),
                "top_confidence": self.top_confidence(),
                "count": self.emitted_count,
            }
        )
        return events

    def _sample_array(self) -> np.ndarray:
        return np.asarray(self.samples, dtype=np.float32)

    def _maybe_update_active(self, n: int) -> None:
        stride = int(self.active_cfg["stride_samples"])
        if n != 1 and n % stride != 0:
            self.active_mask.append(self.active_state)
            return
        window = trailing_window(self._sample_array(), n, int(self.active_cfg["window_samples"]))[None, :, :]
        features = extract_gate_features(window, str(self.active_cfg.get("feature_mode", "periodic")), self.sample_rate)
        probs = self.pipeline.active_clf.predict_proba(self.pipeline.active_scaler.transform(features))
        active_col = {int(cls): idx for idx, cls in enumerate(self.pipeline.active_clf.classes_)}.get(1, 0)
        self.active_prob = float(probs[0, active_col])
        interval = max(1, n - self.last_active_update)
        if not self.active_state:
            if self.active_prob >= float(self.active_cfg["enter_threshold"]):
                self.active_enter_count += interval
                if self.active_enter_count >= int(self.active_cfg["enter_hold_samples"]):
                    self.active_state = True
                    self.active_exit_count = 0
            else:
                self.active_enter_count = 0
        else:
            if self.active_prob < float(self.active_cfg["exit_threshold"]):
                self.active_exit_count += interval
                if self.active_exit_count >= max(1, int(self.active_cfg["exit_hold_samples"])):
                    self.active_state = False
                    self.active_enter_count = 0
                    self.active_exit_count = 0
            else:
                self.active_exit_count = 0
        self.last_active_update = n
        self.active_mask.append(self.active_state)

    def _maybe_update_action(self, n: int) -> None:
        window_size = int(self.action_cfg["window_samples"])
        stride = int(self.action_cfg["stride_samples"])
        if n < window_size or n % stride != 0:
            self.posterior_by_sample[-1] = None if self.posterior is None else self.posterior.copy()
            return
        window = trailing_window(self._sample_array(), n, window_size)[None, :, :]
        window = (window - self.pipeline.action_mean) / self.pipeline.action_std
        features = extract_features_batch(window)
        active_raw = self.pipeline.action_active_rf.predict_proba(features)
        active_col = {int(cls): idx for idx, cls in enumerate(self.pipeline.action_active_rf.classes_)}.get(1, 0)
        self.action_active_prob = float(active_raw[0, active_col])
        action_probs = probs_for_actions(self.pipeline.action_rf, self.pipeline.action_rf.predict_proba(features))[0]
        weight = max(self.action_active_prob, 1e-6)
        self.action_running += action_probs.astype(np.float64) * weight
        self.action_weight += weight
        self.posterior = (self.action_running / max(self.action_weight, 1e-8)).astype(np.float32)
        self.posterior_by_sample[-1] = self.posterior.copy()

    def _update_phase_until(self, n: int, force: bool = False) -> list[dict[str, object]]:
        if n <= len(self.raw_phase_probs):
            return []
        if bool(self.active_mask[-1]):
            values = (self._sample_array() - self.pipeline.phase_mean) / self.pipeline.phase_std
            prob = self.pipeline.phase_runner.predict_tail(trailing_window(values, n, int(self.phase_cfg["window_samples"])))
        else:
            prob = np.asarray([0.5, 0.5], dtype=np.float32)
        while len(self.raw_phase_probs) < n:
            self.raw_phase_probs.append(prob)
            self._append_viterbi_state(len(self.raw_phase_probs) - 1)
        return self._finalize_ready(force=force)

    def _append_viterbi_state(self, idx: int) -> None:
        window = int(self.phase_cfg["smoothing_window"])
        start = max(0, idx - window + 1)
        smoothed = np.mean(np.stack(self.raw_phase_probs[start : idx + 1]), axis=0).astype(np.float32)
        log_prob = np.log(np.clip(smoothed, 1e-8, 1.0))
        if idx == 0:
            self.dp.append(log_prob.astype(np.float64))
            self.back.append(np.zeros((2,), dtype=np.int64))
            return
        prev = self.dp[-1]
        cur = np.zeros((2,), dtype=np.float64)
        back = np.zeros((2,), dtype=np.int64)
        penalty = float(self.phase_cfg["viterbi_penalty"])
        for state in range(2):
            stay = prev[state]
            switch = prev[1 - state] - penalty
            if stay >= switch:
                cur[state] = log_prob[state] + stay
                back[state] = state
            else:
                cur[state] = log_prob[state] + switch
                back[state] = 1 - state
        self.dp.append(cur)
        self.back.append(back)

    def _label_at_with_traceback(self, final_idx: int, current_idx: int) -> int:
        state = int(np.argmax(self.dp[current_idx]))
        for k in range(current_idx, final_idx, -1):
            state = int(self.back[k][state])
        return state

    def _finalize_ready(self, force: bool = False) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        current = len(self.raw_phase_probs) - 1
        if current < 0:
            return events
        lag = 0 if force else int(self.phase_cfg["fixed_lag_samples"])
        target = current if force else current - lag
        while self.finalized_until < target:
            idx = self.finalized_until + 1
            label_idx = self._label_at_with_traceback(idx, current)
            label = None
            if idx < len(self.active_mask) and bool(self.active_mask[idx]):
                label = "eccentric" if int(label_idx) == 0 else "concentric"
            for rep in self.parser.update(idx, label):
                events.extend(self._handle_raw_rep(rep))
            self.finalized_until = idx
        return events

    def _handle_raw_rep(self, rep: RepEvent) -> list[dict[str, object]]:
        if int(self.args.event_confirm_min_reps) <= 0:
            return self._emit_confirmed_rep(rep)
        if self.event_confirmed:
            if self.event_group:
                gap = int(rep.start_idx) - int(self.event_group[-1].end_idx)
                if gap > int(self.args.event_confirm_gap_samples):
                    self.event_group = [rep]
                    self.event_confirmed = False
                    return []
            self.event_group = [rep]
            return self._emit_confirmed_rep(rep)
        if self.event_group:
            gap = int(rep.start_idx) - int(self.event_group[-1].end_idx)
            if gap > int(self.args.event_confirm_gap_samples):
                flushed = self._flush_event_group(final=False)
                self.event_group = [rep]
                self.event_confirmed = False
                events: list[dict[str, object]] = []
                for old_rep in flushed:
                    events.extend(self._emit_confirmed_rep(old_rep))
                return events
        self.event_group.append(rep)
        if not self.event_confirmed and len(self.event_group) >= int(self.args.event_confirm_min_reps):
            self.event_confirmed = True
            group = self.event_group
            self.event_group = []
            return [event for item in group for event in self._emit_confirmed_rep(item)]
        return []

    def _flush_event_group(self, final: bool) -> list[RepEvent]:
        if not self.event_group:
            return []
        if self.event_confirmed:
            self.event_group = []
            self.event_confirmed = False
            return []
        group = self.event_group
        self.event_group = []
        confirmed = self.event_confirmed or (final and len(group) >= int(self.args.event_confirm_min_reps))
        self.event_confirmed = False
        return group if confirmed else []

    def _emit_confirmed_rep(self, rep: RepEvent) -> list[dict[str, object]]:
        posterior_idx = min(max(int(rep.end_idx) + int(self.args.fixed_lag_samples), 0), max(0, len(self.posterior_by_sample) - 1))
        threshold = soft_threshold_from_context(self.posterior_by_sample[posterior_idx], self.soft_cfg.get("duration_priors", {}), self.args)
        if threshold is not None:
            threshold *= float(self.soft_cfg.get("threshold_scale", 0.8))
        return [self._rep_event(item) for item in self.soft_merger.add(rep, threshold)]

    def _rep_event(self, rep: RepEvent) -> dict[str, object]:
        self.emitted_count += 1
        top_action = self.top_action()
        return {
            "type": "rep",
            "count": self.emitted_count,
            "samples_seen": len(self.samples),
            "top_action": top_action,
            "top_confidence": self.top_confidence(),
            **FullAutoWorkoutPipeline._rep_to_dict(rep, self.sample_rate, top_action),
        }

    def top_action(self) -> str | None:
        if self.posterior is None:
            return None
        return ACTIONS[int(np.argmax(self.posterior))]

    def top_confidence(self) -> float:
        if self.posterior is None:
            return 0.0
        return float(np.max(self.posterior))


def run_jsonl_events(
    pipeline: FullAutoWorkoutPipeline,
    input_path: str,
    emit_stride_samples: int,
    release_delay_samples: int | None,
) -> None:
    buffer: list[tuple[float, float, float, float, float, float]] = []
    emitted_end_idx = -1
    emitted_count = 0
    stride = max(1, int(emit_stride_samples))
    phase_cfg = pipeline.config["phase_decoder"]
    if release_delay_samples is None:
        release_delay_samples = int(phase_cfg.get("fixed_lag_samples", 100)) + int(phase_cfg.get("step_samples", 10))

    def process(final: bool = False) -> None:
        nonlocal emitted_end_idx, emitted_count
        if not buffer:
            return
        samples = np.asarray(buffer, dtype=np.float32)
        result = pipeline.run(samples)
        safe_idx = len(samples) if final else max(0, len(samples) - int(release_delay_samples))
        for rep in result.reps:
            end_idx = int(rep["end_idx"])
            if end_idx <= emitted_end_idx or end_idx > safe_idx:
                continue
            emitted_count += 1
            emitted_end_idx = end_idx
            event = {
                "type": "rep",
                "count": emitted_count,
                "samples_seen": int(len(samples)),
                "top_action": result.top_action,
                "top_confidence": result.top_confidence,
                **rep,
            }
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if final:
            summary = {
                "type": "summary",
                "samples": result.samples,
                "active_samples": result.active_samples,
                "top_action": result.top_action,
                "top_confidence": result.top_confidence,
                "count": emitted_count,
                "batch_count_at_end": result.count,
            }
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    for sample in iter_samples(input_path):
        buffer.append(sample)
        if len(buffer) % stride == 0:
            process(final=False)
    process(final=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full automatic workout pipeline bundle.")
    parser.add_argument("--artifact", default="artifacts/deploy/full_auto_realtime_current")
    parser.add_argument("--input", default="-", help="Input CSV path, or '-' for stdin raw zig stream.")
    parser.add_argument("--runtime", choices=["torch", "onnx", "rknn"], default="torch")
    parser.add_argument("--rf-runtime", choices=["auto", "json", "sklearn"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emit-mode", choices=["final", "jsonl-events", "stateful-jsonl"], default="final")
    parser.add_argument("--emit-stride-samples", type=int, default=50)
    parser.add_argument("--release-delay-samples", type=int, default=None)
    args = parser.parse_args()

    pipeline = FullAutoWorkoutPipeline(Path(args.artifact), args.runtime, args.device, args.rf_runtime)
    if args.emit_mode == "jsonl-events":
        run_jsonl_events(pipeline, args.input, args.emit_stride_samples, args.release_delay_samples)
        return
    if args.emit_mode == "stateful-jsonl":
        live = StatefulLivePipeline(pipeline)
        for sample in iter_samples(args.input):
            for event in live.process_sample(sample):
                print(json.dumps(event, ensure_ascii=False), flush=True)
        for event in live.finish():
            print(json.dumps(event, ensure_ascii=False), flush=True)
        return

    samples = load_samples(args.input)
    result = pipeline.run(samples)
    print(json.dumps({**result.__dict__}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
