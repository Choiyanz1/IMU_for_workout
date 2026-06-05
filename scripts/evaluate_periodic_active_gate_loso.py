"""Evaluate active gates for full-session realtime use.

This script isolates the active/rest problem before rerunning the expensive
phase CNN pipeline. It compares a baseline statistical RF gate with a
periodicity-aware RF gate and evaluates deployment-oriented failure modes:
rest-only false activity and set+rest-tail leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import load_non_action_streams  # noqa: E402
from scripts.new_c_pipeline.compare_phase_models import PhaseCompareConfig, _extract_window_features_batch  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import stream_subject  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, set_seed, should_exclude  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


ACTIVE_PHASES = {"concentric", "eccentric"}
REST_STATIC = 0
EXERCISE_ACTIVE = 1
TRANSITION_MOTION = 2


def active_labels_from_df(df: pd.DataFrame) -> np.ndarray:
    if "phase" not in df.columns:
        return np.zeros(len(df), dtype=np.int64)
    phases = df["phase"].astype(str).to_numpy()
    return np.asarray([1 if phase in ACTIVE_PHASES else 0 for phase in phases], dtype=np.int64)


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
    ends = sorted(set([1, n, *range(int(stride_samples), n + 1, int(stride_samples))]))
    return [end for end in ends if end > 0]


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
    base = _extract_window_features_batch(windows)
    if mode == "basic":
        return base
    periodic = periodic_features_batch(windows, sample_rate_hz)
    return np.concatenate([base, periodic], axis=1).astype(np.float32, copy=False)


def motion_energy_batch(windows: np.ndarray) -> np.ndarray:
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    centered = arr - np.mean(arr, axis=1, keepdims=True)
    jerk = np.diff(arr, axis=1, prepend=arr[:, :1, :])
    acc_energy = np.mean(np.sum(centered[:, :, :3] ** 2, axis=2), axis=1)
    gyro_energy = np.mean(np.sum(centered[:, :, 3:6] ** 2, axis=2), axis=1)
    jerk_energy = np.mean(np.sum(jerk ** 2, axis=2), axis=1)
    return (acc_energy + gyro_energy + 0.5 * jerk_energy).astype(np.float32)


def build_window_dataset(streams, imu_columns, args, mode: str):
    x_rows = []
    y_rows = []
    energy_rows = []
    for _stream_id, df in streams:
        if not set(imu_columns).issubset(df.columns) or len(df) == 0:
            continue
        values = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = active_labels_from_df(df)
        ends = window_ends(len(values), args.window_samples, args.stride_samples)
        if not ends:
            continue
        windows = np.stack([trailing_window(values, end, args.window_samples) for end in ends]).astype(np.float32)
        features = extract_gate_features(windows, mode, args.sample_rate_hz)
        energies = motion_energy_batch(windows)
        y = []
        for end in ends:
            start = max(0, int(end) - int(args.window_samples))
            if end - start <= 0:
                y.append(0)
            else:
                y.append(int(float(np.mean(labels[start:end])) >= args.window_active_fraction))
        x_rows.append(features)
        y_rows.append(np.asarray(y, dtype=np.int64))
        energy_rows.append(energies)
    if not x_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    x = np.vstack(x_rows).astype(np.float32)
    y = np.concatenate(y_rows).astype(np.int64)
    if args.label_mode == "tri_motion":
        energies = np.concatenate(energy_rows).astype(np.float32)
        inactive = energies[y == 0]
        threshold = float(np.quantile(inactive, args.transition_energy_quantile)) if len(inactive) else float("inf")
        y = np.where(y == 1, EXERCISE_ACTIVE, np.where(energies >= threshold, TRANSITION_MOTION, REST_STATIC)).astype(np.int64)
    return x, y


def train_gate(train_streams, imu_columns, args, mode: str):
    x_train, y_train = build_window_dataset(train_streams, imu_columns, args, mode)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(x_scaled, y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    return clf, scaler, {
        "train_windows": int(len(y_train)),
        "train_active_rate": float(np.mean(y_train == EXERCISE_ACTIVE)),
        "train_class_counts": {str(int(cls)): int(count) for cls, count in zip(classes, counts)},
    }


def predict_active_prob(df: pd.DataFrame, imu_columns, clf, scaler, args, mode: str) -> np.ndarray:
    values = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(values)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    ends = window_ends(n, args.window_samples, args.stride_samples)
    windows = np.stack([trailing_window(values, end, args.window_samples) for end in ends]).astype(np.float32)
    features = extract_gate_features(windows, mode, args.sample_rate_hz)
    probs = clf.predict_proba(scaler.transform(features))
    class_to_col = {int(cls): idx for idx, cls in enumerate(clf.classes_)}
    active_col = class_to_col.get(EXERCISE_ACTIVE, 0)
    out = np.zeros(n, dtype=np.float32)
    prev = 0
    for i, end in enumerate(ends):
        out[prev:int(end)] = float(probs[i, active_col])
        prev = int(end)
    if prev < n:
        out[prev:] = out[prev - 1] if prev > 0 else 0.0
    return out


def state_machine(prob: np.ndarray, args) -> np.ndarray:
    n = len(prob)
    state = False
    enter_count = 0
    exit_count = 0
    mask = np.zeros(n, dtype=bool)
    enter_hold = max(1, int(args.enter_hold_samples))
    exit_hold = max(1, int(args.exit_hold_samples))
    cooldown_samples = max(0, int(args.cooldown_samples))
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
                    cooldown_until = i + cooldown_samples
                    enter_count = 0
                    exit_count = 0
                else:
                    mask[i] = True
            else:
                exit_count = 0
                mask[i] = True
    return clean_mask(mask, args.min_active_samples, args.bridge_gap_samples)


def clean_mask(mask: np.ndarray, min_active_samples: int, bridge_gap_samples: int) -> np.ndarray:
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


def active_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < len(mask) and mask[i]:
            i += 1
        segments.append((start, i))
    return segments


def append_rest_tail(stream_id: str, df: pd.DataFrame, data_dir: Path, seconds: float, sample_rate_hz: float, imu_columns):
    if seconds <= 0:
        return df, len(df), 0
    parts = [part for part in str(stream_id).split("/") if part]
    if len(parts) < 4:
        return df, len(df), 0
    subject, session, action, set_name = parts[0], parts[1], parts[2], parts[3]
    rest_dir = data_dir / subject / session / action / f"rest_after_{set_name}"
    if not rest_dir.exists():
        return df, len(df), 0
    max_rows = int(round(seconds * sample_rate_hz))
    for csv_path in sorted(rest_dir.glob("*.csv")):
        try:
            rest_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if rest_df.empty or not set(imu_columns).issubset(rest_df.columns):
            continue
        rest_df = rest_df.iloc[:max_rows].copy() if max_rows > 0 else rest_df.copy()
        rest_df["phase"] = "non_action"
        return pd.concat([df.copy(), rest_df], ignore_index=True, sort=False), len(df), len(rest_df)
    return df, len(df), 0


def mask_metrics(gt: np.ndarray, pred: np.ndarray, sample_rate_hz: float) -> dict[str, float | int]:
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
        "false_active_rate": float(np.sum(false_active) / max(1, int(np.sum(gt == 0)))),
        "missed_active_rate": float(np.sum(missed) / max(1, int(np.sum(gt == 1)))),
    }


def aggregate_mask_metrics(rows: list[dict[str, float | int]], sample_rate_hz: float) -> dict[str, float | int]:
    if not rows:
        return {}
    total_samples = int(sum(int(row["samples"]) for row in rows))
    active_samples = int(sum(int(row["active_samples"]) for row in rows))
    pred_active_samples = int(sum(int(row["pred_active_samples"]) for row in rows))
    false_active_sec = float(sum(float(row["false_active_sec"]) for row in rows))
    missed_active_sec = float(sum(float(row["missed_active_sec"]) for row in rows))
    return {
        "streams": len(rows),
        "samples": total_samples,
        "duration_min": float(total_samples / sample_rate_hz / 60.0),
        "active_sample_rate": float(active_samples / max(1, total_samples)),
        "pred_active_sample_rate": float(pred_active_samples / max(1, total_samples)),
        "mean_precision": float(np.mean([float(row["precision"]) for row in rows])),
        "mean_recall": float(np.mean([float(row["recall"]) for row in rows])),
        "mean_f1": float(np.mean([float(row["f1"]) for row in rows])),
        "mean_macro_f1": float(np.mean([float(row["macro_f1"]) for row in rows])),
        "false_active_sec": false_active_sec,
        "missed_active_sec": missed_active_sec,
        "false_active_per_min": float(false_active_sec / max(total_samples / sample_rate_hz / 60.0, 1e-8)),
        "missed_active_per_min": float(missed_active_sec / max(total_samples / sample_rate_hz / 60.0, 1e-8)),
    }


def evaluate_stream(stream_id: str, df: pd.DataFrame, imu_columns, clf, scaler, args, mode: str):
    prob = predict_active_prob(df, imu_columns, clf, scaler, args, mode)
    pred = state_machine(prob, args)
    gt = active_labels_from_df(df)
    row = mask_metrics(gt, pred, args.sample_rate_hz)
    row.update(
        {
            "stream_id": stream_id,
            "segments": len(active_segments(pred)),
            "max_probability": float(np.max(prob)) if len(prob) else 0.0,
            "mean_probability": float(np.mean(prob)) if len(prob) else 0.0,
        }
    )
    return row, pred, prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate periodicity-aware active gates under LOSO.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/active_gate_periodic_loso/summary.json")
    parser.add_argument("--feature-modes", default="basic,periodic")
    parser.add_argument("--label-mode", choices=["binary", "tri_motion"], default="binary")
    parser.add_argument("--transition-energy-quantile", type=float, default=0.7)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=50)
    parser.add_argument("--window-active-fraction", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--enter-threshold", type=float, default=0.7)
    parser.add_argument("--exit-threshold", type=float, default=0.45)
    parser.add_argument("--enter-hold-samples", type=int, default=50)
    parser.add_argument("--exit-hold-samples", type=int, default=100)
    parser.add_argument("--min-active-samples", type=int, default=200)
    parser.add_argument("--bridge-gap-samples", type=int, default=50)
    parser.add_argument("--cooldown-samples", type=int, default=0)
    parser.add_argument("--rest-tail-seconds", type=float, default=20.0)
    parser.add_argument("--train-rest-tail-seconds", type=float, default=0.0)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-rest-streams-per-fold", type=int, default=0)
    args = parser.parse_args()
    modes = [mode.strip() for mode in str(args.feature_modes).split(",") if mode.strip()]
    set_seed(args.seed)

    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = PhaseCompareConfig()
    all_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    subjects = sorted({stream_subject(sid) for sid, _ in set_streams})
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects
    data_dir = Path(raw_cfg.get("data", {}).get("data_dir", "datasets/raw_data"))

    output = {"settings": vars(args), "excluded_sessions": EXCLUDED_SESSIONS, "modes": {}}
    print(f"sets={len(set_streams)} rest={len(rest_streams)} subjects={subjects}", flush=True)
    for mode in modes:
        print(f"\nMode={mode}", flush=True)
        mode_folds = []
        set_rows_all = []
        rest_rows_all = []
        appended_rows_all = []
        for fold_idx, test_subject in enumerate(eval_subjects, start=1):
            train_set = [(sid, df) for sid, df in set_streams if stream_subject(sid) != test_subject]
            test_set = [(sid, df) for sid, df in set_streams if stream_subject(sid) == test_subject]
            train_rest = [(sid, df) for sid, df in rest_streams if stream_subject(sid) != test_subject]
            test_rest = [(sid, df) for sid, df in rest_streams if stream_subject(sid) == test_subject]
            if args.max_rest_streams_per_fold and args.max_rest_streams_per_fold > 0:
                test_rest = test_rest[: args.max_rest_streams_per_fold]
            train_augmented = [*train_set, *train_rest]
            if args.train_rest_tail_seconds > 0:
                for stream_id, df in train_set:
                    combined, _set_len, rest_len = append_rest_tail(
                        stream_id,
                        df,
                        data_dir,
                        args.train_rest_tail_seconds,
                        args.sample_rate_hz,
                        cfg.imu_columns,
                    )
                    if rest_len > 0:
                        train_augmented.append((f"{stream_id}+train_rest_tail", combined))
            clf, scaler, train_info = train_gate(train_augmented, cfg.imu_columns, args, mode)
            fold_set_rows = []
            fold_rest_rows = []
            fold_appended_rows = []

            for stream_id, df in test_set:
                row, _pred, _prob = evaluate_stream(stream_id, df, cfg.imu_columns, clf, scaler, args, mode)
                fold_set_rows.append(row)
                set_rows_all.append(row)
                combined, set_len, rest_len = append_rest_tail(stream_id, df, data_dir, args.rest_tail_seconds, args.sample_rate_hz, cfg.imu_columns)
                if rest_len > 0:
                    app_row, app_pred, _app_prob = evaluate_stream(f"{stream_id}+rest_tail", combined, cfg.imu_columns, clf, scaler, args, mode)
                    rest_pred = app_pred[set_len:]
                    app_row.update(
                        {
                            "stream_id": stream_id,
                            "set_samples": int(set_len),
                            "rest_samples": int(rest_len),
                            "rest_pred_active_samples": int(np.sum(rest_pred)),
                            "rest_pred_active_sec": float(np.sum(rest_pred) / args.sample_rate_hz),
                            "rest_tail_active_rate": float(np.mean(rest_pred)) if len(rest_pred) else 0.0,
                            "rest_tail_segments": len(active_segments(rest_pred)),
                        }
                    )
                    fold_appended_rows.append(app_row)
                    appended_rows_all.append(app_row)

            for stream_id, df in test_rest:
                row, pred, _prob = evaluate_stream(stream_id, df, cfg.imu_columns, clf, scaler, args, mode)
                row.update({"false_active_segments": len(active_segments(pred))})
                fold_rest_rows.append(row)
                rest_rows_all.append(row)

            fold_summary = {
                "fold": fold_idx,
                "test_subject": test_subject,
                "train": train_info,
                "set_summary": aggregate_mask_metrics(fold_set_rows, args.sample_rate_hz),
                "rest_summary": aggregate_mask_metrics(fold_rest_rows, args.sample_rate_hz),
                "appended_summary": aggregate_mask_metrics(fold_appended_rows, args.sample_rate_hz),
                "rest_false_active_segments": int(sum(int(row.get("false_active_segments", 0)) for row in fold_rest_rows)),
                "appended_rest_tail_segments": int(sum(int(row.get("rest_tail_segments", 0)) for row in fold_appended_rows)),
            }
            mode_folds.append(fold_summary)
            print(
                f"  fold {fold_idx}/{len(eval_subjects)} {test_subject}: setF1={fold_summary['set_summary'].get('mean_f1', 0):.3f} restFA/min={fold_summary['rest_summary'].get('false_active_per_min', 0):.2f} appRestRate={np.mean([r['rest_tail_active_rate'] for r in fold_appended_rows]) if fold_appended_rows else 0:.3f}",
                flush=True,
            )

        mode_output = {
            "set_total": aggregate_mask_metrics(set_rows_all, args.sample_rate_hz),
            "rest_total": aggregate_mask_metrics(rest_rows_all, args.sample_rate_hz),
            "appended_total": aggregate_mask_metrics(appended_rows_all, args.sample_rate_hz),
            "rest_false_active_segments": int(sum(int(row.get("false_active_segments", 0)) for row in rest_rows_all)),
            "appended_rest_tail_segments": int(sum(int(row.get("rest_tail_segments", 0)) for row in appended_rows_all)),
            "folds": mode_folds,
            "set_rows": set_rows_all,
            "rest_rows": rest_rows_all,
            "appended_rows": appended_rows_all,
        }
        output["modes"][mode] = mode_output
        print(
            f"TOTAL {mode}: setF1={mode_output['set_total'].get('mean_f1', 0):.3f} setRecall={mode_output['set_total'].get('mean_recall', 0):.3f} restFA/min={mode_output['rest_total'].get('false_active_per_min', 0):.2f} appRestRate={mode_output['appended_total'].get('pred_active_sample_rate', 0):.3f}",
            flush=True,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
