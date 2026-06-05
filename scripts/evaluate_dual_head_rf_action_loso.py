"""Evaluate a simple dual-head RF for streaming active/action recognition.

Head 1 predicts whether a window is known workout motion vs non-action.
Head 2 predicts the 8 workout actions, trained only on workout windows.

This is a lightweight design probe for the planned parallel action branch. It
does not alter the C/E CNN pipeline and does not use rep-completed features.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import stream_action, stream_subject  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, should_exclude  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


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
ACTIVE_PHASES = {"concentric", "eccentric"}
NON_ACTION = "non_action"


@dataclass(frozen=True)
class WindowMeta:
    stream_id: str
    subject: str
    action: str
    start: int
    end: int
    active_fraction: float


def _is_excluded_subject_session(subject: str, session: str) -> bool:
    return subject in EXCLUDED_SESSIONS and session in EXCLUDED_SESSIONS[subject]


def _read_non_action_csv(path: Path, subject: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    required = set(["ax", "ay", "az", "gx", "gy", "gz"])
    if not required.issubset(df.columns):
        return None
    df = df.copy()
    df["subject_id"] = subject
    df["action_type"] = NON_ACTION
    df["phase"] = NON_ACTION
    return df


def load_non_action_streams(raw_cfg: dict) -> list[tuple[str, pd.DataFrame]]:
    data_dir = Path(raw_cfg.get("data", {}).get("data_dir", "datasets/raw_data"))
    streams: list[tuple[str, pd.DataFrame]] = []
    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            session = session_dir.name
            if _is_excluded_subject_session(subject, session):
                continue

            big_rest = session_dir / "big_rest"
            if big_rest.exists():
                for csv_path in sorted(big_rest.glob("session*/*.csv")):
                    df = _read_non_action_csv(csv_path, subject)
                    if df is not None:
                        rel = csv_path.relative_to(data_dir)
                        streams.append((f"{subject}/{session}/{NON_ACTION}/{rel.stem}", df))

            for action in ACTIONS:
                action_dir = session_dir / action
                if not action_dir.exists():
                    continue
                for rest_dir in sorted(action_dir.glob("rest_after_set*")):
                    if not rest_dir.is_dir():
                        continue
                    for csv_path in sorted(rest_dir.glob("*.csv")):
                        df = _read_non_action_csv(csv_path, subject)
                        if df is not None:
                            rel = csv_path.relative_to(data_dir)
                            streams.append((f"{subject}/{session}/{NON_ACTION}/{rel.stem}", df))
    return streams


def normalize_streams(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    eval_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
) -> list[tuple[str, pd.DataFrame]]:
    train_values = [df[list(imu_columns)].to_numpy(dtype=np.float32) for _, df in train_streams if len(df)]
    stacked = np.concatenate(train_values, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out = []
    for stream_id, df in eval_streams:
        copied = df.copy()
        values = copied[list(imu_columns)].to_numpy(dtype=np.float32)
        copied.loc[:, list(imu_columns)] = (values - mean) / std
        out.append((stream_id, copied))
    return out


def _starts_for_length(n: int, window_samples: int, stride_samples: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if n < window_samples:
        return np.asarray([0], dtype=np.int64)
    starts = list(range(0, n - window_samples + 1, stride_samples))
    tail = n - window_samples
    if starts[-1] != tail:
        starts.append(tail)
    return np.asarray(starts, dtype=np.int64)


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """Vectorized statistical features from windows [N, T, 6]."""
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


def build_windows(
    streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    window_samples: int,
    stride_samples: int,
    active_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[WindowMeta]]:
    x_rows: list[np.ndarray] = []
    y_active: list[int] = []
    y_action: list[str] = []
    metas: list[WindowMeta] = []

    for stream_id, df in streams:
        if not set(imu_columns).issubset(df.columns):
            continue
        values = df[list(imu_columns)].to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        phase = df["phase"].astype(str).to_numpy() if "phase" in df.columns else np.full(len(df), NON_ACTION, dtype=object)
        active_mask = np.asarray([p in ACTIVE_PHASES for p in phase], dtype=bool)
        action = stream_action(stream_id)
        if action not in ACTIONS:
            action = NON_ACTION

        pad = max(0, window_samples - len(values))
        padded_values = np.pad(values, ((0, pad), (0, 0)), mode="edge") if pad else values
        padded_active = np.pad(active_mask, (0, pad), constant_values=False) if pad else active_mask
        starts = _starts_for_length(len(values), window_samples, stride_samples)
        stream_windows = []
        stream_labels_active = []
        stream_labels_action = []
        stream_metas = []
        for start in starts:
            end = min(int(start) + window_samples, len(values))
            window = padded_values[int(start) : int(start) + window_samples]
            active_fraction = float(np.mean(padded_active[int(start) : int(start) + window_samples]))
            active_label = int(active_fraction >= active_threshold and action in ACTIONS)
            action_label = action if active_label else NON_ACTION
            stream_windows.append(window)
            stream_labels_active.append(active_label)
            stream_labels_action.append(action_label)
            stream_metas.append(WindowMeta(stream_id, stream_subject(stream_id), action, int(start), end, active_fraction))
        if stream_windows:
            x_rows.extend(extract_features_batch(np.stack(stream_windows)).astype(np.float32))
            y_active.extend(stream_labels_active)
            y_action.extend(stream_labels_action)
            metas.extend(stream_metas)

    return (
        np.vstack(x_rows).astype(np.float32) if x_rows else np.zeros((0, 0), dtype=np.float32),
        np.asarray(y_active, dtype=np.int64),
        np.asarray(y_action, dtype=object),
        metas,
    )


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _proba_for_label(clf: RandomForestClassifier, probs: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(probs), len(labels)), dtype=np.float32)
    class_to_col = {str(cls): idx for idx, cls in enumerate(clf.classes_)}
    for j, label in enumerate(labels):
        col = class_to_col.get(str(label))
        if col is not None:
            out[:, j] = probs[:, col]
    return out


def lock_action_for_stream(
    metas: Sequence[WindowMeta],
    action_probs: np.ndarray,
    active_probs: np.ndarray,
    lock_threshold: float,
    active_threshold: float,
    margin_threshold: float,
    stable_windows: int,
    min_windows: int,
    sample_rate_hz: float,
) -> dict[str, object]:
    if len(metas) == 0:
        return {"locked": False, "locked_action": None, "lock_time_s": None, "confidence": 0.0}
    running = np.zeros((len(ACTIONS),), dtype=np.float64)
    stable: list[str] = []
    total_weight = 0.0
    for idx, (meta, probs, active_prob) in enumerate(zip(metas, action_probs, active_probs), start=1):
        weight = max(float(active_prob), 1e-6)
        running += probs.astype(np.float64) * weight
        total_weight += weight
        posterior = running / max(total_weight, 1e-8)
        order = np.argsort(posterior)[::-1]
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        top_action = ACTIONS[top]
        top_prob = float(posterior[top])
        margin = float(posterior[top] - posterior[second])
        stable.append(top_action)
        stable = stable[-stable_windows:]
        if (
            idx >= min_windows
            and len(stable) == stable_windows
            and len(set(stable)) == 1
            and float(active_prob) >= active_threshold
            and top_prob >= lock_threshold
            and margin >= margin_threshold
        ):
            return {
                "locked": True,
                "locked_action": top_action,
                "lock_time_s": float(meta.end) / float(sample_rate_hz),
                "confidence": top_prob,
                "margin": margin,
                "windows_seen": idx,
            }
    posterior = running / max(total_weight, 1e-8)
    top = int(np.argmax(posterior)) if len(posterior) else 0
    return {
        "locked": False,
        "locked_action": ACTIONS[top] if len(posterior) else None,
        "lock_time_s": None,
        "confidence": float(posterior[top]) if len(posterior) else 0.0,
        "margin": 0.0,
        "windows_seen": len(metas),
    }


def evaluate_locks(
    metas: Sequence[WindowMeta],
    action_probs: np.ndarray,
    active_probs: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    by_stream: dict[str, list[int]] = {}
    for idx, meta in enumerate(metas):
        by_stream.setdefault(meta.stream_id, []).append(idx)
    rows = []
    for stream_id, indices in sorted(by_stream.items()):
        stream_metas = [metas[i] for i in indices]
        result = lock_action_for_stream(
            stream_metas,
            action_probs[indices],
            active_probs[indices],
            args.lock_threshold,
            args.lock_active_threshold,
            args.lock_margin,
            args.stable_windows,
            args.min_lock_windows,
            args.sample_rate_hz,
        )
        true_action = stream_metas[0].action
        is_action_stream = true_action in ACTIONS
        rows.append({"stream_id": stream_id, "true_action": true_action, "is_action_stream": is_action_stream, **result})

    action_rows = [r for r in rows if r["is_action_stream"]]
    non_action_rows = [r for r in rows if not r["is_action_stream"]]
    locked_action_rows = [r for r in action_rows if r["locked"]]
    lock_times = [float(r["lock_time_s"]) for r in locked_action_rows if r["lock_time_s"] is not None]
    return {
        "rows": rows,
        "summary": {
            "action_streams": len(action_rows),
            "action_lock_rate": float(len(locked_action_rows) / max(1, len(action_rows))),
            "action_lock_accuracy": float(
                np.mean([r["locked_action"] == r["true_action"] for r in locked_action_rows])
            )
            if locked_action_rows
            else 0.0,
            "median_lock_time_s": float(np.median(lock_times)) if lock_times else None,
            "mean_lock_time_s": float(np.mean(lock_times)) if lock_times else None,
            "non_action_streams": len(non_action_rows),
            "non_action_false_lock_rate": float(np.mean([bool(r["locked"]) for r in non_action_rows])) if non_action_rows else None,
        },
    }


def evaluate_lock_grid(
    metas: Sequence[WindowMeta],
    action_probs: np.ndarray,
    active_probs: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict[str, float | int | None]]:
    policies: dict[str, dict[str, float | int]] = {
        "balanced_a055_p055_m010_s3": {
            "lock_active_threshold": 0.55,
            "lock_threshold": 0.55,
            "lock_margin": 0.10,
            "stable_windows": 3,
            "min_lock_windows": 3,
        },
        "strict_a070_p065_m015_s4": {
            "lock_active_threshold": 0.70,
            "lock_threshold": 0.65,
            "lock_margin": 0.15,
            "stable_windows": 4,
            "min_lock_windows": 4,
        },
        "stricter_a075_p070_m020_s4": {
            "lock_active_threshold": 0.75,
            "lock_threshold": 0.70,
            "lock_margin": 0.20,
            "stable_windows": 4,
            "min_lock_windows": 4,
        },
        "very_strict_a080_p075_m020_s5": {
            "lock_active_threshold": 0.80,
            "lock_threshold": 0.75,
            "lock_margin": 0.20,
            "stable_windows": 5,
            "min_lock_windows": 5,
        },
        "ultra_a085_p080_m025_s5": {
            "lock_active_threshold": 0.85,
            "lock_threshold": 0.80,
            "lock_margin": 0.25,
            "stable_windows": 5,
            "min_lock_windows": 5,
        },
        "late_a080_p075_m025_s6": {
            "lock_active_threshold": 0.80,
            "lock_threshold": 0.75,
            "lock_margin": 0.25,
            "stable_windows": 6,
            "min_lock_windows": 6,
        },
    }
    summaries: dict[str, dict[str, float | int | None]] = {}
    for name, policy in policies.items():
        policy_args = SimpleNamespace(**vars(args))
        for key, value in policy.items():
            setattr(policy_args, key, value)
        summaries[name] = evaluate_locks(metas, action_probs, active_probs, policy_args)["summary"]
    return summaries


def evaluate_fold(test_subject: str, train_streams, test_streams, imu_columns, args) -> dict[str, object]:
    train_norm = normalize_streams(train_streams, train_streams, imu_columns)
    test_norm = normalize_streams(train_streams, test_streams, imu_columns)
    x_train, y_active_train, y_action_train, _ = build_windows(
        train_norm, imu_columns, args.window_samples, args.stride_samples, args.window_active_threshold
    )
    x_test, y_active_test, y_action_test, metas = build_windows(
        test_norm, imu_columns, args.window_samples, args.stride_samples, args.window_active_threshold
    )
    if len(x_train) == 0 or len(x_test) == 0:
        raise RuntimeError(f"No windows for fold {test_subject}")

    active_rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    active_rf.fit(x_train, y_active_train)

    active_train_idx = np.where(y_active_train == 1)[0]
    action_rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed + 17,
        n_jobs=-1,
    )
    action_rf.fit(x_train[active_train_idx], y_action_train[active_train_idx])

    active_pred = active_rf.predict(x_test)
    active_probs_raw = active_rf.predict_proba(x_test)
    active_class_to_col = {int(cls): idx for idx, cls in enumerate(active_rf.classes_)}
    active_probs = active_probs_raw[:, active_class_to_col.get(1, 0)] if 1 in active_class_to_col else np.zeros(len(x_test))

    action_pred = action_rf.predict(x_test)
    action_probs = _proba_for_label(action_rf, action_rf.predict_proba(x_test), ACTIONS)
    active_mask = y_active_test == 1
    gated_pred = np.where(active_probs >= args.eval_active_threshold, action_pred, NON_ACTION)

    lock_eval = evaluate_locks(metas, action_probs, active_probs, args)
    result = {
        "test_subject": test_subject,
        "train_windows": int(len(x_train)),
        "test_windows": int(len(x_test)),
        "train_active_windows": int(np.sum(y_active_train == 1)),
        "train_non_action_windows": int(np.sum(y_active_train == 0)),
        "test_active_windows": int(np.sum(y_active_test == 1)),
        "test_non_action_windows": int(np.sum(y_active_test == 0)),
        "active_head": _binary_metrics(y_active_test, active_pred),
        "action_head_on_true_active": {
            "accuracy": float(accuracy_score(y_action_test[active_mask], action_pred[active_mask])) if np.any(active_mask) else 0.0,
            "macro_f1": float(f1_score(y_action_test[active_mask], action_pred[active_mask], labels=ACTIONS, average="macro", zero_division=0))
            if np.any(active_mask)
            else 0.0,
        },
        "gated_9class": {
            "accuracy": float(accuracy_score(y_action_test, gated_pred)),
            "macro_f1": float(f1_score(y_action_test, gated_pred, labels=[*ACTIONS, NON_ACTION], average="macro", zero_division=0)),
        },
        "lock": lock_eval["summary"],
        "lock_rows": lock_eval["rows"],
    }
    if args.lock_grid:
        result["lock_grid"] = evaluate_lock_grid(metas, action_probs, active_probs, args)
    return result


def aggregate_folds(folds: Sequence[dict[str, object]]) -> dict[str, object]:
    def mean_path(*keys: str):
        vals = []
        for fold in folds:
            item = fold
            for key in keys:
                item = item[key]
            if item is not None:
                vals.append(float(item))
        return float(np.mean(vals)) if vals else None

    return {
        "folds": len(folds),
        "active_head": {
            "accuracy": mean_path("active_head", "accuracy"),
            "precision": mean_path("active_head", "precision"),
            "recall": mean_path("active_head", "recall"),
            "f1": mean_path("active_head", "f1"),
            "macro_f1": mean_path("active_head", "macro_f1"),
        },
        "action_head_on_true_active": {
            "accuracy": mean_path("action_head_on_true_active", "accuracy"),
            "macro_f1": mean_path("action_head_on_true_active", "macro_f1"),
        },
        "gated_9class": {
            "accuracy": mean_path("gated_9class", "accuracy"),
            "macro_f1": mean_path("gated_9class", "macro_f1"),
        },
        "lock": {
            "action_lock_rate": mean_path("lock", "action_lock_rate"),
            "action_lock_accuracy": mean_path("lock", "action_lock_accuracy"),
            "median_lock_time_s": mean_path("lock", "median_lock_time_s"),
            "non_action_false_lock_rate": mean_path("lock", "non_action_false_lock_rate"),
        },
    }


def aggregate_lock_grid(folds: Sequence[dict[str, object]]) -> dict[str, dict[str, float | None]]:
    policy_names = sorted({name for fold in folds for name in (fold.get("lock_grid") or {}).keys()})
    output: dict[str, dict[str, float | None]] = {}
    for name in policy_names:
        rows = [fold["lock_grid"][name] for fold in folds if name in (fold.get("lock_grid") or {})]
        output[name] = {}
        for key in ["action_lock_rate", "action_lock_accuracy", "median_lock_time_s", "mean_lock_time_s", "non_action_false_lock_rate"]:
            vals = [row.get(key) for row in rows if row.get(key) is not None]
            output[name][key] = float(np.mean(vals)) if vals else None
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-head RF action recognizer with LOSO splits.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/dual_head_rf_loso/summary.json")
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--stride-samples", type=int, default=50)
    parser.add_argument("--window-active-threshold", type=float, default=0.5)
    parser.add_argument("--eval-active-threshold", type=float, default=0.5)
    parser.add_argument("--lock-active-threshold", type=float, default=0.55)
    parser.add_argument("--lock-threshold", type=float, default=0.55)
    parser.add_argument("--lock-margin", type=float, default=0.10)
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--min-lock-windows", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rest-fragments", action="store_true")
    parser.add_argument("--max-folds", type=int, default=0, help="Debug limit; 0 runs all folds.")
    parser.add_argument("--lock-grid", action="store_true", help="Evaluate several preset lock policies from the same RF outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    imu_columns = list(raw_cfg.get("feature", {}).get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    set_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in set_streams if not should_exclude(sid)]
    rest_streams = [] if args.no_rest_fragments else load_non_action_streams(raw_cfg)
    streams = [*set_streams, *rest_streams]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})

    print(f"set_streams={len(set_streams)} rest_streams={len(rest_streams)} subjects={subjects}", flush=True)
    folds = []
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects
    for fold_idx, test_subject in enumerate(eval_subjects, start=1):
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject}", flush=True)
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        fold = evaluate_fold(test_subject, train_streams, test_streams, imu_columns, args)
        folds.append(fold)
        print(
            "  active_f1={:.4f} action_acc={:.4f} gated_macro={:.4f} lock_acc={:.4f} lock_rate={:.4f}".format(
                fold["active_head"]["f1"],
                fold["action_head_on_true_active"]["accuracy"],
                fold["gated_9class"]["macro_f1"],
                fold["lock"]["action_lock_accuracy"],
                fold["lock"]["action_lock_rate"],
            ),
            flush=True,
        )

    output = {
        "settings": {
            "model": "dual_head_random_forest_window_action_probe",
            "input_columns": imu_columns,
            "actions": ACTIONS,
            "non_action_label": NON_ACTION,
            "window_samples": args.window_samples,
            "stride_samples": args.stride_samples,
            "sample_rate_hz": args.sample_rate_hz,
            "set_streams": len(set_streams),
            "rest_streams": len(rest_streams),
            "excluded_sessions": EXCLUDED_SESSIONS,
            "note": "First probe for parallel action branch; non_action uses rest fragments plus phase!=concentric/eccentric windows.",
        },
        "overall": aggregate_folds(folds),
        "lock_grid_overall": aggregate_lock_grid(folds) if args.lock_grid else {},
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nOVERALL")
    print(json.dumps(output["overall"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
