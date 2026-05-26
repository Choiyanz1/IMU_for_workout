from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cb = _load_module(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")

from preprocessing.micro_macro_segments import truth_reps_from_labels


def _action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 2 else "unknown"


def _build_action_rf(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 25,
    n_estimators: int = 200,
    max_depth: int = 15,
) -> RandomForestClassifier:
    x_all, y_all = [], []
    for _, df in train_streams:
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        action_col = "action_type"
        action_vals = df[action_col].astype(str).to_numpy()
        x_batch, starts, ends = cb._build_start_window_matrix(x, int(window_size), int(stride))
        if not len(x_batch):
            continue
        y_batch = []
        for start, end in zip(starts, ends):
            actions_in_window = action_vals[int(start):int(end)]
            counter = Counter(actions_in_window)
            majority = counter.most_common(1)[0][0]
            y_batch.append(majority)
        x_all.append(x_batch)
        y_all.append(np.asarray(y_batch))
    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators), max_depth=int(max_depth),
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    clf.fit(x_all, y_all)
    return clf


def _aggregate_rep_action(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 10,
) -> np.ndarray:
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    x_batch, starts, ends = cb._build_start_window_matrix(x, int(window_size), int(stride))
    if not len(x_batch):
        return np.zeros((len(df),), dtype=object)
    preds = clf.predict(x_batch)
    n = len(df)
    votes = np.full((n,), "", dtype=object)
    counts = np.zeros(n, dtype=np.int64)
    for wi, (start, end) in enumerate(zip(starts, ends)):
        for t in range(int(start), int(end)):
            if votes[t] == "":
                votes[t] = preds[wi]
                counts[t] = 1
            elif counts[t] < 10:
                votes[t] = preds[wi]
                counts[t] += 1
    for t in range(n):
        if votes[t] == "":
            votes[t] = "unknown"
    return votes


def _per_rep_majority_action(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 10,
) -> tuple[list[str], list[str]]:
    per_sample = _aggregate_rep_action(clf, df, imu_columns, int(window_size), int(stride))
    truth = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=1,
    )
    y_true, y_pred = [], []
    for rep in truth:
        labels = per_sample[int(rep.start_idx):int(rep.end_idx)]
        counter = Counter(labels)
        majority = counter.most_common(1)[0][0]
        true_action = str(rep.pred_action_type)
        y_true.append(true_action)
        y_pred.append(majority)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Sliding-window RF action + per-rep majority vote.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/rep_action_rf")
    parser.add_argument("--outer-subject", required=True)
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=25)
    parser.add_argument("--eval-stride", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw_cfg = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    train_cfg = cb.TrainConfig(**{k: v for k, v in train_raw_cfg.items() if k in cb.TrainConfig.__dataclass_fields__})
    cb.set_seed(train_cfg.seed)

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    include_actions = [a for a in str(args.include_actions).split(",") if a.strip()]

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)
    if include_actions:
        streams = [(sid, df) for sid, df in streams if _action_from_stream_id(sid) in include_actions]

    outer_subject = str(args.outer_subject)
    train_subjects = [s for s in sorted(set(subjects)) if s != outer_subject]
    train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
    test_streams = cb._filter_subjects(streams, [outer_subject], subject_column)

    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

    print(f"[INFO] outer_subject={outer_subject} train={len(train_streams)} test={len(test_streams)}", flush=True)

    t0 = time.time()
    clf = _build_action_rf(
        train_streams, imu_columns,
        window_size=int(args.window_size), stride=int(args.train_stride),
        n_estimators=int(args.n_estimators), max_depth=int(args.max_depth),
    )
    train_time = time.time() - t0

    y_true_all, y_pred_all = [], []
    for sid, df in test_streams:
        y_true, y_pred = _per_rep_majority_action(
            clf, df, imu_columns,
            window_size=int(args.window_size), stride=int(args.eval_stride),
        )
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    if y_true_all:
        acc = accuracy_score(y_true_all, y_pred_all)
        macro = f1_score(y_true_all, y_pred_all, average="macro")
    else:
        acc, macro = 0.0, 0.0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "rep_action_rf_majority_vote",
        "outer_subject": str(args.outer_subject),
        "settings": {
            "window_size": int(args.window_size),
            "train_stride": int(args.train_stride),
            "eval_stride": int(args.eval_stride),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
        },
        "n_reps": len(y_true_all),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
    }
    (out_dir / "results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"[OK] acc={acc:.4f} macro_f1={macro:.4f} n_reps={len(y_true_all)}")
    print(f"[OK] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
