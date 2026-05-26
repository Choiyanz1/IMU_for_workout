from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_module(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
crf = _load_module(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")

from preprocessing.micro_macro_segments import RepDetection, rep_metrics, truth_reps_from_labels


EVENT_OTHER = 0
EVENT_START = 1
EVENT_TRANSITION = 2
EVENT_END = 3
EVENT_NAMES = {
    EVENT_OTHER: "other",
    EVENT_START: "start",
    EVENT_TRANSITION: "transition",
    EVENT_END: "end",
}


def _action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 2 else "unknown"


def _mark_event_band(labels: np.ndarray, idx: int, label: int, radius: int) -> None:
    lo = max(0, int(idx) - int(radius))
    hi = min(len(labels), int(idx) + int(radius) + 1)
    labels[lo:hi] = int(label)


def _event_labels_from_truth(df: pd.DataFrame, event_radius: int) -> np.ndarray:
    truth = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=1,
    )
    labels = np.full(len(df), EVENT_OTHER, dtype=np.int64)
    for rep in truth:
        _mark_event_band(labels, int(rep.start_idx), EVENT_START, int(event_radius))
        _mark_event_band(labels, int(rep.transition_idx), EVENT_TRANSITION, int(event_radius))
        end_idx = max(0, min(len(df) - 1, int(rep.end_idx) - 1))
        _mark_event_band(labels, end_idx, EVENT_END, int(event_radius))
    return labels


def _train_event_rf(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    *,
    window_size: int,
    stride: int,
    event_radius: int,
    n_estimators: int,
    max_depth: int,
    max_samples: float,
) -> RandomForestClassifier:
    x_all = []
    y_all = []
    for stream_idx, (_, df) in enumerate(train_streams, start=1):
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = _event_labels_from_truth(df, int(event_radius))
        x_batch, ends = crf._build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(x_batch):
            x_all.append(x_batch)
            y_all.append(labels[np.maximum(0, ends - 1)])
        if stream_idx % 25 == 0 or stream_idx == len(train_streams):
            print(f"  [DirectEventRF] prepared {stream_idx}/{len(train_streams)} train streams", flush=True)
    x_all = np.concatenate(x_all, axis=0) if x_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    print(
        f"  [DirectEventRF] Training on {len(x_all)} windows "
        f"(window={window_size}, stride={stride}, trees={n_estimators}, max_depth={max_depth}, max_samples={max_samples})",
        flush=True,
    )
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples) if max_samples and max_samples < 1.0 else max_samples,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    clf.fit(x_all, y_all)
    clf.verbose = 0
    return clf


def _predict_event_probs(
    clf: RandomForestClassifier,
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    *,
    window_size: int,
    smoothing_window: int,
) -> np.ndarray:
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    x_batch, ends = crf._build_trailing_feature_matrix(x, int(window_size), 1)
    n = len(df)
    probs = np.zeros((n, 4), dtype=np.float32)
    if len(x_batch):
        raw = clf.predict_proba(x_batch)
        class_map = {int(c): i for i, c in enumerate(clf.classes_)}
        full = np.zeros((len(raw), 4), dtype=np.float32)
        for cls_idx, mi in class_map.items():
            full[:, cls_idx] = raw[:, mi]
        probs[np.maximum(0, ends - 1)] = full
    smooth_w = max(1, int(smoothing_window))
    if smooth_w > 1:
        smoothed = np.zeros_like(probs)
        csum = np.cumsum(probs, axis=0)
        for i in range(len(probs)):
            start = max(0, i - smooth_w + 1)
            total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
            smoothed[i] = total / float(i - start + 1)
        probs = smoothed
    return probs


def _extract_events(probs: np.ndarray, event_label: int, threshold: float) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    scores = probs[:, int(event_label)]
    mask = scores >= float(threshold)
    n = len(mask)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(mask[j]):
            j += 1
        region = scores[i:j]
        peak_local = int(np.argmax(region))
        peak_idx = i + peak_local
        out.append((peak_idx, float(region[peak_local])))
        i = j
    return out


def _duration_bounds(train_streams: Sequence[tuple[str, pd.DataFrame]]) -> tuple[int, int]:
    durations = []
    for _, df in train_streams:
        truth = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=1,
        )
        durations.extend(max(1, int(rep.end_idx) - int(rep.start_idx)) for rep in truth)
    if not durations:
        return 20, 400
    low = max(5, int(np.quantile(durations, 0.10)))
    high = max(low + 1, int(np.quantile(durations, 0.90)))
    return low, high


def _decode_reps_from_event_probs(
    probs: np.ndarray,
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    *,
    event_threshold: float,
) -> list[RepDetection]:
    start_events = _extract_events(probs, EVENT_START, float(event_threshold))
    transition_events = _extract_events(probs, EVENT_TRANSITION, float(event_threshold))
    end_events = _extract_events(probs, EVENT_END, float(event_threshold))
    min_rep, max_rep = _duration_bounds(train_streams)
    used_t = set()
    used_e = set()
    reps: list[RepDetection] = []
    for s_idx, _ in sorted(start_events, key=lambda item: item[0]):
        t_candidates = [
            (i, item)
            for i, item in enumerate(transition_events)
            if i not in used_t and item[0] > s_idx and item[0] - s_idx < max_rep
        ]
        if not t_candidates:
            continue
        t_i, (t_idx, _) = max(t_candidates, key=lambda pair: pair[1][1])
        e_candidates = [
            (i, item)
            for i, item in enumerate(end_events)
            if i not in used_e and item[0] > t_idx and min_rep <= item[0] - s_idx <= max_rep
        ]
        if not e_candidates:
            continue
        e_i, (e_idx, _) = max(e_candidates, key=lambda pair: pair[1][1])
        used_t.add(t_i)
        used_e.add(e_i)
        reps.append(
            RepDetection(
                start_idx=int(s_idx),
                transition_idx=int(t_idx),
                end_idx=int(e_idx + 1),
                micro_source="direct_event_rf",
                micro_confidence=float((probs[s_idx, EVENT_START] + probs[t_idx, EVENT_TRANSITION] + probs[e_idx, EVENT_END]) / 3.0),
                pred_action_type="unknown",
                action_confidence=0.0,
            )
        )
    return reps


def _aggregate_rows(rows: Sequence[dict]) -> dict:
    agg = {k: sum(float(r.get(k, 0.0)) for r in rows) for k in ["n_pred", "n_true", "tp", "fp", "fn"]}
    precision = agg["tp"] / max(1.0, agg["tp"] + agg["fp"])
    recall = agg["tp"] / max(1.0, agg["tp"] + agg["fn"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    avg_keys = ["start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_10", "micro_f1_at_25", "micro_f1_at_50"]
    avgs = {}
    for key in avg_keys:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
        avgs[key] = float(np.mean(vals)) if vals else None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "rep_f1": float(f1),
        **agg,
        **avgs,
        "stream_count": len(rows),
        "exact_count_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) == int(r.get("n_true", 0))),
        "over_segmented_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) > int(r.get("n_true", 0))),
        "under_segmented_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) < int(r.get("n_true", 0))),
        "zero_tp_streams": sum(1 for r in rows if int(r.get("tp", 0)) == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick direct event RF probe for rep boundaries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/direct_event_rf_probe")
    parser.add_argument("--outer-subject", required=True)
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--event-radius", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--event-threshold", type=float, default=0.30)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-samples", type=float, default=0.7)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    train_cfg = cb.TrainConfig(**{k: v for k, v in train_raw.items() if k in cb.TrainConfig.__dataclass_fields__})
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

    actions = sorted(set(_action_from_stream_id(sid) for sid, _ in test_streams) & set(_action_from_stream_id(sid) for sid, _ in train_streams))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    per_action = {}
    for action in actions:
        action_train = [(sid, df) for sid, df in train_streams if _action_from_stream_id(sid) == action]
        action_test = [(sid, df) for sid, df in test_streams if _action_from_stream_id(sid) == action]
        if not action_train or not action_test:
            continue
        print(f"[INFO] action={action} train_streams={len(action_train)} test_streams={len(action_test)}", flush=True)
        clf = _train_event_rf(
            action_train,
            imu_columns,
            window_size=int(args.window_size),
            stride=int(args.train_stride),
            event_radius=int(args.event_radius),
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            max_samples=float(args.max_samples),
        )
        action_rows = []
        for stream_id, df in action_test:
            probs = _predict_event_probs(
                clf,
                df,
                imu_columns,
                window_size=int(args.window_size),
                smoothing_window=int(args.smoothing_window),
            )
            pred_reps = _decode_reps_from_event_probs(probs, action_train, event_threshold=float(args.event_threshold))
            truth = truth_reps_from_labels(
                df["phase"].to_numpy(),
                actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
                min_phase_samples=1,
            )
            sample_rate = float(cb.infer_sample_rate_hz(df))
            metrics = rep_metrics(pred_reps, truth, sample_rate_hz=sample_rate)
            row = {
                **metrics,
                "rep_f1": float(metrics.get("f1", 0.0)),
                "stream_id": stream_id,
                "action": action,
                "outer_test_subject": outer_subject,
                "model_name": "direct_event_rf_probe",
                "count_diff": float(metrics.get("n_pred", 0.0) - metrics.get("n_true", 0.0)),
            }
            action_rows.append(row)
            all_rows.append(row)
        summary = _aggregate_rows(action_rows)
        per_action[action] = summary

    overall = _aggregate_rows(all_rows)
    results = {
        "benchmark": "direct_event_rf_probe",
        "outer_subject": outer_subject,
        "config": str(args.config),
        "actions": actions,
        "settings": {
            "window_size": int(args.window_size),
            "train_stride": int(args.train_stride),
            "event_radius": int(args.event_radius),
            "smoothing_window": int(args.smoothing_window),
            "event_threshold": float(args.event_threshold),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "max_samples": float(args.max_samples),
        },
        "overall": overall,
        "per_action": per_action,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(out_dir / "stream_metrics.csv", index=False)
    print(json.dumps(results, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
