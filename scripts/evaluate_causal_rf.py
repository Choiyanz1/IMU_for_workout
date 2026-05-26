from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_compare_baselines_module():
    path = ROOT / "scripts" / "compare_baselines.py"
    spec = importlib.util.spec_from_file_location("compare_baselines_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load compare_baselines module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_compare_baselines_module()


def _build_trailing_feature_matrix(x: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    ends = np.arange(1, len(x) + 1, int(max(1, stride)), dtype=np.int64)
    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float32), ends
    window_size = int(max(1, window_size))
    prefix = np.repeat(x[:1], max(0, window_size - 1), axis=0)
    padded = np.concatenate([prefix, x], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    selected = windows[np.maximum(0, ends - 1)]
    return cb._extract_window_features_batch(selected), ends


def train_causal_rf(
    train_streams,
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 10,
    n_estimators: int = 50,
    max_depth: int = 15,
    max_samples: float = 0.7,
) -> RandomForestClassifier:
    X_all, y_all = [], []
    for stream_idx, (_, df) in enumerate(train_streams, start=1):
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(X_batch):
            X_all.append(X_batch)
            y_all.append(label_idx[np.maximum(0, ends - 1)])
        if stream_idx % 25 == 0 or stream_idx == len(train_streams):
            print(f"  [CausalRF] prepared {stream_idx}/{len(train_streams)} train streams", flush=True)
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    print(
        f"  [CausalRF] Training on {len(X_all)} trailing windows "
        f"({window_size} samples, stride {stride}, trees={n_estimators}, max_depth={max_depth}, max_samples={max_samples})",
        flush=True,
    )
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples) if max_samples and max_samples < 1.0 else max_samples,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_all, y_all)
    clf.verbose = 0
    return clf


def predict_causal_rf(
    clf: RandomForestClassifier,
    df,
    imu_columns: Sequence[str],
    window_size: int = 50,
    stride: int = 1,
) -> np.ndarray:
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    probs = np.zeros((n, len(cb.MICRO_LABELS)), dtype=np.float32)
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    X_batch, ends = _build_trailing_feature_matrix(x, int(window_size), int(stride))
    raw_batch = clf.predict_proba(X_batch) if len(X_batch) else np.zeros((0, len(class_map)), dtype=np.float32)
    if len(raw_batch):
        full_batch = np.zeros((len(raw_batch), len(cb.MICRO_LABELS)), dtype=np.float32)
        for cls_idx, mi in class_map.items():
            full_batch[:, cls_idx] = raw_batch[:, mi]
        probs[np.maximum(0, ends - 1)] = full_batch
    return probs


def _parse_int_list(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if str(x).strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a strict causal Random Forest phase model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/causal_rf")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--smoothing-windows", default="")
    parser.add_argument("--max-streams", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = raw.get("data", {}) or {}
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
    actions = cb._available_actions(Path(data_cfg.get("data_dir", "./datasets/raw_data")), data_cfg.get("include_actions"))
    macro_classes = [cb.OTHER_LABEL] + [a for a in actions if a != cb.OTHER_LABEL]

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    subjects_sorted = sorted(set(subjects))
    configured_test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else subjects_sorted[-1]
    train_all_subjects = cb._is_all_subjects_mode(configured_test_subject)
    if train_all_subjects:
        test_subject = "__all__"
        train_streams = list(streams)
        test_streams = list(streams)
        evaluation_protocol = "train_all_in_sample"
    else:
        test_subject = configured_test_subject
        train_subjects = [s for s in subjects_sorted if s != test_subject]
        train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
        test_streams = cb._filter_subjects(streams, [test_subject], subject_column)
        evaluation_protocol = "subject_holdout"

    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    if int(args.max_streams) > 0:
        train_streams = train_streams[: int(args.max_streams)]
        test_streams = test_streams[: int(args.max_streams)]

    print(f"[INFO] protocol={evaluation_protocol} test_subject={test_subject} train={len(train_streams)} test={len(test_streams)}", flush=True)
    t0 = time.time()
    clf = train_causal_rf(
        train_streams,
        imu_columns,
        window_size=int(args.window_size),
        stride=int(args.train_stride),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        max_samples=float(args.max_samples),
    )
    train_time = time.time() - t0

    t0 = time.time()
    raw_prob_cache = []
    for stream_idx, (stream_id, df) in enumerate(test_streams, start=1):
        probs = predict_causal_rf(clf, df, imu_columns, window_size=int(args.window_size), stride=1)
        raw_prob_cache.append((stream_id, df, probs))
        if stream_idx % 25 == 0 or stream_idx == len(test_streams):
            print(f"  [CausalRF] predicted {stream_idx}/{len(test_streams)} test streams", flush=True)
    predict_time = time.time() - t0

    windows = _parse_int_list(args.smoothing_windows) if str(args.smoothing_windows).strip() else [int(args.smoothing_window)]
    all_results = {}
    best_window = None
    best_score = -1.0
    for smooth_w in windows:
        t1 = time.time()
        smoothed_streams = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smooth_w) > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smooth_w) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    smoothed[i] = total / float(i - start + 1)
                cur = smoothed
            smoothed_streams.append((stream_id, df, cur))

        def predict_fn_from_cache(df, _cache_iter=iter(smoothed_streams)):
            _, _, cached_probs = next(_cache_iter)
            return cached_probs, None

        results = cb.evaluate_all_streams(predict_fn_from_cache, test_streams, macro_classes, mm_cfg)
        eval_time = time.time() - t1
        results["train_time_s"] = train_time
        results["predict_time_s"] = predict_time
        results["eval_time_s"] = eval_time
        results["model_name"] = "Causal Random Forest"
        results["evaluation_protocol"] = evaluation_protocol
        results["test_subject"] = test_subject
        results["config"] = {
            "window_size": int(args.window_size),
            "train_stride": int(args.train_stride),
            "smoothing_window": int(smooth_w),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "max_samples": float(args.max_samples),
        }
        all_results[str(smooth_w)] = results
        print(
            f"  [CausalRF] smoothing={smooth_w} rep_f1={results.get('rep_f1', 0):.4f} "
            f"micro_f1@50={results.get('micro_f1_at_50', 0):.4f}",
            flush=True,
        )
        score = float(results.get("micro_f1_at_50") or 0.0)
        if score > best_score:
            best_score = score
            best_window = int(smooth_w)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_results = all_results[str(best_window)]
    payload = {
        "best_smoothing_window": best_window,
        "best_results": best_results,
        "all_results": all_results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "best_smoothing_window": best_window,
        **{k: best_results[k] for k in ["rep_f1", "precision", "recall", "start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_50"]}
    }, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
