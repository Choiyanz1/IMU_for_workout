from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor


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

from evaluation.rep_segmentation import _write_segmentation_svg
from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    OTHER_LABEL,
    RepDetection,
    SegmentRun,
    labels_to_runs,
    match_segments,
    rep_metrics,
    segment_iou_f1,
    truth_reps_from_labels,
)
from preprocessing.sdtw_rep_segmentation import SegmentDetection, infer_sample_rate_hz


def _phase_labels_from_reps(n: int, reps: Sequence[RepDetection]) -> np.ndarray:
    labels = np.full(n, OTHER_LABEL, dtype=object)
    for rep in reps:
        s = max(0, min(n, int(rep.start_idx)))
        t = max(s, min(n, int(rep.transition_idx)))
        e = max(t, min(n, int(rep.end_idx)))
        labels[s:t] = CONCENTRIC_LABEL
        labels[t:e] = ECCENTRIC_LABEL
    return labels


def _edge_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    if arr.size == 0:
        return feats
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    vmax = arr.max(axis=0)
    vmin = arr.min(axis=0)
    rng = vmax - vmin
    rms = np.sqrt(np.mean(arr ** 2, axis=0))
    for i in range(arr.shape[1]):
        feats[f"{prefix}_mean_{i}"] = float(mean[i])
        feats[f"{prefix}_std_{i}"] = float(std[i])
        feats[f"{prefix}_max_{i}"] = float(vmax[i])
        feats[f"{prefix}_min_{i}"] = float(vmin[i])
        feats[f"{prefix}_range_{i}"] = float(rng[i])
        feats[f"{prefix}_rms_{i}"] = float(rms[i])
    return feats


def _edge_prob_stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    if arr.size == 0:
        return feats
    mean = arr.mean(axis=0)
    vmax = arr.max(axis=0)
    vmin = arr.min(axis=0)
    for i, name in enumerate(("other", "concentric", "eccentric")):
        feats[f"{prefix}_{name}_mean"] = float(mean[i])
        feats[f"{prefix}_{name}_max"] = float(vmax[i])
        feats[f"{prefix}_{name}_min"] = float(vmin[i])
    return feats


def _build_edge_features(df: pd.DataFrame, probs: np.ndarray, rep: RepDetection, edge_name: str, half_window: int, imu_columns: Sequence[str]) -> dict[str, float]:
    if edge_name == "start":
        idx = int(rep.start_idx)
    elif edge_name == "transition":
        idx = int(rep.transition_idx)
    else:
        idx = int(rep.end_idx)
    arr = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(arr)
    lo = max(0, idx - half_window)
    hi = min(n, idx + half_window + 1)
    before = arr[lo:idx]
    after = arr[idx:hi]
    before_p = probs[lo:idx]
    after_p = probs[idx:hi]
    feats: dict[str, float] = {
        "edge_idx": float(idx),
        "stream_pos": float(idx) / max(1.0, float(n)),
        "rep_start": float(rep.start_idx),
        "rep_transition": float(rep.transition_idx),
        "rep_end": float(rep.end_idx),
        "rep_duration": float(rep.end_idx - rep.start_idx),
        "concentric_duration": float(rep.transition_idx - rep.start_idx),
        "eccentric_duration": float(rep.end_idx - rep.transition_idx),
        "micro_confidence": float(rep.micro_confidence),
    }
    feats.update(_edge_stats(before, "imu_before"))
    feats.update(_edge_stats(after, "imu_after"))
    feats.update(_edge_prob_stats(before_p, "prob_before"))
    feats.update(_edge_prob_stats(after_p, "prob_after"))
    if len(before) and len(after):
        before_mean = before.mean(axis=0)
        after_mean = after.mean(axis=0)
        for i in range(len(before_mean)):
            feats[f"imu_delta_mean_{i}"] = float(after_mean[i] - before_mean[i])
    if len(before_p) and len(after_p):
        db = after_p.mean(axis=0) - before_p.mean(axis=0)
        for i, name in enumerate(("other", "concentric", "eccentric")):
            feats[f"prob_delta_mean_{name}"] = float(db[i])
    return feats


def _rows_to_matrix(rows: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    key_set = set()
    for row in rows:
        key_set.update(row.keys())
    keys = sorted(key_set)
    x = np.asarray([[float(row.get(k, 0.0)) for k in keys] for row in rows], dtype=np.float32)
    return x, keys


def _fit_regressor(x: np.ndarray, y: np.ndarray) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x, y)
    return model


def _coarse_predict_reps(df: pd.DataFrame, probs: np.ndarray, mm_cfg) -> tuple[list[RepDetection], list[SegmentRun]]:
    labels = cb._decode_phase_labels(probs, mm_cfg)
    runs = labels_to_runs(
        labels,
        positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
        probabilities=probs,
        min_length=mm_cfg.min_phase_samples,
    )
    reps, _ = cb.pair_concentric_eccentric_reps(runs, micro_source="causal_rf", max_gap_samples=mm_cfg.max_phase_gap_samples)
    reps = cb._filter_predicted_reps(
        reps,
        sample_rate_hz=infer_sample_rate_hz(df),
        min_duration_seconds=mm_cfg.min_rep_duration_seconds,
        min_confidence=mm_cfg.min_rep_confidence,
    )
    return reps, runs


def _train_refiners(
    train_streams,
    clf,
    imu_columns: Sequence[str],
    mm_cfg,
    window_size: int,
    edge_window: int,
    match_iou: float,
    max_shift: int,
    target_matched_reps: int,
    max_refiner_train_streams: int,
):
    start_rows = []
    trans_rows = []
    end_rows = []
    y_start = []
    y_trans = []
    y_end = []

    train_subset = list(train_streams)
    if int(max_refiner_train_streams) > 0:
        train_subset = train_subset[: int(max_refiner_train_streams)]

    for stream_idx, (stream_id, df) in enumerate(train_subset, start=1):
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
        coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
        truth = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        matches = match_segments(
            [(r.start_idx, r.end_idx) for r in coarse_reps],
            [(r.start_idx, r.end_idx) for r in truth],
            iou_threshold=match_iou,
        )
        for pi, ti, _ in matches:
            p = coarse_reps[pi]
            t = truth[ti]
            ds = int(np.clip(t.start_idx - p.start_idx, -max_shift, max_shift))
            dt = int(np.clip(t.transition_idx - p.transition_idx, -max_shift, max_shift))
            de = int(np.clip(t.end_idx - p.end_idx, -max_shift, max_shift))
            start_rows.append(_build_edge_features(df, probs, p, "start", edge_window, imu_columns))
            trans_rows.append(_build_edge_features(df, probs, p, "transition", edge_window, imu_columns))
            end_rows.append(_build_edge_features(df, probs, p, "end", edge_window, imu_columns))
            y_start.append(ds)
            y_trans.append(dt)
            y_end.append(de)
        if stream_idx % 10 == 0 or stream_idx == len(train_subset):
            print(
                f"  [RFRefiner] collected {stream_idx}/{len(train_subset)} train streams "
                f"matched_reps_so_far={len(y_start)}",
                flush=True,
            )
        if int(target_matched_reps) > 0 and len(y_start) >= int(target_matched_reps):
            print(
                f"  [RFRefiner] early stop: reached target_matched_reps={target_matched_reps}",
                flush=True,
            )
            break

    x_start, feature_keys = _rows_to_matrix(start_rows)
    x_trans, _ = _rows_to_matrix(trans_rows)
    x_end, _ = _rows_to_matrix(end_rows)
    print(f"  [RFRefiner] matched reps for training: {len(y_start)}", flush=True)
    start_model = _fit_regressor(x_start, np.asarray(y_start, dtype=np.float32))
    trans_model = _fit_regressor(x_trans, np.asarray(y_trans, dtype=np.float32))
    end_model = _fit_regressor(x_end, np.asarray(y_end, dtype=np.float32))
    return {
        "start": start_model,
        "transition": trans_model,
        "end": end_model,
        "feature_keys": feature_keys,
    }


def _predict_delta(model: ExtraTreesRegressor, feats: dict[str, float], feature_keys: Sequence[str], max_shift: int) -> int:
    x = np.asarray([[float(feats.get(k, 0.0)) for k in feature_keys]], dtype=np.float32)
    pred = float(model.predict(x)[0])
    return int(np.clip(np.round(pred), -max_shift, max_shift))


def _refine_reps(df: pd.DataFrame, probs: np.ndarray, coarse_reps: Sequence[RepDetection], refiner, imu_columns: Sequence[str], edge_window: int, max_shift: int) -> list[RepDetection]:
    out: list[RepDetection] = []
    keys = refiner["feature_keys"]
    for rep in coarse_reps:
        fs = _build_edge_features(df, probs, rep, "start", edge_window, imu_columns)
        ft = _build_edge_features(df, probs, rep, "transition", edge_window, imu_columns)
        fe = _build_edge_features(df, probs, rep, "end", edge_window, imu_columns)
        ds = _predict_delta(refiner["start"], fs, keys, max_shift)
        dt = _predict_delta(refiner["transition"], ft, keys, max_shift)
        de = _predict_delta(refiner["end"], fe, keys, max_shift)
        start_idx = max(0, int(rep.start_idx) + ds)
        transition_idx = max(start_idx + 1, int(rep.transition_idx) + dt)
        end_idx = max(transition_idx + 1, int(rep.end_idx) + de)
        n = len(df)
        start_idx = min(start_idx, n - 2)
        transition_idx = min(max(transition_idx, start_idx + 1), n - 1)
        end_idx = min(max(end_idx, transition_idx + 1), n)
        out.append(
            RepDetection(
                start_idx=int(start_idx),
                transition_idx=int(transition_idx),
                end_idx=int(end_idx),
                micro_source=f"{rep.micro_source}_refined",
                micro_confidence=float(rep.micro_confidence),
                pred_action_type=str(rep.pred_action_type),
                action_confidence=float(rep.action_confidence),
            )
        )
    return out


def _evaluate_stream(df: pd.DataFrame, probs: np.ndarray, reps: Sequence[RepDetection], sample_rate_hz: float):
    truth = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=1,
    )
    metrics = rep_metrics(reps, truth, sample_rate_hz)
    pred_labels = _phase_labels_from_reps(len(df), reps)
    gt_labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
    gt_runs = labels_to_runs(gt_labels, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=1)
    pred_runs = labels_to_runs(pred_labels, positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL), min_length=1)
    metrics.update({f"micro_f1_at_{k}": v for k, v in [(10, segment_iou_f1(gt_runs, pred_runs)["f1_at_10"]), (25, segment_iou_f1(gt_runs, pred_runs)["f1_at_25"]), (50, segment_iou_f1(gt_runs, pred_runs)["f1_at_50"])]})
    return truth, metrics


def _aggregate_rows(rows: list[dict]) -> dict:
    agg = {k: sum(float(r.get(k, 0.0)) for r in rows) for k in ["n_pred", "n_true", "tp", "fp", "fn"]}
    p = agg["tp"] / max(1.0, agg["tp"] + agg["fp"])
    r = agg["tp"] / max(1.0, agg["tp"] + agg["fn"])
    f1 = 2 * p * r / max(1e-9, p + r)
    avg_keys = ["start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_10", "micro_f1_at_25", "micro_f1_at_50"]
    avgs = {}
    for k in avg_keys:
        vals = [float(r[k]) for r in rows if k in r and np.isfinite(r[k])]
        avgs[k] = float(np.mean(vals)) if vals else None
    return {
        "precision": p,
        "recall": r,
        "rep_f1": f1,
        **agg,
        **avgs,
        "stream_count": len(rows),
        "exact_count_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) == int(r.get("n_true", 0))),
        "over_segmented_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) > int(r.get("n_true", 0))),
        "under_segmented_streams": sum(1 for r in rows if int(r.get("n_pred", 0)) < int(r.get("n_true", 0))),
        "zero_tp_streams": sum(1 for r in rows if int(r.get("tp", 0)) == 0),
        "stream_rows": rows,
    }


def _write_index_html(path: Path, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: float(r.get("micro_f1_at_50", 0.0)), reverse=True)
    trs = []
    for row in rows_sorted:
        rel_svg = html.escape(str(row.get("svg_rel", "")))
        trs.append(
            f"<tr><td><a href=\"{rel_svg}\">{html.escape(str(row['stream_id']))}</a></td>"
            f"<td>{float(row.get('rep_f1', 0.0)):.4f}</td>"
            f"<td>{float(row.get('micro_f1_at_50', 0.0)):.4f}</td>"
            f"<td>{float(row.get('start_mae_ms', float('nan'))):.1f}</td>"
            f"<td>{float(row.get('end_mae_ms', float('nan'))):.1f}</td></tr>"
        )
    html_text = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>RF Boundary Refiner Results</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:24px}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px 8px}} th{{background:#f3f4f6}}</style>
</head><body><h1>RF Boundary Refiner Stream Results</h1><table><thead><tr><th>Stream</th><th>Rep F1</th><th>micro_f1@50</th><th>Start MAE</th><th>End MAE</th></tr></thead><tbody>{''.join(trs)}</tbody></table></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an RF boundary refiner on top of causal RF.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/rf_boundary_refiner")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--rf-smoothing-window", type=int, default=1)
    parser.add_argument("--edge-window", type=int, default=20)
    parser.add_argument("--match-iou-train", type=float, default=0.3)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--target-matched-reps", type=int, default=1200)
    parser.add_argument("--max-refiner-train-streams", type=int, default=100)
    parser.add_argument("--max-train-streams-total", type=int, default=0)
    parser.add_argument("--max-test-streams", type=int, default=0)
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
    if int(args.max_train_streams_total) > 0:
        train_streams = train_streams[: int(args.max_train_streams_total)]
    if int(args.max_test_streams) > 0:
        test_streams = test_streams[: int(args.max_test_streams)]

    print(
        f"[INFO] resources cpu_count={os.cpu_count()} sklearn_cpu_parallel=-1 cuda_available=False_for_sklearn",
        flush=True,
    )
    print(f"[INFO] protocol={evaluation_protocol} test_subject={test_subject} train={len(train_streams)} test={len(test_streams)}", flush=True)
    t0 = time.time()
    clf = crf.train_causal_rf(
        train_streams,
        imu_columns,
        window_size=int(args.window_size),
        stride=int(args.train_stride),
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        max_samples=float(args.max_samples),
    )
    rf_train_time = time.time() - t0

    t0 = time.time()
    refiner = _train_refiners(
        train_streams,
        clf,
        imu_columns,
        mm_cfg,
        window_size=int(args.window_size),
        edge_window=int(args.edge_window),
        match_iou=float(args.match_iou_train),
        max_shift=int(args.max_shift),
        target_matched_reps=int(args.target_matched_reps),
        max_refiner_train_streams=int(args.max_refiner_train_streams),
    )
    refiner_train_time = time.time() - t0

    out_dir = Path(args.output)
    svg_root = out_dir / "stream_replays"
    svg_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for stream_idx, (stream_id, df) in enumerate(test_streams, start=1):
        probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(args.window_size), stride=1)
        coarse_reps, _ = _coarse_predict_reps(df, probs, mm_cfg)
        refined_reps = _refine_reps(
            df,
            probs,
            coarse_reps,
            refiner,
            imu_columns,
            edge_window=int(args.edge_window),
            max_shift=int(args.max_shift),
        )
        sample_rate = infer_sample_rate_hz(df)
        truth, metrics = _evaluate_stream(df, probs, refined_reps, sample_rate)
        row = {
            **metrics,
            "stream_id": stream_id,
            "count_diff": float(metrics.get("n_pred", 0.0) - metrics.get("n_true", 0.0)),
        }

        rel_parts = [p for p in stream_id.split("/") if p]
        svg_path = svg_root.joinpath(*rel_parts).with_suffix(".svg")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        detections = [
            SegmentDetection(
                start_idx=int(rep.start_idx),
                end_idx=int(rep.end_idx),
                cost=0.0,
                feature="rf",
                action_type=str(getattr(rep, "pred_action_type", "unknown")),
                template_id="rf_refined",
                exemplar_source=stream_id,
                normalized_cost=0.0,
            )
            for rep in refined_reps
        ]
        _write_segmentation_svg(
            svg_path,
            stream_id,
            df,
            [(r.start_idx, r.end_idx) for r in truth],
            detections,
            {
                "f1": float(metrics.get("f1", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "n_true": float(metrics.get("n_true", 0.0)),
                "n_pred": float(metrics.get("n_pred", 0.0)),
            },
            sample_rate,
        )
        row["svg_rel"] = svg_path.relative_to(out_dir).as_posix()
        rows.append(row)
        if stream_idx % 10 == 0 or stream_idx == len(test_streams):
            print(f"  [RFRefiner] evaluated {stream_idx}/{len(test_streams)} test streams", flush=True)

    results = _aggregate_rows(rows)
    results["model_name"] = "Causal RF + Boundary Refiner"
    results["evaluation_protocol"] = evaluation_protocol
    results["test_subject"] = test_subject
    results["rf_train_time_s"] = rf_train_time
    results["refiner_train_time_s"] = refiner_train_time
    results["config"] = {
        "window_size": int(args.window_size),
        "train_stride": int(args.train_stride),
        "rf_smoothing_window": int(args.rf_smoothing_window),
        "edge_window": int(args.edge_window),
        "match_iou_train": float(args.match_iou_train),
        "max_shift": int(args.max_shift),
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "max_samples": float(args.max_samples),
        "target_matched_reps": int(args.target_matched_reps),
        "max_refiner_train_streams": int(args.max_refiner_train_streams),
        "max_train_streams_total": int(args.max_train_streams_total),
        "max_test_streams": int(args.max_test_streams),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "stream_metrics.csv", index=False)
    _write_index_html(out_dir / "index.html", rows)
    print(json.dumps({k: results[k] for k in ["rep_f1", "precision", "recall", "start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_50"]}, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")
    print(f"[OK] open {out_dir / 'index.html'} to inspect stream-level replay SVGs")


if __name__ == "__main__":
    main()
