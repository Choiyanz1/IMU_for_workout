from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


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
base = _load_module(ROOT / "scripts" / "train_rf_boundary_refiner.py", "rf_boundary_refiner_mod")


def _action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) < 2:
        return "unknown"
    return parts[-2]


def _aggregate_per_action_results(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_action_from_stream_id(str(row["stream_id"]))].append(row)
    out = {}
    for action, action_rows in sorted(grouped.items()):
        out[action] = base._aggregate_rows(action_rows)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train one RF boundary refiner per action under subject-wise split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/per_action_rf_boundary_refiner")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--edge-window", type=int, default=20)
    parser.add_argument("--match-iou-train", type=float, default=0.3)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--target-matched-reps", type=int, default=500)
    parser.add_argument("--max-refiner-train-streams", type=int, default=40)
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--imu-columns", default="")
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
    if str(args.imu_columns).strip():
        imu_columns = [c.strip() for c in str(args.imu_columns).split(",") if c.strip()]
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    subjects_sorted = sorted(set(subjects))
    configured_test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else subjects_sorted[-1]
    if cb._is_all_subjects_mode(configured_test_subject):
        raise ValueError("Per-action evaluation should use subject_holdout, not test_subject=__all__")
    test_subject = configured_test_subject
    train_subjects = [s for s in subjects_sorted if s != test_subject]
    train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
    test_streams = cb._filter_subjects(streams, [test_subject], subject_column)
    evaluation_protocol = "subject_holdout"

    # Global train-fold normalization for fairer comparison.
    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

    requested_actions = [a for a in str(args.include_actions).split(",") if a.strip()] if str(args.include_actions).strip() else []
    all_actions = sorted(set(_action_from_stream_id(sid) for sid, _ in train_streams + test_streams))
    actions = [a for a in all_actions if (not requested_actions or a in requested_actions)]

    print(f"[INFO] resources cpu_count={os.cpu_count()} sklearn_cpu_parallel=-1 cuda_available=False_for_sklearn", flush=True)
    print(f"[INFO] protocol={evaluation_protocol} test_subject={test_subject} actions={actions}", flush=True)

    out_dir = Path(args.output)
    all_stream_rows = []
    action_summaries = {}

    for action in actions:
        action_train = [(sid, df) for sid, df in train_streams if _action_from_stream_id(sid) == action]
        action_test = [(sid, df) for sid, df in test_streams if _action_from_stream_id(sid) == action]
        if not action_train or not action_test:
            print(f"[WARN] skip action={action} train={len(action_train)} test={len(action_test)}", flush=True)
            continue

        print(f"\n[INFO] action={action} train_streams={len(action_train)} test_streams={len(action_test)}", flush=True)
        t0 = time.time()
        clf = base.crf.train_causal_rf(
            action_train,
            imu_columns,
            window_size=int(args.window_size),
            stride=int(args.train_stride),
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            max_samples=float(args.max_samples),
        )
        rf_train_time = time.time() - t0

        t0 = time.time()
        refiner = base._train_refiners(
            action_train,
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

        action_out_dir = out_dir / action
        svg_root = action_out_dir / "stream_replays"
        svg_root.mkdir(parents=True, exist_ok=True)

        rows = []
        for stream_idx, (stream_id, df) in enumerate(action_test, start=1):
            probs = base.crf.predict_causal_rf(clf, df, imu_columns, window_size=int(args.window_size), stride=1)
            coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
            refined_reps = base._refine_reps(
                df,
                probs,
                coarse_reps,
                refiner,
                imu_columns,
                edge_window=int(args.edge_window),
                max_shift=int(args.max_shift),
            )
            sample_rate = base.infer_sample_rate_hz(df)
            truth, metrics = base._evaluate_stream(df, probs, refined_reps, sample_rate)
            row = {
                **metrics,
                "stream_id": stream_id,
                "action": action,
                "count_diff": float(metrics.get("n_pred", 0.0) - metrics.get("n_true", 0.0)),
            }
            rel_parts = [p for p in stream_id.split("/") if p]
            svg_path = svg_root.joinpath(*rel_parts).with_suffix(".svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            detections = [
                base.SegmentDetection(
                    start_idx=int(rep.start_idx),
                    end_idx=int(rep.end_idx),
                    cost=0.0,
                    feature="rf_per_action",
                    action_type=action,
                    template_id=f"{action}_rf_refined",
                    exemplar_source=stream_id,
                    normalized_cost=0.0,
                )
                for rep in refined_reps
            ]
            base._write_segmentation_svg(
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
            if stream_idx % 5 == 0 or stream_idx == len(action_test):
                print(f"  [PerActionRF] action={action} evaluated {stream_idx}/{len(action_test)} test streams", flush=True)

        summary = base._aggregate_rows(rows)
        summary["rf_train_time_s"] = rf_train_time
        summary["refiner_train_time_s"] = refiner_train_time
        summary["action"] = action
        action_summaries[action] = summary
        (action_out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(rows).to_csv(action_out_dir / "stream_metrics.csv", index=False)
        base._write_index_html(action_out_dir / "index.html", rows)
        all_stream_rows.extend(rows)

    overall = base._aggregate_rows(all_stream_rows)
    overall["model_name"] = "Per-action Causal RF + Boundary Refiner"
    overall["evaluation_protocol"] = evaluation_protocol
    overall["test_subject"] = test_subject
    overall["per_action"] = action_summaries
    overall["config"] = {
        "imu_columns": imu_columns,
        "window_size": int(args.window_size),
        "train_stride": int(args.train_stride),
        "edge_window": int(args.edge_window),
        "match_iou_train": float(args.match_iou_train),
        "max_shift": int(args.max_shift),
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "max_samples": float(args.max_samples),
        "target_matched_reps": int(args.target_matched_reps),
        "max_refiner_train_streams": int(args.max_refiner_train_streams),
        "actions": actions,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(overall, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(all_stream_rows).to_csv(out_dir / "stream_metrics.csv", index=False)
    base._write_index_html(out_dir / "index.html", all_stream_rows)
    print(json.dumps({k: overall[k] for k in ["rep_f1", "precision", "recall", "start_mae_ms", "end_mae_ms", "transition_mae_ms", "micro_f1_at_50"]}, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")
    print(f"[OK] open {out_dir / 'index.html'} to inspect all stream replay SVGs")


if __name__ == "__main__":
    main()
