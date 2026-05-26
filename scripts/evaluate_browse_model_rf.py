#!/usr/bin/env python3
"""Batch evaluate browse_model_replay.py RF model (with duration prior + boundary refiner).

Usage:
    python scripts/evaluate_browse_model_rf.py
Output:
    artifacts/evaluation/bmrf_refiner/results.json
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from preprocessing.micro_macro_segments import rep_metrics
import compare_baselines as cb
import importlib.util


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_mod(ROOT / "scripts" / "train_rf_boundary_refiner.py", "train_rf_boundary_refiner_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")

DATA_ROOT = ROOT / "datasets" / "raw_data"
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
OUT_DIR = ROOT / "artifacts" / "evaluation" / "bmrf_refiner"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_all_subjects(data_root: Path) -> List[str]:
    return sorted([d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_")])


def _get_all_actions(data_root: Path) -> List[str]:
    actions = set()
    for csv_path in data_root.rglob("*.csv"):
        parts = csv_path.relative_to(data_root).parts
        if len(parts) >= 4:
            actions.add(parts[2])
    return sorted([a for a in actions if a != "big_rest"])


def _load_action_streams(data_root: Path, action: str) -> List[Tuple[str, pd.DataFrame]]:
    streams = []
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        parts = rel.parts
        if len(parts) >= 4 and parts[2] == action:
            try:
                df = pd.read_csv(csv_path)
                sid = "/".join(parts)
                streams.append((sid, df))
            except Exception:
                pass
    return streams


def _compute_zscore_stats(train_streams: List[Tuple[str, pd.DataFrame]], imu_columns: List[str]):
    sequences = [df for _, df in train_streams]
    if not sequences:
        return None
    return compute_train_stats(sequences, imu_columns)


def _fit_refiner(train_z: List[Tuple[str, pd.DataFrame]], train_prob_cache: Dict[str, np.ndarray],
                 mm_cfg, imu_columns: List[str], edge_window: int = 20, max_shift: int = 20):
    matched_examples = []
    for sid, df in train_z:
        probs = train_prob_cache.get(sid)
        if probs is None:
            continue
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        truth = cb.truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=mm_cfg.min_phase_samples,
        )
        matches = cb.match_segments(
            [(r.start_idx, r.end_idx) for r in coarse_reps],
            [(r.start_idx, r.end_idx) for r in truth],
            iou_threshold=0.3,
        )
        for pred_idx, true_idx, _ in matches:
            p = coarse_reps[pred_idx]
            t = truth[true_idx]
            matched_examples.append({
                "df": df, "probs": probs, "pred_rep": p,
                "start_shift": int(np.clip(t.start_idx - p.start_idx, -max_shift, max_shift)),
                "transition_shift": int(np.clip(t.transition_idx - p.transition_idx, -max_shift, max_shift)),
                "end_shift": int(np.clip(t.end_idx - p.end_idx, -max_shift, max_shift)),
            })
    if len(matched_examples) < 5:
        return None

    start_rows, trans_rows, end_rows = [], [], []
    y_start, y_trans, y_end = [], [], []
    for ex in matched_examples:
        start_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "start", edge_window, imu_columns))
        trans_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "transition", edge_window, imu_columns))
        end_rows.append(base._build_edge_features(ex["df"], ex["probs"], ex["pred_rep"], "end", edge_window, imu_columns))
        y_start.append(float(ex["start_shift"]))
        y_trans.append(float(ex["transition_shift"]))
        y_end.append(float(ex["end_shift"]))

    # Ensure all three edge types use the same feature keys
    all_keys = set()
    for row in start_rows + trans_rows + end_rows:
        all_keys.update(row.keys())
    all_keys = sorted(all_keys)
    
    def _to_matrix(rows, keys):
        return np.asarray([[float(row.get(k, 0.0)) for k in keys] for row in rows], dtype=np.float32)
    
    x_start = _to_matrix(start_rows, all_keys)
    x_trans = _to_matrix(trans_rows, all_keys)
    x_end = _to_matrix(end_rows, all_keys)

    from sklearn.ensemble import ExtraTreesRegressor

    def _fit_reg(x, y):
        model = ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42)
        model.fit(x, y)
        return model

    return {
        "start": _fit_reg(x_start, np.asarray(y_start, dtype=np.float32)),
        "transition": _fit_reg(x_trans, np.asarray(y_trans, dtype=np.float32)),
        "end": _fit_reg(x_end, np.asarray(y_end, dtype=np.float32)),
        "feature_keys": all_keys,
        "edge_window": edge_window,
        "max_shift": max_shift,
    }


def _apply_duration_prior(reps, min_dur: int, max_dur: int):
    if min_dur <= 0 and max_dur <= 0:
        return reps
    out = []
    for r in reps:
        dur = int(r.end_idx) - int(r.start_idx)
        if min_dur > 0 and dur < min_dur:
            continue
        if max_dur > 0 and dur > max_dur:
            continue
        out.append(r)
    return out


def evaluate_action_loso(action: str, data_root: Path, imu_columns: List[str],
                         window_size: int = 100, stride: int = 10,
                         n_estimators: int = 100, max_depth: int = 15,
                         max_samples: float = 0.7, smoothing_window: int = 15,
                         edge_window: int = 20, max_shift: int = 20):
    """Evaluate one action with LOSO. Returns list of per-fold metrics."""
    all_streams = _load_action_streams(data_root, action)
    subjects = sorted(set(sid.split("/")[0] for sid, _ in all_streams))
    mm_cfg = cb.MicroMacroConfig()

    fold_results = []
    for test_subject in subjects:
        train_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams if sid.split("/")[0] == test_subject]
        if not test_streams:
            continue

        stats = _compute_zscore_stats(train_streams, imu_columns)
        if stats is None:
            continue

        train_z = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

        # Duration prior from training GT
        train_durations = []
        for _, df in train_streams:
            gt_reps = cb.truth_reps_from_labels(
                df["phase"].to_numpy(),
                actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
                min_phase_samples=mm_cfg.min_phase_samples,
            )
            for r in gt_reps:
                train_durations.append(int(r.end_idx) - int(r.start_idx))

        if train_durations:
            min_dur = int(np.quantile(train_durations, 0.05)) if len(train_durations) >= 10 else min(train_durations)
            max_dur = int(np.quantile(train_durations, 0.95)) if len(train_durations) >= 10 else max(train_durations)
            min_rep_dur = max(1, int(min_dur * 0.5))
            max_rep_dur = max(min_rep_dur + 1, int(max_dur * 2.0))
        else:
            min_rep_dur, max_rep_dur = 0, 0

        # Train RF
        clf = crf.train_causal_rf(
            train_z, imu_columns,
            window_size=window_size, stride=stride,
            n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples,
        )

        # Cache train probs (sample subset for speed) and fit refiner
        # Only use up to 30 train streams for refiner to keep runtime reasonable
        refiner_train_z = train_z[:30] if len(train_z) > 30 else train_z
        train_prob_cache = {}
        for sid, df in refiner_train_z:
            probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
            if smoothing_window > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    st = max(0, i - smoothing_window + 1)
                    total = csum[i] - (csum[st - 1] if st > 0 else 0.0)
                    smoothed[i] = total / float(i - st + 1)
                probs = smoothed
            train_prob_cache[sid] = probs

        refiner = _fit_refiner(refiner_train_z, train_prob_cache, mm_cfg, imu_columns, edge_window=edge_window, max_shift=max_shift)

        # Predict test streams
        fold_metrics_list = []
        for sid, df in test_z:
            probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=window_size, stride=1)
            if smoothing_window > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    st = max(0, i - smoothing_window + 1)
                    total = csum[i] - (csum[st - 1] if st > 0 else 0.0)
                    smoothed[i] = total / float(i - st + 1)
                probs = smoothed

            gt_reps = cb.truth_reps_from_labels(
                df["phase"].to_numpy(),
                actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
                min_phase_samples=mm_cfg.min_phase_samples,
            )

            coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
            coarse_reps = _apply_duration_prior(coarse_reps, min_rep_dur, max_rep_dur)

            if refiner is not None:
                pred_reps = base._refine_reps(df, probs, coarse_reps, refiner, imu_columns, edge_window=edge_window, max_shift=max_shift)
            else:
                pred_reps = list(coarse_reps)

            pred_reps = _apply_duration_prior(pred_reps, min_rep_dur, max_rep_dur)

            sample_rate = cb.infer_sample_rate_hz(df)
            metrics = rep_metrics(pred_reps, gt_reps, sample_rate_hz=sample_rate)

            n_pred = int(metrics.get("n_pred", 0))
            n_true = int(metrics.get("n_true", 0))
            metrics["count_diff"] = float(n_pred - n_true)
            metrics["exact_count"] = 1.0 if n_pred == n_true else 0.0
            metrics["over_segmented"] = 1.0 if n_pred > n_true else 0.0
            metrics["under_segmented"] = 1.0 if n_pred < n_true else 0.0
            metrics["stream_id"] = sid
            fold_metrics_list.append(metrics)

        # Aggregate fold-level metrics
        if fold_metrics_list:
            total_pred = sum(m["n_pred"] for m in fold_metrics_list)
            total_true = sum(m["n_true"] for m in fold_metrics_list)
            total_tp = sum(m["tp"] for m in fold_metrics_list)
            precision = total_tp / total_pred if total_pred > 0 else 0.0
            recall = total_tp / total_true if total_true > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            exact_count = sum(1 for m in fold_metrics_list if m["exact_count"] > 0)
            over_count = sum(1 for m in fold_metrics_list if m["over_segmented"] > 0)
            under_count = sum(1 for m in fold_metrics_list if m["under_segmented"] > 0)
            count_diffs = [abs(m["n_pred"] - m["n_true"]) for m in fold_metrics_list]
            mean_abs_diff = np.mean(count_diffs) if count_diffs else 0.0

            start_mae_vals = [m["start_mae_ms"] for m in fold_metrics_list if m["tp"] > 0 and np.isfinite(m["start_mae_ms"])]
            end_mae_vals = [m["end_mae_ms"] for m in fold_metrics_list if m["tp"] > 0 and np.isfinite(m["end_mae_ms"])]

            fold_results.append({
                "test_subject": test_subject,
                "action": action,
                "stream_count": len(fold_metrics_list),
                "n_pred": total_pred,
                "n_true": total_true,
                "tp": total_tp,
                "precision": precision,
                "recall": recall,
                "rep_f1": f1,
                "exact_count_streams": exact_count,
                "over_segmented_streams": over_count,
                "under_segmented_streams": under_count,
                "exact_count_ratio": exact_count / len(fold_metrics_list) if fold_metrics_list else 0.0,
                "mean_abs_count_diff": mean_abs_diff,
                "start_mae_ms": float(np.mean(start_mae_vals)) if start_mae_vals else float("nan"),
                "end_mae_ms": float(np.mean(end_mae_vals)) if end_mae_vals else float("nan"),
                "stream_metrics": fold_metrics_list,
            })

    return fold_results


def main():
    print("=" * 80)
    print("Evaluating browse_model_replay.py RF model (Duration Prior + Boundary Refiner)")
    print("=" * 80)

    actions = _get_all_actions(DATA_ROOT)
    print(f"Actions: {actions}")

    all_results = []
    for action in actions:
        print(f"\nEvaluating action: {action} ...")
        fold_results = evaluate_action_loso(
            action, DATA_ROOT, IMU_COLUMNS,
            window_size=100, stride=10,
            n_estimators=100, max_depth=15,
            max_samples=0.7, smoothing_window=15,
            edge_window=20, max_shift=20,
        )
        all_results.extend(fold_results)
        if fold_results:
            exact_ratios = [f["exact_count_ratio"] for f in fold_results]
            print(f"  Folds: {len(fold_results)}, Exact Count Ratio: {np.mean(exact_ratios):.1%}")

    # Save results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Overall summary
    total_streams = sum(r["stream_count"] for r in all_results)
    total_pred = sum(r["n_pred"] for r in all_results)
    total_true = sum(r["n_true"] for r in all_results)
    total_tp = sum(r["tp"] for r in all_results)
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_true if total_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    total_exact = sum(r["exact_count_streams"] for r in all_results)
    total_over = sum(r["over_segmented_streams"] for r in all_results)
    total_under = sum(r["under_segmented_streams"] for r in all_results)
    exact_ratio = total_exact / total_streams if total_streams > 0 else 0.0
    mean_diff = np.mean([r["mean_abs_count_diff"] for r in all_results])

    summary = {
        "model": "browse_model_replay RF (Duration Prior + Boundary Refiner)",
        "n_folds": len(all_results),
        "total_streams": total_streams,
        "n_pred": total_pred,
        "n_true": total_true,
        "tp": total_tp,
        "precision": precision,
        "recall": recall,
        "rep_f1": f1,
        "exact_count_streams": total_exact,
        "over_segmented_streams": total_over,
        "under_segmented_streams": total_under,
        "exact_count_ratio": exact_ratio,
        "mean_abs_count_diff": mean_diff,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Total Folds:         {len(all_results)}")
    print(f"  Total Streams:       {total_streams}")
    print(f"  Rep F1:              {f1:.4f}")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall:              {recall:.4f}")
    print(f"  Exact Count Ratio:   {exact_ratio:.1%} ({total_exact}/{total_streams})")
    print(f"  Over-segmented:      {total_over} streams")
    print(f"  Under-segmented:     {total_under} streams")
    print(f"  Mean Abs Count Diff: {mean_diff:.2f} reps/stream")
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
