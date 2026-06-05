"""Integrate predicted RF action locks with the raw6 CNN + top5_p5 decoder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import (  # noqa: E402
    ACTIONS,
    WindowMeta,
    _starts_for_length,
    build_windows,
    extract_features_batch,
    load_non_action_streams,
    lock_action_for_stream,
    normalize_streams,
)
from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    _build_start_window_matrix,
    _prepare_active_labels,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from scripts.new_c_pipeline.duration_merge_decoder_9fold import (  # noqa: E402
    build_duration_priors,
    evaluate_with_reps,
    merge_short_reps,
    threshold_for_action,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    aggregate_rich,
    group_aggregate,
    stream_action,
    stream_subject,
    train_raw6_model,
)
from scripts.new_c_pipeline.selective_duration_merge_decoder_9fold import ACTION_SETS  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def train_action_rf(train_set_streams, train_rest_streams, imu_columns, args):
    train_streams = [*train_set_streams, *train_rest_streams]
    train_norm = normalize_streams(train_streams, train_streams, imu_columns)
    x_train, y_active_train, y_action_train, _ = build_windows(
        train_norm,
        imu_columns,
        args.action_window_samples,
        args.action_stride_samples,
        args.window_active_threshold,
    )
    active_rf = RandomForestClassifier(
        n_estimators=args.action_n_estimators,
        max_depth=args.action_max_depth,
        min_samples_leaf=args.action_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    active_rf.fit(x_train, y_active_train)

    active_idx = np.where(y_active_train == 1)[0]
    action_rf = RandomForestClassifier(
        n_estimators=args.action_n_estimators,
        max_depth=args.action_max_depth,
        min_samples_leaf=args.action_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed + 17,
        n_jobs=-1,
    )
    action_rf.fit(x_train[active_idx], y_action_train[active_idx])
    return active_rf, action_rf, train_streams


def train_global_active_detector(train_set_streams, train_rest_streams, cfg, args):
    x_rows = []
    y_rows = []
    for _stream_id, df in [*train_set_streams, *train_rest_streams]:
        if not set(cfg.imu_columns).issubset(df.columns):
            continue
        values = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        if "phase" in df.columns:
            active_labels = _prepare_active_labels(df["phase"].to_numpy())
        else:
            active_labels = np.zeros(len(values), dtype=np.int64)
        features, starts, ends = _build_start_window_matrix(values, cfg.active_window_size, cfg.active_stride)
        if len(features) == 0:
            continue
        y_batch = [int(np.bincount(active_labels[int(s) : int(e)], minlength=2).argmax()) for s, e in zip(starts, ends)]
        x_rows.append(features)
        y_rows.append(np.asarray(y_batch, dtype=np.int64))

    if not x_rows:
        raise RuntimeError("No windows available to train global active detector")
    x_train = np.concatenate(x_rows, axis=0)
    y_train = np.concatenate(y_rows, axis=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    clf = RandomForestClassifier(
        n_estimators=cfg.active_n_estimators,
        max_depth=cfg.active_max_depth,
        max_samples=cfg.active_max_samples,
        class_weight="balanced_subsample",
        random_state=args.seed + 31,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf, scaler


def predict_global_active(model, scaler, df, cfg):
    values = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(values)
    features, starts, ends = _build_start_window_matrix(values, cfg.active_window_size, cfg.active_stride)
    if len(features) == 0:
        return np.zeros(n, dtype=np.float64)
    probs = model.predict_proba(scaler.transform(features))
    active_idx = list(model.classes_).index(1) if 1 in model.classes_ else 0
    prob_accum = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for wi, (start, end) in enumerate(zip(starts, ends)):
        prob_accum[int(start) : int(end)] += probs[wi, active_idx]
        counts[int(start) : int(end)] += 1.0
    counts = np.where(counts < 1e-8, 1.0, counts)
    return prob_accum / counts


def _probs_for_actions(action_rf, raw_probs):
    out = np.zeros((len(raw_probs), len(ACTIONS)), dtype=np.float32)
    class_to_col = {str(cls): idx for idx, cls in enumerate(action_rf.classes_)}
    for j, action in enumerate(ACTIONS):
        col = class_to_col.get(action)
        if col is not None:
            out[:, j] = raw_probs[:, col]
    return out


def build_inference_windows(streams, imu_columns, window_samples, stride_samples):
    x_rows = []
    metas = []
    for stream_id, df in streams:
        if not set(imu_columns).issubset(df.columns):
            continue
        values = df[list(imu_columns)].to_numpy(dtype=np.float32)
        if len(values) == 0:
            continue
        pad = max(0, window_samples - len(values))
        padded_values = np.pad(values, ((0, pad), (0, 0)), mode="edge") if pad else values
        starts = _starts_for_length(len(values), window_samples, stride_samples)
        windows = []
        for start in starts:
            end = min(int(start) + window_samples, len(values))
            windows.append(padded_values[int(start) : int(start) + window_samples])
            metas.append(WindowMeta(stream_id, subject="", action="", start=int(start), end=end, active_fraction=0.0))
        if windows:
            x_rows.extend(extract_features_batch(np.stack(windows)).astype(np.float32))
    x = np.vstack(x_rows).astype(np.float32) if x_rows else np.zeros((0, 0), dtype=np.float32)
    return x, metas


def predict_action_contexts(active_rf, action_rf, train_action_streams, test_set_streams, imu_columns, args):
    test_norm = normalize_streams(train_action_streams, test_set_streams, imu_columns)
    x_test, metas = build_inference_windows(
        test_norm,
        imu_columns,
        args.action_window_samples,
        args.action_stride_samples,
    )
    active_raw = active_rf.predict_proba(x_test)
    active_class_to_col = {int(cls): idx for idx, cls in enumerate(active_rf.classes_)}
    active_probs = active_raw[:, active_class_to_col.get(1, 0)] if 1 in active_class_to_col else np.zeros(len(x_test))
    action_probs = _probs_for_actions(action_rf, action_rf.predict_proba(x_test))

    by_stream = {}
    for idx, meta in enumerate(metas):
        by_stream.setdefault(meta.stream_id, []).append(idx)

    locks = {}
    contexts = {}
    for stream_id, indices in sorted(by_stream.items()):
        stream_metas = [metas[i] for i in indices]
        stream_action_probs = action_probs[indices]
        stream_active_probs = active_probs[indices]
        locks[stream_id] = lock_action_for_stream(
            stream_metas,
            stream_action_probs,
            stream_active_probs,
            args.lock_threshold,
            args.lock_active_threshold,
            args.lock_margin,
            args.stable_windows,
            args.min_lock_windows,
            args.sample_rate_hz,
        )

        weights = np.clip(stream_active_probs - args.soft_active_threshold, 0.0, None)
        if float(weights.sum()) <= 1e-8:
            weights = np.maximum(stream_active_probs, 1e-3)
        posterior = np.average(stream_action_probs, axis=0, weights=weights)
        order = np.argsort(posterior)[::-1]
        top_idx = int(order[0]) if len(order) else 0
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top5_indices = [ACTIONS.index(action) for action in ACTION_SETS["top5"] if action in ACTIONS]
        top5_mass = float(np.sum(posterior[top5_indices])) if top5_indices else 0.0
        contexts[stream_id] = {
            "action_posterior": {action: float(posterior[i]) for i, action in enumerate(ACTIONS)},
            "top_action": ACTIONS[top_idx],
            "top_confidence": float(posterior[top_idx]),
            "margin": float(posterior[top_idx] - posterior[second_idx]),
            "top5_mass": top5_mass,
        }
    return locks, contexts


def action_lock_summary(locks, test_streams):
    rows = []
    for stream_id, _df in test_streams:
        truth = stream_action(stream_id)
        lock = locks.get(stream_id, {"locked": False, "locked_action": None, "lock_time_s": None, "confidence": 0.0})
        rows.append(
            {
                "stream_id": stream_id,
                "true_action": truth,
                "locked": bool(lock.get("locked")),
                "locked_action": lock.get("locked_action"),
                "correct": bool(lock.get("locked")) and lock.get("locked_action") == truth,
                "lock_time_s": lock.get("lock_time_s"),
                "confidence": lock.get("confidence"),
            }
        )
    locked_rows = [row for row in rows if row["locked"]]
    lock_times = [float(row["lock_time_s"]) for row in locked_rows if row["lock_time_s"] is not None]
    return {
        "rows": rows,
        "summary": {
            "streams": len(rows),
            "lock_rate": float(len(locked_rows) / max(1, len(rows))),
            "locked_accuracy": float(np.mean([row["correct"] for row in locked_rows])) if locked_rows else 0.0,
            "median_lock_time_s": float(np.median(lock_times)) if lock_times else None,
        },
    }


def aggregate_action_locks(folds):
    rows = []
    for fold in folds:
        rows.extend(fold.get("action_lock_rows", []))
    locked_rows = [row for row in rows if row.get("locked")]
    lock_times = [float(row["lock_time_s"]) for row in locked_rows if row.get("lock_time_s") is not None]
    return {
        "streams": len(rows),
        "lock_rate": float(len(locked_rows) / max(1, len(rows))),
        "locked_accuracy": float(np.mean([row.get("correct", False) for row in locked_rows])) if locked_rows else 0.0,
        "median_lock_time_s": float(np.median(lock_times)) if lock_times else None,
    }


def apply_top5_merge(base_reps, duration_priors, action, args):
    if action not in ACTION_SETS["top5"]:
        return base_reps
    threshold = threshold_for_action(duration_priors, action, 5)
    return merge_short_reps(base_reps, threshold, args.max_gap_samples)


def apply_soft_top5_merge(base_reps, duration_priors, context, args):
    if not context:
        return base_reps, {"soft_merged": False, "soft_threshold_samples": None}
    if context["top5_mass"] < args.soft_top5_mass_threshold:
        return base_reps, {"soft_merged": False, "soft_threshold_samples": None}
    if context["top_confidence"] < args.soft_action_confidence_threshold:
        return base_reps, {"soft_merged": False, "soft_threshold_samples": None}
    if context["margin"] < args.soft_margin_threshold:
        return base_reps, {"soft_merged": False, "soft_threshold_samples": None}

    posterior = context["action_posterior"]
    weights = np.asarray([posterior.get(action, 0.0) for action in ACTION_SETS["top5"]], dtype=np.float32)
    total = float(weights.sum())
    if total <= 1e-8:
        return base_reps, {"soft_merged": False, "soft_threshold_samples": None}
    thresholds = np.asarray([threshold_for_action(duration_priors, action, 5) for action in ACTION_SETS["top5"]], dtype=np.float32)
    threshold = float(np.dot(weights, thresholds) / total)
    return merge_short_reps(base_reps, threshold, args.max_gap_samples), {
        "soft_merged": True,
        "soft_threshold_samples": threshold,
        "soft_top5_mass": context["top5_mass"],
        "soft_top_action": context["top_action"],
        "soft_top_confidence": context["top_confidence"],
        "soft_margin": context["margin"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted-action top5_p5 integration.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/predicted_action_top5_pipeline/summary.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--action-window-samples", type=int, default=200)
    parser.add_argument("--action-stride-samples", type=int, default=100)
    parser.add_argument("--window-active-threshold", type=float, default=0.5)
    parser.add_argument("--action-n-estimators", type=int, default=50)
    parser.add_argument("--action-max-depth", type=int, default=12)
    parser.add_argument("--action-min-samples-leaf", type=int, default=2)
    parser.add_argument("--lock-policy", choices=["stricter", "very_strict", "ultra"], default="stricter")
    parser.add_argument("--soft-active-threshold", type=float, default=0.55)
    parser.add_argument("--soft-top5-mass-threshold", type=float, default=0.65)
    parser.add_argument("--soft-action-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--soft-margin-threshold", type=float, default=0.05)
    parser.add_argument("--active-detector", choices=["global", "per_action_oracle"], default="global")
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    policies = {
        "stricter": {"lock_active_threshold": 0.75, "lock_threshold": 0.70, "lock_margin": 0.20, "stable_windows": 4, "min_lock_windows": 4},
        "very_strict": {"lock_active_threshold": 0.80, "lock_threshold": 0.75, "lock_margin": 0.20, "stable_windows": 5, "min_lock_windows": 5},
        "ultra": {"lock_active_threshold": 0.85, "lock_threshold": 0.80, "lock_margin": 0.25, "stable_windows": 5, "min_lock_windows": 5},
    }
    for key, value in policies[args.lock_policy].items():
        setattr(args, key, value)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    all_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    subjects = sorted({stream_subject(sid) for sid, _ in set_streams})
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects

    print(f"set_streams={len(set_streams)} rest_streams={len(rest_streams)} subjects={subjects} device={device}", flush=True)
    print(f"phase epochs={args.epochs} hidden={args.hidden} action_policy={args.lock_policy} active_detector={args.active_detector}", flush=True)

    raw_results = []
    oracle_results = []
    predicted_results = []
    soft_results = []
    folds = []

    for fold_idx, test_subject in enumerate(eval_subjects, start=1):
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject}", flush=True)
        train_set_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) == test_subject]
        train_rest_streams = [(sid, df) for sid, df in rest_streams if stream_subject(sid) != test_subject]

        print("  Training action RF...", flush=True)
        active_rf, action_rf, train_action_streams = train_action_rf(train_set_streams, train_rest_streams, cfg.imu_columns, args)
        locks, action_contexts = predict_action_contexts(active_rf, action_rf, train_action_streams, test_streams, cfg.imu_columns, args)
        lock_eval = action_lock_summary(locks, test_streams)
        print(f"  Action locks: rate={lock_eval['summary']['lock_rate']:.3f} acc={lock_eval['summary']['locked_accuracy']:.3f}", flush=True)

        print(f"  Training C/E CNN and {args.active_detector} active detector...", flush=True)
        duration_priors = build_duration_priors(train_set_streams, [5])
        model, mean, std, n_segments = train_raw6_model(train_set_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        if args.active_detector == "global":
            active_model, active_scaler = train_global_active_detector(train_set_streams, train_rest_streams, cfg, args)
            active_models = active_scalers = None
        else:
            active_models, active_scalers = train_active_detector(train_set_streams, cfg)
            active_model = active_scaler = None
        print(f"  Train active segments={n_segments}", flush=True)

        fold_raw = []
        fold_oracle = []
        fold_predicted = []
        fold_soft = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            gt_action = stream_action(stream_id)
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            if args.active_detector == "global":
                active_probs = predict_global_active(active_model, active_scaler, df, cfg)
            else:
                active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            base_reps = parse_reps(np.argmax(phase_probs, axis=1))

            raw = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, base_reps)
            oracle_reps = apply_top5_merge(base_reps, duration_priors, gt_action, args)
            oracle = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, oracle_reps)

            lock = locks.get(stream_id, {})
            pred_action = lock.get("locked_action") if lock.get("locked") else None
            predicted_reps = apply_top5_merge(base_reps, duration_priors, pred_action, args) if pred_action else base_reps
            predicted = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, predicted_reps)
            predicted["locked_action"] = pred_action
            predicted["action_locked"] = bool(lock.get("locked"))
            predicted["action_lock_correct"] = bool(lock.get("locked")) and pred_action == gt_action
            predicted["action_lock_time_s"] = lock.get("lock_time_s")

            soft_reps, soft_meta = apply_soft_top5_merge(base_reps, duration_priors, action_contexts.get(stream_id), args)
            soft = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, soft_reps)
            soft.update(soft_meta)

            raw_results.append(raw)
            oracle_results.append(oracle)
            predicted_results.append(predicted)
            soft_results.append(soft)
            fold_raw.append(raw)
            fold_oracle.append(oracle)
            fold_predicted.append(predicted)
            fold_soft.append(soft)

        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "action_lock": lock_eval["summary"],
            "raw": aggregate_rich(fold_raw),
            "oracle_top5": aggregate_rich(fold_oracle),
            "predicted_top5": aggregate_rich(fold_predicted),
            "soft_top5": aggregate_rich(fold_soft),
            "action_lock_rows": lock_eval["rows"],
        }
        folds.append(fold_summary)
        for name in ["raw", "oracle_top5", "predicted_top5", "soft_top5"]:
            agg = fold_summary[name]
            print(
                f"  {name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
                flush=True,
            )

    output = {
        "settings": {
            "model": "raw6_cnn_with_predicted_rf_action_top5_p5",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "action_lock_policy": args.lock_policy,
            "action_policy_values": policies[args.lock_policy],
            "active_detector": args.active_detector,
            "soft_policy_values": {
                "soft_active_threshold": args.soft_active_threshold,
                "soft_top5_mass_threshold": args.soft_top5_mass_threshold,
                "soft_action_confidence_threshold": args.soft_action_confidence_threshold,
                "soft_margin_threshold": args.soft_margin_threshold,
            },
            "top5_actions": ACTION_SETS["top5"],
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw_total": aggregate_rich(raw_results),
        "oracle_top5_total": aggregate_rich(oracle_results),
        "predicted_top5_total": aggregate_rich(predicted_results),
        "soft_top5_total": aggregate_rich(soft_results),
        "action_lock_total": aggregate_action_locks(folds),
        "predicted_top5_per_action": group_aggregate(predicted_results, lambda item: item["action"]),
        "soft_top5_per_action": group_aggregate(soft_results, lambda item: item["action"]),
        "folds": folds,
        "predicted_streams": predicted_results,
        "soft_streams": soft_results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nTOTAL", flush=True)
    for name in ["raw_total", "oracle_top5_total", "predicted_top5_total", "soft_top5_total"]:
        agg = output[name]
        print(
            f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
            flush=True,
        )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
