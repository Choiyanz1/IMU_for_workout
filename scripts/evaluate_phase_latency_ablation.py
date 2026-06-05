"""Ablate offline vs causal/bounded-latency phase decoding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import load_non_action_streams  # noqa: E402
from scripts.evaluate_predicted_action_top5_pipeline import (  # noqa: E402
    predict_global_active,
    train_global_active_detector,
)
from scripts.evaluate_realtime_soft_top5_pipeline import OnlineRepParser, trailing_window  # noqa: E402
from scripts.new_c_pipeline.compare_phase_models import PhaseCompareConfig, extract_active_segments  # noqa: E402
from scripts.new_c_pipeline.duration_merge_decoder_9fold import evaluate_with_reps  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import aggregate_rich, stream_subject, train_raw6_model  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
    smooth_ma,
    viterbi_decode,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def centered_ma(probs: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return probs.copy()
    out = probs.copy()
    n = len(probs)
    for c in range(probs.shape[1]):
        cumsum = np.concatenate([[0.0], np.cumsum(probs[:, c])])
        for i in range(n):
            start = max(0, i - radius)
            end = min(n, i + radius + 1)
            out[i, c] = (cumsum[end] - cumsum[start]) / max(1, end - start)
    return out


def fixed_lag_viterbi_decode(phase_probs: np.ndarray, penalty: float, lag_samples: int) -> np.ndarray:
    """Viterbi with bounded lookahead.

    Label i is finalized when frame i + lag_samples is available. This is a
    legal fixed-latency streaming approximation of full-sequence Viterbi.
    """
    n = len(phase_probs)
    if n == 0:
        return phase_probs.copy()
    lag = max(0, int(lag_samples))
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2), dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int64)
    dp[0] = log_probs[0]
    labels = np.full(n, -1, dtype=np.int64)

    for t in range(1, n):
        for s in range(2):
            stay = dp[t - 1, s]
            switch = dp[t - 1, 1 - s] - penalty
            if stay >= switch:
                dp[t, s] = log_probs[t, s] + stay
                back[t, s] = s
            else:
                dp[t, s] = log_probs[t, s] + switch
                back[t, s] = 1 - s

        finalize_idx = t - lag
        if finalize_idx >= 0:
            state = int(np.argmax(dp[t]))
            for k in range(t, finalize_idx, -1):
                state = int(back[k, state])
            labels[finalize_idx] = state

    final_state = int(np.argmax(dp[-1]))
    path = np.zeros(n, dtype=np.int64)
    path[-1] = final_state
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    labels[labels < 0] = path[labels < 0]

    result = np.zeros((n, 2), dtype=np.float32)
    result[labels == 0, 0] = 1.0
    result[labels == 1, 1] = 1.0
    return result


def causal_phase_probs(model, df, active_probs, cfg, mean, std, args, device):
    values = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(values)
    x_std = (values - mean) / std
    phase_probs = np.ones((n, 2), dtype=np.float32) * 0.5
    ends = sorted(set([1, n, *range(args.phase_step_samples, n + 1, args.phase_step_samples)]))
    phase_windows = []
    event_indices = []
    for end in ends:
        idx = end - 1
        if active_probs[idx] >= args.active_threshold:
            phase_windows.append(trailing_window(x_std, end, args.phase_window_samples))
            event_indices.append(idx)
    event_probs = []
    if phase_windows:
        batch = np.stack(phase_windows).astype(np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(batch), args.phase_batch_size):
                x_np = batch[start : start + args.phase_batch_size]
                x = torch.from_numpy(x_np).float().transpose(1, 2).to(device)
                probs = F.softmax(model(x), dim=1).cpu().numpy()[:, :, -1].astype(np.float32)
                event_probs.extend([p for p in probs])
    prev = -1
    event_cursor = 0
    active_events = set(event_indices)
    for end in ends:
        idx = end - 1
        start_fill = max(0, prev + 1)
        if idx in active_events:
            prob = event_probs[event_cursor]
            event_cursor += 1
        else:
            prob = np.asarray([0.5, 0.5], dtype=np.float32)
        phase_probs[start_fill : idx + 1] = prob
        prev = idx
    return phase_probs


def stateful_parse(labels: np.ndarray):
    parser = OnlineRepParser()
    reps = []
    for idx, label_idx in enumerate(labels):
        if int(label_idx) < 0:
            label = None
        else:
            label = "eccentric" if int(label_idx) == 0 else "concentric"
        reps.extend(parser.update(idx, label))
    reps.extend(parser.finish(len(labels)))
    return reps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline, causal, and bounded-latency phase decoding.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/phase_latency_ablation/summary_e5_h64.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    parser.add_argument("--phase-window-samples", type=int, default=300)
    parser.add_argument("--phase-step-samples", type=int, default=10)
    parser.add_argument("--phase-batch-size", type=int, default=256)
    parser.add_argument("--lookahead-samples", type=int, default=100)
    parser.add_argument("--fixed-lags-samples", default="50,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    all_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    subjects = sorted({stream_subject(sid) for sid, _ in set_streams})
    eval_subjects = subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else subjects

    fixed_lags = [int(x.strip()) for x in args.fixed_lags_samples.split(",") if x.strip()]
    variants = {
        "offline_predict_fast": [],
        "causal_raw_stateful": [],
        "causal_past_ma25_parse": [],
        "causal_full_viterbi_parse": [],
        "lookahead_center_ma_parse": [],
    }
    for lag in fixed_lags:
        variants[f"fixed_lag_viterbi_{lag}"] = []
    folds = []
    print(f"streams={len(set_streams)} subjects={subjects} device={device} step={args.phase_step_samples} lookahead={args.lookahead_samples}", flush=True)

    for fold_idx, test_subject in enumerate(eval_subjects, 1):
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject}", flush=True)
        train_set_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) == test_subject]
        train_rest_streams = [(sid, df) for sid, df in rest_streams if stream_subject(sid) != test_subject]
        active_model, active_scaler = train_global_active_detector(train_set_streams, train_rest_streams, cfg, args)
        model, mean, std, n_segments = train_raw6_model(train_set_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"  trained phase segments={n_segments}", flush=True)
        fold_results = {name: [] for name in variants}

        for stream_id, df in test_streams:
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_global_active(active_model, active_scaler, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=args.active_threshold, min_consecutive=3)

            offline_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            offline_reps = parse_reps(np.argmax(offline_probs, axis=1))
            result = evaluate_with_reps(stream_id, offline_probs, gt_reps, gt_phases, offline_reps)
            variants["offline_predict_fast"].append(result)
            fold_results["offline_predict_fast"].append(result)

            causal_probs = causal_phase_probs(model, df, active_probs, cfg, mean, std, args, device)
            causal_labels = np.argmax(causal_probs, axis=1)
            inactive = np.max(causal_probs, axis=1) <= 0.5001
            causal_labels_masked = causal_labels.copy()
            causal_labels_masked[inactive] = -1
            reps = stateful_parse(causal_labels_masked)
            result = evaluate_with_reps(stream_id, causal_probs, gt_reps, gt_phases, reps)
            variants["causal_raw_stateful"].append(result)
            fold_results["causal_raw_stateful"].append(result)

            past_ma = smooth_ma(causal_probs, 25)
            reps = parse_reps(np.argmax(past_ma, axis=1))
            result = evaluate_with_reps(stream_id, past_ma, gt_reps, gt_phases, reps)
            variants["causal_past_ma25_parse"].append(result)
            fold_results["causal_past_ma25_parse"].append(result)

            full_viterbi = viterbi_decode(smooth_ma(causal_probs, 25), 0.3)
            reps = parse_reps(np.argmax(full_viterbi, axis=1))
            result = evaluate_with_reps(stream_id, full_viterbi, gt_reps, gt_phases, reps)
            variants["causal_full_viterbi_parse"].append(result)
            fold_results["causal_full_viterbi_parse"].append(result)

            for lag in fixed_lags:
                fixed = fixed_lag_viterbi_decode(smooth_ma(causal_probs, 25), 0.3, lag)
                reps = parse_reps(np.argmax(fixed, axis=1))
                result = evaluate_with_reps(stream_id, fixed, gt_reps, gt_phases, reps)
                name = f"fixed_lag_viterbi_{lag}"
                variants[name].append(result)
                fold_results[name].append(result)

            lookahead = centered_ma(causal_probs, args.lookahead_samples)
            reps = parse_reps(np.argmax(lookahead, axis=1))
            result = evaluate_with_reps(stream_id, lookahead, gt_reps, gt_phases, reps)
            variants["lookahead_center_ma_parse"].append(result)
            fold_results["lookahead_center_ma_parse"].append(result)

        fold_summary = {name: aggregate_rich(rows) for name, rows in fold_results.items()}
        fold_summary["fold"] = fold_idx
        fold_summary["test_subject"] = test_subject
        folds.append(fold_summary)
        for name in variants:
            agg = fold_summary[name]
            print(f"  {name}: RepF1={agg['rep_f1']:.4f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f}", flush=True)

    output = {
        "settings": vars(args),
        "excluded_sessions": EXCLUDED_SESSIONS,
        "totals": {name: aggregate_rich(rows) for name, rows in variants.items()},
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nTOTAL", flush=True)
    for name, agg in output["totals"].items():
        print(f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
