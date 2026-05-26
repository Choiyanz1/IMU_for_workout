"""Decoder-only experiment: merge too-short predicted reps using train-fold duration priors."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import RepDetection  # noqa: E402
from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    evaluate_phase,
    evaluate_reps,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from scripts.new_c_pipeline.master_eval import (  # noqa: E402
    compute_ce_ratio_metrics,
    compute_rep_ce_ratios,
    evaluate_phase_segments,
    extract_phase_segments,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    aggregate_rich,
    stream_action,
    stream_subject,
    train_raw6_model,
)
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def build_duration_priors(train_streams, percentiles):
    by_action = defaultdict(list)
    global_durations = []
    for stream_id, df in train_streams:
        if "phase" not in df.columns:
            continue
        action = stream_action(stream_id)
        reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=3)
        durations = [int(rep.end_idx) - int(rep.start_idx) for rep in reps]
        by_action[action].extend(durations)
        global_durations.extend(durations)

    global_thresholds = {
        str(p): float(np.percentile(global_durations, p)) if global_durations else 0.0
        for p in percentiles
    }
    priors = {}
    for action, durations in by_action.items():
        priors[action] = {
            str(p): float(np.percentile(durations, p)) if durations else global_thresholds[str(p)]
            for p in percentiles
        }
    priors["__global__"] = global_thresholds
    return priors


def threshold_for_action(priors, action, percentile):
    key = str(percentile)
    if action in priors and key in priors[action]:
        return float(priors[action][key])
    return float(priors.get("__global__", {}).get(key, 0.0))


def clone_rep(start_idx, transition_idx, end_idx, source_rep):
    return RepDetection(
        start_idx=int(start_idx),
        transition_idx=int(transition_idx),
        end_idx=int(end_idx),
        micro_source=source_rep.micro_source,
        micro_confidence=float(source_rep.micro_confidence),
        pred_action_type=source_rep.pred_action_type,
        action_confidence=source_rep.action_confidence,
    )


def merge_short_reps(reps, min_duration_samples, max_gap_samples):
    if not reps:
        return []
    min_duration_samples = int(round(min_duration_samples))
    max_gap_samples = int(max_gap_samples)
    merged = []
    i = 0
    while i < len(reps):
        cur = reps[i]
        cur_duration = int(cur.end_idx) - int(cur.start_idx)
        if cur_duration < min_duration_samples and i + 1 < len(reps):
            nxt = reps[i + 1]
            gap = int(nxt.start_idx) - int(cur.end_idx)
            if gap <= max_gap_samples:
                merged.append(clone_rep(cur.start_idx, cur.transition_idx, nxt.end_idx, cur))
                i += 2
                continue
        if cur_duration < min_duration_samples and merged:
            prev = merged[-1]
            gap = int(cur.start_idx) - int(prev.end_idx)
            if gap <= max_gap_samples:
                merged[-1] = clone_rep(prev.start_idx, prev.transition_idx, cur.end_idx, prev)
                i += 1
                continue
        merged.append(cur)
        i += 1
    return merged


def evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, pred_reps):
    pred_labels = np.argmax(phase_probs, axis=1)
    rep_m = evaluate_reps(pred_reps, gt_reps)
    phase_m = evaluate_phase(phase_probs, gt_phases)
    pred_phase_arr = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])

    gt_c_segs = extract_phase_segments(gt_phases, "concentric")
    gt_e_segs = extract_phase_segments(gt_phases, "eccentric")
    pred_c_segs = extract_phase_segments(pred_phase_arr, "concentric")
    pred_e_segs = extract_phase_segments(pred_phase_arr, "eccentric")
    c_seg = evaluate_phase_segments(pred_c_segs, gt_c_segs)
    e_seg = evaluate_phase_segments(pred_e_segs, gt_e_segs)
    pred_ratios = compute_rep_ce_ratios(pred_reps, pred_phase_arr)
    gt_ratios = compute_rep_ce_ratios(gt_reps, gt_phases)
    ce_metrics = compute_ce_ratio_metrics(pred_ratios, gt_ratios)

    return {
        "stream_id": stream_id,
        "subject": stream_subject(stream_id),
        "action": stream_action(stream_id),
        "pred_count": rep_m["pred_count"],
        "gt_count": rep_m["gt_count"],
        "count_error": abs(rep_m["pred_count"] - rep_m["gt_count"]),
        **{k: v for k, v in rep_m.items() if k not in ["pred_count", "gt_count"]},
        **phase_m,
        "concentric_seg_f1": c_seg["f1"],
        "eccentric_seg_f1": e_seg["f1"],
        **ce_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--percentiles", default="5,10,15,20,25")
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/duration_merge_decoder_9fold_gpu_h64e20.json")
    args = parser.parse_args()

    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})

    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Remaining streams: {len(streams)}")
    print(f"Subjects: {subjects}")
    print(f"Settings: raw6 CNN, hidden={args.hidden}, epochs={args.epochs}, percentiles={percentiles}, max_gap={args.max_gap_samples}, device={device}")

    raw_results = []
    merged_results = {p: [] for p in percentiles}
    folds = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        duration_priors = build_duration_priors(train_streams, percentiles)

        print("Training raw 6-axis CNN...")
        model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"Train active segments={n_segments}")

        print("Training active detector and evaluating merge configs...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        fold_raw = []
        fold_merged = {p: [] for p in percentiles}

        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            action = stream_action(stream_id)
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            pred_labels = np.argmax(phase_probs, axis=1)
            base_reps = parse_reps(pred_labels)

            raw_result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, base_reps)
            raw_results.append(raw_result)
            fold_raw.append(raw_result)

            for p in percentiles:
                threshold = threshold_for_action(duration_priors, action, p)
                merged_reps = merge_short_reps(base_reps, threshold, args.max_gap_samples)
                merged_result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, merged_reps)
                merged_result["duration_threshold_samples"] = threshold
                merged_results[p].append(merged_result)
                fold_merged[p].append(merged_result)

        fold_summary = {"fold": fold_idx, "test_subject": test_subject, "raw": aggregate_rich(fold_raw)}
        for p in percentiles:
            fold_summary[f"merge_p{p}"] = aggregate_rich(fold_merged[p])
        folds.append(fold_summary)

        raw_agg = fold_summary["raw"]
        print(f"RAW: RepF1={raw_agg['rep_f1']:.4f} Exact={raw_agg['exact_count_acc']:.3f} MAE={raw_agg['mean_abs_count_error']:.2f} Over={raw_agg['over_rate']:.3f}")
        for p in percentiles:
            agg = fold_summary[f"merge_p{p}"]
            print(f"p{p:02d}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Over={agg['over_rate']:.3f}")

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "decoder_base": "MA25 + Viterbi penalty=0.3",
            "merge_rule": "merge predicted reps shorter than train-fold per-action GT duration percentile",
            "percentiles": percentiles,
            "max_gap_samples": args.max_gap_samples,
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw_total": aggregate_rich(raw_results),
        "merge_totals": {f"p{p}": aggregate_rich(results) for p, results in merged_results.items()},
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    raw_total = output["raw_total"]
    print(f"RAW: RepF1={raw_total['rep_f1']:.4f} Exact={raw_total['exact_count_acc']:.3f} MAE={raw_total['mean_abs_count_error']:.2f} Over={raw_total['over_rate']:.3f} Bias={raw_total['count_bias_pred_minus_gt']:.2f}")
    for p in percentiles:
        agg = output["merge_totals"][f"p{p}"]
        print(f"p{p:02d}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Over={agg['over_rate']:.3f} Bias={agg['count_bias_pred_minus_gt']:.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
