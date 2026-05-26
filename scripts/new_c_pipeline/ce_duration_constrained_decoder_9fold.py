"""C/E duration-constrained decoder for raw6 CNN phase predictions.

This decoder-only experiment keeps the trained raw 6-axis causal CNN and active
detector unchanged. It suppresses too-short concentric/eccentric fragments using
train-fold per-action, per-phase duration percentiles, then re-parses reps.
"""
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

from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
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


PHASE_LABELS = {0: "eccentric", 1: "concentric"}
PHASE_TO_INDEX = {v: k for k, v in PHASE_LABELS.items()}


def one_hot_phase(labels):
    probs = np.zeros((len(labels), 2), dtype=float)
    probs[np.arange(len(labels)), labels.astype(int)] = 1.0
    return probs


def phase_run_lengths(phase_arr, target):
    lengths = []
    in_run = False
    start = 0
    for i, phase in enumerate(phase_arr):
        if str(phase) == target and not in_run:
            start = i
            in_run = True
        elif str(phase) != target and in_run:
            lengths.append(i - start)
            in_run = False
    if in_run:
        lengths.append(len(phase_arr) - start)
    return lengths


def build_phase_duration_priors(train_streams, percentiles):
    by_action_phase = defaultdict(list)
    global_by_phase = defaultdict(list)
    for stream_id, df in train_streams:
        if "phase" not in df.columns:
            continue
        action = stream_action(stream_id)
        phase_arr = df["phase"].to_numpy()
        for phase in ["concentric", "eccentric"]:
            lengths = phase_run_lengths(phase_arr, phase)
            by_action_phase[(action, phase)].extend(lengths)
            global_by_phase[phase].extend(lengths)

    priors = {}
    for action in sorted({action for action, _ in by_action_phase}):
        priors[action] = {}
        for phase in ["concentric", "eccentric"]:
            values = by_action_phase.get((action, phase), []) or global_by_phase.get(phase, [])
            priors[action][phase] = {
                str(p): float(np.percentile(values, p)) if values else 0.0
                for p in percentiles
            }
    priors["__global__"] = {
        phase: {
            str(p): float(np.percentile(global_by_phase.get(phase, []), p)) if global_by_phase.get(phase) else 0.0
            for p in percentiles
        }
        for phase in ["concentric", "eccentric"]
    }
    return priors


def phase_threshold(priors, action, phase, percentile):
    key = str(percentile)
    if action in priors and phase in priors[action] and key in priors[action][phase]:
        return float(priors[action][phase][key])
    return float(priors.get("__global__", {}).get(phase, {}).get(key, 0.0))


def label_runs(labels):
    if len(labels) == 0:
        return []
    runs = []
    start = 0
    cur = int(labels[0])
    for i in range(1, len(labels)):
        label = int(labels[i])
        if label != cur:
            runs.append({"label": cur, "start": start, "end": i})
            start = i
            cur = label
    runs.append({"label": cur, "start": start, "end": len(labels)})
    return runs


def collapse_runs(runs):
    collapsed = []
    cursor = 0
    for run in runs:
        length = int(run["end"] - run["start"])
        if length <= 0:
            continue
        if collapsed and collapsed[-1]["label"] == run["label"]:
            collapsed[-1]["end"] += length
        else:
            collapsed.append({"label": int(run["label"]), "start": cursor, "end": cursor + length})
        cursor = collapsed[-1]["end"]
    return collapsed


def runs_to_labels(runs, n):
    labels = np.zeros(n, dtype=np.int64)
    for run in runs:
        labels[int(run["start"]):int(run["end"])] = int(run["label"])
    return labels


def suppress_short_phase_fragments(labels, priors, action, percentile, max_iters=5):
    n = len(labels)
    runs = label_runs(labels)
    if len(runs) <= 1:
        return labels.copy()

    for _ in range(max_iters):
        changed = False
        i = 0
        while i < len(runs):
            run = runs[i]
            phase = PHASE_LABELS[int(run["label"])]
            threshold = int(round(phase_threshold(priors, action, phase, percentile)))
            length = int(run["end"] - run["start"])
            if threshold <= 0 or length >= threshold or len(runs) <= 1:
                i += 1
                continue

            if i == 0:
                target = 1
            elif i == len(runs) - 1:
                target = i - 1
            else:
                prev_len = int(runs[i - 1]["end"] - runs[i - 1]["start"])
                next_len = int(runs[i + 1]["end"] - runs[i + 1]["start"])
                target = i - 1 if prev_len >= next_len else i + 1
            runs[i]["label"] = int(runs[target]["label"])
            runs = collapse_runs(runs)
            changed = True
            break
        if not changed:
            break
    return runs_to_labels(runs, n)


def evaluate_labels(stream_id, labels, gt_reps, gt_phases):
    phase_probs = one_hot_phase(labels)
    reps = parse_reps(labels)
    return evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, reps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--percentiles", default="1,5,10,15")
    parser.add_argument("--top5-percentile", type=int, default=5)
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/ce_duration_constrained_decoder_9fold_gpu_h64e20.json")
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
    print(f"Settings: raw6 CNN hidden={args.hidden}, epochs={args.epochs}, CE duration percentiles={percentiles}, device={device}")

    raw_results = []
    top5_results = []
    ce_results = {f"ce_p{p}": [] for p in percentiles}
    folds = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        rep_duration_priors = build_duration_priors(train_streams, [args.top5_percentile])
        phase_duration_priors = build_phase_duration_priors(train_streams, percentiles)

        print("Training raw 6-axis CNN...")
        model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"Train active segments={n_segments}")

        print("Training active detector and evaluating C/E duration constraints...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        fold_raw = []
        fold_top5 = []
        fold_ce = {f"ce_p{p}": [] for p in percentiles}

        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            action = stream_action(stream_id)
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            raw_labels = phase_probs.argmax(axis=1)
            raw_reps = parse_reps(raw_labels)

            raw_result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, raw_reps)
            raw_results.append(raw_result)
            fold_raw.append(raw_result)

            top5_reps = raw_reps
            if action in ACTION_SETS["top5"]:
                threshold = threshold_for_action(rep_duration_priors, action, args.top5_percentile)
                top5_reps = merge_short_reps(raw_reps, threshold, args.max_gap_samples)
            top5_result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, top5_reps)
            top5_results.append(top5_result)
            fold_top5.append(top5_result)

            for p in percentiles:
                name = f"ce_p{p}"
                constrained_labels = suppress_short_phase_fragments(raw_labels, phase_duration_priors, action, p)
                result = evaluate_labels(stream_id, constrained_labels, gt_reps, gt_phases)
                ce_results[name].append(result)
                fold_ce[name].append(result)

        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "raw": aggregate_rich(fold_raw),
            "top5_p5": aggregate_rich(fold_top5),
        }
        for name in fold_ce:
            fold_summary[name] = aggregate_rich(fold_ce[name])
        folds.append(fold_summary)

        print_summary = {"raw": fold_summary["raw"], "top5_p5": fold_summary["top5_p5"]}
        print_summary.update({name: fold_summary[name] for name in fold_ce})
        for name, agg in print_summary.items():
            print(
                f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
                f"MAE={agg['mean_abs_count_error']:.2f} CE={agg['ce_ratio_mae']:.3f} "
                f"Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}"
            )

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "decoder_base": "MA25 + Viterbi penalty=0.3",
            "experiment": "suppress predicted C/E fragments shorter than train-fold per-action per-phase duration percentile",
            "ce_duration_percentiles": percentiles,
            "top5_reference": {
                "actions": ACTION_SETS["top5"],
                "rep_duration_percentile": args.top5_percentile,
                "max_gap_samples": args.max_gap_samples,
            },
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw_total": aggregate_rich(raw_results),
        "top5_p5_total": aggregate_rich(top5_results),
        "ce_totals": {name: aggregate_rich(results) for name, results in ce_results.items()},
        "raw_per_action": group_aggregate(raw_results, lambda item: item["action"]),
        "top5_p5_per_action": group_aggregate(top5_results, lambda item: item["action"]),
        "ce_per_action": {
            name: group_aggregate(results, lambda item: item["action"])
            for name, results in ce_results.items()
        },
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    totals = {"raw": output["raw_total"], "top5_p5": output["top5_p5_total"]}
    totals.update(output["ce_totals"])
    for name, agg in totals.items():
        print(
            f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
            f"MAE={agg['mean_abs_count_error']:.3f} CE={agg['ce_ratio_mae']:.3f} "
            f"Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
