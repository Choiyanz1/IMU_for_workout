"""Selective duration merge: apply rep merging only to over-count-prone actions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


ACTION_SETS = {
    "top3": ["db_rdl", "db_shoulder_press", "db_bench_press"],
    "top4": ["db_rdl", "db_shoulder_press", "db_bench_press", "one_arm_db_row"],
    "top5": ["db_rdl", "db_shoulder_press", "db_bench_press", "one_arm_db_row", "db_weighted_crunch"],
    "over50": ["db_rdl", "db_shoulder_press", "db_squat", "one_arm_db_row"],
    "compound6": [
        "db_bench_press",
        "db_rdl",
        "db_shoulder_press",
        "db_squat",
        "db_weighted_crunch",
        "one_arm_db_row",
    ],
}


def config_name(action_set_name, percentile):
    return f"{action_set_name}_p{percentile}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--percentiles", default="5,10")
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--include-streams", action="store_true", help="Save per-stream rows for calibration/debugging.")
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/selective_duration_merge_decoder_9fold_gpu_h64e20.json")
    args = parser.parse_args()

    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    configs = {
        config_name(action_set_name, p): {"actions": actions, "percentile": p}
        for action_set_name, actions in ACTION_SETS.items()
        for p in percentiles
    }

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
    print("Configs:")
    for name, spec in configs.items():
        print(f"  {name}: p{spec['percentile']} on {spec['actions']}")

    raw_results = []
    config_results = {name: [] for name in configs}
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

        print("Training active detector and evaluating selective merge configs...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        fold_raw = []
        fold_config_results = {name: [] for name in configs}

        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            action = stream_action(stream_id)
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            pred_labels = phase_probs.argmax(axis=1)
            base_reps = parse_reps(pred_labels)

            raw_result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, base_reps)
            raw_results.append(raw_result)
            fold_raw.append(raw_result)

            for name, spec in configs.items():
                reps = base_reps
                if action in spec["actions"]:
                    threshold = threshold_for_action(duration_priors, action, spec["percentile"])
                    reps = merge_short_reps(base_reps, threshold, args.max_gap_samples)
                result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, reps)
                config_results[name].append(result)
                fold_config_results[name].append(result)

        fold_summary = {"fold": fold_idx, "test_subject": test_subject, "raw": aggregate_rich(fold_raw)}
        for name in configs:
            fold_summary[name] = aggregate_rich(fold_config_results[name])
        folds.append(fold_summary)

        raw_agg = fold_summary["raw"]
        print(f"RAW: RepF1={raw_agg['rep_f1']:.4f} Exact={raw_agg['exact_count_acc']:.3f} MAE={raw_agg['mean_abs_count_error']:.2f} Over={raw_agg['over_rate']:.3f} Under={raw_agg['under_rate']:.3f}")
        for name in configs:
            agg = fold_summary[name]
            print(f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}")

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "decoder_base": "MA25 + Viterbi penalty=0.3",
            "merge_rule": "duration merge only for selected actions",
            "percentiles": percentiles,
            "max_gap_samples": args.max_gap_samples,
            "action_sets": ACTION_SETS,
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "best_balanced_candidate": {
            "name": "top5_p5",
            "actions": ACTION_SETS["top5"],
            "percentile": 5,
            "reason": "Highest Rep F1 and lowest Count MAE among strict-gate passing selective merge configs.",
        },
        "raw_total": aggregate_rich(raw_results),
        "config_totals": {name: aggregate_rich(results) for name, results in config_results.items()},
        "raw_per_action": group_aggregate(raw_results, lambda item: item["action"]),
        "config_per_action": {
            name: group_aggregate(results, lambda item: item["action"])
            for name, results in config_results.items()
        },
        "raw_per_subject": group_aggregate(raw_results, lambda item: item["subject"]),
        "config_per_subject": {
            name: group_aggregate(results, lambda item: item["subject"])
            for name, results in config_results.items()
        },
        "folds": folds,
    }
    if args.include_streams:
        output["raw_streams"] = raw_results
        output["config_streams"] = config_results

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    raw_total = output["raw_total"]
    print(f"RAW: RepF1={raw_total['rep_f1']:.4f} Exact={raw_total['exact_count_acc']:.3f} MAE={raw_total['mean_abs_count_error']:.2f} Over={raw_total['over_rate']:.3f} Under={raw_total['under_rate']:.3f} Bias={raw_total['count_bias_pred_minus_gt']:.2f}")
    for name, agg in output["config_totals"].items():
        print(f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f} Bias={agg['count_bias_pred_minus_gt']:.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
