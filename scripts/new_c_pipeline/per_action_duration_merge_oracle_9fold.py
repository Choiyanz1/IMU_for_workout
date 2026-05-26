"""Per-action duration merge oracle grid for raw6 1D Causal CNN.

This is an exploratory upper-bound decoder search. It selects the best merge
percentile per action using held-out results, so it is not a deployable protocol
until confirmed with nested tuning.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def option_sort_key(metrics):
    return (
        metrics.get("mean_abs_count_error", 999.0),
        -metrics.get("exact_count_acc", 0.0),
        -metrics.get("rep_f1", 0.0),
        abs(metrics.get("count_bias_pred_minus_gt", 999.0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--percentiles", default="5,10,15,20,25,30")
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/per_action_duration_merge_oracle_9fold_gpu_h64e20.json")
    args = parser.parse_args()

    percentiles = [int(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    options = ["none"] + [f"p{p}" for p in percentiles]

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
    print(f"Settings: raw6 CNN, hidden={args.hidden}, epochs={args.epochs}, options={options}, max_gap={args.max_gap_samples}, device={device}")
    print("NOTE: per-action selected result is an oracle upper bound, not a deployment-valid nested tuning result.")

    option_results = {option: [] for option in options}
    action_option_results = defaultdict(lambda: {option: [] for option in options})
    stream_option_results = {}
    stream_actions = {}
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

        print("Training active detector and evaluating per-action option grid...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        fold_option_results = {option: [] for option in options}

        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            action = stream_action(stream_id)
            stream_actions[stream_id] = action
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
            phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
            pred_labels = phase_probs.argmax(axis=1)
            base_reps = parse_reps(pred_labels)

            for option in options:
                reps = base_reps
                if option != "none":
                    percentile = int(option[1:])
                    threshold = threshold_for_action(duration_priors, action, percentile)
                    reps = merge_short_reps(base_reps, threshold, args.max_gap_samples)
                result = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, reps)
                result["merge_option"] = option
                option_results[option].append(result)
                action_option_results[action][option].append(result)
                stream_option_results[(stream_id, option)] = result
                fold_option_results[option].append(result)

        fold_summary = {"fold": fold_idx, "test_subject": test_subject}
        for option in options:
            fold_summary[option] = aggregate_rich(fold_option_results[option])
        folds.append(fold_summary)
        for option in options:
            agg = fold_summary[option]
            print(f"{option:>4}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Bias={agg['count_bias_pred_minus_gt']:.2f}")

    option_totals = {option: aggregate_rich(results) for option, results in option_results.items()}
    per_action_option_totals = {
        action: {option: aggregate_rich(results) for option, results in option_map.items()}
        for action, option_map in sorted(action_option_results.items())
    }

    selected_by_action = {}
    for action, option_map in per_action_option_totals.items():
        selected_by_action[action] = min(option_map, key=lambda option: option_sort_key(option_map[option]))

    oracle_results = []
    for stream_id, action in stream_actions.items():
        option = selected_by_action[action]
        oracle_results.append(stream_option_results[(stream_id, option)])
    oracle_total = aggregate_rich(oracle_results)

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "decoder_base": "MA25 + Viterbi penalty=0.3",
            "merge_rule": "per-action oracle selection over none and duration percentiles",
            "percentiles": percentiles,
            "max_gap_samples": args.max_gap_samples,
            "warning": "oracle uses held-out labels to select per-action options; use only as upper bound",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "option_totals": option_totals,
        "per_action_option_totals": per_action_option_totals,
        "selected_by_action": selected_by_action,
        "oracle_total": oracle_total,
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL OPTIONS")
    print(f"{'=' * 72}")
    for option, agg in option_totals.items():
        print(f"{option:>4}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} Bias={agg['count_bias_pred_minus_gt']:.2f} Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}")
    print("\nSelected by action:")
    for action, option in selected_by_action.items():
        metrics = per_action_option_totals[action][option]
        print(f"  {action}: {option} MAE={metrics['mean_abs_count_error']:.3f} Exact={metrics['exact_count_acc']:.3f} RepF1={metrics['rep_f1']:.3f}")
    print("\nORACLE PER-ACTION TOTAL:")
    print(f"RepF1={oracle_total['rep_f1']:.4f} Exact={oracle_total['exact_count_acc']:.3f} Within1={oracle_total['within_1_count_acc']:.3f} MAE={oracle_total['mean_abs_count_error']:.2f} Bias={oracle_total['count_bias_pred_minus_gt']:.2f} Over={oracle_total['over_rate']:.3f} Under={oracle_total['under_rate']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
