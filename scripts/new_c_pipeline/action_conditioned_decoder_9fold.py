"""Action-conditioned decoder policy search for the raw6 C/E CNN.

The CNN is still global. Only decoder parameters are selected per action using
train-fold streams, then applied to the held-out subject.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    set_seed,
    should_exclude,
    smooth_ma,
    viterbi_decode,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def predict_model_probs(model, df, active_segments, imu_columns, mean, std):
    """Predict raw per-sample phase probabilities before MA/Viterbi decoding."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    x_std = (x - mean) / std
    n = len(x_std)
    phase_probs = np.ones((n, 2), dtype=np.float32) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end:
                continue
            seg_x = x_std[seg_start:seg_end]
            seg_len = len(seg_x)
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x, ((0, pad_len), (0, 0)), mode="edge")
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                probs = F.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150
                starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len:
                    starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    probs = F.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
                    gs = seg_start + start
                    phase_probs[gs:gs + 300, :] += probs.T
                    phase_counts[gs:gs + 300] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    return phase_probs


def decode_probs(raw_probs, policy):
    probs = smooth_ma(raw_probs, int(policy["ma_window"]))
    return viterbi_decode(probs, float(policy["viterbi_penalty"]))


def reps_for_policy(decoded_probs, policy, duration_priors, action, max_merge_gap_samples):
    labels = np.argmax(decoded_probs, axis=1)
    reps = parse_reps(labels, min_phase=int(policy["min_phase_samples"]), max_gap=int(policy["max_phase_gap_samples"]))
    merge_percentile = policy.get("merge_percentile")
    if merge_percentile is not None:
        threshold = threshold_for_action(duration_priors, action, int(merge_percentile))
        reps = merge_short_reps(reps, threshold, max_merge_gap_samples)
    return reps


def policy_name(policy):
    merge = "none" if policy.get("merge_percentile") is None else f"p{policy['merge_percentile']}"
    return f"ma{policy['ma_window']}_v{str(policy['viterbi_penalty']).replace('.', '')}_m{policy['min_phase_samples']}_{merge}"


def make_policy_grid(ma_windows, penalties, min_phases, merge_percentiles):
    grid = []
    merge_options = [None] + list(merge_percentiles)
    for ma, penalty, min_phase, merge in product(ma_windows, penalties, min_phases, merge_options):
        grid.append({
            "ma_window": int(ma),
            "viterbi_penalty": float(penalty),
            "min_phase_samples": int(min_phase),
            "max_phase_gap_samples": 3,
            "merge_percentile": None if merge is None else int(merge),
        })
    return grid


def cache_predictions(streams, cfg, active_models, active_scalers, model, mean, std):
    cached = []
    for stream_id, df in streams:
        if "phase" not in df.columns:
            continue
        gt_phases = df["phase"].to_numpy()
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        raw_probs = predict_model_probs(model, df, active_segments, cfg.imu_columns, mean, std)
        cached.append({
            "stream_id": stream_id,
            "action": stream_action(stream_id),
            "gt_phases": gt_phases,
            "gt_reps": truth_reps_from_labels(gt_phases, min_phase_samples=3),
            "raw_probs": raw_probs,
        })
    return cached


def select_policy_validation_streams(train_streams, per_action):
    if per_action <= 0:
        return train_streams
    by_action = defaultdict(list)
    for stream in train_streams:
        by_action[stream_action(stream[0])].append(stream)
    selected = []
    for action in sorted(by_action):
        selected.extend(sorted(by_action[action], key=lambda item: item[0])[:per_action])
    return selected


def eval_cached_item(item, policy, duration_priors, max_merge_gap_samples):
    decoded = decode_probs(item["raw_probs"], policy)
    reps = reps_for_policy(decoded, policy, duration_priors, item["action"], max_merge_gap_samples)
    return evaluate_with_reps(item["stream_id"], decoded, item["gt_reps"], item["gt_phases"], reps)


def choose_action_policies(
    train_cached,
    policies,
    duration_priors,
    max_merge_gap_samples,
    min_rep_f1_drop,
    min_mae_improvement,
    max_exact_drop,
):
    by_action = defaultdict(list)
    for item in train_cached:
        by_action[item["action"]].append(item)

    baseline_policy = {
        "ma_window": 25,
        "viterbi_penalty": 0.3,
        "min_phase_samples": 3,
        "max_phase_gap_samples": 3,
        "merge_percentile": None,
    }
    chosen = {}
    summaries = {}
    for action, items in sorted(by_action.items()):
        baseline_results = [eval_cached_item(item, baseline_policy, duration_priors, max_merge_gap_samples) for item in items]
        baseline_agg = aggregate_rich(baseline_results)
        candidates = []
        for policy in policies:
            results = [eval_cached_item(item, policy, duration_priors, max_merge_gap_samples) for item in items]
            agg = aggregate_rich(results)
            rep_f1_ok = agg["rep_f1"] >= baseline_agg["rep_f1"] - min_rep_f1_drop
            exact_ok = agg["exact_count_acc"] >= baseline_agg["exact_count_acc"] - max_exact_drop
            mae_ok = agg["mean_abs_count_error"] <= baseline_agg["mean_abs_count_error"] - min_mae_improvement
            allowed = rep_f1_ok and exact_ok and mae_ok
            # Count stability is primary, but only accept policies that pass train-fold gates.
            objective = (
                0 if allowed else 1,
                agg["mean_abs_count_error"],
                -agg["exact_count_acc"],
                agg["ce_ratio_mae"] if agg.get("ce_ratio_mae") is not None else 999.0,
                -agg["rep_f1"],
            )
            candidates.append((objective, policy, agg))
        candidates.sort(key=lambda x: x[0])
        best_obj, best_policy, best_agg = candidates[0]
        if best_obj[0] != 0:
            best_policy = baseline_policy
            best_agg = baseline_agg
            best_obj = (0, baseline_agg["mean_abs_count_error"], -baseline_agg["exact_count_acc"], baseline_agg.get("ce_ratio_mae") or 999.0, -baseline_agg["rep_f1"])
        chosen[action] = best_policy
        summaries[action] = {
            "baseline": baseline_agg,
            "selected_policy": best_policy,
            "selected_policy_name": policy_name(best_policy),
            "selected_train_summary": best_agg,
            "objective": list(best_obj),
        }
    return chosen, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--ma-windows", default="15,25,35")
    parser.add_argument("--penalties", default="0.3,0.5,0.7")
    parser.add_argument("--min-phases", default="3,5")
    parser.add_argument("--merge-percentiles", default="5,10")
    parser.add_argument("--max-merge-gap-samples", type=int, default=50)
    parser.add_argument("--min-rep-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-mae-improvement", type=float, default=0.05)
    parser.add_argument("--max-exact-drop", type=float, default=0.0)
    parser.add_argument("--policy-val-per-action", type=int, default=5)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/action_conditioned_decoder_9fold_fast.json")
    args = parser.parse_args()

    ma_windows = [int(x.strip()) for x in args.ma_windows.split(",") if x.strip()]
    penalties = [float(x.strip()) for x in args.penalties.split(",") if x.strip()]
    min_phases = [int(x.strip()) for x in args.min_phases.split(",") if x.strip()]
    merge_percentiles = [int(x.strip()) for x in args.merge_percentiles.split(",") if x.strip()]
    policies = make_policy_grid(ma_windows, penalties, min_phases, merge_percentiles)

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
    print(f"Policy grid size: {len(policies)}")
    print(f"Settings: hidden={args.hidden}, epochs={args.epochs}, device={device}")

    baseline_policy = {
        "ma_window": 25,
        "viterbi_penalty": 0.3,
        "min_phase_samples": 3,
        "max_phase_gap_samples": 3,
        "merge_percentile": None,
    }
    all_baseline = []
    all_selected = []
    folds = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        duration_priors = build_duration_priors(train_streams, merge_percentiles)

        print("Training raw6 CNN...")
        model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"Train active segments={n_segments}")
        print("Training active detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)

        policy_streams = select_policy_validation_streams(train_streams, args.policy_val_per_action)
        print(f"Caching policy-val/test raw phase probabilities... policy_val_streams={len(policy_streams)}")
        train_cached = cache_predictions(policy_streams, cfg, active_models, active_scalers, model, mean, std)
        test_cached = cache_predictions(test_streams, cfg, active_models, active_scalers, model, mean, std)

        print("Selecting per-action decoder policies on train fold...")
        chosen_policies, train_policy_summaries = choose_action_policies(
            train_cached,
            policies,
            duration_priors,
            args.max_merge_gap_samples,
            args.min_rep_f1_drop,
            args.min_mae_improvement,
            args.max_exact_drop,
        )
        for action, policy in sorted(chosen_policies.items()):
            print(f"  {action}: {policy_name(policy)}")

        baseline_results = [eval_cached_item(item, baseline_policy, duration_priors, args.max_merge_gap_samples) for item in test_cached]
        selected_results = []
        for item in test_cached:
            policy = chosen_policies.get(item["action"], baseline_policy)
            selected_results.append(eval_cached_item(item, policy, duration_priors, args.max_merge_gap_samples))

        baseline_agg = aggregate_rich(baseline_results)
        selected_agg = aggregate_rich(selected_results)
        all_baseline.extend(baseline_results)
        all_selected.extend(selected_results)
        folds.append({
            "fold": fold_idx,
            "test_subject": test_subject,
            "baseline": baseline_agg,
            "action_conditioned": selected_agg,
            "selected_policies": chosen_policies,
            "train_policy_summaries": train_policy_summaries,
        })
        print(
            f"BASE: RepF1={baseline_agg['rep_f1']:.4f} Exact={baseline_agg['exact_count_acc']:.3f} "
            f"MAE={baseline_agg['mean_abs_count_error']:.2f} CE={baseline_agg['ce_ratio_mae']:.3f}"
        )
        print(
            f"ACT : RepF1={selected_agg['rep_f1']:.4f} Exact={selected_agg['exact_count_acc']:.3f} "
            f"MAE={selected_agg['mean_abs_count_error']:.2f} CE={selected_agg['ce_ratio_mae']:.3f}"
        )

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "policy_grid": {
                "ma_windows": ma_windows,
                "viterbi_penalties": penalties,
                "min_phase_samples": min_phases,
                "merge_percentiles": merge_percentiles,
                "max_merge_gap_samples": args.max_merge_gap_samples,
                "min_rep_f1_drop": args.min_rep_f1_drop,
                "min_mae_improvement": args.min_mae_improvement,
                "max_exact_drop": args.max_exact_drop,
                "policy_val_per_action": args.policy_val_per_action,
            },
            "selection": "per-action train-fold objective: prefer low MAE while constraining Rep F1 drop",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "baseline_total": aggregate_rich(all_baseline),
        "action_conditioned_total": aggregate_rich(all_selected),
        "baseline_per_action": group_aggregate(all_baseline, lambda item: item["action"]),
        "action_conditioned_per_action": group_aggregate(all_selected, lambda item: item["action"]),
        "folds": folds,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    base = output["baseline_total"]
    act = output["action_conditioned_total"]
    print(f"BASE: RepF1={base['rep_f1']:.4f} Exact={base['exact_count_acc']:.3f} MAE={base['mean_abs_count_error']:.2f} CE={base['ce_ratio_mae']:.3f}")
    print(f"ACT : RepF1={act['rep_f1']:.4f} Exact={act['exact_count_acc']:.3f} MAE={act['mean_abs_count_error']:.2f} CE={act['ce_ratio_mae']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
