"""Dataset replay demo for the current top5_p5 streaming-style pipeline.

This is a visualization/demo script, not a new evaluation protocol. It trains the
current raw6 CNN on all non-held-out subjects, replays held-out streams, and
shows the display count that a user would see after the top5_p5 merge rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
from scripts.new_c_pipeline.plot_current_segmentation_examples import (  # noqa: E402
    draw_phase_bar,
    draw_phase_spans,
    draw_rep_boundaries,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
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


def find_streams(streams, subject, actions):
    selected = []
    for action in actions:
        candidates = [
            (sid, df)
            for sid, df in streams
            if stream_subject(sid) == subject and stream_action(sid) == action and "phase" in df.columns
        ]
        if candidates:
            selected.append(candidates[0])
    return selected


def count_series(reps, n_samples):
    values = np.zeros(n_samples, dtype=int)
    count = 0
    events = []
    for rep in sorted(reps, key=lambda item: item.end_idx):
        end = min(n_samples, int(rep.end_idx))
        values[end:] = count + 1
        count += 1
        events.append(rep)
    return values, events


def build_event_log(raw_reps, display_reps, gt_reps, sample_rate):
    raw_events = {int(rep.end_idx): i + 1 for i, rep in enumerate(sorted(raw_reps, key=lambda item: item.end_idx))}
    display_events = {int(rep.end_idx): i + 1 for i, rep in enumerate(sorted(display_reps, key=lambda item: item.end_idx))}
    gt_events = {int(rep.end_idx): i + 1 for i, rep in enumerate(sorted(gt_reps, key=lambda item: item.end_idx))}
    all_times = sorted(set(raw_events) | set(display_events) | set(gt_events))
    rows = []
    raw_count = 0
    display_count = 0
    gt_count = 0
    for idx in all_times:
        if idx in raw_events:
            raw_count = raw_events[idx]
        if idx in display_events:
            display_count = display_events[idx]
        if idx in gt_events:
            gt_count = gt_events[idx]
        rows.append({
            "time_sec": round(idx / sample_rate, 2),
            "display_count_top5_p5": int(display_count),
            "raw_count_before_merge": int(raw_count),
            "gt_count_for_demo_only": int(gt_count),
        })
    return rows


def plot_demo(stream_id, df, cfg, mean, std, active_probs, phase_probs, raw_reps, display_reps, gt_reps, gt_phases, metrics, output_dir, sample_rate):
    pred_labels = phase_probs.argmax(axis=1)
    pred_phase = np.array(["eccentric" if label == 0 else "concentric" for label in pred_labels])
    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    t = np.arange(len(df)) / sample_rate
    acc_mag = np.linalg.norm(x[:, :3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)
    raw_count, _ = count_series(raw_reps, len(df))
    display_count, _ = count_series(display_reps, len(df))
    gt_count, _ = count_series(gt_reps, len(df))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 0.8, 1.0, 1.2]},
    )
    draw_phase_spans(axes[0], pred_phase, sample_rate, alpha=0.18)
    axes[0].plot(t, acc_mag, color="#1d4ed8", linewidth=1.0, label="acc magnitude")
    axes[0].plot(t, gyro_mag, color="#dc2626", linewidth=0.9, alpha=0.8, label="gyro magnitude")
    draw_rep_boundaries(axes[0], gt_reps, sample_rate, "#111827", "--", "GT rep boundary")
    draw_rep_boundaries(axes[0], raw_reps, sample_rate, "#ef4444", ":", "Raw predicted reps")
    draw_rep_boundaries(axes[0], display_reps, sample_rate, "#16a34a", "-", "Displayed top5_p5 reps")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_ylim(0, 2.3)
    draw_phase_bar(axes[1], gt_phases, sample_rate, 1.25, 0.55, "GT")
    draw_phase_bar(axes[1], pred_phase, sample_rate, 0.35, 0.55, "Pred")
    axes[1].set_yticks([])
    axes[1].set_title("C/E phase cuts: blue=concentric, yellow=eccentric", fontsize=10)

    axes[2].plot(t, active_probs, color="#7c3aed", linewidth=1.0, label="active probability")
    axes[2].axhline(0.5, color="#6b7280", linestyle="--", linewidth=0.8)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Active")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="upper right", frameon=False, fontsize=8)

    axes[3].step(t, raw_count, where="post", color="#ef4444", linewidth=1.4, label="raw count before merge")
    axes[3].step(t, display_count, where="post", color="#16a34a", linewidth=1.8, label="display count top5_p5")
    axes[3].step(t, gt_count, where="post", color="#111827", linestyle="--", linewidth=1.1, label="GT count (demo only)")
    axes[3].set_ylabel("Count")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, axis="y", alpha=0.25)
    axes[3].legend(loc="upper left", frameon=False, fontsize=8)

    title = (
        f"Streaming-style top5_p5 replay | {stream_id}\n"
        f"Display reps={metrics['pred_count']} GT={metrics['gt_count']} "
        f"RepF1={metrics['f1']:.3f} C/E MAE={metrics.get('ce_ratio_mae', 0):.3f} "
        f"Count error={metrics['count_error']}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    safe = stream_id.replace("/", "__")
    path = output_dir / f"streaming_top5_p5_{safe}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def replay_stream(stream_id, df, cfg, active_models, active_scalers, model, mean, std, duration_priors, output_dir, sample_rate):
    action = stream_action(stream_id)
    gt_phases = df["phase"].to_numpy()
    gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
    active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
    active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
    phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
    raw_labels = phase_probs.argmax(axis=1)
    raw_reps = parse_reps(raw_labels)
    display_reps = raw_reps
    merge_applied = False
    threshold = None
    if action in ACTION_SETS["top5"]:
        threshold = threshold_for_action(duration_priors, action, 5)
        display_reps = merge_short_reps(raw_reps, threshold, max_gap_samples=50)
        merge_applied = True

    metrics = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, display_reps)
    event_log = build_event_log(raw_reps, display_reps, gt_reps, sample_rate)
    figure_path = plot_demo(
        stream_id,
        df,
        cfg,
        mean,
        std,
        active_probs,
        phase_probs,
        raw_reps,
        display_reps,
        gt_reps,
        gt_phases,
        metrics,
        output_dir,
        sample_rate,
    )
    return {
        "stream_id": stream_id,
        "action": action,
        "merge_applied": merge_applied,
        "top5_p5_min_rep_duration_samples": threshold,
        "raw_pred_count_before_merge": len(raw_reps),
        "display_count_top5_p5": len(display_reps),
        "gt_count_demo_only": len(gt_reps),
        "metrics": metrics,
        "figure": str(figure_path),
        "event_log": event_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="yushuan")
    parser.add_argument("--actions", default="db_rdl,db_weighted_crunch,db_biceps_curl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--output-dir", default="artifacts/figures/streaming_top5_p5_demo")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != args.subject]
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    selected_streams = find_streams(streams, args.subject, actions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Held-out subject: {args.subject}")
    print(f"Selected streams: {[sid for sid, _ in selected_streams]}")
    print(f"Training current raw6 CNN on {len(train_streams)} train streams, device={device}")
    model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
    print(f"Train active segments={n_segments}")
    print("Training active detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    duration_priors = build_duration_priors(train_streams, [5])

    demos = []
    for stream_id, df in selected_streams:
        demo = replay_stream(
            stream_id,
            df,
            cfg,
            active_models,
            active_scalers,
            model,
            mean,
            std,
            duration_priors,
            output_dir,
            args.sample_rate,
        )
        demos.append(demo)
        print(
            f"{stream_id}: raw={demo['raw_pred_count_before_merge']} "
            f"display={demo['display_count_top5_p5']} gt={demo['gt_count_demo_only']} "
            f"CE={demo['metrics'].get('ce_ratio_mae', 0):.3f} figure={demo['figure']}"
        )

    out_json = output_dir / "streaming_top5_p5_demo.json"
    out_json.write_text(json.dumps({"demos": demos}, indent=2), encoding="utf-8")

    out_txt = output_dir / "streaming_top5_p5_event_log.txt"
    lines = []
    for demo in demos:
        lines.append(f"# {demo['stream_id']}")
        lines.append(f"raw_count={demo['raw_pred_count_before_merge']} display_count={demo['display_count_top5_p5']} gt_demo_only={demo['gt_count_demo_only']}")
        lines.append("time_sec | display_top5_p5 | raw_before_merge | gt_demo_only")
        for row in demo["event_log"]:
            lines.append(
                f"{row['time_sec']:>7.2f} | {row['display_count_top5_p5']:>16d} | "
                f"{row['raw_count_before_merge']:>16d} | {row['gt_count_for_demo_only']:>12d}"
            )
        lines.append("")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved JSON: {out_json}")
    print(f"Saved event log: {out_txt}")


if __name__ == "__main__":
    main()
