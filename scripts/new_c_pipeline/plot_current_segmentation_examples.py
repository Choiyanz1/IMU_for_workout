"""Plot current CNN phase/rep segmentation examples with concentric/eccentric cuts."""
from __future__ import annotations

import argparse
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
    merge_short_reps,
    threshold_for_action,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    stream_action,
    stream_subject,
    train_raw6_model,
)
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


SELECTIVE_ACTIONS = {
    "db_rdl",
    "db_shoulder_press",
    "db_bench_press",
    "one_arm_db_row",
    "db_weighted_crunch",
}


def phase_runs(labels, target):
    runs = []
    in_run = False
    start = 0
    for i, label in enumerate(labels):
        if str(label) == target and not in_run:
            start = i
            in_run = True
        elif str(label) != target and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(labels)))
    return runs


def draw_phase_spans(ax, labels, sample_rate, alpha=0.35):
    for start, end in phase_runs(labels, "concentric"):
        ax.axvspan(start / sample_rate, end / sample_rate, color="#60a5fa", alpha=alpha, linewidth=0)
    for start, end in phase_runs(labels, "eccentric"):
        ax.axvspan(start / sample_rate, end / sample_rate, color="#fbbf24", alpha=alpha, linewidth=0)


def draw_phase_bar(ax, labels, sample_rate, y, height, label):
    for start, end in phase_runs(labels, "concentric"):
        ax.broken_barh([(start / sample_rate, (end - start) / sample_rate)], (y, height), facecolors="#2563eb")
    for start, end in phase_runs(labels, "eccentric"):
        ax.broken_barh([(start / sample_rate, (end - start) / sample_rate)], (y, height), facecolors="#f59e0b")
    ax.text(-0.02, y + height / 2, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10)


def draw_rep_boundaries(ax, reps, sample_rate, color, linestyle, label):
    first = True
    for rep in reps:
        start = rep.start_idx / sample_rate
        trans = rep.transition_idx / sample_rate
        end = rep.end_idx / sample_rate
        ax.axvline(start, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9, label=label if first else None)
        ax.axvline(trans, color=color, linestyle=":", linewidth=0.9, alpha=0.8)
        ax.axvline(end, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.9)
        first = False


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


def plot_stream(stream_id, df, cfg, active_models, active_scalers, model, mean, std, duration_priors, output_dir, merge_mode, sample_rate=100.0):
    action = stream_action(stream_id)
    gt_phases = df["phase"].to_numpy()
    gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)

    active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
    active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
    phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
    pred_labels_idx = phase_probs.argmax(axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels_idx])
    pred_reps_raw = parse_reps(pred_labels_idx)
    pred_reps = pred_reps_raw
    merge_note = "raw MA25+Viterbi, no per-action duration merge"
    if merge_mode == "selective_top5_p5" and action in SELECTIVE_ACTIONS:
        threshold = threshold_for_action(duration_priors, action, 5)
        pred_reps = merge_short_reps(pred_reps_raw, threshold, max_gap_samples=50)
        merge_note = f"top5_p5 merge, min duration={threshold:.0f} samples"

    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    t = np.arange(len(df)) / sample_rate
    acc_mag = np.linalg.norm(x[:, :3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 0.8, 1.0]})

    draw_phase_spans(axes[0], pred_phase, sample_rate, alpha=0.22)
    axes[0].plot(t, acc_mag, color="#1d4ed8", linewidth=1.1, label="acc magnitude (z-score space)")
    axes[0].plot(t, gyro_mag, color="#dc2626", linewidth=1.0, alpha=0.85, label="gyro magnitude (z-score space)")
    draw_rep_boundaries(axes[0], gt_reps, sample_rate, "#111827", "--", "GT rep boundary")
    draw_rep_boundaries(axes[0], pred_reps, sample_rate, "#16a34a", "-", "Pred rep boundary")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"{stream_id} | GT reps={len(gt_reps)} Pred reps={len(pred_reps)} Raw pred reps={len(pred_reps_raw)} | {merge_note}")
    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_ylim(0, 2.3)
    draw_phase_bar(axes[1], gt_phases, sample_rate, 1.25, 0.55, "GT")
    draw_phase_bar(axes[1], pred_phase, sample_rate, 0.35, 0.55, "Pred")
    axes[1].set_yticks([])
    axes[1].set_title("Concentric/eccentric phase cuts (blue=concentric, yellow=eccentric)", fontsize=10)
    axes[1].grid(False)

    axes[2].plot(t, active_probs, color="#7c3aed", linewidth=1.1, label="active detector probability")
    axes[2].axhline(0.5, color="#6b7280", linestyle="--", linewidth=0.9, label="active threshold")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Active prob")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend(loc="upper right", frameon=False, fontsize=9)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    safe = stream_id.replace("/", "__")
    out_path = output_dir / f"segmentation_{merge_mode}_{safe}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="yushuan")
    parser.add_argument("--actions", default="db_rdl,db_weighted_crunch,db_biceps_curl,db_shoulder_press")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--merge-mode", choices=["raw", "selective_top5_p5"], default="raw")
    parser.add_argument("--output-dir", default="artifacts/figures/current_segmentation_examples")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != args.subject]
    selected_streams = find_streams(streams, args.subject, actions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Held-out subject: {args.subject}")
    print(f"Selected streams: {[sid for sid, _ in selected_streams]}")
    print(f"Training raw6 CNN on {len(train_streams)} train streams, device={device}")
    model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
    print(f"Train active segments={n_segments}")
    print("Training active detector...")
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    duration_priors = build_duration_priors(train_streams, [5])

    written = []
    for stream_id, df in selected_streams:
        out_path = plot_stream(stream_id, df, cfg, active_models, active_scalers, model, mean, std, duration_priors, output_dir, args.merge_mode)
        written.append(str(out_path))
        print(f"Wrote {out_path}")

    print("\nGenerated figures:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
