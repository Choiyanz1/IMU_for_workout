"""Visualize the active detector on full streams, including rest periods.

This script trains the existing per-action RF active detector on non-held-out
subjects, then plots held-out streams with GT active/rest regions and predicted
active probability. It does not train or use the C/E CNN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import (  # noqa: E402
    CONCENTRIC_LABEL,
    ECCENTRIC_LABEL,
    micro_labels_from_phase,
)
from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    stream_action,
    stream_subject,
)
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    EXCLUDED_SESSIONS,
    set_seed,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams  # noqa: E402


def active_labels_from_phase(phases: np.ndarray) -> np.ndarray:
    labels = micro_labels_from_phase(phases)
    active = [str(label) in {CONCENTRIC_LABEL, ECCENTRIC_LABEL} for label in labels]
    return np.asarray(active, dtype=np.int64)


def mask_runs(mask: np.ndarray):
    runs = []
    in_run = False
    start = 0
    for i, value in enumerate(mask):
        if value and not in_run:
            start = i
            in_run = True
        elif not value and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def draw_binary_bar(ax, mask: np.ndarray, sample_rate: float, y: float, height: float, color: str, label: str):
    for start, end in mask_runs(mask):
        ax.broken_barh(
            [(start / sample_rate, (end - start) / sample_rate)],
            (y, height),
            facecolors=color,
        )
    ax.text(-0.02, y + height / 2, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10)


def find_streams(streams, subject: str, actions: list[str], max_per_action: int):
    selected = []
    for action in actions:
        candidates = [
            (sid, df)
            for sid, df in streams
            if stream_subject(sid) == subject and stream_action(sid) == action and "phase" in df.columns
        ]
        selected.extend(candidates[:max_per_action])
    return selected


def append_rest_after_set(stream_id: str, df: pd.DataFrame, data_dir: Path, rest_seconds: float, sample_rate: float):
    if rest_seconds <= 0:
        return stream_id, df

    parts = [part for part in stream_id.split("/") if part]
    if len(parts) < 4:
        return stream_id, df
    subject, session, action, set_name = parts[0], parts[1], parts[2], parts[3]
    rest_dir = data_dir / subject / session / action / f"rest_after_{set_name}"
    if not rest_dir.exists():
        return stream_id, df

    frames = []
    max_rows = int(round(rest_seconds * sample_rate))
    for csv_path in sorted(rest_dir.glob("*.csv")):
        try:
            rest_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if rest_df.empty or "phase" not in rest_df.columns:
            continue
        rest_df = rest_df.copy()
        if max_rows > 0:
            rest_df = rest_df.iloc[:max_rows]
        rest_df["action_type"] = "other"
        rest_df["subject_id"] = subject
        rest_df["_split_subject"] = subject
        rest_df["_source_file"] = csv_path.name
        frames.append(rest_df)
        break

    if not frames:
        return stream_id, df
    combined = pd.concat([df.copy(), *frames], ignore_index=True, sort=False)
    return f"{stream_id}+rest_after", combined


def active_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray, sample_rate: float):
    rest_mask = gt_mask == 0
    active_mask = gt_mask == 1
    false_active = np.logical_and(rest_mask, pred_mask == 1)
    missed_active = np.logical_and(active_mask, pred_mask == 0)
    return {
        "accuracy": float(accuracy_score(gt_mask, pred_mask)),
        "precision": float(precision_score(gt_mask, pred_mask, zero_division=0)),
        "recall": float(recall_score(gt_mask, pred_mask, zero_division=0)),
        "f1": float(f1_score(gt_mask, pred_mask, zero_division=0)),
        "rest_samples": int(rest_mask.sum()),
        "active_samples": int(active_mask.sum()),
        "false_active_rest_samples": int(false_active.sum()),
        "missed_active_samples": int(missed_active.sum()),
        "false_active_rest_sec": float(false_active.sum() / sample_rate),
        "missed_active_sec": float(missed_active.sum() / sample_rate),
        "false_active_rest_rate": float(false_active.sum() / max(1, rest_mask.sum())),
        "missed_active_rate": float(missed_active.sum() / max(1, active_mask.sum())),
    }


def plot_stream(stream_id, df, cfg, active_models, active_scalers, output_dir: Path, sample_rate: float, threshold: float, min_consecutive: int):
    phases = df["phase"].to_numpy()
    gt_mask = active_labels_from_phase(phases)
    active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
    threshold_mask = (active_probs >= threshold).astype(np.int64)
    active_segments = extract_active_segments(active_probs, threshold=threshold, min_consecutive=min_consecutive)
    segment_mask = np.zeros(len(df), dtype=np.int64)
    for start, end in active_segments:
        segment_mask[int(start):int(end)] = 1

    x = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    t = np.arange(len(df)) / sample_rate
    acc_mag = np.linalg.norm(x[:, :3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1)

    metrics = active_metrics(gt_mask, segment_mask, sample_rate)
    metrics["threshold_f1"] = float(f1_score(gt_mask, threshold_mask, zero_division=0))
    metrics["gt_active_segments"] = len(mask_runs(gt_mask.astype(bool)))
    metrics["pred_active_segments"] = len(active_segments)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 0.9, 1.4]},
    )

    for start, end in mask_runs(gt_mask.astype(bool)):
        axes[0].axvspan(start / sample_rate, end / sample_rate, color="#dbeafe", alpha=0.7, linewidth=0)
    axes[0].plot(t, acc_mag, color="#1d4ed8", linewidth=0.9, label="acc magnitude")
    axes[0].plot(t, gyro_mag, color="#dc2626", linewidth=0.8, alpha=0.75, label="gyro magnitude")
    axes[0].set_ylabel("Raw magnitude")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_ylim(0, 2.6)
    draw_binary_bar(axes[1], gt_mask.astype(bool), sample_rate, 1.55, 0.55, "#2563eb", "GT active")
    draw_binary_bar(axes[1], segment_mask.astype(bool), sample_rate, 0.55, 0.55, "#16a34a", "Pred active")
    axes[1].set_yticks([])
    axes[1].set_title("Active/rest segmentation bars", fontsize=10)

    axes[2].plot(t, active_probs, color="#7c3aed", linewidth=1.0, label="active probability")
    axes[2].axhline(threshold, color="#6b7280", linestyle="--", linewidth=0.9, label=f"threshold={threshold:g}")
    for start, end in active_segments:
        axes[2].axvspan(start / sample_rate, end / sample_rate, color="#86efac", alpha=0.18, linewidth=0)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Active prob")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend(loc="upper right", frameon=False, fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.25)

    title = (
        f"Active detector rest check | {stream_id}\n"
        f"F1={metrics['f1']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"false-active rest={metrics['false_active_rest_sec']:.2f}s ({metrics['false_active_rest_rate']:.1%}) "
        f"missed-active={metrics['missed_active_sec']:.2f}s ({metrics['missed_active_rate']:.1%})"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    safe = stream_id.replace("/", "__")
    path = output_dir / f"active_detector_{safe}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)

    return {
        "stream_id": stream_id,
        "subject": stream_subject(stream_id),
        "action": stream_action(stream_id),
        "figure": str(path),
        "metrics": metrics,
    }


def aggregate(results):
    if not results:
        return {}
    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_active_rest_rate",
        "missed_active_rate",
        "false_active_rest_sec",
        "missed_active_sec",
    ]
    return {key: float(np.mean([item["metrics"][key] for item in results])) for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="yushuan")
    parser.add_argument("--actions", default="db_rdl,db_weighted_crunch,db_biceps_curl,db_shoulder_press")
    parser.add_argument("--max-per-action", type=int, default=1)
    parser.add_argument("--append-rest-sec", type=float, default=20.0)
    parser.add_argument("--train-rest-sec", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--output-dir", default="artifacts/figures/active_detector_rest_examples")
    args = parser.parse_args()

    set_seed(42)
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    data_dir = Path((raw_cfg.get("data", {}) or {}).get("data_dir", "./datasets/raw_data"))
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != args.subject]
    active_train_streams = [
        append_rest_after_set(sid, df, data_dir, args.train_rest_sec, args.sample_rate)
        for sid, df in train_streams
    ]
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    selected_streams = [
        append_rest_after_set(sid, df, data_dir, args.append_rest_sec, args.sample_rate)
        for sid, df in find_streams(streams, args.subject, actions, args.max_per_action)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Held-out subject: {args.subject}")
    print(f"Training active detector on {len(active_train_streams)} streams; train_rest_sec={args.train_rest_sec}")
    print(f"Selected streams: {[sid for sid, _ in selected_streams]}")
    active_models, active_scalers = train_active_detector(active_train_streams, cfg)

    results = []
    for stream_id, df in selected_streams:
        result = plot_stream(
            stream_id,
            df,
            cfg,
            active_models,
            active_scalers,
            output_dir,
            args.sample_rate,
            args.threshold,
            args.min_consecutive,
        )
        results.append(result)
        m = result["metrics"]
        print(
            f"{stream_id}: F1={m['f1']:.3f} precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"false-active-rest={m['false_active_rest_sec']:.2f}s missed-active={m['missed_active_sec']:.2f}s "
            f"figure={result['figure']}"
        )

    summary = {
        "settings": {
            "active_detector": "per-action RandomForestClassifier over 1.0s windows, 0.1s stride",
            "window_size_samples": cfg.active_window_size,
            "stride_samples": cfg.active_stride,
            "threshold": args.threshold,
            "min_consecutive_samples": args.min_consecutive,
            "appended_rest_after_set_sec": args.append_rest_sec,
            "train_rest_after_set_sec": args.train_rest_sec,
            "held_out_subject": args.subject,
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "overall": aggregate(results),
        "streams": results,
    }
    out_json = output_dir / "active_detector_rest_examples.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
