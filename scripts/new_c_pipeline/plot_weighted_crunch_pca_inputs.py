"""Plot weighted-crunch windows before/after PCA as seen by the 1D CNN."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    extract_active_segments_data,
    normalize,
    should_exclude,
)
from train.micro_macro_recognition import _load_streams  # noqa: E402


IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
RAW_COLORS = ["#2563eb", "#60a5fa", "#1e40af", "#dc2626", "#f87171", "#991b1b"]
PCA_COLORS = ["#0f766e", "#14b8a6", "#7c3aed", "#a855f7"]


def stream_subject(stream_id: str) -> str:
    return stream_id.split("/")[0]


def stream_action(stream_id: str) -> str:
    parts = [p for p in stream_id.split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def active_runs(phase_arr):
    active = np.array([str(p) in {"concentric", "eccentric"} for p in phase_arr])
    runs = []
    in_run = False
    start = 0
    for i, is_active in enumerate(active):
        if is_active and not in_run:
            start = i
            in_run = True
        elif not is_active and in_run:
            if i - start >= 10:
                runs.append((start, i))
            in_run = False
    if in_run and len(active) - start >= 10:
        runs.append((start, len(active)))
    return runs


def fit_fold_transforms(train_streams):
    train_segments, _ = extract_active_segments_data(train_streams, IMU_COLUMNS)
    raw_mean, raw_std, _ = normalize(train_segments)

    all_train_samples = np.concatenate(train_segments, axis=0)
    scaler = StandardScaler()
    all_train_std = scaler.fit_transform(all_train_samples)
    pca = PCA(n_components=4)
    pca.fit(all_train_std)

    pca_segments = [pca.transform(scaler.transform(seg)) for seg in train_segments]
    pca_mean, pca_std, _ = normalize(pca_segments)
    return raw_mean, raw_std, scaler, pca, pca_mean, pca_std


def select_window(df, raw_mean, raw_std, scaler, pca, pca_mean, pca_std, slice_len=300):
    phase_arr = df["phase"].to_numpy()
    runs = active_runs(phase_arr)
    if not runs:
        return None
    start, end = max(runs, key=lambda item: item[1] - item[0])
    length = end - start
    if length >= slice_len:
        win_start = start + max(0, (length - slice_len) // 2)
        win_end = win_start + slice_len
    else:
        win_start, win_end = start, end

    x = df[IMU_COLUMNS].to_numpy(dtype=np.float32)[win_start:win_end]
    labels = phase_arr[win_start:win_end]
    if len(x) < slice_len:
        pad_len = slice_len - len(x)
        x = np.pad(x, ((0, pad_len), (0, 0)), mode="edge")
        labels = np.pad(labels, (0, pad_len), mode="edge")

    raw_norm = (x - raw_mean) / raw_std
    pca_norm = pca.transform(scaler.transform(x))
    pca_norm = (pca_norm - pca_mean) / pca_std
    return raw_norm, pca_norm, labels[:slice_len], win_start


def shade_phases(ax, labels, sample_rate_hz=100.0):
    colors = {"eccentric": "#fef3c7", "concentric": "#dbeafe"}
    start = 0
    current = str(labels[0])
    for i in range(1, len(labels) + 1):
        label = str(labels[i]) if i < len(labels) else None
        if label != current:
            if current in colors:
                ax.axvspan(start / sample_rate_hz, i / sample_rate_hz, color=colors[current], alpha=0.55, linewidth=0)
            start = i
            current = label


def plot_stream(stream_id, df, transforms, output_dir):
    raw_mean, raw_std, scaler, pca, pca_mean, pca_std = transforms
    selected = select_window(df, raw_mean, raw_std, scaler, pca, pca_mean, pca_std)
    if selected is None:
        return None
    raw_norm, pca_norm, labels, win_start = selected
    t = np.arange(len(labels)) / 100.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    for ax in axes:
        shade_phases(ax, labels)
        ax.set_xlim(0, len(labels) / 100.0)
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.35)
        ax.grid(True, axis="y", alpha=0.22)

    for i, col in enumerate(IMU_COLUMNS):
        axes[0].plot(t, raw_norm[:, i], label=col, linewidth=1.25, color=RAW_COLORS[i])
    axes[0].set_title("Before PCA: raw 6-axis IMU after training-fold z-score")
    axes[0].set_ylabel("z-score")
    axes[0].legend(ncol=6, loc="upper right", frameon=False)

    axes[1].plot(t, pca_norm[:, 0], label="PC1", linewidth=1.7, color=PCA_COLORS[0])
    axes[1].set_title("After PCA: PCA-1 input channel (PC1 only)")
    axes[1].set_ylabel("z-score")
    axes[1].legend(ncol=1, loc="upper right", frameon=False)

    for i in range(4):
        axes[2].plot(t, pca_norm[:, i], label=f"PC{i + 1}", linewidth=1.4, color=PCA_COLORS[i])
    axes[2].set_title("After PCA: PCA-4 input channels")
    axes[2].set_xlabel("Time in CNN window (seconds)")
    axes[2].set_ylabel("z-score")
    axes[2].legend(ncol=4, loc="upper right", frameon=False)

    subject = stream_subject(stream_id)
    fig.suptitle(
        f"Weighted crunch CNN input window: {stream_id} | start sample {win_start}",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.018,
        "Background: yellow=eccentric, blue=concentric. Window length: 300 samples (3s @ 100Hz).",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))

    safe_id = stream_id.replace("/", "__")
    out_path = output_dir / f"weighted_crunch_pca_input_{subject}_{safe_id}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="hsianshun,yanz,kevin")
    parser.add_argument("--output-dir", default="artifacts/figures/weighted_crunch_pca_inputs")
    args = parser.parse_args()

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    target_subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for subject in target_subjects:
        candidates = [
            (sid, df)
            for sid, df in streams
            if stream_subject(sid) == subject and stream_action(sid) == "db_weighted_crunch" and "phase" in df.columns
        ]
        if not candidates:
            print(f"No weighted crunch stream found for subject={subject}")
            continue
        stream_id, df = candidates[0]
        train_streams = [(sid, data) for sid, data in streams if stream_subject(sid) != subject]
        transforms = fit_fold_transforms(train_streams)
        out_path = plot_stream(stream_id, df, transforms, output_dir)
        if out_path is not None:
            written.append(str(out_path))
            print(f"Wrote {out_path}")

    print("\nGenerated figures:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
