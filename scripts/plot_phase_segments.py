"""Plot IMU signals overview with phase transition lines.

Generates a combined overview plot showing one representative rep per action
with phase transition visualization.

Usage:
    python -m scripts.plot_phase_segments [--config config.yaml] [--out-dir artifacts_phase]
"""
from __future__ import annotations

import argparse
import fnmatch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml


PHASE_COLORS = {
    "eccentric": "#4C9BE8",       # blue
    "concentric": "#E8854C",      # orange
    "inter_set_rest": "#AAAAAA",  # grey
    "none": "#DDDDDD",            # light grey
}

PHASE_LABELS = {
    "eccentric": "Eccentric",
    "concentric": "Concentric",
    "inter_set_rest": "Rest",
    "none": "None",
}

IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
ACCEL_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]


def _find_transitions(phases: np.ndarray) -> List[int]:
    """Return indices where phase changes."""
    transitions = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            transitions.append(i)
    return transitions


def _rle(phases: np.ndarray) -> List[Tuple[str, int, int]]:
    """Run-length encoding: list of (phase, start_idx, end_idx)."""
    runs = []
    cur, start = phases[0], 0
    for i in range(1, len(phases)):
        if phases[i] != cur:
            runs.append((str(cur), start, i))
            cur, start = phases[i], i
    runs.append((str(cur), start, len(phases)))
    return runs


def _plot_rep(
    df: pd.DataFrame,
    title: str,
    ax_accel: plt.Axes,
    ax_gyro: plt.Axes,
    sample_rate_hz: int = 50,
) -> None:
    """Plot one rep's IMU data with phase background and transition lines."""
    phases = df["phase"].astype(str).to_numpy()
    n = len(df)
    t = np.arange(n) / sample_rate_hz  # time in seconds

    runs = _rle(phases)
    transitions = _find_transitions(phases)

    for ax, cols, ylabel in [
        (ax_accel, ACCEL_COLS, "Accel (g)"),
        (ax_gyro, GYRO_COLS, "Gyro (°/s)"),
    ]:
        # Phase background spans
        for phase, s, e in runs:
            color = PHASE_COLORS.get(phase, "#EEEEEE")
            ax.axvspan(t[s], t[min(e, n) - 1], alpha=0.15, color=color, linewidth=0)

        # Transition lines
        for idx in transitions:
            ax.axvline(t[idx], color="red", linestyle="--", linewidth=1.2, alpha=0.8)

        # IMU signals
        for col in cols:
            if col in df.columns:
                ax.plot(t, df[col].to_numpy(), linewidth=0.8, label=col)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.tick_params(labelsize=8)

    ax_accel.set_title(title, fontsize=10, fontweight="bold")
    ax_gyro.set_xlabel("Time (s)", fontsize=9)


def _get_timestamped_dir(base_dir: Path) -> Path:
    """Create a timestamped subdirectory for organizing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


def main(config_path: Path, out_dir: Path, use_timestamp: bool = True) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    exclude_patterns = data_cfg.get("exclude_patterns", [])
    include_actions = set(data_cfg.get("include_actions", []))
    sample_rate_hz = cfg.get("window", {}).get("sample_rate_hz", 50)

    out_dir = Path(out_dir)
    if use_timestamp:
        out_dir = _get_timestamped_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {out_dir}")

    # Collect CSV files grouped by action
    action_files: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(data_dir.rglob("*.csv")):
        rel = f.relative_to(data_dir)
        parts = rel.parts
        if any(fnmatch.fnmatch(p, pat) for p in parts for pat in exclude_patterns):
            continue
        if len(parts) < 2:
            continue
        action = parts[1]
        if include_actions and action not in include_actions:
            continue
        # Skip rest-only files
        if "rest" in parts[-1].lower() or any("rest" in p.lower() for p in parts[2:]):
            continue
        action_files[action].append(f)

    if not action_files:
        print("[WARN] No action files found.")
        return

    # Generate a combined overview: 1 rep per action
    fig, axes = plt.subplots(
        len(action_files) * 2, 1,
        figsize=(14, 3.5 * len(action_files)),
        sharex=False,
        constrained_layout=True,
    )
    if len(action_files) == 1:
        axes = np.array(axes).reshape(-1)

    idx = 0
    for action in sorted(action_files):
        files = action_files[action]
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "phase" not in df.columns:
                continue
            phases = df["phase"].astype(str).to_numpy()
            if "eccentric" in set(phases) and "concentric" in set(phases):
                subject = f.relative_to(data_dir).parts[0]
                title = f"{action} ({subject})"
                _plot_rep(df, title, axes[idx * 2], axes[idx * 2 + 1], sample_rate_hz=100)
                idx += 1
                break

    patches = [
        mpatches.Patch(color=PHASE_COLORS["eccentric"], alpha=0.4, label="Eccentric"),
        mpatches.Patch(color=PHASE_COLORS["concentric"], alpha=0.4, label="Concentric"),
        mpatches.Patch(color=PHASE_COLORS["inter_set_rest"], alpha=0.4, label="Rest"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.2, label="Transition"),
    ]
    fig.legend(handles=patches, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Phase Segmentation Overview — All Actions", fontsize=13,
                 fontweight="bold", y=1.04)
    save_path = out_dir / "phase_segments_overview.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/phase_segmentation/plots"))
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    args = parser.parse_args()
    main(args.config, args.out_dir, use_timestamp=not args.no_timestamp)
