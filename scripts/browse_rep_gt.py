"""Interactive browser for set-level GT waveforms.

Navigate: subject -> session -> action -> set, then view a whole-set overview.
The display is meant to be closer to the per-action causal RF + boundary
refiner input: it shows the configured IMU channels after train-fold z-score
normalization, concatenated rep by rep with GT phase colouring.

Usage:
    python scripts/browse_rep_gt.py
    python scripts/browse_rep_gt.py --config configs/micro_macro_recognition_8act_test_yushuan.yaml
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.window_pipeline import ZScoreStats, apply_zscore, compute_train_stats


PHASE_COLORS = {
    "concentric": "#3b82f6",
    "eccentric": "#ef4444",
    "none": "#d1d5db",
}

GROUP_SPECS = [
    ("accel_z", ["ax", "ay", "az"], ["#1d4ed8", "#2563eb", "#60a5fa"]),
    ("gyro_z", ["gx", "gy", "gz"], ["#6d28d9", "#7c3aed", "#a78bfa"]),
    ("mag_z", ["mx", "my", "mz"], ["#047857", "#10b981", "#6ee7b7"]),
]
MAG_COLS = ["mx", "my", "mz"]


def _natural_sort_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _subdirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted([d for d in parent.iterdir() if d.is_dir()], key=_natural_sort_key)


def _csv_files(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted(parent.glob("*.csv"), key=_natural_sort_key)


def _choose(prompt: str, options: List[str], allow_back: bool = True) -> int | None:
    print()
    print(f"--- {prompt} ---")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    if allow_back:
        print("  [b] <- back")
    print("  [q] quit")
    while True:
        raw = input(">>> ").strip().lower()
        if raw == "q":
            sys.exit(0)
        if raw == "b" and allow_back:
            return None
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  Invalid input, try again.")


def _load_browser_config(config_path: Path, data_dir_arg: str) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = raw.get("data", {}) or {}
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_cfg = raw.get("train", {}) or {}
    data_dir = Path(data_dir_arg) if str(data_dir_arg).strip() else Path(data_cfg.get("data_dir", "datasets/raw_data"))
    model_imu_columns = [str(c) for c in feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"])]
    inspection_columns = list(dict.fromkeys(model_imu_columns + MAG_COLS))
    return {
        "data_dir": data_dir,
        "imu_columns": model_imu_columns,
        "inspection_columns": inspection_columns,
        "time_column": str(feature_cfg.get("time_column", "sensor_ts")),
        "sample_rate_hz": float(window_cfg.get("sample_rate_hz", 100.0)),
        "config_test_subject": str(train_cfg.get("test_subject", "")).strip(),
        "include_actions": set(str(x) for x in (data_cfg.get("include_actions") or [])),
        "exclude_patterns": [str(x) for x in (data_cfg.get("exclude_patterns") or [])],
    }


def _path_matches_patterns(path: Path, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(part, pat) for part in path.parts for pat in patterns)


def _path_contains_action(path: Path, include_actions: set[str]) -> bool:
    if not include_actions:
        return True
    return any(part in include_actions for part in path.parts)


def _apply_zscore_subset(df: pd.DataFrame, columns: List[str], stats: ZScoreStats) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        return df
    if len(available) == len(columns):
        return apply_zscore(df, columns, stats)
    out = df.copy()
    mean_map = {col: float(stats.mean[i]) for i, col in enumerate(columns)}
    std_map = {col: float(stats.std[i]) for i, col in enumerate(columns)}
    for col in available:
        out[col] = (out[col].to_numpy(dtype=np.float32) - mean_map[col]) / std_map[col]
    return out


def _load_rep_frames(rep_files: List[Path], zscore_columns: List[str], stats: ZScoreStats | None) -> List[pd.DataFrame | None]:
    frames: List[pd.DataFrame | None] = []
    for csv_path in rep_files:
        try:
            df = pd.read_csv(csv_path)
            if stats is not None:
                df = _apply_zscore_subset(df, zscore_columns, stats)
            frames.append(df)
        except Exception:
            frames.append(None)
    return frames


def _time_axis(df: pd.DataFrame, time_column: str, sample_rate_hz: float) -> np.ndarray:
    if time_column in df.columns:
        t = (df[time_column] - df[time_column].iloc[0]).to_numpy(dtype=float) / 1e6
    else:
        t = np.arange(len(df), dtype=float) / max(sample_rate_hz, 1e-6)
    return t


def _add_phase_spans(ax: plt.Axes, t: np.ndarray, phases: np.ndarray) -> None:
    i = 0
    while i < len(phases):
        j = i + 1
        while j < len(phases) and phases[j] == phases[i]:
            j += 1
        phase = str(phases[i]).lower()
        color = PHASE_COLORS.get(phase, "#e5e7eb")
        ax.axvspan(t[i], t[min(j - 1, len(t) - 1)], alpha=0.24, color=color, linewidth=0)
        i = j


def _training_stats_key(cfg: dict, selected_subject: str) -> str:
    configured = str(cfg.get("config_test_subject") or "").strip()
    return configured or str(selected_subject)


def _compute_train_fold_stats(data_root: Path, heldout_subject: str, imu_columns: List[str], include_actions: set[str], exclude_patterns: List[str]) -> ZScoreStats | None:
    sequences: List[pd.DataFrame] = []
    for csv_path in sorted(data_root.rglob("*.csv")):
        rel = csv_path.relative_to(data_root)
        if rel.parts and rel.parts[0] == heldout_subject:
            continue
        if _path_matches_patterns(rel, exclude_patterns):
            continue
        if not _path_contains_action(rel, include_actions):
            continue
        try:
            df = pd.read_csv(csv_path, usecols=imu_columns)
        except Exception:
            continue
        if not df.empty:
            sequences.append(df)
    if not sequences:
        return None
    return compute_train_stats(sequences, imu_columns)


def _plot_group_series(
    ax: plt.Axes,
    rep_frames: List[pd.DataFrame | None],
    rep_files: List[Path],
    time_column: str,
    sample_rate_hz: float,
    cols: List[str],
    ylabel: str,
    colors: List[str],
    rep_texts: Dict[int, plt.Text] | None = None,
) -> Dict[int, plt.Text]:
    local_texts = rep_texts or {}
    available_cols = [c for c in cols if any(df is not None and c in df.columns for df in rep_frames)]
    if not available_cols:
        ax.text(0.5, 0.5, f"No {ylabel} channels found", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#64748b")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        return local_texts

    offset_s = 0.0
    gap_s = 0.15
    for idx, (_, df) in enumerate(zip(rep_files, rep_frames)):
        if df is None or df.empty:
            continue
        t_local = _time_axis(df, time_column, sample_rate_hz)
        t = t_local + offset_s
        phases = df["phase"].astype(str).to_numpy() if "phase" in df.columns else np.asarray(["none"] * len(df), dtype=object)
        _add_phase_spans(ax, t, phases)
        for col, color in zip(available_cols, colors):
            if col in df.columns:
                ax.plot(
                    t,
                    df[col].to_numpy(dtype=float),
                    linewidth=0.95,
                    color=color,
                    label=col if idx == 0 else "_nolegend_",
                )

        mid_t = t[0] + (t[-1] - t[0]) * 0.5 if len(t) else offset_s
        if rep_texts is None:
            local_texts[idx] = ax.text(mid_t, 1.02, str(idx), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)

        boundary_t = t[-1] if len(t) else offset_s
        ax.axvline(boundary_t, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.85)
        duration = float(t_local[-1]) if len(t_local) > 1 else 0.0
        offset_s = boundary_t + gap_s + max(0.0, duration / max(1, len(t_local) - 1))

    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=8, ncol=max(1, len(available_cols)))
    return local_texts


def _plot_set(
    set_dir: Path,
    rep_files: List[Path],
    cfg: dict,
    stats: ZScoreStats | None,
    rf_window_size: int,
    edge_window: int,
    stats_subject: str,
) -> List[Path]:
    n = len(rep_files)
    if n == 0:
        print("  (no rep CSVs found)")
        return []

    imu_columns = list(cfg["imu_columns"])
    inspection_columns = list(cfg["inspection_columns"])
    rep_frames = _load_rep_frames(rep_files, inspection_columns, stats)
    groups = [(name, [c for c in cols if c in inspection_columns], colors) for name, cols, colors in GROUP_SPECS]
    groups = [g for g in groups if g[1]]
    if not groups:
        print(f"  No configured IMU columns found in config: {imu_columns}")
        return []

    fig, axes = plt.subplots(len(groups), 1, figsize=(18, 3.25 * len(groups) + 1.5), sharex=True)
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle(str(set_dir), fontsize=11, fontweight="bold")

    rep_texts: Dict[int, plt.Text] | None = None
    for ax, (name, cols, colors) in zip(axes, groups):
        rep_texts = _plot_group_series(
            ax,
            rep_frames,
            rep_files,
            str(cfg["time_column"]),
            float(cfg["sample_rate_hz"]),
            cols,
            name,
            colors,
            rep_texts=rep_texts,
        )

    axes[0].set_title(
        f"Whole-set overview (z-scored model channels + inspection mag; RF window={rf_window_size} samples; edge window={edge_window}; stats holdout={stats_subject})",
        fontsize=10,
        fontweight="bold",
    )
    axes[-1].set_xlabel("time (s)", fontsize=9)

    legend_elements = [
        Patch(facecolor=PHASE_COLORS["concentric"], alpha=0.4, label="concentric"),
        Patch(facecolor=PHASE_COLORS["eccentric"], alpha=0.4, label="eccentric"),
        Patch(facecolor=PHASE_COLORS["none"], alpha=0.4, label="none/other"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show(block=False)
    plt.pause(0.1)

    print()
    print(f"  Viewing {n} reps in: {set_dir}")
    print(f"  Plot uses model imu_columns={imu_columns} plus inspection mag if present; train-fold z-score holdout={stats_subject}")
    for idx, rep_file in enumerate(rep_files):
        print(f"    [{idx}] {rep_file.name}")
    print("  Commands:")
    print("    d <idx> [idx...]  - mark rep(s) for deletion (e.g. 'd 3 5 7')")
    print("    u <idx>           - unmark rep")
    print("    l                 - list marked reps")
    print("    confirm           - DELETE marked files and continue")
    print("    enter / n         - continue without deleting")

    to_delete: set[int] = set()
    rep_texts = rep_texts or {}

    def _update_marks() -> None:
        for idx, txt in rep_texts.items():
            if idx in to_delete:
                txt.set_color("red")
                txt.set_fontstyle("italic")
            else:
                txt.set_color("black")
                txt.set_fontstyle("normal")
        fig.canvas.draw_idle()

    while True:
        raw = input("  >> ").strip().lower()
        if raw in ("", "n", "next"):
            plt.close(fig)
            return []
        if raw == "q":
            plt.close(fig)
            sys.exit(0)
        if raw == "l":
            if to_delete:
                for di in sorted(to_delete):
                    print(f"    [x] [{di}] {rep_files[di].name}")
            else:
                print("    (none marked)")
            continue
        if raw.startswith("d ") or raw.startswith("d\t"):
            for p in raw.split()[1:]:
                try:
                    idx = int(p)
                    if 0 <= idx < n:
                        to_delete.add(idx)
                        print(f"    marked [{idx}] {rep_files[idx].name}")
                    else:
                        print(f"    [{idx}] out of range")
                except ValueError:
                    print(f"    '{p}' not a number")
            _update_marks()
            continue
        if raw.startswith("u "):
            for p in raw.split()[1:]:
                try:
                    idx = int(p)
                    to_delete.discard(idx)
                    print(f"    unmarked [{idx}]")
                except ValueError:
                    pass
            _update_marks()
            continue
        if raw == "confirm":
            if not to_delete:
                print("    nothing to delete, continuing.")
                plt.close(fig)
                return []
            print(f"    About to DELETE {len(to_delete)} file(s):")
            for di in sorted(to_delete):
                print(f"      {rep_files[di]}")
            yn = input("    Type 'yes' to confirm: ").strip().lower()
            if yn == "yes":
                deleted = []
                for di in sorted(to_delete):
                    fp = rep_files[di]
                    try:
                        fp.unlink()
                        deleted.append(fp)
                        print(f"      DELETED: {fp.name}")
                    except Exception as exc:
                        print(f"      ERROR deleting {fp.name}: {exc}")
                plt.close(fig)
                return deleted
            print("    cancelled.")
            continue
        print("    unknown command")


def main() -> None:
    default_config = Path("configs/micro_macro_recognition_8act_test_yushuan.yaml")
    parser = argparse.ArgumentParser(description="Browse dataset waveforms in a model-closer view")
    parser.add_argument("--config", default=str(default_config), help="Config used to resolve imu_columns, sample rate, and test subject")
    parser.add_argument("--data-dir", default="", help="Override data root directory")
    parser.add_argument("--rf-window-size", type=int, default=50, help="Trailing RF window size in samples for title/context")
    parser.add_argument("--edge-window", type=int, default=20, help="Boundary refiner edge window in samples for title/context")
    args = parser.parse_args()

    cfg = _load_browser_config(Path(args.config), str(args.data_dir))
    data_root = Path(cfg["data_dir"])
    if not data_root.is_dir():
        print(f"Data directory not found: {data_root}")
        sys.exit(1)

    stats_cache: Dict[str, ZScoreStats | None] = {}

    while True:
        subjects = _subdirs(data_root)
        if not subjects:
            print("No subjects found.")
            break
        idx = _choose("Select SUBJECT", [s.name for s in subjects], allow_back=False)
        if idx is None:
            continue
        subject_dir = subjects[idx]
        stats_subject = _training_stats_key(cfg, subject_dir.name)
        if stats_subject not in stats_cache:
            print(f"[INFO] computing train-fold z-score stats using holdout={stats_subject}")
            stats_cache[stats_subject] = _compute_train_fold_stats(
                data_root,
                stats_subject,
                list(cfg["inspection_columns"]),
                set(cfg["include_actions"]),
                list(cfg["exclude_patterns"]),
            )
            if stats_cache[stats_subject] is None:
                print("[WARN] could not compute train-fold stats; plotting raw values instead")

        while True:
            sessions = _subdirs(subject_dir)
            if not sessions:
                print("  No sessions found.")
                break
            idx = _choose(f"Select SESSION ({subject_dir.name})", [s.name for s in sessions])
            if idx is None:
                break
            session_dir = sessions[idx]

            while True:
                actions = _subdirs(session_dir)
                if not actions:
                    print("  No actions found.")
                    break
                display = []
                for action_dir in actions:
                    n_sets = len([d for d in action_dir.iterdir() if d.is_dir() and d.name.startswith("set")])
                    display.append(f"{action_dir.name}  ({n_sets} sets)")
                idx = _choose(f"Select ACTION ({subject_dir.name}/{session_dir.name})", display)
                if idx is None:
                    break
                action_dir = actions[idx]

                while True:
                    sets = [d for d in _subdirs(action_dir) if d.name.startswith("set")]
                    if not sets:
                        print("  No sets found.")
                        break
                    display = []
                    for set_dir in sets:
                        csvs = _csv_files(set_dir)
                        display.append(f"{set_dir.name}  ({len(csvs)} reps)")
                    idx = _choose(f"Select SET ({subject_dir.name}/{session_dir.name}/{action_dir.name})", display)
                    if idx is None:
                        break
                    set_dir = sets[idx]
                    rep_files = _csv_files(set_dir)
                    _plot_set(
                        set_dir,
                        rep_files,
                        cfg,
                        stats_cache.get(stats_subject),
                        rf_window_size=int(args.rf_window_size),
                        edge_window=int(args.edge_window),
                        stats_subject=stats_subject,
                    )


if __name__ == "__main__":
    main()
