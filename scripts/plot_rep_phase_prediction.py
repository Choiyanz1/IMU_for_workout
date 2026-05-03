"""Plot one-rep phase segmentation predictions.

This helper matches the intended runtime pipeline:
rep segmentation crops one repetition first, then the phase model predicts the
eccentric/concentric transition inside that rep.

Usage:
    python -m scripts.plot_rep_phase_prediction \
        --config configs/phase_segmentation.yaml \
        --model-dir artifacts/phase_segmentation/20260428_191118/models \
        --out-dir artifacts/phase_segmentation/plots
"""
from __future__ import annotations

import argparse
import fnmatch
import re
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - lightweight runtime fallback
    yaml = None

from train.phase_segmentation import (
    normalize_rep_for_phase,
    rep_to_phase_feature_rows,
)

PHASE_COLORS = {
    "eccentric": "#4C9BE8",
    "concentric": "#E8854C",
    "none": "#EEEEEE",
}


def _load_config(config_path: Path) -> dict:
    if yaml is not None:
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    from evaluation.rep_segmentation import _load_config as fallback_load_config
    return fallback_load_config(config_path)


def _natural_sort_key(path: Path):
    nums = re.findall(r"(\d+)", path.stem)
    return [int(n) for n in nums] if nums else [0]


def _rle(arr: Sequence[str]) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    cur, start = str(arr[0]), 0
    for i in range(1, len(arr)):
        value = str(arr[i])
        if value != cur:
            runs.append((cur, start, i))
            cur, start = value, i
    runs.append((cur, start, len(arr)))
    return runs


def _find_transitions(arr: Sequence[str]) -> List[int]:
    return [i for i in range(1, len(arr)) if str(arr[i]) != str(arr[i - 1])]


def _active_rep_slice(df: pd.DataFrame) -> pd.DataFrame:
    phases = df["phase"].astype(str).to_numpy()
    active = np.isin(phases, ["eccentric", "concentric"])
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return pd.DataFrame()
    return df.iloc[int(idx[0]): int(idx[-1]) + 1].reset_index(drop=True)


def _predict_rep(
    rep_df: pd.DataFrame,
    predictor,
    imu_cols: Sequence[str],
    action_type: str,
    window_size: int,
    stride_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if predictor is None:
        return np.array([]), np.array([])
    rep_norm = normalize_rep_for_phase(rep_df, imu_cols)
    features, meta = rep_to_phase_feature_rows(
        rep_norm,
        imu_cols,
        action_type,
        window_size,
        stride_size,
    )
    if features.empty:
        return np.array([]), np.array([])
    pred = predictor.predict(features, as_pandas=True).astype(str).to_numpy()
    centers = np.asarray([row["center"] for row in meta], dtype=int)
    return centers, pred


def _plot_rep(
    rep_df: pd.DataFrame,
    gt_phases: np.ndarray,
    pred_centers: np.ndarray,
    pred_labels: np.ndarray,
    imu_cols: Sequence[str],
    title: str,
    save_path: Path,
    sample_rate_hz: int,
) -> None:
    n = len(rep_df)
    t = np.arange(n) / sample_rate_hz
    accel_cols = [col for col in imu_cols if col.startswith("a")]
    gyro_cols = [col for col in imu_cols if col.startswith("g")]

    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, constrained_layout=True)

    for phase, start, end in _rle(gt_phases):
        color = PHASE_COLORS.get(phase, "#EEEEEE")
        for ax in (ax_a, ax_g):
            ax.axvspan(t[start], t[min(end, n) - 1], alpha=0.16, color=color, linewidth=0)

    for idx in _find_transitions(gt_phases):
        for ax in (ax_a, ax_g):
            ax.axvline(t[idx], color="red", linestyle="--", linewidth=1.0, alpha=0.75)

    if len(pred_centers) > 0 and len(pred_labels) > 0:
        for tr_idx in _find_transitions(pred_labels):
            sample_idx = int(pred_centers[tr_idx])
            if sample_idx < n:
                for ax in (ax_a, ax_g):
                    ax.axvline(t[sample_idx], color="limegreen", linestyle="-.", linewidth=1.5, alpha=0.9)

    for col in accel_cols:
        if col in rep_df.columns:
            ax_a.plot(t, rep_df[col].to_numpy(), linewidth=0.7, label=col)
    for col in gyro_cols:
        if col in rep_df.columns:
            ax_g.plot(t, rep_df[col].to_numpy(), linewidth=0.7, label=col)

    ax_a.set_ylabel("Accel")
    ax_g.set_ylabel("Gyro")
    ax_g.set_xlabel("Time (s)")
    ax_a.set_title(title, fontsize=12, fontweight="bold")
    for ax in (ax_a, ax_g):
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    handles = [
        mpatches.Patch(color=PHASE_COLORS["eccentric"], alpha=0.3, label="Eccentric GT"),
        mpatches.Patch(color=PHASE_COLORS["concentric"], alpha=0.3, label="Concentric GT"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.0, label="GT transition"),
        plt.Line2D([0], [0], color="limegreen", linestyle="-.", linewidth=1.5, label="Pred transition"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.04))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {save_path}")


def _timestamped_dir(base_dir: Path) -> Path:
    return base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")


def main(config_path: Path, model_dir: Path, out_dir: Path, use_timestamp: bool, max_plots: int) -> None:
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    feature_cfg = cfg.get("feature", {})
    phase_cfg = cfg.get("phase", {})
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    include_actions = set(data_cfg.get("include_actions") or [])
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])
    imu_cols = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    sample_rate_hz = int(cfg.get("window", {}).get("sample_rate_hz", 50))
    window_size = int(float(phase_cfg.get("window_seconds", 0.6)) * sample_rate_hz)
    stride_size = int(float(phase_cfg.get("stride_seconds", 0.1)) * sample_rate_hz)

    out_dir = _timestamped_dir(out_dir) if use_timestamp else out_dir
    plots_dir = out_dir / "rep_predictions"
    plots_dir.mkdir(parents=True, exist_ok=True)

    predictor = None
    if model_dir.exists():
        try:
            from autogluon.tabular import TabularPredictor
            predictor = TabularPredictor.load(str(model_dir))
            print(f"[INFO] Loaded phase model from {model_dir}")
        except Exception as exc:
            print(f"[WARN] Could not load phase model: {exc}")
    else:
        print(f"[WARN] Model dir {model_dir} not found, plotting GT only")

    plotted = 0
    seen = set()
    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for action_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            action = action_dir.name
            if include_actions and action not in include_actions:
                continue
            for set_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("set")):
                if any(fnmatch.fnmatch(part, pattern) for part in set_dir.parts for pattern in exclude_patterns):
                    continue
                key = (subject, action)
                if key in seen:
                    continue
                for csv_path in sorted(set_dir.glob("*.csv"), key=_natural_sort_key):
                    try:
                        raw_df = pd.read_csv(csv_path)
                    except Exception:
                        continue
                    if "phase" not in raw_df.columns:
                        continue
                    rep_df = _active_rep_slice(raw_df)
                    if rep_df.empty or len(rep_df) < window_size:
                        continue
                    phases = rep_df["phase"].astype(str).to_numpy()
                    if not {"eccentric", "concentric"}.issubset(set(phases)):
                        continue

                    centers, labels = _predict_rep(rep_df, predictor, imu_cols, action, window_size, stride_size)
                    title = f"{action} - {subject}/{set_dir.name}/{csv_path.stem}"
                    save_path = plots_dir / f"rep_phase_{action}_{subject}_{set_dir.name}_{csv_path.stem}.png"
                    _plot_rep(rep_df, phases, centers, labels, imu_cols, title, save_path, sample_rate_hz)
                    seen.add(key)
                    plotted += 1
                    break
                if max_plots > 0 and plotted >= max_plots:
                    print(f"[OK] Wrote {plotted} plots to {plots_dir}")
                    return

    print(f"[OK] Wrote {plotted} plots to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/phase_segmentation.yaml"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/phase_segmentation/models"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/phase_segmentation/plots"))
    parser.add_argument("--max-plots", type=int, default=0, help="Maximum plots to write; 0 = no limit")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    args = parser.parse_args()
    main(args.config, args.model_dir, args.out_dir, use_timestamp=not args.no_timestamp, max_plots=args.max_plots)
