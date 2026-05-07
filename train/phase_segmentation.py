"""Train a per-rep model for phase segmentation (eccentric / concentric).

The intended pipeline is:
1. detect repetition boundaries,
2. crop one rep,
3. run this model inside that single rep to cut eccentric/concentric phases.

Training uses sliding windows over the active part of each labelled rep
sequence. Each window is labelled with the phase at the **center** of the
window. Evaluation reports both window classification metrics and per-rep
transition-boundary error.

Usage:
    python -m train.phase_segmentation --config config.yaml [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import yaml
except Exception:  # pragma: no cover - lightweight runtime fallback
    yaml = None

from datasets.custom_resistance_dataset import (
    FeatureConfig,
    filter_sequences_by_subject,
    prepare_sequences_from_folder,
)
from evaluation.reporting import write_standard_run_outputs
from preprocessing.window_pipeline import (
    set_seed,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASE_MAP = {
    "eccentric": "eccentric",
    "concentric": "concentric",
    "inter_set_rest": None,
    "none": None,
}

FeatureMode = Literal["stats", "rich"]


@dataclass
class PhaseConfig:
    """Phase-segmentation specific settings (read from config.yaml -> phase)."""
    feature_mode: FeatureMode = "rich"
    window_seconds: float = 0.4
    stride_seconds: float = 0.1
    presets: str = "medium_quality_faster_train"
    time_limit_s: int = 600
    eval_metric: str = "accuracy"
    num_cpus: int | None = None
    included_model_types: List[str] | None = None
    excluded_model_types: List[str] | None = None
    num_bag_folds: int = 0
    num_stack_levels: int = 0


def build_configs(config_path: Path):
    if yaml is not None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        from evaluation.rep_segmentation import _load_config
        raw = _load_config(config_path)
    feature_cfg = FeatureConfig(**raw.get("feature", {}))
    # We only need sample_rate_hz from the window section
    sample_rate_hz = raw.get("window", {}).get("sample_rate_hz", 50)
    phase_cfg = PhaseConfig(**raw.get("phase", {}))
    return raw, feature_cfg, sample_rate_hz, phase_cfg


# ---------------------------------------------------------------------------
# Per-rep sliding-window extraction with center-point phase label
# ---------------------------------------------------------------------------

def _active_rep_bounds(phases: np.ndarray) -> Tuple[int, int] | None:
    active = np.isin(phases.astype(str), ["eccentric", "concentric"])
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def _first_phase_transition(phases: np.ndarray) -> int | None:
    """Return the first eccentric/concentric transition index inside a rep."""
    mapped = [PHASE_MAP.get(str(p)) for p in phases]
    for i in range(1, len(mapped)):
        if mapped[i] is not None and mapped[i - 1] is not None and mapped[i] != mapped[i - 1]:
            return i
    return None


def _extract_phase_windows(
    df: pd.DataFrame,
    imu_columns: Sequence[str],
    phase_column: str,
    subject_column: str,
    label_column: str,
    window_size: int,
    stride_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """Extract sliding windows from one active rep; label = phase at center.

    Returns (windows, phase_labels, action_types, meta).
    meta includes 'rel_pos' = relative position in the active rep [0, 1].
    """
    phases = df[phase_column].astype(str).to_numpy()
    active_bounds = _active_rep_bounds(phases)
    if active_bounds is None:
        empty_x = np.empty((0, window_size, len(imu_columns)), dtype=np.float32)
        return empty_x, np.empty((0,), dtype=object), [], []

    active_start, active_end = active_bounds
    active_df = df.iloc[active_start:active_end].reset_index(drop=True)
    active_phases = active_df[phase_column].astype(str).to_numpy()
    transition_idx = _first_phase_transition(active_phases)
    if transition_idx is None:
        empty_x = np.empty((0, window_size, len(imu_columns)), dtype=np.float32)
        return empty_x, np.empty((0,), dtype=object), [], []

    x = active_df[list(imu_columns)].to_numpy(dtype=np.float32)
    action = str(df.iloc[0][label_column])
    subj = str(df.iloc[0][subject_column])
    seq_len = len(active_df)
    source_id = str(df.attrs.get("source_path", f"{subj}/{action}/rep"))

    if len(active_df) < window_size:
        empty_x = np.empty((0, window_size, len(imu_columns)), dtype=np.float32)
        return empty_x, np.empty((0,), dtype=object), [], []

    windows, labels, actions, meta = [], [], [], []

    for start in range(0, len(active_df) - window_size + 1, stride_size):
        end = start + window_size
        center = start + window_size // 2
        raw_phase = active_phases[center]
        mapped = PHASE_MAP.get(raw_phase)
        if mapped is None:
            continue  # skip rest / none windows

        windows.append(x[start:end])
        labels.append(mapped)
        actions.append(action)
        meta.append(
            {
                "subject_id": subj,
                "source_id": source_id,
                "action_type": action,
                "start": start,
                "end": end,
                "center": center,
                "active_start_original": active_start,
                "active_end_original": active_end,
                "active_length": seq_len,
                "gt_transition_idx": transition_idx,
                "rel_pos": center / max(seq_len - 1, 1),
            }
        )

    if not windows:
        empty_x = np.empty((0, window_size, len(imu_columns)), dtype=np.float32)
        return empty_x, np.empty((0,), dtype=object), [], []
    return np.stack(windows), np.asarray(labels, dtype=object), actions, meta


def _collect_phase_windows(
    sequences: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    phase_column: str,
    subject_column: str,
    label_column: str,
    window_size: int,
    stride_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    all_w, all_y, all_a, all_m = [], [], [], []
    for seq in sequences:
        w, y, a, m = _extract_phase_windows(
            seq, imu_columns, phase_column, subject_column,
            label_column, window_size, stride_size,
        )
        if len(w) > 0:
            all_w.append(w)
            all_y.append(y)
            all_a.extend(a)
            all_m.extend(m)
    if not all_w:
        raise RuntimeError("No phase windows extracted.")
    return np.concatenate(all_w), np.concatenate(all_y), all_a, all_m


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _rms(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(arr ** 2, axis=1))


def _iqr(arr: np.ndarray) -> np.ndarray:
    return np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)


def _zcr(arr: np.ndarray) -> np.ndarray:
    signs = np.sign(arr)
    return np.abs(np.diff(signs, axis=1)).sum(axis=1) / (arr.shape[1] - 1)


def _autocorr1(arr: np.ndarray) -> np.ndarray:
    m = arr.mean(axis=1, keepdims=True)
    c = arr - m
    var = np.sum(c ** 2, axis=1)
    var = np.where(var < 1e-12, 1.0, var)
    return np.sum(c[:, :-1] * c[:, 1:], axis=1) / var


def _fft_features(arr: np.ndarray) -> Dict[str, np.ndarray]:
    fft_vals = np.abs(np.fft.rfft(arr, axis=1))
    feats: Dict[str, np.ndarray] = {}
    feats["fft_mean"] = fft_vals.mean(axis=1)
    feats["fft_std"] = fft_vals.std(axis=1)
    feats["fft_max"] = fft_vals.max(axis=1)
    feats["fft_energy"] = np.sum(fft_vals ** 2, axis=1)
    psd = fft_vals ** 2
    psd_s = psd.sum(axis=1, keepdims=True)
    psd_s = np.where(psd_s < 1e-12, 1.0, psd_s)
    psd_n = psd / psd_s
    psd_n = np.where(psd_n < 1e-12, 1e-12, psd_n)
    feats["fft_entropy"] = -np.sum(psd_n * np.log(psd_n), axis=1)
    return feats


# ---------------------------------------------------------------------------
# Phase-specific feature set
# ---------------------------------------------------------------------------

def _compute_phase_features(
    windows: np.ndarray, imu_columns: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Rich features + half-window delta + gradient features.

    Designed for detecting phase transitions within a window.
    """
    feats: Dict[str, np.ndarray] = {}
    n, t, c = windows.shape
    half = t // 2

    for ci, col in enumerate(imu_columns):
        ch = windows[:, :, ci]  # (N, T)

        # --- Standard stats ---
        feats[f"{col}_mean"] = ch.mean(axis=1)
        feats[f"{col}_std"] = ch.std(axis=1)
        feats[f"{col}_min"] = ch.min(axis=1)
        feats[f"{col}_max"] = ch.max(axis=1)
        feats[f"{col}_median"] = np.median(ch, axis=1)
        feats[f"{col}_rms"] = _rms(ch)
        feats[f"{col}_iqr"] = _iqr(ch)
        feats[f"{col}_zcr"] = _zcr(ch)
        feats[f"{col}_autocorr1"] = _autocorr1(ch)

        # --- FFT ---
        for k, v in _fft_features(ch).items():
            feats[f"{col}_{k}"] = v

        # --- Half-window delta features (key for transition detection) ---
        first_half = ch[:, :half]
        second_half = ch[:, half:]
        feats[f"{col}_delta_mean"] = second_half.mean(axis=1) - first_half.mean(axis=1)
        feats[f"{col}_delta_std"] = second_half.std(axis=1) - first_half.std(axis=1)
        feats[f"{col}_delta_energy"] = (
            np.sum(second_half ** 2, axis=1) - np.sum(first_half ** 2, axis=1)
        )

        # --- Gradient (first derivative) features ---
        grad = np.diff(ch, axis=1)  # (N, T-1)
        feats[f"{col}_grad_mean"] = grad.mean(axis=1)
        feats[f"{col}_grad_std"] = grad.std(axis=1)
        feats[f"{col}_grad_max"] = grad.max(axis=1)
        feats[f"{col}_grad_min"] = grad.min(axis=1)
        feats[f"{col}_grad_abs_mean"] = np.abs(grad).mean(axis=1)

        # --- Linear slope (movement direction — key for ecc vs con) ---
        # Fit y = slope * t + intercept for each window via least squares
        t_axis = np.arange(t, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = np.sum((t_axis - t_mean) ** 2)
        ch_mean = ch.mean(axis=1, keepdims=True)
        slope = np.sum((t_axis[None, :] - t_mean) * (ch - ch_mean), axis=1) / t_var
        feats[f"{col}_slope"] = slope

        # --- Temporal ratio: end vs start (another direction indicator) ---
        quarter = max(1, t // 4)
        feats[f"{col}_end_start_ratio"] = (
            ch[:, -quarter:].mean(axis=1) - ch[:, :quarter].mean(axis=1)
        )

    # --- Inter-axis correlations ---
    for i in range(c):
        for j in range(i + 1, c):
            ci_d = windows[:, :, i]
            cj_d = windows[:, :, j]
            ci_m = ci_d - ci_d.mean(axis=1, keepdims=True)
            cj_m = cj_d - cj_d.mean(axis=1, keepdims=True)
            num = np.sum(ci_m * cj_m, axis=1)
            den = np.sqrt(np.sum(ci_m ** 2, axis=1) * np.sum(cj_m ** 2, axis=1))
            den = np.where(den < 1e-12, 1.0, den)
            feats[f"corr_{imu_columns[i]}_{imu_columns[j]}"] = num / den

    # --- Magnitude features ---
    acc_idx = [i for i, c in enumerate(imu_columns) if c.startswith("a")]
    gyro_idx = [i for i, c in enumerate(imu_columns) if c.startswith("g")]
    for name, idx_list in [("acc_mag", acc_idx), ("gyro_mag", gyro_idx)]:
        if len(idx_list) >= 2:
            mag = np.sqrt(np.sum(windows[:, :, idx_list] ** 2, axis=2))
            feats[f"{name}_mean"] = mag.mean(axis=1)
            feats[f"{name}_std"] = mag.std(axis=1)
            feats[f"{name}_max"] = mag.max(axis=1)
            feats[f"{name}_min"] = mag.min(axis=1)
            feats[f"{name}_rms"] = _rms(mag)
            # Magnitude gradient
            mag_grad = np.diff(mag, axis=1)
            feats[f"{name}_grad_mean"] = mag_grad.mean(axis=1)
            feats[f"{name}_grad_std"] = mag_grad.std(axis=1)

    return feats


def windows_to_phase_features(
    windows: np.ndarray, imu_columns: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(_compute_phase_features(windows, imu_columns))


def normalize_rep_for_phase(df: pd.DataFrame, imu_columns: Sequence[str]) -> pd.DataFrame:
    """Normalize one rep independently before phase inference."""
    out = df.copy()
    for col in imu_columns:
        mean = out[col].mean()
        std = out[col].std()
        if std < 1e-8:
            std = 1.0
        out[col] = (out[col] - mean) / std
    return out


def rep_to_phase_feature_rows(
    rep_df: pd.DataFrame,
    imu_columns: Sequence[str],
    action_type: str,
    window_size: int,
    stride_size: int,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Convert one already-cropped rep into model feature rows.

    This is the inference-time helper used after rep segmentation. It does not
    require ground-truth phase labels; the full provided rep is treated as the
    active region.
    """
    if len(rep_df) < window_size:
        return pd.DataFrame(), []

    x = rep_df[list(imu_columns)].to_numpy(dtype=np.float32)
    windows, meta = [], []
    for start in range(0, len(rep_df) - window_size + 1, stride_size):
        end = start + window_size
        center = start + window_size // 2
        windows.append(x[start:end])
        meta.append(
            {
                "start": start,
                "end": end,
                "center": center,
                "active_length": len(rep_df),
                "rel_pos": center / max(len(rep_df) - 1, 1),
            }
        )

    if not windows:
        return pd.DataFrame(), []

    features = windows_to_phase_features(np.stack(windows), imu_columns)
    features["action_type"] = pd.Categorical([action_type] * len(features))
    features["rel_pos"] = [row["rel_pos"] for row in meta]
    return features, meta


def _transition_from_labels(centers: Sequence[int], labels: Sequence[str]) -> float:
    """Estimate transition sample from center-labelled window predictions."""
    if len(labels) < 2:
        return float("nan")
    for i in range(1, len(labels)):
        if str(labels[i]) != str(labels[i - 1]):
            return float((centers[i - 1] + centers[i]) / 2.0)
    return float("nan")


def segment_rep_phases(
    rep_df: pd.DataFrame,
    predictor,
    imu_columns: Sequence[str],
    action_type: str,
    window_size: int,
    stride_size: int,
) -> Dict[str, object]:
    """Predict the eccentric/concentric boundary for one cropped rep."""
    rep_norm = normalize_rep_for_phase(rep_df, imu_columns)
    feature_df, meta = rep_to_phase_feature_rows(
        rep_norm,
        imu_columns,
        action_type,
        window_size,
        stride_size,
    )
    if feature_df.empty:
        return {"transition_idx": float("nan"), "centers": [], "labels": []}

    pred = predictor.predict(feature_df, as_pandas=True).astype(str).tolist()
    centers = [int(row["center"]) for row in meta]
    transition_idx = _transition_from_labels(centers, pred)
    return {"transition_idx": transition_idx, "centers": centers, "labels": pred}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _save_evaluation(predictor, df_test, label_col, output_dir):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_true = df_test[label_col].values
    y_pred = predictor.predict(df_test.drop(columns=[label_col]), as_pandas=True).values
    labels_sorted = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    report_text = classification_report(y_true, y_pred, labels=labels_sorted)
    report_dict = classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True)
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2), encoding="utf-8"
    )

    try:
        importance = predictor.feature_importance(df_test, silent=True)
        importance.to_csv(output_dir / "feature_importance.csv")
    except Exception as e:
        print(f"[WARN] feature importance failed: {e}")

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    print("\n" + "=" * 60)
    print("PHASE CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)
    return overall


def _save_per_rep_transition_evaluation(
    predictor,
    df_test_features: pd.DataFrame,
    meta: Sequence[Dict],
    output_dir: Path,
    sample_rate_hz: float,
) -> Dict[str, float]:
    pred = predictor.predict(df_test_features, as_pandas=True).astype(str).tolist()
    rows: List[Dict[str, object]] = []

    grouped: Dict[str, List[Tuple[Dict, str]]] = {}
    for meta_row, pred_label in zip(meta, pred):
        grouped.setdefault(str(meta_row["source_id"]), []).append((meta_row, pred_label))

    for source_id, items in grouped.items():
        items = sorted(items, key=lambda pair: int(pair[0]["center"]))
        centers = [int(pair[0]["center"]) for pair in items]
        labels = [pair[1] for pair in items]
        first = items[0][0]
        gt_transition = float(first["gt_transition_idx"])
        pred_transition = _transition_from_labels(centers, labels)
        error_samples = (
            abs(pred_transition - gt_transition)
            if np.isfinite(pred_transition)
            else float("nan")
        )
        rows.append(
            {
                "source_id": source_id,
                "subject_id": first["subject_id"],
                "action_type": first["action_type"],
                "active_length": first["active_length"],
                "gt_transition_idx": gt_transition,
                "pred_transition_idx": pred_transition,
                "transition_abs_error_samples": error_samples,
                "transition_abs_error_ms": error_samples / sample_rate_hz * 1000.0
                if np.isfinite(error_samples) and sample_rate_hz > 0
                else float("nan"),
                "n_windows": len(items),
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "rep_transition_metrics.csv", index=False)
    valid = metrics_df["transition_abs_error_ms"].dropna() if not metrics_df.empty else pd.Series(dtype=float)
    summary = {
        "n_reps": int(len(metrics_df)),
        "n_reps_with_prediction": int(valid.shape[0]),
        "transition_mae_ms": float(valid.mean()) if len(valid) else float("nan"),
        "transition_median_ae_ms": float(valid.median()) if len(valid) else float("nan"),
    }
    (output_dir / "rep_transition_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _get_timestamped_dir(base_dir: Path) -> Path:
    """Create a timestamped subdirectory for organizing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_phase(config_path: Path, dry_run: bool = False, use_timestamp: bool = True) -> None:
    raw_cfg, feature_cfg, sample_rate_hz, phase_cfg = build_configs(config_path)

    seed = int(raw_cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = raw_cfg.get("data", {})
    io_cfg = raw_cfg.get("io", {})

    data_dir = Path(data_cfg.get("data_dir", "./data"))
    csv_glob = data_cfg.get("csv_glob", "*.csv")
    exclude_patterns = data_cfg.get("exclude_patterns", None)
    include_actions = data_cfg.get("include_actions", None)
    base_output_dir = Path(io_cfg.get("phase_output_dir", "./artifacts/phase_segmentation"))

    # Create timestamped subdirectory
    if use_timestamp:
        output_dir = _get_timestamped_dir(base_output_dir)
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")

    # Window params
    ws = int(phase_cfg.window_seconds * sample_rate_hz)
    ss = int(phase_cfg.stride_seconds * sample_rate_hz)
    print(f"[INFO] Phase window: {phase_cfg.window_seconds}s = {ws} samples, "
          f"stride: {phase_cfg.stride_seconds}s = {ss} samples")

    # --- Load ---
    sequences, subjects = prepare_sequences_from_folder(
        data_dir=data_dir,
        feature_cfg=feature_cfg,
        sample_rate_hz=sample_rate_hz,
        csv_glob=csv_glob,
        exclude_patterns=exclude_patterns,
        include_actions=include_actions,
    )
    print(f"[INFO] Loaded {len(sequences)} sequences from {len(set(subjects))} subjects")

    # --- Subject split: leave-one-out (3 train, 1 test) ---
    import random as _rng
    unique_subj = sorted(set(subjects))
    rng = _rng.Random(seed)
    rng.shuffle(unique_subj)
    test_subj = [unique_subj[-1]]
    train_subj = unique_subj[:-1]
    print(f"[INFO] train subjects: {train_subj}, test subject: {test_subj}")

    train_seqs = filter_sequences_by_subject(sequences, train_subj, feature_cfg.subject_column)
    test_seqs = filter_sequences_by_subject(sequences, test_subj, feature_cfg.subject_column)
    del sequences
    gc.collect()

    # Each labelled rep is normalized independently. This matches inference,
    # where rep segmentation will hand this model one cropped rep at a time.
    train_seqs = [normalize_rep_for_phase(s, feature_cfg.imu_columns) for s in train_seqs]
    test_seqs = [normalize_rep_for_phase(s, feature_cfg.imu_columns) for s in test_seqs]

    # --- Sliding windows with phase label ---
    phase_col = "phase"
    x_train, y_train, a_train, m_train = _collect_phase_windows(
        train_seqs, feature_cfg.imu_columns, phase_col,
        feature_cfg.subject_column, feature_cfg.label_column, ws, ss)
    x_test, y_test, a_test, m_test = _collect_phase_windows(
        test_seqs, feature_cfg.imu_columns, phase_col,
        feature_cfg.subject_column, feature_cfg.label_column, ws, ss)
    del train_seqs, test_seqs
    gc.collect()

    print(f"[INFO] Per-rep phase windows: train={x_train.shape}, test={x_test.shape} (subject split)")

    # Class distribution
    for name, y in [("train", y_train), ("test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join(f"{u}={c}" for u, c in zip(unique, counts))
        print(f"[INFO] {name} class dist: {dist}")

    # --- Features ---
    df_train = windows_to_phase_features(x_train, feature_cfg.imu_columns)
    df_test = windows_to_phase_features(x_test, feature_cfg.imu_columns)
    del x_train, x_test
    gc.collect()

    # Add action_type as categorical feature (phase patterns differ by exercise)
    df_train["action_type"] = pd.Categorical(a_train)
    df_test["action_type"] = pd.Categorical(a_test)

    # Add relative position within the rep (0=start, 1=end)
    # Concentric is always the first phase, eccentric the second.
    df_train["rel_pos"] = [m["rel_pos"] for m in m_train]
    df_test["rel_pos"] = [m["rel_pos"] for m in m_test]

    label_col = "label"
    df_train[label_col] = y_train
    df_test[label_col] = y_test

    n_feats = df_train.shape[1] - 1
    print(f"[INFO] feature_mode = {phase_cfg.feature_mode}, feature dim = {n_feats}")

    (output_dir / "dataset_shapes.json").write_text(
        json.dumps({
            "n_train": len(df_train),
            "n_test": len(df_test),
            "n_features": n_feats,
            "window_size": ws,
            "stride_size": ss,
            "unit": "one active repetition",
            "phase_map": {k: v for k, v in PHASE_MAP.items() if v is not None},
            "train_subjects": train_subj,
            "test_subjects": test_subj,
        }, indent=2),
        encoding="utf-8",
    )

    if dry_run:
        df_train.head(200).to_csv(output_dir / "train_preview.csv", index=False)
        pd.DataFrame(m_train).head(200).to_csv(output_dir / "train_meta_preview.csv", index=False)
        print(f"[DRY RUN] df_train shape: {df_train.shape}")
        print(f"[DRY RUN] preview saved to {output_dir / 'train_preview.csv'}")
        return

    # --- AutoGluon ---
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        raise RuntimeError("AutoGluon not installed.") from e

    predictor_path = str(output_dir / "models")
    if Path(predictor_path).exists():
        print(f"[INFO] Removing old {predictor_path}")
        shutil.rmtree(predictor_path, ignore_errors=True)

    fit_kwargs: Dict = {
        "presets": phase_cfg.presets,
        "time_limit": int(phase_cfg.time_limit_s) if phase_cfg.time_limit_s else None,
    }
    if phase_cfg.num_cpus is not None:
        fit_kwargs["num_cpus"] = int(phase_cfg.num_cpus)
    if phase_cfg.included_model_types:
        fit_kwargs["included_model_types"] = phase_cfg.included_model_types
    if phase_cfg.excluded_model_types:
        fit_kwargs["excluded_model_types"] = phase_cfg.excluded_model_types
    if phase_cfg.num_bag_folds > 0:
        fit_kwargs["num_bag_folds"] = phase_cfg.num_bag_folds
    if phase_cfg.num_stack_levels > 0:
        fit_kwargs["num_stack_levels"] = phase_cfg.num_stack_levels
    fit_kwargs["ag_args_fit"] = {"num_gpus": 0}

    print(f"[INFO] AutoGluon fit_kwargs: {fit_kwargs}")

    predictor = TabularPredictor(
        label=label_col,
        path=predictor_path,
        problem_type="binary",
        eval_metric=phase_cfg.eval_metric,
    ).fit(
        train_data=df_train,
        **{k: v for k, v in fit_kwargs.items() if v is not None},
    )

    # --- Evaluation ---
    leaderboard = predictor.leaderboard(df_test, silent=True)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    print("\n" + "=" * 60)
    print("LEADERBOARD (test set)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))

    overall = _save_evaluation(predictor, df_test, label_col, output_dir)
    transition_summary = _save_per_rep_transition_evaluation(
        predictor,
        df_test.drop(columns=[label_col]),
        m_test,
        output_dir,
        float(sample_rate_hz),
    )

    summary = {
        "seed": seed,
        "feature_mode": phase_cfg.feature_mode,
        "window_seconds": phase_cfg.window_seconds,
        "stride_seconds": phase_cfg.stride_seconds,
        "presets": phase_cfg.presets,
        "n_features": n_feats,
        "train_subjects": train_subj,
        "test_subjects": test_subj,
        "test_metrics": overall,
        "rep_transition_metrics": transition_summary,
        "best_model": predictor.model_best,
    }
    (output_dir / "phase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    standard_overall = dict(overall)
    standard_overall.update(
        {
            "transition_mae_ms": transition_summary.get("transition_mae_ms", float("nan")),
            "transition_median_abs_error_ms": transition_summary.get("transition_median_abs_error_ms", float("nan")),
        }
    )
    write_standard_run_outputs(
        output_dir,
        task="phase_segmentation",
        model_name=f"autogluon_{predictor.model_best}",
        title="Phase Segmentation Report",
        overall=standard_overall,
        details={"legacy_summary": summary},
        artifacts={
            "Legacy summary": (output_dir / "phase_summary.json").as_posix(),
            "Leaderboard": (output_dir / "leaderboard.csv").as_posix(),
            "Confusion matrix": (output_dir / "confusion_matrix.csv").as_posix(),
            "Classification report": (output_dir / "classification_report.json").as_posix(),
            "Rep transition metrics": (output_dir / "rep_transition_metrics.csv").as_posix(),
            "Models": (output_dir / "models").as_posix(),
        },
        config={
            "feature_mode": phase_cfg.feature_mode,
            "window_seconds": phase_cfg.window_seconds,
            "stride_seconds": phase_cfg.stride_seconds,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
        },
        config_path=config_path,
        notes=[
            "Standardized report added for cross-model comparison; original phase artifacts are unchanged.",
        ],
    )

    print(f"\nbest_model: {summary['best_model']}")
    print(f"test accuracy: {overall['accuracy']:.4f}")
    print(f"test macro_f1: {overall['macro_f1']:.4f}")
    print(f"rep transition MAE: {transition_summary['transition_mae_ms']:.1f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase segmentation training")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_phase(args.config, dry_run=args.dry_run, use_timestamp=not args.no_timestamp)


if __name__ == "__main__":
    main()
