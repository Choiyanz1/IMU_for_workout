from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from evaluation.reporting import write_standard_run_outputs
from preprocessing.window_pipeline import (
    WindowConfig,
    apply_zscore,
    compute_train_stats,
    extract_windows,
    set_seed,
    split_subjects,
)


FeatureMode = Literal["stats", "flatten", "rich"]


@dataclass
class AutoGluonConfig:
    feature_mode: FeatureMode = "rich"
    presets: str = "medium_quality_faster_train"
    time_limit_s: int = 600
    eval_metric: str = "accuracy"
    num_cpus: int | None = None
    # Which model families to include (None = all AutoGluon defaults)
    included_model_types: List[str] | None = None
    # Which model families to exclude
    excluded_model_types: List[str] | None = None
    # Number of folds for bagging (0 = no bagging)
    num_bag_folds: int = 0
    # Stack levels (0 = no stacking, 1 = single stack)
    num_stack_levels: int = 0
    # Memory limit in GB (None = no limit)
    memory_limit_gb: float | None = None


def build_configs(config_path: Path) -> tuple[dict, FeatureConfig, WindowConfig, AutoGluonConfig]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    feature_cfg = FeatureConfig(**raw.get("feature", {}))
    window_cfg = WindowConfig(**raw.get("window", {}))
    ag_cfg = AutoGluonConfig(**raw.get("autogluon", {}))

    return raw, feature_cfg, window_cfg, ag_cfg


def _collect_windows_as_xy(
    sequences: Sequence[pd.DataFrame],
    feature_cfg: FeatureConfig,
    window_cfg: WindowConfig,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    windows_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    meta_all: List[Dict[str, str]] = []

    for seq in sequences:
        w, y, meta = extract_windows(
            df=seq,
            imu_columns=feature_cfg.imu_columns,
            label_column=feature_cfg.label_column,
            subject_column=feature_cfg.subject_column,
            window_cfg=window_cfg,
        )
        if len(w) == 0:
            continue
        windows_all.append(w)
        labels_all.append(y.astype(str))
        meta_all.extend(meta)

    if not windows_all:
        raise RuntimeError("No windows extracted. Check sample rate, window length, and CSV quality.")

    x = np.concatenate(windows_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    return x, y, meta_all


def _build_rep_features(
    sequences: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    label_column: str,
    feature_mode: FeatureMode,
) -> pd.DataFrame:
    """Extract features from each sequence (rep) as a whole unit.

    Each CSV file = 1 rep = 1 row in the output DataFrame.
    Variable-length sequences are each treated as a single (1, T_i, C) window.
    """
    if feature_mode == "flatten":
        raise ValueError("flatten mode not supported for per-rep features (variable length)")

    rows: List[Dict[str, float]] = []
    labels: List[str] = []

    for seq in sequences:
        data = seq[list(imu_columns)].to_numpy(dtype=np.float32)  # (T, C)
        if len(data) < 2:
            continue
        window = data[np.newaxis, :, :]  # (1, T, C)

        if feature_mode == "rich":
            feats = _compute_rich_features(window, imu_columns)
        else:
            feats = _compute_stats_features(window, imu_columns)

        rows.append({k: float(v[0]) for k, v in feats.items()})
        labels.append(str(seq.iloc[0][label_column]))

    if not rows:
        raise RuntimeError("No valid sequences for feature extraction.")

    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


def _zero_crossing_rate(arr: np.ndarray) -> np.ndarray:
    """Per-window zero-crossing rate.  arr shape (N, T)."""
    signs = np.sign(arr)
    diff = np.abs(np.diff(signs, axis=1))
    return diff.sum(axis=1) / (arr.shape[1] - 1)


def _rms(arr: np.ndarray) -> np.ndarray:
    """Root mean square along time axis.  arr shape (N, T)."""
    return np.sqrt(np.mean(arr ** 2, axis=1))


def _iqr(arr: np.ndarray) -> np.ndarray:
    """Inter-quartile range along time axis.  arr shape (N, T)."""
    q75 = np.percentile(arr, 75, axis=1)
    q25 = np.percentile(arr, 25, axis=1)
    return q75 - q25


def _autocorr_lag1(arr: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation per window.  arr shape (N, T)."""
    m = arr.mean(axis=1, keepdims=True)
    centered = arr - m
    var = np.sum(centered ** 2, axis=1)
    var = np.where(var < 1e-12, 1.0, var)
    ac = np.sum(centered[:, :-1] * centered[:, 1:], axis=1) / var
    return ac


def _fft_features(arr: np.ndarray, top_k: int = 3) -> Dict[str, np.ndarray]:
    """Compute FFT-based features per window.  arr shape (N, T)."""
    fft_vals = np.abs(np.fft.rfft(arr, axis=1))
    n_freq = fft_vals.shape[1]
    feats: Dict[str, np.ndarray] = {}
    feats["fft_mean"] = fft_vals.mean(axis=1)
    feats["fft_std"] = fft_vals.std(axis=1)
    feats["fft_max"] = fft_vals.max(axis=1)
    feats["fft_argmax"] = fft_vals.argmax(axis=1).astype(np.float32)
    # Spectral energy
    feats["fft_energy"] = np.sum(fft_vals ** 2, axis=1)
    # Spectral entropy
    psd = fft_vals ** 2
    psd_sum = psd.sum(axis=1, keepdims=True)
    psd_sum = np.where(psd_sum < 1e-12, 1.0, psd_sum)
    psd_norm = psd / psd_sum
    psd_norm = np.where(psd_norm < 1e-12, 1e-12, psd_norm)
    feats["fft_entropy"] = -np.sum(psd_norm * np.log(psd_norm), axis=1)
    # Top-K dominant frequencies
    for k in range(min(top_k, n_freq)):
        idx = np.argsort(fft_vals, axis=1)[:, -(k + 1)]
        feats[f"fft_top{k + 1}_freq"] = idx.astype(np.float32)
        feats[f"fft_top{k + 1}_mag"] = np.take_along_axis(fft_vals, idx[:, None], axis=1).squeeze(1)
    return feats


def _compute_stats_features(
    windows: np.ndarray, imu_columns: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Basic statistical features (same as original 'stats' mode)."""
    feats: Dict[str, np.ndarray] = {}
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    vmin = windows.min(axis=1)
    vmax = windows.max(axis=1)
    med = np.median(windows, axis=1)

    for ci, col in enumerate(imu_columns):
        feats[f"{col}_mean"] = mean[:, ci]
        feats[f"{col}_std"] = std[:, ci]
        feats[f"{col}_min"] = vmin[:, ci]
        feats[f"{col}_max"] = vmax[:, ci]
        feats[f"{col}_median"] = med[:, ci]
    return feats


def _compute_rich_features(
    windows: np.ndarray, imu_columns: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Rich feature set: stats + RMS + IQR + ZCR + autocorrelation + FFT + inter-axis correlation."""
    feats = _compute_stats_features(windows, imu_columns)
    n, t, c = windows.shape

    for ci, col in enumerate(imu_columns):
        ch = windows[:, :, ci]  # (N, T)
        feats[f"{col}_rms"] = _rms(ch)
        feats[f"{col}_iqr"] = _iqr(ch)
        feats[f"{col}_zcr"] = _zero_crossing_rate(ch)
        feats[f"{col}_autocorr1"] = _autocorr_lag1(ch)
        feats[f"{col}_skew"] = pd.DataFrame(ch).skew(axis=1).to_numpy(dtype=np.float32)
        feats[f"{col}_kurt"] = pd.DataFrame(ch).kurt(axis=1).to_numpy(dtype=np.float32)
        # FFT features per channel
        fft_f = _fft_features(ch)
        for fname, fval in fft_f.items():
            feats[f"{col}_{fname}"] = fval

    # Inter-axis correlations (pairwise among IMU channels)
    for i in range(c):
        for j in range(i + 1, c):
            ci_data = windows[:, :, i]
            cj_data = windows[:, :, j]
            # Pearson correlation per window
            ci_m = ci_data - ci_data.mean(axis=1, keepdims=True)
            cj_m = cj_data - cj_data.mean(axis=1, keepdims=True)
            num = np.sum(ci_m * cj_m, axis=1)
            den = np.sqrt(np.sum(ci_m ** 2, axis=1) * np.sum(cj_m ** 2, axis=1))
            den = np.where(den < 1e-12, 1.0, den)
            feats[f"corr_{imu_columns[i]}_{imu_columns[j]}"] = num / den

    # Magnitude features (accel & gyro separately if applicable)
    acc_cols = [i for i, c in enumerate(imu_columns) if c.startswith("a")]
    gyro_cols = [i for i, c in enumerate(imu_columns) if c.startswith("g")]
    for name, idx_list in [("acc_mag", acc_cols), ("gyro_mag", gyro_cols)]:
        if len(idx_list) >= 2:
            mag = np.sqrt(np.sum(windows[:, :, idx_list] ** 2, axis=2))  # (N, T)
            feats[f"{name}_mean"] = mag.mean(axis=1)
            feats[f"{name}_std"] = mag.std(axis=1)
            feats[f"{name}_max"] = mag.max(axis=1)
            feats[f"{name}_min"] = mag.min(axis=1)
            feats[f"{name}_rms"] = _rms(mag)

    return feats


def windows_to_features(
    windows: np.ndarray,
    imu_columns: Sequence[str],
    feature_mode: FeatureMode,
) -> pd.DataFrame:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, T, C), got {windows.shape}")

    n, t, c = windows.shape
    if c != len(imu_columns):
        raise ValueError(f"windows channels={c} != len(imu_columns)={len(imu_columns)}")

    if feature_mode == "flatten":
        flat = windows.reshape(n, t * c)
        cols: List[str] = []
        for ti in range(t):
            for ci, col in enumerate(imu_columns):
                cols.append(f"{col}_t{ti:03d}")
        return pd.DataFrame(flat, columns=cols)

    if feature_mode == "stats":
        return pd.DataFrame(_compute_stats_features(windows, imu_columns))

    if feature_mode == "rich":
        return pd.DataFrame(_compute_rich_features(windows, imu_columns))

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _save_detailed_evaluation(
    predictor,
    df_test: pd.DataFrame,
    label_col: str,
    output_dir: Path,
) -> Dict:
    """Generate confusion matrix, per-class metrics, and feature importance."""
    y_true = df_test[label_col].values
    y_pred = predictor.predict(df_test.drop(columns=[label_col]), as_pandas=True).values

    # Per-class classification report
    labels_sorted = sorted(set(y_true) | set(y_pred))
    cls_report = classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True)
    cls_report_text = classification_report(y_true, y_pred, labels=labels_sorted)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Save classification report
    (output_dir / "classification_report.txt").write_text(cls_report_text, encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(cls_report, indent=2), encoding="utf-8"
    )

    # Feature importance (permutation-based)
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
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(cls_report_text)
    print(f"\nConfusion matrix saved to: {output_dir / 'confusion_matrix.csv'}")

    return overall


def _get_timestamped_dir(base_dir: Path) -> Path:
    """Create a timestamped subdirectory for organizing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


def train_action_classification(config_path: Path, dry_run: bool = False, use_timestamp: bool = True) -> None:
    raw_cfg, feature_cfg, window_cfg, ag_cfg = build_configs(config_path)

    seed = int(raw_cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    data_cfg = raw_cfg.get("data", {})
    io_cfg = raw_cfg.get("io", {})

    data_dir = Path(data_cfg.get("data_dir", "./data"))
    csv_glob = data_cfg.get("csv_glob", "*.csv")
    exclude_patterns = data_cfg.get("exclude_patterns", None)
    include_actions = data_cfg.get("include_actions", None)
    base_output_dir = Path(io_cfg.get("output_dir", "./artifacts/action_classification"))

    # Create timestamped subdirectory
    if use_timestamp:
        output_dir = _get_timestamped_dir(base_output_dir)
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")

    sequences, subjects = prepare_sequences_from_folder(
        data_dir=data_dir,
        feature_cfg=feature_cfg,
        sample_rate_hz=window_cfg.sample_rate_hz,
        csv_glob=csv_glob,
        exclude_patterns=exclude_patterns,
        include_actions=include_actions,
        subject_aliases=data_cfg.get("subject_aliases", None),
    )
    print(f"[INFO] Loaded {len(sequences)} sequences from {len(set(subjects))} subjects")

    # Leave-one-out split: 3 subjects train, 1 test
    import random as _rng
    unique_subj = sorted(set(subjects))
    rng = _rng.Random(seed)
    rng.shuffle(unique_subj)
    test_subj = [unique_subj[-1]]
    train_subj = unique_subj[:-1]
    print(f"[INFO] train subjects: {train_subj}, test subject: {test_subj}")

    train_seqs = filter_sequences_by_subject(sequences, train_subj, feature_cfg.subject_column)
    test_seqs = filter_sequences_by_subject(sequences, test_subj, feature_cfg.subject_column)

    stats = compute_train_stats(train_seqs, feature_cfg.imu_columns)
    stats.save(output_dir / "zscore_stats.json")

    train_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in train_seqs]
    test_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in test_seqs]

    # Free raw sequences to reclaim memory
    del sequences
    gc.collect()

    # Per-rep feature extraction: each CSV (rep) → 1 row of features
    print(f"[INFO] feature_mode = {ag_cfg.feature_mode}")
    label_col = "label"

    df_train = _build_rep_features(train_seqs, feature_cfg.imu_columns, feature_cfg.label_column, ag_cfg.feature_mode)
    df_test = _build_rep_features(test_seqs, feature_cfg.imu_columns, feature_cfg.label_column, ag_cfg.feature_mode)

    # Free sequence DataFrames
    del train_seqs, test_seqs
    gc.collect()

    print(f"[INFO] reps: train={len(df_train)}, test={len(df_test)}")
    print(f"[INFO] feature dim = {df_train.shape[1] - 1} (excluding label column)")

    (output_dir / "dataset_shapes.json").write_text(
        json.dumps(
            {
                "n_train": len(df_train),
                "n_test": len(df_test),
                "n_features": int(df_train.shape[1] - 1),
                "feature_mode": ag_cfg.feature_mode,
                "train_subjects": train_subj,
                "test_subjects": test_subj,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if dry_run:
        out_path = output_dir / "train_preview.csv"
        df_train.head(200).to_csv(out_path, index=False)
        print(f"[DRY RUN] wrote preview: {out_path}")
        print(f"[DRY RUN] df_train shape: {df_train.shape}")
        print(f"[DRY RUN] feature columns: {[c for c in df_train.columns if c != label_col]}")
        return

    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "AutoGluon is not installed. Install with: pip install autogluon.tabular\n"
            "Then rerun: python -m train.action_classification --config config.yaml"
        ) from e

    predictor_path = str(output_dir / "models")

    # Clean up stale models from previous failed runs
    if Path(predictor_path).exists():
        print(f"[INFO] Removing old {predictor_path}")
        shutil.rmtree(predictor_path, ignore_errors=True)

    fit_kwargs: Dict = {
        "presets": ag_cfg.presets,
        "time_limit": int(ag_cfg.time_limit_s) if ag_cfg.time_limit_s else None,
    }
    ag_args_fit: Dict = {"num_gpus": 0}
    if ag_cfg.num_cpus is not None:
        fit_kwargs["num_cpus"] = int(ag_cfg.num_cpus)
    if ag_cfg.included_model_types:
        fit_kwargs["included_model_types"] = ag_cfg.included_model_types
    if ag_cfg.excluded_model_types:
        fit_kwargs["excluded_model_types"] = ag_cfg.excluded_model_types
    if ag_cfg.num_bag_folds > 0:
        fit_kwargs["num_bag_folds"] = ag_cfg.num_bag_folds
    if ag_cfg.num_stack_levels > 0:
        fit_kwargs["num_stack_levels"] = ag_cfg.num_stack_levels
    fit_kwargs["ag_args_fit"] = ag_args_fit

    print(f"[INFO] AutoGluon fit_kwargs: {fit_kwargs}")

    predictor = TabularPredictor(
        label=label_col,
        path=predictor_path,
        problem_type="multiclass",
        eval_metric=ag_cfg.eval_metric,
    ).fit(
        train_data=df_train,
        **{k: v for k, v in fit_kwargs.items() if v is not None},
    )

    # --- Leaderboard ---
    leaderboard = predictor.leaderboard(df_test, silent=True)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    print("\n" + "=" * 60)
    print("LEADERBOARD (test set)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))

    # --- Detailed evaluation ---
    overall_metrics = _save_detailed_evaluation(predictor, df_test, label_col, output_dir)

    # --- Predict probabilities for distillation ---
    prob_train = predictor.predict_proba(df_train.drop(columns=[label_col]), as_pandas=True)
    prob_train.to_csv(output_dir / "train_soft_labels.csv", index=False)
    prob_test = predictor.predict_proba(df_test.drop(columns=[label_col]), as_pandas=True)
    prob_test.to_csv(output_dir / "test_soft_labels.csv", index=False)

    summary = {
        "seed": seed,
        "feature_mode": ag_cfg.feature_mode,
        "presets": ag_cfg.presets,
        "time_limit_s": ag_cfg.time_limit_s,
        "eval_metric": ag_cfg.eval_metric,
        "num_bag_folds": ag_cfg.num_bag_folds,
        "num_stack_levels": ag_cfg.num_stack_levels,
        "n_features": int(df_train.shape[1] - 1),
        "train_subjects": train_subj,
        "test_subjects": test_subj,
        "test_metrics": overall_metrics,
        "best_model": predictor.model_best,
        "model_names": predictor.model_names(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_standard_run_outputs(
        output_dir,
        task="action_classification",
        model_name=f"autogluon_{predictor.model_best}",
        title="Action Classification Report",
        overall=overall_metrics,
        details={"legacy_summary": summary},
        artifacts={
            "Legacy summary": (output_dir / "summary.json").as_posix(),
            "Leaderboard": (output_dir / "leaderboard.csv").as_posix(),
            "Confusion matrix": (output_dir / "confusion_matrix.csv").as_posix(),
            "Classification report": (output_dir / "classification_report.json").as_posix(),
            "Models": (output_dir / "models").as_posix(),
            "Train soft labels": (output_dir / "train_soft_labels.csv").as_posix(),
            "Test soft labels": (output_dir / "test_soft_labels.csv").as_posix(),
        },
        config={
            "feature_mode": ag_cfg.feature_mode,
            "eval_metric": ag_cfg.eval_metric,
            "train_subjects": train_subj,
            "test_subjects": test_subj,
        },
        config_path=config_path,
        notes=[
            "Standardized report added for cross-model comparison; original AutoGluon artifacts are unchanged.",
        ],
    )

    print(f"\nbest_model: {summary['best_model']}")
    print(f"test accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"test macro_f1: {overall_metrics['macro_f1']:.4f}")
    print(f"leaderboard saved to: {output_dir / 'leaderboard.csv'}")
    print(f"soft labels saved to: {output_dir / 'train_soft_labels.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Action classification training with AutoGluon tabular models on windowed IMU data.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only build features and write a small preview CSV")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_action_classification(args.config, dry_run=args.dry_run, use_timestamp=not args.no_timestamp)


if __name__ == "__main__":
    main()
