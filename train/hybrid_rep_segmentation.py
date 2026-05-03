"""Hybrid rep segmentation: SDTW candidates + AutoGluon binary classifier.

For each Leave-One-Subject-Out fold and each action:
  1. Build SDTW templates from the training subjects.
  2. Generate raw SDTW candidates (no NMS, optionally relaxed cost threshold)
     on every training subject's set streams. Each candidate becomes one row of
     features; the label is 1 if its best IoU vs ground-truth >= label_iou.
  3. Train an AutoGluon TabularPredictor (binary classifier) on this dataset.
  4. Apply the classifier to the held-out subject's streams: predict P(rep),
     drop candidates below `prob_threshold`, then NMS by descending probability.
  5. Score against ground truth at the same IoU threshold used elsewhere.

Outputs (timestamped folder under io.rep_segmentation_output_dir/hybrid/):
  models/{action}/{test_subject}/    AutoGluon predictor for that fold
  metrics/stream_metrics.csv          per-stream precision/recall/F1
  metrics/summary.json                overall + per-action + per-subject
  detections/detections.csv           kept detections after classifier + NMS
  candidates/labeled_candidates.csv   all training candidates with labels
  templates/templates.csv             SDTW template metadata per fold
  plots/{action}/{subject}/*.svg      same SVG style as evaluation/rep_segmentation
  metadata/                           run manifest + config snapshot

Usage:
  python -m train.hybrid_rep_segmentation --config configs/hybrid_rep_segmentation.yaml
  python -m train.hybrid_rep_segmentation --config configs/hybrid_rep_segmentation.yaml --mode whole
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.rep_segmentation import (
    _aggregate_metrics,
    _detection_rows,
    _load_config,
    _load_rep_csvs,
    _load_set_streams,
    _load_whole_streams,
    _prepare_output_dirs,
    _safe_name,
    _subject_dirs,
    _truth_segments_for_stream,
    _write_segmentation_svg,
)
from preprocessing.hybrid_rep_features import (
    best_iou_per_candidate,
    best_truth_match,
    compute_candidate_features,
    label_candidates_by_iou,
)
from preprocessing.sdtw_rep_segmentation import (
    SDTWConfig,
    SegmentDetection,
    _nms_by_cost,
    fit_sdtw_templates,
    generate_candidates_for_templates,
    infer_sample_rate_hz,
    summarize_detection_metrics,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig:
    label_iou: float = 0.50
    # `prob_threshold` is only used as a fallback / hard floor. When
    # `use_calibrated_threshold` is True (default), the AutoGluon-calibrated
    # decision_threshold from each fold's tuning set is preferred.
    prob_threshold: float = 0.50
    use_calibrated_threshold: bool = True
    cost_threshold_scale: float = 1.5
    nms_iou: float = 0.20
    presets: str = "medium_quality_faster_train"
    time_limit_s: int = 300
    eval_metric: str = "f1"
    excluded_model_types: List[str] = field(default_factory=lambda: ["NN_TORCH", "FASTAI", "GBM", "XGB"])
    included_model_types: List[str] | None = None
    num_bag_folds: int = 0
    num_stack_levels: int = 0
    num_cpus: int | None = None
    memory_limit_gb: float | None = None
    # Disk-saving knobs (each AutoGluon predictor can be 100-500 MB; 32 folds
    # therefore quickly grow to multi-GB on a 4-subject x 4-action x 2-mode run).
    keep_only_best: bool = True       # AutoGluon: drop non-winning models after fit
    save_space: bool = True           # AutoGluon: delete intermediate caches
    delete_predictor_after_eval: bool = True  # remove fold dir once test scoring is done
    # Use one held-out *training* subject as AutoGluon's validation set instead
    # of a random 10% holdout. This simulates the outer LOSO at fit time so the
    # classifier's internal model selection / threshold calibration generalises
    # better across subjects. Falls back to random holdout when there are <2
    # train subjects available.
    subject_stratified_validation: bool = True
    # Boundary refinement (third stage)
    enable_boundary_refiner: bool = True
    edge_window_samples: int = 20
    refiner_train_iou: float = 0.30
    refiner_time_limit_s: int = 90
    refiner_max_shift_samples: int = 30


def _build_hybrid_config(raw: Dict) -> HybridConfig:
    section = dict(raw.get("hybrid", {}) or {})
    return HybridConfig(**section)


# ---------------------------------------------------------------------------
# Candidate dataset construction
# ---------------------------------------------------------------------------


def _build_labeled_rows(
    stream_id: str,
    subject: str,
    action: str,
    stream_df: pd.DataFrame,
    candidates: Sequence[SegmentDetection],
    imu_columns: Sequence[str],
    sample_rate_hz: float,
    label_iou: float,
    edge_window_samples: int = 0,
) -> List[Dict[str, object]]:
    truth = _truth_segments_for_stream(stream_df)
    if not candidates:
        return []
    labels = label_candidates_by_iou(candidates, truth, label_iou)
    rows: List[Dict[str, object]] = []
    for cand, label in zip(candidates, labels):
        best_iou, matched = best_truth_match(cand, truth)
        feats = compute_candidate_features(
            stream_df, cand, imu_columns, sample_rate_hz,
            edge_window_samples=edge_window_samples,
        )
        truth_start = matched[0] if matched is not None else int(cand.start_idx)
        truth_end = matched[1] if matched is not None else int(cand.end_idx)
        feats.update(
            {
                "subject_id": subject,
                "action_type": action,
                "stream_id": stream_id,
                "start_idx": int(cand.start_idx),
                "end_idx": int(cand.end_idx),
                "best_iou": float(best_iou),
                "truth_start_idx": int(truth_start),
                "truth_end_idx": int(truth_end),
                "delta_start_samples": int(truth_start) - int(cand.start_idx),
                "delta_end_samples": int(truth_end) - int(cand.end_idx),
                "label": int(label),
            }
        )
        rows.append(feats)
    return rows


# ---------------------------------------------------------------------------
# AutoGluon helpers
# ---------------------------------------------------------------------------


_NON_FEATURE_COLS = {
    "subject_id", "action_type", "stream_id",
    "start_idx", "end_idx", "best_iou", "label",
    "truth_start_idx", "truth_end_idx",
    "delta_start_samples", "delta_end_samples",
}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _pick_validation_subject(train_rows: pd.DataFrame) -> str | None:
    """Pick the train subject most representative of the overall train pool.

    We choose the subject whose own positive rate is closest to the overall
    pool's positive rate. The intuition: AutoGluon's internal model selection
    and decision-threshold calibration should be done on a fold whose label
    distribution mirrors what the model will see at test time. Picking the
    'median' subject avoids letting an extreme subject (very high or very low
    pos rate) bias the calibration.
    """
    candidates = []
    overall_pos_rate = float(train_rows["label"].mean())
    for subj in sorted(train_rows["subject_id"].dropna().unique().tolist()):
        val_part = train_rows[train_rows["subject_id"] == subj]
        train_part = train_rows[train_rows["subject_id"] != subj]
        if len(val_part) < 50 or len(train_part) < 50:
            continue
        if val_part["label"].nunique() < 2 or train_part["label"].nunique() < 2:
            continue
        pos_rate = float(val_part["label"].mean())
        candidates.append((abs(pos_rate - overall_pos_rate), subj))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _train_predictor(
    train_rows: pd.DataFrame,
    cfg: HybridConfig,
    output_path: Path,
):
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "AutoGluon is not installed. `pip install autogluon.tabular`"
        ) from exc

    feature_cols = _feature_columns(train_rows)
    full = train_rows[feature_cols + ["label", "subject_id"]].copy()
    full["label"] = full["label"].astype(int)
    if full["label"].nunique() < 2:
        raise RuntimeError(
            f"Training data has a single class only ({int(full['label'].iloc[0])}); "
            "increase cost_threshold_scale or label_iou tolerance."
        )

    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fit_kwargs: Dict = {
        "presets": cfg.presets,
        "time_limit": int(cfg.time_limit_s) if cfg.time_limit_s else None,
        "ag_args_fit": {"num_gpus": 0},
    }
    if cfg.num_cpus is not None:
        fit_kwargs["num_cpus"] = int(cfg.num_cpus)
    if cfg.included_model_types:
        fit_kwargs["included_model_types"] = list(cfg.included_model_types)
    if cfg.excluded_model_types:
        fit_kwargs["excluded_model_types"] = list(cfg.excluded_model_types)
    if cfg.num_bag_folds and cfg.num_bag_folds > 0:
        fit_kwargs["num_bag_folds"] = int(cfg.num_bag_folds)
    if cfg.num_stack_levels and cfg.num_stack_levels > 0:
        fit_kwargs["num_stack_levels"] = int(cfg.num_stack_levels)

    train_data: pd.DataFrame
    tuning_data: pd.DataFrame | None = None
    if cfg.subject_stratified_validation:
        val_subject = _pick_validation_subject(full)
        if val_subject is not None:
            tuning_data = full[full["subject_id"] == val_subject][feature_cols + ["label"]].copy()
            train_data = full[full["subject_id"] != val_subject][feature_cols + ["label"]].copy()
            print(f"[INFO]   subject-stratified val: held-out={val_subject} "
                  f"val_rows={len(tuning_data)} train_rows={len(train_data)} "
                  f"train_pos_rate={train_data['label'].mean():.3f} "
                  f"val_pos_rate={tuning_data['label'].mean():.3f}")
        else:
            train_data = full[feature_cols + ["label"]].copy()
    else:
        train_data = full[feature_cols + ["label"]].copy()

    predictor_kwargs = {"train_data": train_data}
    if tuning_data is not None and len(tuning_data) > 0 and tuning_data["label"].nunique() == 2:
        predictor_kwargs["tuning_data"] = tuning_data
        # When tuning_data is supplied, AutoGluon will not split off a random
        # holdout from train_data, so internal selection now matches LOSO.

    predictor = TabularPredictor(
        label="label",
        path=str(output_path),
        problem_type="binary",
        eval_metric=cfg.eval_metric,
    ).fit(**predictor_kwargs, **{k: v for k, v in fit_kwargs.items() if v is not None})

    # Aggressively shrink the on-disk predictor footprint. With LOSO across
    # subjects x actions x modes the model directories otherwise reach
    # multiple GB (~300 MB per fold for medium_quality_faster_train).
    if cfg.keep_only_best:
        try:
            predictor.delete_models(models_to_keep="best", dry_run=False)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[WARN] keep_only_best failed: {exc}")
    if cfg.save_space:
        try:
            predictor.save_space()
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] save_space failed: {exc}")

    return predictor, feature_cols


def _predict_proba(predictor, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Return P(label=1) for each row."""
    proba = predictor.predict_proba(df[feature_cols], as_pandas=True)
    if 1 in proba.columns:
        return proba[1].to_numpy()
    if "1" in proba.columns:
        return proba["1"].to_numpy()
    # Fallback: assume the second column is positive class
    return proba.iloc[:, -1].to_numpy()


# ---------------------------------------------------------------------------
# Boundary refinement (third stage)
# ---------------------------------------------------------------------------


def _train_boundary_refiner(
    train_rows: pd.DataFrame,
    cfg: HybridConfig,
    output_path: Path,
    target_col: str,
):
    """Train an AutoGluon regressor that predicts a boundary correction (in
    samples) for SDTW candidates. Trained only on rows where the candidate has
    a meaningful overlap with a ground-truth segment (best_iou >= refiner_train_iou).
    Returns (predictor, feature_cols) or (None, []) if training is skipped.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("AutoGluon is not installed.") from exc

    feature_cols = _feature_columns(train_rows)
    mask = (
        (train_rows["best_iou"] >= float(cfg.refiner_train_iou))
        & train_rows[target_col].notna()
    )
    eligible = train_rows.loc[mask].copy()
    if len(eligible) < 20:
        print(f"[WARN]   refiner({target_col}) skipped: only {len(eligible)} eligible rows")
        return None, []

    full = eligible[feature_cols + [target_col, "subject_id"]].copy()

    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fit_kwargs: Dict = {
        "presets": cfg.presets,
        "time_limit": int(cfg.refiner_time_limit_s) if cfg.refiner_time_limit_s else None,
        "ag_args_fit": {"num_gpus": 0},
    }
    if cfg.num_cpus is not None:
        fit_kwargs["num_cpus"] = int(cfg.num_cpus)
    if cfg.included_model_types:
        fit_kwargs["included_model_types"] = list(cfg.included_model_types)
    if cfg.excluded_model_types:
        fit_kwargs["excluded_model_types"] = list(cfg.excluded_model_types)

    train_data: pd.DataFrame
    tuning_data: pd.DataFrame | None = None
    if cfg.subject_stratified_validation:
        # Reuse the binary picker (it just needs `label` to exist; map MAE
        # buckets is overkill here, so we just pick by label parity).
        # Simpler: pick the subject whose row count is closest to the median.
        subj_counts = full.groupby("subject_id").size()
        if len(subj_counts) >= 2:
            median = subj_counts.median()
            val_subject = (subj_counts - median).abs().idxmin()
            tuning_data = full[full["subject_id"] == val_subject][feature_cols + [target_col]].copy()
            train_data = full[full["subject_id"] != val_subject][feature_cols + [target_col]].copy()
        else:
            train_data = full[feature_cols + [target_col]].copy()
    else:
        train_data = full[feature_cols + [target_col]].copy()

    predictor_kwargs = {"train_data": train_data}
    if tuning_data is not None and len(tuning_data) >= 20:
        predictor_kwargs["tuning_data"] = tuning_data

    predictor = TabularPredictor(
        label=target_col,
        path=str(output_path),
        problem_type="regression",
        eval_metric="mean_absolute_error",
    ).fit(**predictor_kwargs, **{k: v for k, v in fit_kwargs.items() if v is not None})

    if cfg.keep_only_best:
        try:
            predictor.delete_models(models_to_keep="best", dry_run=False)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] refiner keep_only_best failed: {exc}")
    if cfg.save_space:
        try:
            predictor.save_space()
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] refiner save_space failed: {exc}")

    return predictor, feature_cols


def _predict_regression(predictor, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    if predictor is None:
        return np.zeros(len(df), dtype=np.float64)
    pred = predictor.predict(df[feature_cols], as_pandas=False)
    return np.asarray(pred, dtype=np.float64)


# ---------------------------------------------------------------------------
# Inference NMS by classifier probability
# ---------------------------------------------------------------------------


def _nms_by_prob(
    candidates: Sequence[SegmentDetection],
    probs: Sequence[float],
    max_overlap_iou: float,
) -> Tuple[List[SegmentDetection], List[float]]:
    order = np.argsort(np.asarray(probs, dtype=np.float64))[::-1]
    selected_idx: List[int] = []
    for i in order:
        cand = candidates[int(i)]
        keep = True
        for j in selected_idx:
            other = candidates[j]
            left = max(cand.start_idx, other.start_idx)
            right = min(cand.end_idx, other.end_idx)
            inter = max(0, right - left)
            union = max(cand.end_idx, other.end_idx) - min(cand.start_idx, other.start_idx)
            iou = inter / union if union > 0 else 0.0
            if iou > max_overlap_iou:
                keep = False
                break
        if keep:
            selected_idx.append(int(i))
    selected_idx.sort(key=lambda k: candidates[k].start_idx)
    return [candidates[k] for k in selected_idx], [float(probs[k]) for k in selected_idx]


# ---------------------------------------------------------------------------
# Stream loaders by mode
# ---------------------------------------------------------------------------


def _load_streams(
    mode: str,
    data_dir: Path,
    subject: str,
    action: str,
    exclude_patterns: Sequence[str],
) -> List[Tuple[str, pd.DataFrame]]:
    if mode == "sets":
        return _load_set_streams(data_dir, subject, action, exclude_patterns)
    if mode == "whole":
        return _load_whole_streams(data_dir, subject, action)
    raise ValueError(f"Unsupported mode: {mode}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


_DEFAULT_OUT_DIR = Path("artifacts/rep_segmentation")


def _resolve_base_dir(out_dir: Path, io_cfg: Dict) -> Path:
    """If the caller didn't override --out-dir, prefer the path from config."""
    if out_dir == _DEFAULT_OUT_DIR:
        configured = io_cfg.get("rep_segmentation_output_dir")
        if configured:
            return Path(configured)
    return out_dir


def evaluate_hybrid(
    config_path: Path,
    mode: str,
    out_dir: Path,
    use_timestamp: bool,
    iou_threshold: float,
    make_plots: bool,
    max_plots: int,
    _prebuilt_run_dir: Path | None = None,
) -> None:
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {}) or {}
    feature_cfg = cfg.get("feature", {}) or {}
    seg_cfg = cfg.get("segmentation", {}) or {}
    io_cfg = cfg.get("io", {}) or {}
    hybrid_cfg = _build_hybrid_config(cfg)

    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    include_actions = list(data_cfg.get("include_actions") or [])
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])

    raw_sdtw_cfg = dict(seg_cfg.get("sdtw", {}) or {})
    motion_columns = raw_sdtw_cfg.pop(
        "motion_columns",
        feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]),
    )
    sdtw_cfg = SDTWConfig(**raw_sdtw_cfg)
    imu_columns = tuple(motion_columns)

    # Output layout: <base>/<timestamp>/hybrid/<mode>/...
    # When called from evaluate_both, the run_dir is already created at
    # <base>/<timestamp>/hybrid; we just append the mode here.
    if _prebuilt_run_dir is not None:
        out_dir = _prebuilt_run_dir / mode
    else:
        base = _resolve_base_dir(out_dir, io_cfg)
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = base / timestamp / "hybrid" / mode
        else:
            out_dir = base / "hybrid" / mode
    dirs = _prepare_output_dirs(out_dir)
    candidates_dir = out_dir / "candidates"
    models_dir = out_dir / "models"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    subjects = [p.name for p in _subject_dirs(data_dir)]
    if not include_actions:
        action_names = sorted(
            {p.name for subject in _subject_dirs(data_dir) for p in subject.iterdir() if p.is_dir()}
        )
        include_actions = [a for a in action_names if "rest" not in a]

    print(f"[INFO] mode={mode} subjects={subjects} actions={include_actions}")
    print(f"[INFO] hybrid_cfg={asdict(hybrid_cfg)}")
    print(f"[INFO] output={out_dir}")

    metrics_rows: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []
    templates_summary: List[Dict[str, object]] = []
    labeled_rows_all: List[Dict[str, object]] = []
    plot_count = 0

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        for action in include_actions:
            train_reps: List[pd.DataFrame] = []
            for subject in train_subjects:
                train_reps.extend(_load_rep_csvs(data_dir, subject, action, exclude_patterns, sdtw_cfg))
            if len(train_reps) < 3:
                print(f"[WARN] Skip action={action} fold={test_subject}: only {len(train_reps)} train reps")
                continue

            try:
                templates = fit_sdtw_templates(action, train_reps, imu_columns, sdtw_cfg)
            except Exception as exc:
                print(f"[WARN] Template fit failed action={action} fold={test_subject}: {exc}")
                continue

            for template in templates:
                templates_summary.append(
                    {
                        "test_subject": test_subject,
                        "action_type": action,
                        "train_subjects": train_subjects,
                        "template_id": template.template_id,
                        "dtw_feature": template.dtw_feature,
                        "active_duration": template.active_duration,
                        "min_duration": template.min_duration,
                        "max_duration": template.max_duration,
                        "cost_threshold": template.cost_threshold,
                        "n_calibration_costs": len(template.calibration_costs),
                        "exemplar_source": template.exemplar_source,
                    }
                )

            # 1. Build labeled candidate dataset from train subjects
            train_candidate_rows: List[Dict[str, object]] = []
            for subject in train_subjects:
                streams = _load_streams(mode, data_dir, subject, action, exclude_patterns)
                for stream_id, stream_df in streams:
                    sample_rate = infer_sample_rate_hz(stream_df)
                    candidates = generate_candidates_for_templates(
                        stream_df, templates, imu_columns, sdtw_cfg,
                        cost_threshold_scale=hybrid_cfg.cost_threshold_scale,
                    )
                    rows = _build_labeled_rows(
                        stream_id=stream_id,
                        subject=subject,
                        action=action,
                        stream_df=stream_df,
                        candidates=candidates,
                        imu_columns=imu_columns,
                        sample_rate_hz=sample_rate,
                        label_iou=hybrid_cfg.label_iou,
                        edge_window_samples=(
                            hybrid_cfg.edge_window_samples
                            if hybrid_cfg.enable_boundary_refiner else 0
                        ),
                    )
                    train_candidate_rows.extend(rows)

            if not train_candidate_rows:
                print(f"[WARN] No training candidates action={action} fold={test_subject}")
                continue

            train_df = pd.DataFrame(train_candidate_rows)
            pos = int(train_df["label"].sum())
            neg = int(len(train_df) - pos)
            print(
                f"[INFO] action={action} fold={test_subject} "
                f"train_candidates={len(train_df)} pos={pos} neg={neg}"
            )
            labeled_rows_all.extend(train_candidate_rows)

            predictor_path = models_dir / action / test_subject
            try:
                predictor, feature_cols = _train_predictor(train_df, hybrid_cfg, predictor_path)
            except Exception as exc:
                print(f"[WARN] Classifier fit failed action={action} fold={test_subject}: {exc}")
                continue

            # 2. (optional) Train boundary refiners — two regressors that predict
            #    sample-level corrections to the SDTW-proposed start / end indices.
            refiner_start = refiner_end = None
            refiner_feature_cols: List[str] = []
            if hybrid_cfg.enable_boundary_refiner:
                start_path = models_dir / action / test_subject / "_refiner_start"
                end_path = models_dir / action / test_subject / "_refiner_end"
                try:
                    refiner_start, refiner_feature_cols = _train_boundary_refiner(
                        train_df, hybrid_cfg, start_path, "delta_start_samples"
                    )
                    refiner_end, _ = _train_boundary_refiner(
                        train_df, hybrid_cfg, end_path, "delta_end_samples"
                    )
                except Exception as exc:
                    print(f"[WARN] Refiner fit failed action={action} fold={test_subject}: {exc}")
                    refiner_start = refiner_end = None

            # 3. Apply classifier (and refiners) on test subject's streams
            test_streams = _load_streams(mode, data_dir, test_subject, action, exclude_patterns)
            for stream_id, stream_df in test_streams:
                truth = _truth_segments_for_stream(stream_df)
                if not truth:
                    continue
                sample_rate = infer_sample_rate_hz(stream_df)
                candidates = generate_candidates_for_templates(
                    stream_df, templates, imu_columns, sdtw_cfg,
                    cost_threshold_scale=hybrid_cfg.cost_threshold_scale,
                )
                if not candidates:
                    metrics = summarize_detection_metrics([], truth, sample_rate, iou_threshold)
                    metrics_rows.append({
                        "test_subject": test_subject, "action_type": action,
                        "stream_id": stream_id, "sample_rate_hz": sample_rate,
                        "n_candidates": 0, "n_kept": 0, **metrics,
                    })
                    continue

                edge_w = hybrid_cfg.edge_window_samples if hybrid_cfg.enable_boundary_refiner else 0
                rows = [
                    compute_candidate_features(stream_df, c, imu_columns, sample_rate, edge_window_samples=edge_w)
                    for c in candidates
                ]
                test_df = pd.DataFrame(rows)
                # Ensure missing cols are filled with 0.0 to match training schema
                for col in feature_cols:
                    if col not in test_df.columns:
                        test_df[col] = 0.0
                probs = _predict_proba(predictor, test_df, feature_cols)

                # Use AutoGluon's per-fold calibrated decision threshold when
                # available (it was tuned on the held-out validation subject and
                # generalises better than a fixed cutoff). The configured
                # `prob_threshold` acts as a hard floor so we never accept a
                # candidate the model has very low confidence in.
                fold_threshold = hybrid_cfg.prob_threshold
                if hybrid_cfg.use_calibrated_threshold:
                    cal = getattr(predictor, "decision_threshold", None)
                    if cal is not None:
                        try:
                            fold_threshold = max(float(cal), float(hybrid_cfg.prob_threshold) * 0.3)
                        except Exception:
                            fold_threshold = hybrid_cfg.prob_threshold
                kept_indices = [i for i, p in enumerate(probs) if p >= fold_threshold]
                kept_candidates = [candidates[i] for i in kept_indices]
                kept_probs = [float(probs[i]) for i in kept_indices]

                # 3a. Boundary refinement on kept candidates.
                refined_candidates = list(kept_candidates)
                refine_log: List[Tuple[int, int]] = []  # (delta_start, delta_end) per kept
                if (
                    hybrid_cfg.enable_boundary_refiner
                    and refiner_start is not None
                    and refiner_end is not None
                    and kept_candidates
                ):
                    kept_df = test_df.iloc[kept_indices].copy()
                    for col in refiner_feature_cols:
                        if col not in kept_df.columns:
                            kept_df[col] = 0.0
                    delta_start = _predict_regression(refiner_start, kept_df, refiner_feature_cols)
                    delta_end = _predict_regression(refiner_end, kept_df, refiner_feature_cols)
                    cap = int(hybrid_cfg.refiner_max_shift_samples)
                    refined_candidates = []
                    stream_n = len(stream_df)
                    for cand, ds, de in zip(kept_candidates, delta_start, delta_end):
                        ds_i = int(round(float(ds)))
                        de_i = int(round(float(de)))
                        # Safety cap: ignore huge predicted shifts (likely garbage).
                        if cap > 0 and (abs(ds_i) > cap or abs(de_i) > cap):
                            refined_candidates.append(cand)
                            refine_log.append((0, 0))
                            continue
                        new_start = max(0, min(stream_n - 1, int(cand.start_idx) + ds_i))
                        new_end = max(0, min(stream_n, int(cand.end_idx) + de_i))
                        if new_end <= new_start:
                            refined_candidates.append(cand)
                            refine_log.append((0, 0))
                            continue
                        refined_candidates.append(replace(cand, start_idx=new_start, end_idx=new_end))
                        refine_log.append((ds_i, de_i))

                detections, det_probs = _nms_by_prob(refined_candidates, kept_probs, hybrid_cfg.nms_iou)
                # Build a mapping from refined_candidate -> applied delta so the
                # detection CSV can record the per-detection refinement.
                if refine_log:
                    refined_to_delta = {id(c): d for c, d in zip(refined_candidates, refine_log)}
                else:
                    refined_to_delta = {}

                metrics = summarize_detection_metrics(detections, truth, sample_rate, iou_threshold)
                metrics_rows.append({
                    "test_subject": test_subject, "action_type": action,
                    "stream_id": stream_id, "sample_rate_hz": sample_rate,
                    "n_candidates": int(len(candidates)),
                    "n_kept": int(len(detections)),
                    **metrics,
                })

                rows_for_csv = _detection_rows(stream_id, detections)
                for row, det, prob in zip(rows_for_csv, detections, det_probs):
                    row["classifier_prob"] = prob
                    row["test_subject"] = test_subject
                    delta = refined_to_delta.get(id(det), (0, 0))
                    row["refiner_delta_start_samples"] = int(delta[0])
                    row["refiner_delta_end_samples"] = int(delta[1])
                detection_rows.extend(rows_for_csv)

                if make_plots and (max_plots <= 0 or plot_count < max_plots):
                    plot_path = dirs["plots"] / action / test_subject / (_safe_name(stream_id) + ".svg")
                    _write_segmentation_svg(
                        plot_path, stream_id, stream_df, truth, detections, metrics, sample_rate,
                    )
                    plot_count += 1

            # 4. Drop the predictor directory if requested. Reproducibility is
            #    preserved via labeled_candidates.csv + the config snapshot. The
            #    classifier directory contains both the binary predictor and any
            #    `_refiner_*` subfolders.
            if hybrid_cfg.delete_predictor_after_eval and predictor_path.exists():
                shutil.rmtree(predictor_path, ignore_errors=True)

    # ----- Save artefacts -----
    metrics_df = pd.DataFrame(metrics_rows)
    detections_df = pd.DataFrame(detection_rows)
    templates_df = pd.DataFrame(templates_summary)
    labeled_df = pd.DataFrame(labeled_rows_all)

    metrics_df.to_csv(dirs["metrics"] / "stream_metrics.csv", index=False)
    detections_df.to_csv(dirs["detections"] / "detections.csv", index=False)
    templates_df.to_csv(dirs["templates"] / "templates.csv", index=False)
    labeled_df.to_csv(candidates_dir / "labeled_candidates.csv", index=False)

    summary: Dict[str, object] = {
        "mode": mode,
        "iou_threshold": iou_threshold,
        "hybrid": asdict(hybrid_cfg),
        "sdtw": asdict(sdtw_cfg),
        "overall": _aggregate_metrics(metrics_rows),
        "by_action": {},
        "by_subject": {},
        "outputs": {
            "metrics": str(dirs["metrics"]),
            "detections": str(dirs["detections"]),
            "templates": str(dirs["templates"]),
            "candidates": str(candidates_dir),
            "models": str(models_dir),
            "plots": str(dirs["plots"]) if make_plots else None,
        },
    }
    if not metrics_df.empty:
        for action, group in metrics_df.groupby("action_type"):
            summary["by_action"][str(action)] = _aggregate_metrics(group.to_dict("records"))
        for subject, group in metrics_df.groupby("test_subject"):
            summary["by_subject"][str(subject)] = _aggregate_metrics(group.to_dict("records"))

    (dirs["metrics"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (dirs["metadata"] / "run_manifest.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "config_path": str(config_path),
                "iou_threshold": iou_threshold,
                "make_plots": make_plots,
                "max_plots": max_plots,
                "plot_count": plot_count,
                "subjects": subjects,
                "actions": include_actions,
                "motion_columns": list(imu_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        shutil.copy2(config_path, dirs["metadata"] / "config_snapshot.yaml")
    except Exception:
        pass

    (dirs["root"] / "README.md").write_text(
        "\n".join(
            [
                f"# Hybrid Rep Segmentation ({mode})",
                "",
                "SDTW candidates filtered by an AutoGluon binary classifier.",
                "",
                "- `metrics/summary.json`: overall, by-action, by-subject scores.",
                "- `metrics/stream_metrics.csv`: per-stream metrics.",
                "- `detections/detections.csv`: kept detections (post-classifier + NMS) with `classifier_prob`.",
                "- `candidates/labeled_candidates.csv`: every training candidate with features and TP/FP label.",
                "- `models/{action}/{test_subject}/`: trained AutoGluon predictor per LOSO fold.",
                "- `templates/templates.csv`: SDTW template metadata per fold.",
                "- `plots/{action}/{subject}/*.svg`: visual GT vs predicted reps.",
                "- `metadata/`: run manifest + config snapshot.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("HYBRID REP SEGMENTATION SUMMARY")
    print("=" * 72)
    print(json.dumps(summary["overall"], indent=2))
    if not metrics_df.empty:
        cols = ["action_type", "n_true", "n_pred", "precision", "recall", "f1", "start_mae_ms", "end_mae_ms"]
        by_action = metrics_df.groupby("action_type")[cols[1:]].mean(numeric_only=True).reset_index()
        print("\n[BY ACTION]")
        print(by_action[cols].to_string(index=False))
    print(f"\n[OK] Wrote outputs to {out_dir}")


def evaluate_both(
    config_path: Path,
    out_dir: Path,
    use_timestamp: bool,
    iou_threshold: float,
    make_plots: bool,
    max_plots: int,
) -> None:
    cfg = _load_config(config_path)
    io_cfg = cfg.get("io", {}) or {}
    base = _resolve_base_dir(out_dir, io_cfg)
    # Layout: <base>/<timestamp>/hybrid/{sets,whole}/  (parallel to plain SDTW
    # eval which writes <base>/<timestamp>/{sets,whole}/).
    if use_timestamp:
        run_dir = base / datetime.now().strftime("%Y%m%d_%H%M%S") / "hybrid"
    else:
        run_dir = base / "hybrid"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] combined hybrid output={run_dir}")

    evaluate_hybrid(
        config_path=config_path, mode="sets", out_dir=out_dir,
        use_timestamp=False, iou_threshold=iou_threshold,
        make_plots=make_plots, max_plots=max_plots,
        _prebuilt_run_dir=run_dir,
    )
    evaluate_hybrid(
        config_path=config_path, mode="whole", out_dir=out_dir,
        use_timestamp=False, iou_threshold=iou_threshold,
        make_plots=make_plots, max_plots=max_plots,
        _prebuilt_run_dir=run_dir,
    )

    combined: Dict[str, object] = {"mode": "both", "iou_threshold": iou_threshold,
                                   "outputs": {"sets": str(run_dir / "sets"), "whole": str(run_dir / "whole")}}
    for sub in ("sets", "whole"):
        summary_path = run_dir / sub / "metrics" / "summary.json"
        if summary_path.exists():
            try:
                combined[sub] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                combined[sub] = {"error": f"Could not read {summary_path}"}
    (run_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\n[OK] Combined hybrid sets+whole outputs at {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid SDTW + AutoGluon rep segmentation")
    parser.add_argument("--config", type=Path, default=Path("configs/hybrid_rep_segmentation.yaml"))
    parser.add_argument("--mode", choices=["sets", "whole", "both"], default="both")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Base output directory. Outputs are written to "
             "<out-dir>/<timestamp>/hybrid/{sets,whole}/ (parallel to plain SDTW eval).",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-plots", type=int, default=0)
    parser.add_argument("--no-timestamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "both":
        evaluate_both(
            config_path=args.config, out_dir=args.out_dir,
            use_timestamp=not args.no_timestamp, iou_threshold=args.iou_threshold,
            make_plots=not args.no_plots, max_plots=args.max_plots,
        )
    else:
        evaluate_hybrid(
            config_path=args.config, mode=args.mode, out_dir=args.out_dir,
            use_timestamp=not args.no_timestamp, iou_threshold=args.iou_threshold,
            make_plots=not args.no_plots, max_plots=args.max_plots,
        )


if __name__ == "__main__":
    main()
