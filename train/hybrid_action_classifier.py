"""Hybrid action classifier: macro-confidence routing + rep-complete fallback.

This module trains a lightweight sklearn classifier on per-rep rich features
extracted from non-test training data.  At runtime it combines the DS-MS-TCN
macro-stage confidence with the classifier prediction:

    if macro_confidence >= threshold: use macro label
    else:                            use classifier label

Usage::

    clf = HybridActionClassifier.from_config(config_path, test_subject="kevin")
    label, confidence = clf.hybrid_label(
        macro_label, macro_confidence, segment_df, already_zscored=True
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasets.custom_resistance_dataset import (
    FeatureConfig,
    filter_sequences_by_subject,
    prepare_sequences_from_folder,
)
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features, _compute_rich_features


def _is_all_subjects_mode(test_subject: object) -> bool:
    value = str(test_subject or "").strip().lower()
    return value in {"all", "__all__", "*", ""}


class HybridActionClassifier:
    """Trains a per-rep action classifier and provides hybrid prediction."""

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(
        self,
        model,
        stats,
        imu_columns: Sequence[str],
        label_column: str,
        feature_mode: str = "rich",
    ) -> None:
        self.model = model
        self.stats = stats
        self.imu_columns = list(imu_columns)
        self.label_column = label_column
        self.feature_mode = feature_mode

    @classmethod
    def from_config(cls, config_path: Path, test_subject: str) -> "HybridActionClassifier":
        """Train classifier on all non-test-subject reps from config."""
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        feature_cfg = FeatureConfig(**(raw.get("feature", {}) or {}))
        data_cfg = raw.get("data", {}) or {}
        window_cfg = raw.get("window", {}) or {}

        data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
        sequences, subjects = prepare_sequences_from_folder(
            data_dir=data_dir,
            feature_cfg=feature_cfg,
            sample_rate_hz=int(window_cfg.get("sample_rate_hz", 100)),
            csv_glob=data_cfg.get("csv_glob", "*.csv"),
            exclude_patterns=data_cfg.get("exclude_patterns", None),
            include_actions=data_cfg.get("include_actions", None),
            subject_aliases=data_cfg.get("subject_aliases", None),
        )
        if _is_all_subjects_mode(test_subject):
            train_subjects = sorted(set(subjects))
        else:
            train_subjects = [s for s in sorted(set(subjects)) if s != str(test_subject)]
        if not train_subjects:
            raise ValueError(f"No training subjects found after excluding {test_subject}")

        train_seqs = filter_sequences_by_subject(sequences, train_subjects, feature_cfg.subject_column)
        if not train_seqs:
            raise ValueError("No training sequences loaded.")

        stats = compute_train_stats(train_seqs, feature_cfg.imu_columns)
        train_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in train_seqs]
        train_df = _build_rep_features(train_seqs, feature_cfg.imu_columns, feature_cfg.label_column, "rich")

        x_train = train_df.drop(columns=["label"])
        y_train = train_df["label"].astype(str)

        candidates = {
            "logreg": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=4000, multi_class="auto", random_state=42),
            ),
            "rf": RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
                n_jobs=-1,
            ),
        }
        best_model = None
        best_score = -1.0
        for name, candidate in candidates.items():
            candidate.fit(x_train, y_train)
            pred = candidate.predict(x_train)
            score = float(f1_score(y_train, pred, average="macro"))
            if score > best_score:
                best_score = score
                best_model = candidate

        if best_model is None:
            raise RuntimeError("Failed to train any classifier candidate.")

        return cls(best_model, stats, feature_cfg.imu_columns, feature_cfg.label_column)

    def predict_segment(
        self,
        segment_df: pd.DataFrame,
        already_zscored: bool = False,
    ) -> tuple[str, float]:
        """Predict action for a single rep segment DataFrame.

        Args:
            segment_df: DataFrame containing IMU columns (and optionally label).
            already_zscored: If False, applies training z-score stats before feature
                extraction.  In streaming eval the input is already z-scored, so
                pass True.

        Returns:
            (predicted_label, confidence)
        """
        if not already_zscored:
            segment_df = apply_zscore(segment_df, self.imu_columns, self.stats)

        data = segment_df[self.imu_columns].to_numpy(dtype=np.float32)
        if len(data) < 2:
            return "unknown", 0.0

        window = data[np.newaxis, :, :]  # (1, T, C)
        feats = _compute_rich_features(window, self.imu_columns)
        row = {k: float(v[0]) for k, v in feats.items()}
        x = pd.DataFrame([row])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            best_idx = int(np.argmax(proba))
            label = str(self.model.classes_[best_idx])
            confidence = float(proba[best_idx])
        else:
            label = str(self.model.predict(x)[0])
            confidence = 0.6

        return label, confidence

    def hybrid_label(
        self,
        macro_label: str,
        macro_confidence: float,
        segment_df: pd.DataFrame,
        already_zscored: bool = False,
    ) -> tuple[str, float]:
        """Return hybrid action label.

        Uses macro label when its confidence is high; otherwise falls back to
        the rep-complete classifier.
        """
        if macro_confidence >= self.CONFIDENCE_THRESHOLD:
            return str(macro_label), float(macro_confidence)

        clf_label, clf_confidence = self.predict_segment(segment_df, already_zscored)
        return clf_label, clf_confidence
