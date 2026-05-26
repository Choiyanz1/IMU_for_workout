"""
Quick single-fold test: 8 subjects train, 1 test (kevin)
Verify if more training data improves per-rep macro_f1 above 0.70.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features

DATA_DIR = Path("datasets/raw_data")
FEATURE_CFG = FeatureConfig(
    imu_columns=("ax", "ay", "az", "gx", "gy", "gz"),
    label_column="action_type",
    subject_column="subject_id",
    time_column="sensor_ts",
)
EXCLUDE_PATTERNS = ["*whole_session*", "*_w", "*rest_after*"]
INCLUDE_ACTIONS = [
    "db_bench_press", "db_biceps_curl", "db_rdl", "db_shoulder_press",
    "db_squat", "db_triceps_curl", "db_weighted_crunch", "one_arm_db_row",
]
SAMPLE_RATE_HZ = 100


def main():
    sequences, subjects = prepare_sequences_from_folder(
        data_dir=DATA_DIR,
        feature_cfg=FEATURE_CFG,
        sample_rate_hz=SAMPLE_RATE_HZ,
        csv_glob="*.csv",
        exclude_patterns=EXCLUDE_PATTERNS,
        include_actions=INCLUDE_ACTIONS,
    )
    all_subjects = sorted(set(subjects))
    print(f"Subjects: {all_subjects}")

    test_subj = "kevin"
    train_subjs = [s for s in all_subjects if s != test_subj]

    train_seqs = filter_sequences_by_subject(sequences, train_subjs, FEATURE_CFG.subject_column)
    test_seqs = filter_sequences_by_subject(sequences, [test_subj], FEATURE_CFG.subject_column)
    print(f"train={len(train_seqs)}, test={len(test_seqs)}")

    stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
    train_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
    test_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

    for feature_mode in ["stats", "rich"]:
        print(f"\n{'='*60}")
        print(f"Feature mode: {feature_mode}")
        print(f"{'='*60}")

        df_train = _build_rep_features(train_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, feature_mode)
        df_test = _build_rep_features(test_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, feature_mode)

        X_train = df_train.drop(columns=["label"]).values
        y_train = df_train["label"].values
        X_test = df_test.drop(columns=["label"]).values
        y_test = df_test["label"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"accuracy: {acc:.4f}")
        print(f"macro_f1: {macro_f1:.4f}")
        print(f"weighted_f1: {weighted_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
