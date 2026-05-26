"""
Action Classification 9-Fold LOSO (RF only, fast baseline)

Quickly evaluate per-rep action classification across all 9 subjects
using RandomForest with both stats and rich features.

Usage:
    conda run -n imu_for_workout python scripts/action_classification_9fold_rf.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features

# --- Configuration ---
DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_9fold_rf")
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


def evaluate_fold_rf(train_seqs, test_seqs, test_subject, feature_mode):
    stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
    train_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
    test_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

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

    return {
        "test_subject": test_subject,
        "feature_mode": feature_mode,
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sequences, subjects = prepare_sequences_from_folder(
        data_dir=DATA_DIR,
        feature_cfg=FEATURE_CFG,
        sample_rate_hz=SAMPLE_RATE_HZ,
        csv_glob="*.csv",
        exclude_patterns=EXCLUDE_PATTERNS,
        include_actions=INCLUDE_ACTIONS,
    )
    all_subjects = sorted(set(subjects))
    print(f"[INFO] Loaded {len(sequences)} sequences from {len(all_subjects)} subjects: {all_subjects}")

    results = []

    for test_subj in all_subjects:
        print(f"\n{'='*60}")
        print(f"Fold: test={test_subj}")
        print(f"{'='*60}")

        train_subjs = [s for s in all_subjects if s != test_subj]
        train_seqs = filter_sequences_by_subject(sequences, train_subjs, FEATURE_CFG.subject_column)
        test_seqs = filter_sequences_by_subject(sequences, [test_subj], FEATURE_CFG.subject_column)
        print(f"[INFO] train={len(train_seqs)}, test={len(test_seqs)}")

        for feature_mode in ["stats", "rich"]:
            print(f"  [RF {feature_mode}] ...", end="")
            res = evaluate_fold_rf(train_seqs, test_seqs, test_subj, feature_mode)
            results.append(res)
            print(f" acc={res['accuracy']:.4f}, macro_f1={res['macro_f1']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("9-FOLD LOSO SUMMARY (RF)")
    print(f"{'='*60}")

    for feature_mode in ["stats", "rich"]:
        mode_res = [r for r in results if r["feature_mode"] == feature_mode]
        avg_acc = np.mean([r["accuracy"] for r in mode_res])
        avg_macro = np.mean([r["macro_f1"] for r in mode_res])
        avg_weighted = np.mean([r["weighted_f1"] for r in mode_res])

        print(f"\nRF ({feature_mode}):")
        print(f"  avg accuracy:    {avg_acc:.4f}")
        print(f"  avg macro_f1:    {avg_macro:.4f}")
        print(f"  avg weighted_f1: {avg_weighted:.4f}")
        print(f"  per-fold:")
        for r in mode_res:
            print(f"    {r['test_subject']}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")

    summary = {
        "subjects": all_subjects,
        "results": results,
    }
    summary_path = OUTPUT_DIR / "9fold_rf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[INFO] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
