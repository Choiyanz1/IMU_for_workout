"""
Action Classification 9-Fold LOSO (LinearSVC ONLY - fast linear kernel)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features

DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_9fold_linearsvc_only")
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/3] Loading sequences...")
    sequences, subjects = prepare_sequences_from_folder(
        data_dir=DATA_DIR,
        feature_cfg=FEATURE_CFG,
        sample_rate_hz=100,
        csv_glob="*.csv",
        exclude_patterns=EXCLUDE_PATTERNS,
        include_actions=INCLUDE_ACTIONS,
    )
    all_subjects = sorted(set(subjects))
    print(f"[2/3] Loaded {len(sequences)} sequences from {len(all_subjects)} subjects")

    results = []
    for i, test_subj in enumerate(all_subjects):
        print(f"\n[3/3] Fold {i+1}/{len(all_subjects)}: test={test_subj}")
        train_subjs = [s for s in all_subjects if s != test_subj]
        train_seqs = filter_sequences_by_subject(sequences, train_subjs, FEATURE_CFG.subject_column)
        test_seqs = filter_sequences_by_subject(sequences, [test_subj], FEATURE_CFG.subject_column)
        print(f"      train={len(train_seqs)}, test={len(test_seqs)}")

        stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
        train_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
        test_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

        df_train = _build_rep_features(train_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, "rich")
        df_test = _build_rep_features(test_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, "rich")

        X_train = df_train.drop(columns=["label"]).values
        y_train = df_train["label"].values
        X_test = df_test.drop(columns=["label"]).values
        y_test = df_test["label"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LinearSVC(C=1.0, max_iter=2000, random_state=42, dual="auto")
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"      => acc={acc:.4f}, macro_f1={macro_f1:.4f}, weighted_f1={weighted_f1:.4f}")
        results.append({"test_subject": test_subj, "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1})

    print(f"\n{'='*70}")
    print("9-FOLD LOSO SUMMARY (LinearSVC)")
    print(f"{'='*70}")
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_macro = np.mean([r["macro_f1"] for r in results])
    avg_weighted = np.mean([r["weighted_f1"] for r in results])
    print(f"avg accuracy:    {avg_acc:.4f}")
    print(f"avg macro_f1:    {avg_macro:.4f}")
    print(f"avg weighted_f1: {avg_weighted:.4f}")
    print("per-fold:")
    for r in results:
        print(f"  {r['test_subject']:<20}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")

    summary = {"results": results, "avg": {"accuracy": avg_acc, "macro_f1": avg_macro, "weighted_f1": avg_weighted}}
    (OUTPUT_DIR / "9fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT_DIR / '9fold_summary.json'}")


if __name__ == "__main__":
    main()
