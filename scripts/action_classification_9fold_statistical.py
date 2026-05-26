"""
Action Classification 9-Fold LOSO (Statistical Methods Only)

Methods:
1. RF (rich, n=100, depth=15)
2. SVM (RBF, C=1.0)
3. Logistic Regression (multinomial)

All use subject-wise split (9-fold LOSO), z-score fit on training subjects only.

Usage:
    conda run -n imu_for_workout python scripts/action_classification_9fold_statistical.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features

# --- Configuration ---
DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_9fold_statistical")
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
FEATURE_MODE = "rich"


def evaluate_fold_method(train_seqs, test_seqs, test_subject, method_name, clf):
    """Evaluate one fold with one method."""
    stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
    train_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
    test_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

    df_train = _build_rep_features(train_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, FEATURE_MODE)
    df_test = _build_rep_features(test_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, FEATURE_MODE)

    X_train = df_train.drop(columns=["label"]).values
    y_train = df_train["label"].values
    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=INCLUDE_ACTIONS)

    print(f"    {method_name}: acc={acc:.4f}, macro_f1={macro_f1:.4f}")

    return {
        "test_subject": test_subject,
        "method": method_name,
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "labels": INCLUDE_ACTIONS,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Action Classification: 9-Fold LOSO (Statistical Methods)")
    print("="*70)

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

    methods = {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    }

    results = []

    for test_subj in all_subjects:
        print(f"\n{'='*70}")
        print(f"Fold: test={test_subj}")
        print(f"{'='*70}")

        train_subjs = [s for s in all_subjects if s != test_subj]
        train_seqs = filter_sequences_by_subject(sequences, train_subjs, FEATURE_CFG.subject_column)
        test_seqs = filter_sequences_by_subject(sequences, [test_subj], FEATURE_CFG.subject_column)
        print(f"[INFO] train={len(train_seqs)}, test={len(test_seqs)}")

        for method_name, clf in methods.items():
            try:
                res = evaluate_fold_method(train_seqs, test_seqs, test_subj, method_name, clf)
                results.append(res)
            except Exception as e:
                print(f"    ERROR {method_name}: {e}")
                results.append({
                    "test_subject": test_subj,
                    "method": method_name,
                    "error": str(e),
                })

    # Aggregate results
    print(f"\n{'='*70}")
    print("9-FOLD LOSO SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<15} {'avg Acc':>10} {'avg MacroF1':>12} {'avg WeightedF1':>15}")
    print("-"*70)

    for method_name in methods.keys():
        method_results = [r for r in results if r.get("method") == method_name and "macro_f1" in r]
        if not method_results:
            continue

        avg_acc = np.mean([r["accuracy"] for r in method_results])
        avg_macro = np.mean([r["macro_f1"] for r in method_results])
        avg_weighted = np.mean([r["weighted_f1"] for r in method_results])

        print(f"{method_name:<15} {avg_acc:>10.4f} {avg_macro:>12.4f} {avg_weighted:>15.4f}")
        print(f"  per-fold:")
        for r in method_results:
            print(f"    {r['test_subject']:<20}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")

    # Save results
    summary = {
        "subjects": all_subjects,
        "feature_mode": FEATURE_MODE,
        "results": results,
    }
    summary_path = OUTPUT_DIR / "9fold_statistical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[INFO] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
