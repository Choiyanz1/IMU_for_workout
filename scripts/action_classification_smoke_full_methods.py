"""
Action Classification Single-Fold Smoke Test (8 train, 1 test: kevin)
Tests all statistical learning methods + AutoGluon on ONE fold to verify pipeline.

Methods:
1. RF (rich features)
2. SVM (rich features)
3. Logistic Regression (rich features)
4. AutoGluon (rich features, 120s)

Usage:
    conda run -n imu_for_workout python scripts/action_classification_smoke_full_methods.py
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import shutil

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features, AutoGluonConfig

# --- Configuration ---
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
FEATURE_MODE = "rich"


def prepare_data():
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
    print(f"[INFO] train={len(train_seqs)}, test={len(test_seqs)}")

    stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
    train_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
    test_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

    df_train = _build_rep_features(train_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, FEATURE_MODE)
    df_test = _build_rep_features(test_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, FEATURE_MODE)

    print(f"[INFO] feature_mode={FEATURE_MODE}, train reps={len(df_train)}, test reps={len(df_test)}")
    print(f"[INFO] feature dim={df_train.shape[1] - 1}")

    return df_train, df_test, test_subj


def evaluate_sklearn(clf, name, X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"accuracy:    {acc:.4f}")
    print(f"macro_f1:    {macro_f1:.4f}")
    print(f"weighted_f1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "method": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def evaluate_autogluon(df_train, df_test, test_subj):
    ag_cfg = AutoGluonConfig(
        feature_mode="rich",
        presets="medium_quality_faster_train",
        time_limit_s=120,
        eval_metric="accuracy",
        excluded_model_types=["NN_TORCH", "FASTAI", "GBM", "XGB"],
    )

    predictor_path = f"artifacts/action_classification_smoke_full_methods/ag_{test_subj}"
    if Path(predictor_path).exists():
        shutil.rmtree(predictor_path, ignore_errors=True)

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(
        label="label",
        path=predictor_path,
        problem_type="multiclass",
        eval_metric=ag_cfg.eval_metric,
    ).fit(
        train_data=df_train,
        presets=ag_cfg.presets,
        time_limit=ag_cfg.time_limit_s,
        excluded_model_types=ag_cfg.excluded_model_types,
        ag_args_fit={"num_gpus": 0},
    )

    y_true = df_test["label"].values
    y_pred = predictor.predict(df_test.drop(columns=["label"])).values

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print("AutoGluon (rich, 120s)")
    print(f"{'='*60}")
    print(f"accuracy:    {acc:.4f}")
    print(f"macro_f1:    {macro_f1:.4f}")
    print(f"weighted_f1: {weighted_f1:.4f}")
    print(f"best_model:  {predictor.model_best}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    lb = predictor.leaderboard(df_test, silent=True)
    print("\nLeaderboard:")
    print(lb[["model", "score_test"]].to_string(index=False))

    return {
        "method": "AutoGluon",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "best_model": predictor.model_best,
    }


def main():
    print("="*70)
    print("Action Classification: Single-Fold Smoke Test (ALL METHODS)")
    print("Train: 8 subjects, Test: kevin")
    print("="*70)

    df_train, df_test, test_subj = prepare_data()

    X_train = df_train.drop(columns=["label"]).values
    y_train = df_train["label"].values
    X_test = df_test.drop(columns=["label"]).values
    y_test = df_test["label"].values

    results = []

    # 1. RF
    results.append(evaluate_sklearn(
        RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        "RF (rich, n=100, depth=15)", X_train, y_train, X_test, y_test
    ))

    # 2. SVM
    results.append(evaluate_sklearn(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "SVM (rich, RBF, C=1.0)", X_train, y_train, X_test, y_test
    ))

    # 3. LogReg
    results.append(evaluate_sklearn(
        LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42, n_jobs=-1),
        "LogReg (rich, multinomial)", X_train, y_train, X_test, y_test
    ))

    # 4. AutoGluon
    results.append(evaluate_autogluon(df_train, df_test, test_subj))

    # Summary
    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['method']:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}")

    print("\n[OK] All methods passed smoke test. Ready for 9-fold LOSO.")


if __name__ == "__main__":
    main()
