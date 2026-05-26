"""
Action Classification Smoke Test (3 subjects, 3-fold LOSO)

Tests two approaches:
1. AutoGluon (rich features, medium_quality, 60s time limit)
2. sklearn RF (stats features, 100 trees, depth 15)

Usage:
    conda run -n imu_for_workout python scripts/action_classification_smoke_test.py
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import (
    _build_rep_features,
    AutoGluonConfig,
)

# --- Configuration ---
SMOKE_SUBJECTS = ["kevin", "yushuan", "yoru"]
DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_smoke_test")
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


def evaluate_fold(
    train_seqs: List[pd.DataFrame],
    test_seqs: List[pd.DataFrame],
    test_subject: str,
    method: str,
) -> Dict:
    """Evaluate one fold with one method."""
    print(f"\n  [fold] test={test_subject}, method={method}")

    # Z-score normalization
    stats = compute_train_stats(train_seqs, FEATURE_CFG.imu_columns)
    train_seqs_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in train_seqs]
    test_seqs_norm = [apply_zscore(seq, FEATURE_CFG.imu_columns, stats) for seq in test_seqs]

    if method == "autogluon":
        ag_cfg = AutoGluonConfig(
            feature_mode="rich",
            presets="medium_quality_faster_train",
            time_limit_s=60,
            eval_metric="accuracy",
            excluded_model_types=["NN_TORCH", "FASTAI", "GBM", "XGB"],
        )
        df_train = _build_rep_features(train_seqs_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, ag_cfg.feature_mode)
        df_test = _build_rep_features(test_seqs_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, ag_cfg.feature_mode)

        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            print("    AutoGluon not installed, skipping")
            return {"skipped": True}

        predictor = TabularPredictor(
            label="label",
            path=str(OUTPUT_DIR / f"ag_{test_subject}"),
            problem_type="multiclass",
            eval_metric=ag_cfg.eval_metric,
        ).fit(
            train_data=df_train,
            presets=ag_cfg.presets,
            time_limit=ag_cfg.time_limit_s,
            excluded_model_types=ag_cfg.excluded_model_types,
            ag_args_fit={"num_gpus": 0},
        )

        y_pred = predictor.predict(df_test.drop(columns=["label"])).values
        y_true = df_test["label"].values
        best_model = predictor.model_best

        del predictor
        gc.collect()

    elif method == "rf_stats":
        df_train = _build_rep_features(train_seqs_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, "stats")
        df_test = _build_rep_features(test_seqs_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, "stats")

        X_train = df_train.drop(columns=["label"]).values
        y_train = df_train["label"].values
        X_test = df_test.drop(columns=["label"]).values
        y_true = df_test["label"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        best_model = "RandomForest_stats"

        del clf
        gc.collect()

    else:
        raise ValueError(f"Unknown method: {method}")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=INCLUDE_ACTIONS)

    print(f"    acc={acc:.4f}, macro_f1={macro_f1:.4f}, weighted_f1={weighted_f1:.4f}")

    return {
        "test_subject": test_subject,
        "method": method,
        "best_model": best_model,
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

    print("=" * 60)
    print("Action Classification Smoke Test")
    print(f"Subjects: {SMOKE_SUBJECTS}")
    print("=" * 60)

    # Load all sequences
    sequences, subjects = prepare_sequences_from_folder(
        data_dir=DATA_DIR,
        feature_cfg=FEATURE_CFG,
        sample_rate_hz=SAMPLE_RATE_HZ,
        csv_glob="*.csv",
        exclude_patterns=EXCLUDE_PATTERNS,
        include_actions=INCLUDE_ACTIONS,
    )
    print(f"[INFO] Loaded {len(sequences)} sequences from {len(set(subjects))} subjects")

    # Filter to smoke subjects
    smoke_seqs = filter_sequences_by_subject(sequences, SMOKE_SUBJECTS, FEATURE_CFG.subject_column)
    print(f"[INFO] Smoke subjects: {len(smoke_seqs)} sequences")

    # 3-fold LOSO
    results = []
    methods = ["autogluon", "rf_stats"]

    for test_subj in SMOKE_SUBJECTS:
        print(f"\n{'='*60}")
        print(f"Fold: test={test_subj}")
        print(f"{'='*60}")

        train_subjs = [s for s in SMOKE_SUBJECTS if s != test_subj]
        train_seqs = filter_sequences_by_subject(smoke_seqs, train_subjs, FEATURE_CFG.subject_column)
        test_seqs = filter_sequences_by_subject(smoke_seqs, [test_subj], FEATURE_CFG.subject_column)
        print(f"[INFO] train={len(train_seqs)}, test={len(test_seqs)}")

        for method in methods:
            try:
                fold_result = evaluate_fold(train_seqs, test_seqs, test_subj, method)
                results.append(fold_result)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "test_subject": test_subj,
                    "method": method,
                    "error": str(e),
                })

    # Aggregate results
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")

    for method in methods:
        method_results = [r for r in results if r.get("method") == method and "macro_f1" in r]
        if not method_results:
            print(f"\n{method}: NO VALID RESULTS")
            continue

        avg_acc = np.mean([r["accuracy"] for r in method_results])
        avg_macro_f1 = np.mean([r["macro_f1"] for r in method_results])
        avg_weighted_f1 = np.mean([r["weighted_f1"] for r in method_results])

        print(f"\n{method.upper()}:")
        print(f"  avg accuracy:     {avg_acc:.4f}")
        print(f"  avg macro_f1:     {avg_macro_f1:.4f}")
        print(f"  avg weighted_f1:  {avg_weighted_f1:.4f}")
        print(f"  per-fold:")
        for r in method_results:
            print(f"    {r['test_subject']}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}")

    # Save results
    summary = {
        "subjects": SMOKE_SUBJECTS,
        "methods": methods,
        "results": results,
    }
    summary_path = OUTPUT_DIR / "smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[INFO] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
