"""
Action Classification 9-Fold LOSO (AutoGluon ONLY)
Each fold: ~77s, total ~11-12min for 9 folds.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features, AutoGluonConfig
from autogluon.tabular import TabularPredictor

DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_9fold_autogluon_only")
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

    ag_cfg = AutoGluonConfig(
        feature_mode="rich",
        presets="medium_quality_faster_train",
        time_limit_s=120,
        eval_metric="accuracy",
        excluded_model_types=["NN_TORCH", "FASTAI", "GBM", "XGB"],
    )

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
        print(f"      train reps={len(df_train)}, test reps={len(df_test)}")

        predictor_path = str(OUTPUT_DIR / f"ag_{test_subj}")
        if Path(predictor_path).exists():
            shutil.rmtree(predictor_path, ignore_errors=True)

        print("      Training AutoGluon (120s)...")
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

        print(f"      => acc={acc:.4f}, macro_f1={macro_f1:.4f}, best_model={predictor.model_best}")
        results.append({
            "test_subject": test_subj,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "best_model": predictor.model_best,
        })

    print(f"\n{'='*70}")
    print("9-FOLD LOSO SUMMARY (AutoGluon)")
    print(f"{'='*70}")
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_macro = np.mean([r["macro_f1"] for r in results])
    avg_weighted = np.mean([r["weighted_f1"] for r in results])
    print(f"avg accuracy:    {avg_acc:.4f}")
    print(f"avg macro_f1:    {avg_macro:.4f}")
    print(f"avg weighted_f1: {avg_weighted:.4f}")
    print("per-fold:")
    for r in results:
        print(f"  {r['test_subject']:<20}: acc={r['accuracy']:.4f}, macro_f1={r['macro_f1']:.4f}, model={r['best_model']}")

    summary = {"results": results, "avg": {"accuracy": avg_acc, "macro_f1": avg_macro, "weighted_f1": avg_weighted}}
    (OUTPUT_DIR / "9fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to: {OUTPUT_DIR / '9fold_summary.json'}")


if __name__ == "__main__":
    main()
