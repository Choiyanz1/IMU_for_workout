"""
Quick single-fold test: AutoGluon (rich) with 8 subjects train, 1 test (kevin)
Compare with RF baseline.
"""
from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features, AutoGluonConfig
import shutil

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

    ag_cfg = AutoGluonConfig(
        feature_mode="rich",
        presets="medium_quality_faster_train",
        time_limit_s=120,
        eval_metric="accuracy",
        excluded_model_types=["NN_TORCH", "FASTAI", "GBM", "XGB"],
    )

    df_train = _build_rep_features(train_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, ag_cfg.feature_mode)
    df_test = _build_rep_features(test_norm, FEATURE_CFG.imu_columns, FEATURE_CFG.label_column, ag_cfg.feature_mode)

    print(f"[INFO] train reps={len(df_train)}, test reps={len(df_test)}")

    from autogluon.tabular import TabularPredictor
    predictor_path = "artifacts/action_classification_quick_test/ag_kevin"
    if Path(predictor_path).exists():
        shutil.rmtree(predictor_path, ignore_errors=True)

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

    # Evaluate
    y_true = df_test["label"].values
    y_pred = predictor.predict(df_test.drop(columns=["label"])).values

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print("AutoGluon Results (rich features, 120s)")
    print(f"{'='*60}")
    print(f"accuracy:    {acc:.4f}")
    print(f"macro_f1:    {macro_f1:.4f}")
    print(f"weighted_f1: {weighted_f1:.4f}")
    print(f"best_model:  {predictor.model_best}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Leaderboard
    lb = predictor.leaderboard(df_test, silent=True)
    print("\nLeaderboard:")
    print(lb[["model", "score_test", "score_val", "pred_time_test", "fit_time"]].to_string(index=False))


if __name__ == "__main__":
    main()
