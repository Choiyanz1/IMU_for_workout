"""
Action Classification: Real-time Post-processing Evaluation

Goal: Boost per-rep ~84% to set-level 95%+ using sliding-window strategies.

Strategies:
1. No post-processing (baseline)
2. Sliding Window Majority Vote (window=N reps)
3. Confidence-Aware Vote (only predictions with confidence > threshold)
4. Online Agreement (require K consecutive same predictions)

All evaluated on 9-fold LOSO LogReg predictions.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features

DATA_DIR = Path("datasets/raw_data")
OUTPUT_DIR = Path("artifacts/action_classification_postprocessing")
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


def extract_set_id_from_path(csv_path: Path, data_dir: Path) -> str:
    """Extract set identifier from path: subject/session/action/set/rep.csv"""
    try:
        rel = csv_path.relative_to(data_dir)
    except ValueError:
        rel = csv_path
    parts = rel.parts
    if len(parts) >= 4:
        # parts: [subject, session, action, set, rep.csv]
        return f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"
    return str(rel)


def prepare_sequences_with_paths():
    """Load sequences and also return their original file paths for set grouping."""
    import fnmatch
    sequences = []
    paths = []
    subjects = []

    for csv_path in sorted(DATA_DIR.rglob("*.csv")):
        # Skip excluded patterns
        skip = False
        for pat in EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(csv_path.name, pat) or any(fnmatch.fnmatch(p, pat) for p in csv_path.relative_to(DATA_DIR).parts):
                skip = True
                break
        if skip:
            continue

        df = pd.read_csv(csv_path)

        # Infer metadata from path if missing
        rel = csv_path.relative_to(DATA_DIR)
        parts = rel.parts
        if len(parts) >= 3:
            if FEATURE_CFG.subject_column not in df.columns:
                df[FEATURE_CFG.subject_column] = parts[0]
            if FEATURE_CFG.label_column not in df.columns:
                df[FEATURE_CFG.label_column] = parts[2]

        # Filter to included actions
        if df[FEATURE_CFG.label_column].iloc[0] not in INCLUDE_ACTIONS:
            continue

        sequences.append(df)
        paths.append(csv_path)
        subjects.append(str(df[FEATURE_CFG.subject_column].iloc[0]))

    return sequences, paths, subjects


def evaluate_fold_with_predictions(train_seqs, test_seqs, test_paths, test_subj):
    """Train LogReg and return per-rep predictions with set IDs."""
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

    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)
    max_confidence = y_proba.max(axis=1)

    # Build per-rep records with set IDs
    records = []
    for i, path in enumerate(test_paths):
        set_id = extract_set_id_from_path(path, DATA_DIR)
        records.append({
            "set_id": set_id,
            "subject": test_subj,
            "true_label": y_test[i],
            "pred_label": y_pred[i],
            "confidence": float(max_confidence[i]),
        })

    return records


def sliding_window_vote(df_fold: pd.DataFrame, window_size: int) -> float:
    """
    Simulate real-time sliding window majority vote.
    For each rep, use the majority vote of [current - W + 1 : current + 1] reps
    within the same set.
    """
    all_correct = []
    for set_id, group in df_fold.groupby("set_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        votes = []
        for i in range(n):
            start = max(0, i - window_size + 1)
            window_preds = group.loc[start:i+1, "pred_label"].tolist()
            vote = Counter(window_preds).most_common(1)[0][0]
            votes.append(vote)

        # Set-level accuracy: did the final vote (at the end of the set) match GT?
        # Actually, for real-time, we want per-rep accuracy after applying vote
        correct = sum(v == t for v, t in zip(votes, group["true_label"]))
        all_correct.append(correct)
        all_correct.append(len(group))  # total reps for this set

    # Calculate per-rep accuracy after sliding window
    # Actually let's return both per-rep and set-level
    total_reps = 0
    correct_reps = 0
    correct_sets = 0
    total_sets = 0

    for set_id, group in df_fold.groupby("set_id"):
        group = group.reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        votes = []
        for i in range(n):
            start = max(0, i - window_size + 1)
            window_preds = group.loc[start:i+1, "pred_label"].tolist()
            vote = Counter(window_preds).most_common(1)[0][0]
            votes.append(vote)

        # Per-rep accuracy
        correct_reps += sum(v == t for v, t in zip(votes, group["true_label"]))
        total_reps += n

        # Set-level: final vote at end of set
        final_vote = votes[-1]
        true_label = group["true_label"].iloc[0]
        if final_vote == true_label:
            correct_sets += 1
        total_sets += 1

    rep_acc = correct_reps / total_reps if total_reps > 0 else 0
    set_acc = correct_sets / total_sets if total_sets > 0 else 0
    return rep_acc, set_acc


def confidence_threshold_vote(df_fold: pd.DataFrame, threshold: float) -> float:
    """Only accept predictions with confidence > threshold."""
    total_sets = 0
    correct_sets = 0
    total_reps = 0
    correct_reps = 0

    for set_id, group in df_fold.groupby("set_id"):
        n = len(group)
        if n == 0:
            continue

        # High-confidence predictions only
        valid_preds = group[group["confidence"] >= threshold]["pred_label"]
        if len(valid_preds) == 0:
            # Fallback: use all predictions
            valid_preds = group["pred_label"]

        vote = Counter(valid_preds).most_common(1)[0][0]
        true_label = group["true_label"].iloc[0]

        # Count correct reps (those that match the voted label)
        correct_reps += sum(group["pred_label"] == vote)
        total_reps += n

        if vote == true_label:
            correct_sets += 1
        total_sets += 1

    rep_acc = correct_reps / total_reps if total_reps > 0 else 0
    set_acc = correct_sets / total_sets if total_sets > 0 else 0
    return rep_acc, set_acc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Action Classification: Post-processing Evaluation")
    print("="*70)

    print("\n[1/3] Loading sequences with paths...")
    sequences, paths, subjects = prepare_sequences_with_paths()
    all_subjects = sorted(set(subjects))
    print(f"[2/3] Loaded {len(sequences)} sequences from {len(all_subjects)} subjects")

    # Collect all predictions across 9 folds
    all_records = []
    for i, test_subj in enumerate(all_subjects):
        print(f"\n[3/3] Fold {i+1}/{len(all_subjects)}: test={test_subj}")
        train_subjs = [s for s in all_subjects if s != test_subj]

        train_seqs = [seq for seq, subj in zip(sequences, subjects) if subj in train_subjs]
        test_seqs = [seq for seq, subj in zip(sequences, subjects) if subj == test_subj]
        test_paths = [path for path, subj in zip(paths, subjects) if subj == test_subj]

        print(f"      train={len(train_seqs)}, test={len(test_seqs)}")
        records = evaluate_fold_with_predictions(train_seqs, test_seqs, test_paths, test_subj)
        all_records.extend(records)
        print(f"      => collected {len(records)} rep predictions")

    df_all = pd.DataFrame(all_records)
    print(f"\n[OK] Total predictions collected: {len(df_all)} reps across {df_all['set_id'].nunique()} sets")

    # Baseline (no post-processing)
    baseline_rep_acc = accuracy_score(df_all["true_label"], df_all["pred_label"])
    baseline_set_acc = 0
    for set_id, group in df_all.groupby("set_id"):
        vote = Counter(group["pred_label"]).most_common(1)[0][0]
        if vote == group["true_label"].iloc[0]:
            baseline_set_acc += 1
    baseline_set_acc /= df_all["set_id"].nunique()

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nBaseline (no post-processing):")
    print(f"  Per-rep accuracy:  {baseline_rep_acc:.4f}")
    print(f"  Set-level accuracy: {baseline_set_acc:.4f}")

    # Strategy 1: Sliding Window Majority Vote
    print(f"\n--- Sliding Window Majority Vote ---")
    print(f"{'Window Size':<15} {'Per-rep Acc':>12} {'Set-level Acc':>15}")
    print("-" * 45)
    for w in [1, 3, 5, 7, 10, 999]:
        rep_acc, set_acc = sliding_window_vote(df_all, w)
        label = f"W={w}" if w < 999 else "W=ALL (full set)"
        print(f"{label:<15} {rep_acc:>12.4f} {set_acc:>15.4f}")

    # Strategy 2: Confidence Thresholding
    print(f"\n--- Confidence-Aware Majority Vote ---")
    print(f"{'Threshold':<15} {'Per-rep Acc':>12} {'Set-level Acc':>15}")
    print("-" * 45)
    for thresh in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        rep_acc, set_acc = confidence_threshold_vote(df_all, thresh)
        label = f"thresh={thresh}"
        print(f"{label:<15} {rep_acc:>12.4f} {set_acc:>15.4f}")

    # Save detailed results
    df_all.to_csv(OUTPUT_DIR / "all_predictions.csv", index=False)
    print(f"\n[OK] Predictions saved to: {OUTPUT_DIR / 'all_predictions.csv'}")


if __name__ == "__main__":
    main()
