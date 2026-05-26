"""
Action Classification: Improved Post-processing Strategies

Key insight: Set-level accuracy requires (1) good base model + (2) proper voting.
Current LogReg gives 87% set-level. Need 95%.

Strategies tested:
1. Cumulative Vote (vote grows stronger as more reps are seen)
2. Delayed Confirmation (only output after N reps with agreement)
3. Confidence-Weighted Vote (weight by prediction confidence)
4. Hybrid: Cumulative + Confidence threshold
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
OUTPUT_DIR = Path("artifacts/action_classification_postprocessing_v2")
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


def prepare_sequences_with_paths():
    import fnmatch
    sequences = []
    paths = []
    subjects = []
    for csv_path in sorted(DATA_DIR.rglob("*.csv")):
        skip = False
        for pat in EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(csv_path.name, pat) or any(fnmatch.fnmatch(p, pat) for p in csv_path.relative_to(DATA_DIR).parts):
                skip = True
                break
        if skip:
            continue
        df = pd.read_csv(csv_path)
        rel = csv_path.relative_to(DATA_DIR)
        parts = rel.parts
        if len(parts) >= 3:
            if FEATURE_CFG.subject_column not in df.columns:
                df[FEATURE_CFG.subject_column] = parts[0]
            if FEATURE_CFG.label_column not in df.columns:
                df[FEATURE_CFG.label_column] = parts[2]
        if df[FEATURE_CFG.label_column].iloc[0] not in INCLUDE_ACTIONS:
            continue
        sequences.append(df)
        paths.append(csv_path)
        subjects.append(str(df[FEATURE_CFG.subject_column].iloc[0]))
    return sequences, paths, subjects


def extract_set_id(path: Path) -> str:
    rel = path.relative_to(DATA_DIR)
    parts = rel.parts
    if len(parts) >= 4:
        return "/".join(parts[:4])
    return str(rel)


def get_predictions(train_seqs, test_seqs, test_paths, test_subj):
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
    max_conf = y_proba.max(axis=1)

    records = []
    for i, path in enumerate(test_paths):
        records.append({
            "set_id": extract_set_id(path),
            "subject": test_subj,
            "true_label": y_test[i],
            "pred_label": y_pred[i],
            "confidence": float(max_conf[i]),
            "rep_idx": i,
        })
    return records


def eval_strategy(df_all: pd.DataFrame, strategy_name: str, strategy_fn) -> dict:
    """Evaluate a post-processing strategy across all sets."""
    total_sets = 0
    correct_sets = 0
    total_reps = 0
    correct_reps = 0

    for set_id, group in df_all.groupby("set_id"):
        group = group.sort_values("rep_idx").reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue

        true_label = group["true_label"].iloc[0]
        preds = group["pred_label"].tolist()
        confs = group["confidence"].tolist()

        voted_reps, final_vote = strategy_fn(preds, confs, true_label)

        total_reps += n
        correct_reps += sum(voted_reps)
        total_sets += 1
        if final_vote == true_label:
            correct_sets += 1

    return {
        "strategy": strategy_name,
        "per_rep_acc": correct_reps / total_reps if total_reps > 0 else 0,
        "set_acc": correct_sets / total_sets if total_sets > 0 else 0,
        "total_sets": total_sets,
    }


def strategy_baseline(preds, confs, true_label):
    """No post-processing."""
    voted = [p == true_label for p in preds]
    final = Counter(preds).most_common(1)[0][0]
    return voted, final


def strategy_cumulative_vote(preds, confs, true_label):
    """At each rep, vote using all reps seen so far in the set."""
    voted = []
    for i in range(len(preds)):
        vote = Counter(preds[:i+1]).most_common(1)[0][0]
        voted.append(vote == true_label)
    final = Counter(preds).most_common(1)[0][0]
    return voted, final


def strategy_sliding_window(preds, confs, true_label, window=5):
    """Sliding window majority vote."""
    voted = []
    for i in range(len(preds)):
        start = max(0, i - window + 1)
        vote = Counter(preds[start:i+1]).most_common(1)[0][0]
        voted.append(vote == true_label)
    final = Counter(preds).most_common(1)[0][0]
    return voted, final


def strategy_confidence_weighted(preds, confs, true_label):
    """Weight each rep's vote by its confidence."""
    voted = []
    for i in range(len(preds)):
        # Cumulative confidence-weighted vote
        weights = {}
        for j in range(i+1):
            p = preds[j]
            weights[p] = weights.get(p, 0) + confs[j]
        vote = max(weights, key=weights.get)
        voted.append(vote == true_label)
    final = max(weights, key=weights.get)
    return voted, final


def strategy_delayed_confirmation(preds, confs, true_label, min_reps=3):
    """Only start outputting after min_reps, then use cumulative vote."""
    voted = []
    for i in range(len(preds)):
        if i < min_reps - 1:
            # Before threshold: use individual prediction
            voted.append(preds[i] == true_label)
        else:
            # After threshold: use cumulative vote
            vote = Counter(preds[:i+1]).most_common(1)[0][0]
            voted.append(vote == true_label)
    final = Counter(preds).most_common(1)[0][0]
    return voted, final


def strategy_high_conf_only(preds, confs, true_label, thresh=0.9):
    """Only use predictions with confidence >= thresh for final vote."""
    voted = []
    # Per-rep: if this rep's confidence is high, use it; otherwise use cumulative vote
    for i in range(len(preds)):
        if confs[i] >= thresh:
            vote = preds[i]
        else:
            vote = Counter(preds[:i+1]).most_common(1)[0][0]
        voted.append(vote == true_label)

    # Final vote: only high-confidence predictions
    high_conf_preds = [p for p, c in zip(preds, confs) if c >= thresh]
    if len(high_conf_preds) == 0:
        final = Counter(preds).most_common(1)[0][0]
    else:
        final = Counter(high_conf_preds).most_common(1)[0][0]
    return voted, final


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("Action Classification: Post-processing V2")
    print("="*70)

    print("\n[1/3] Loading...")
    sequences, paths, subjects = prepare_sequences_with_paths()
    all_subjects = sorted(set(subjects))
    print(f"[2/3] Loaded {len(sequences)} sequences")

    all_records = []
    for i, test_subj in enumerate(all_subjects):
        print(f"\n[3/3] Fold {i+1}/{len(all_subjects)}: test={test_subj}")
        train_subjs = [s for s in all_subjects if s != test_subj]
        train_seqs = [seq for seq, subj in zip(sequences, subjects) if subj in train_subjs]
        test_seqs = [seq for seq, subj in zip(sequences, subjects) if subj == test_subj]
        test_paths = [path for path, subj in zip(paths, subjects) if subj == test_subj]
        records = get_predictions(train_seqs, test_seqs, test_paths, test_subj)
        all_records.extend(records)

    df_all = pd.DataFrame(all_records)
    print(f"\n[OK] {len(df_all)} reps, {df_all['set_id'].nunique()} sets")

    # Evaluate all strategies
    results = []
    results.append(eval_strategy(df_all, "Baseline (no post-proc)", strategy_baseline))
    results.append(eval_strategy(df_all, "Cumulative Vote", strategy_cumulative_vote))
    results.append(eval_strategy(df_all, "Sliding W=3", lambda p, c, t: strategy_sliding_window(p, c, t, 3)))
    results.append(eval_strategy(df_all, "Sliding W=5", lambda p, c, t: strategy_sliding_window(p, c, t, 5)))
    results.append(eval_strategy(df_all, "Sliding W=7", lambda p, c, t: strategy_sliding_window(p, c, t, 7)))
    results.append(eval_strategy(df_all, "Confidence-Weighted", strategy_confidence_weighted))
    results.append(eval_strategy(df_all, "Delayed (min=3)", lambda p, c, t: strategy_delayed_confirmation(p, c, t, 3)))
    results.append(eval_strategy(df_all, "Delayed (min=5)", lambda p, c, t: strategy_delayed_confirmation(p, c, t, 5)))
    results.append(eval_strategy(df_all, "High-Conf (thresh=0.9)", strategy_high_conf_only))

    print(f"\n{'='*70}")
    print("POST-PROCESSING COMPARISON (LogReg Base Model)")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'Per-rep Acc':>12} {'Set-level Acc':>15}")
    print("-" * 60)
    for r in results:
        print(f"{r['strategy']:<30} {r['per_rep_acc']:>12.4f} {r['set_acc']:>15.4f}")

    # Save
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "postprocessing_comparison.csv", index=False)
    print(f"\n[OK] Saved to: {OUTPUT_DIR / 'postprocessing_comparison.csv'}")

    # Best strategy
    best = max(results, key=lambda x: x["set_acc"])
    print(f"\n[RECOMMENDATION]")
    print(f"Best strategy: {best['strategy']}")
    print(f"  Set-level accuracy: {best['set_acc']:.1%}")
    print(f"  Per-rep accuracy:   {best['per_rep_acc']:.1%}")
    print(f"  Note: Need AutoGluon base model (94.8% per-rep) to reach 95%+ set-level.")


if __name__ == "__main__":
    main()
