"""Compute feature importance WITH velocity features for per-action RF.

Trains one model per action on ALL data (no LOSO) using velocity-augmented
features, then extracts and ranks feature importances.

Usage:
    FEATURE_MODE=velocity python scripts/compute_feature_importance_with_velocity.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compare_baselines as cb
import importlib.util

def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")


def _build_feature_names(imu_columns: List[str], use_velocity: bool = False) -> List[str]:
    """Build feature name list matching _extract_window_features_batch order."""
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "argmax", "argmin", "total_variation"]
    names = []
    for col in imu_columns:
        for s in stats:
            names.append(f"{col}_{s}")
    names.extend(["mag_mean", "mag_std", "mag_max"])

    if use_velocity:
        for col in imu_columns:
            for s in stats:
                names.append(f"vel_{col}_{s}")
        names.extend(["vel_mag_mean", "vel_mag_std", "vel_mag_max"])

    return names


def train_rf_all_data(streams, imu_columns, window_size=100, stride=10):
    """Train RF on all provided streams."""
    X_all, y_all = [], []
    for _, df in streams:
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, ends = crf._build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(X_batch):
            X_all.append(X_batch)
            y_all.append(label_idx[np.maximum(0, ends - 1)])
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, 0), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, max_samples=0.7, n_jobs=-1, random_state=42)
    clf.fit(X_all, y_all)
    return clf


def main():
    os.environ["FEATURE_MODE"] = "velocity"

    config_path = ROOT / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)

    # Z-score normalize using all data
    stats = cb.compute_train_stats([df for _, df in all_streams], imu_columns)
    all_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in all_streams]

    # Group by action
    action_streams = {}
    for sid, df in all_z:
        action = sid.split("/")[-2] if len(sid.split("/")) >= 3 else "unknown"
        if action not in action_streams:
            action_streams[action] = []
        action_streams[action].append((sid, df))

    feature_names = _build_feature_names(imu_columns, use_velocity=True)
    print(f"Total features: {len(feature_names)}")
    print(f"Feature names (first 10): {feature_names[:10]}")
    print(f"Feature names (velocity first 10): {feature_names[63:73]}")

    results = {}
    for action, streams in action_streams.items():
        if action == cb.OTHER_LABEL:
            continue
        print(f"\n{'='*60}")
        print(f"Action: {action} | Streams: {len(streams)}")
        print(f"{'='*60}")

        clf = train_rf_all_data(streams, imu_columns, window_size=100, stride=10)
        importances = clf.feature_importances_

        # Rank all features
        ranked = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)

        print(f"\nTop 20 features:")
        for idx, (fi, imp) in enumerate(ranked[:20], 1):
            name = feature_names[fi] if fi < len(feature_names) else f"feat_{fi}"
            print(f"  {idx:2d}. {name:25s} {imp:.6f}")

        # Show which velocity features are in top 30
        print(f"\nVelocity features in top 30:")
        vel_in_top30 = [(feature_names[fi], imp) for fi, imp in ranked[:30] if fi >= 63]
        if vel_in_top30:
            for name, imp in vel_in_top30:
                print(f"  - {name:25s} {imp:.6f}")
        else:
            print("  (none)")

        # Show top 5 velocity features overall
        print(f"\nTop 10 velocity features (ranked among all 126):")
        vel_ranked = [(fi, imp) for fi, imp in ranked if fi >= 63]
        for idx, (fi, imp) in enumerate(vel_ranked[:10], 1):
            name = feature_names[fi]
            overall_rank = next(i for i, (f, _) in enumerate(ranked, 1) if f == fi)
            print(f"  {idx:2d}. {name:25s} {imp:.6f} (overall rank #{overall_rank})")

        results[action] = {
            "top_20": [(feature_names[fi], float(imp)) for fi, imp in ranked[:20]],
            "velocity_in_top_30": [(feature_names[fi], float(imp)) for fi, imp in ranked[:30] if fi >= 63],
            "top_10_velocity": [(feature_names[fi], float(imp)) for fi, imp in vel_ranked[:10]],
        }

    # Save results
    output_dir = ROOT / "artifacts" / "feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "velocity_feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Saved: {output_dir / 'velocity_feature_importance.json'}")


if __name__ == "__main__":
    main()
