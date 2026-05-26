"""Per-Action LOSO evaluation of Causal RF with feature subset selection.

Given the action is known before Rep Segmentation, this script trains one model per action
using only the top-K most important features for that specific action.

Usage:
    python scripts/evaluate_per_action_feature_subset_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/per_action_feature_subset \
        --window-size 100 \
        --top-k 30 \
        --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")


def _get_feature_names(imu_columns: Sequence[str]) -> List[str]:
    """Generate feature names matching _extract_window_features_batch."""
    names = []
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "argmax", "argmin", "tv"]
    for col in imu_columns:
        for stat in stats:
            names.append(f"{col}_{stat}")
    names.extend(["mag_mean", "mag_std", "mag_max"])
    return names


def _select_top_k_features(
    per_action_importance: Dict,
    action: str,
    top_k: int,
    feature_names: List[str],
) -> Tuple[List[int], List[str]]:
    """Return column indices and names of top-K features for an action."""
    action_data = per_action_importance.get(action, {})
    top_features = action_data.get("top_features", [])
    
    if not top_features:
        # Fallback: use all features
        return list(range(len(feature_names))), feature_names
    
    # Sort by importance descending
    sorted_features = sorted(top_features, key=lambda x: x[1], reverse=True)
    selected_names = [name for name, _ in sorted_features[:top_k]]
    
    # Map names to indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_indices = []
    for name in selected_names:
        if name in name_to_idx:
            selected_indices.append(name_to_idx[name])
    
    # If somehow none matched, fallback to all
    if not selected_indices:
        return list(range(len(feature_names))), feature_names
    
    return selected_indices, selected_names


def _extract_action_from_stream_id(stream_id: str) -> str:
    """Extract action from stream_id like 'kevin/kevin/db_bench_press/set0'."""
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def _extract_subject_from_stream_id(stream_id: str) -> str:
    """Extract subject from stream_id."""
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def train_causal_rf_with_feature_subset(
    train_streams,
    imu_columns: Sequence[str],
    feature_indices: List[int],
    window_size: int = 100,
    stride: int = 10,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
) -> object:
    """Train causal RF using only selected feature columns."""
    X_all, y_all = [], []
    for stream_idx, (_, df) in enumerate(train_streams, start=1):
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        label_idx = np.array([cb.MICRO_LABELS.index(str(l)) for l in labels], dtype=np.int64)
        X_batch, ends = crf._build_trailing_feature_matrix(x, int(window_size), int(stride))
        if len(X_batch):
            # Select only the specified feature columns
            X_selected = X_batch[:, feature_indices]
            X_all.append(X_selected)
            y_all.append(label_idx[np.maximum(0, ends - 1)])
        if stream_idx % 25 == 0 or stream_idx == len(train_streams):
            print(f"  [CausalRF] prepared {stream_idx}/{len(train_streams)} train streams", flush=True)
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, len(feature_indices)), dtype=np.float32)
    y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.int64)
    print(
        f"  [CausalRF] Training on {len(X_all)} trailing windows "
        f"({window_size} samples, stride {stride}, trees={n_estimators}, max_depth={max_depth}, "
        f"features={len(feature_indices)}/{len(_get_feature_names(imu_columns))})",
        flush=True,
    )
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples) if max_samples and max_samples < 1.0 else max_samples,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    clf.fit(X_all, y_all)
    clf.verbose = 0
    return clf


def predict_causal_rf_with_feature_subset(
    clf,
    df,
    imu_columns: Sequence[str],
    feature_indices: List[int],
    window_size: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Predict using only selected feature columns."""
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    n = len(x)
    probs = np.zeros((n, len(cb.MICRO_LABELS)), dtype=np.float32)
    class_map = {int(c): i for i, c in enumerate(clf.classes_)}
    X_batch, ends = crf._build_trailing_feature_matrix(x, int(window_size), int(stride))
    if len(X_batch):
        X_selected = X_batch[:, feature_indices]
        raw_batch = clf.predict_proba(X_selected)
        full_batch = np.zeros((len(raw_batch), len(cb.MICRO_LABELS)), dtype=np.float32)
        for cls_idx, mi in class_map.items():
            full_batch[:, cls_idx] = raw_batch[:, mi]
        probs[np.maximum(0, ends - 1)] = full_batch
    return probs


def run_action_loso(
    action: str,
    all_streams: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    subjects: List[str],
    imu_columns: Sequence[str],
    mm_cfg,
    per_action_importance: Dict,
    all_actions: List[str],
    window_size: int = 100,
    top_k: int = 30,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
) -> Dict:
    """Run LOSO for a single action with feature subset selection."""
    print(f"\n{'='*60}")
    print(f"ACTION: {action} | Top-{top_k} features")
    print(f"{'='*60}")
    
    # Filter streams for this action
    action_streams = [(sid, df) for sid, df in all_streams if _extract_action_from_stream_id(sid) == action]
    print(f"[INFO] Total {action} streams: {len(action_streams)}")
    
    if not action_streams:
        return {"action": action, "error": "No streams found"}
    
    # Get subjects that have this action
    action_subjects = sorted({_extract_subject_from_stream_id(sid) for sid, _ in action_streams})
    print(f"[INFO] Subjects with {action}: {action_subjects}")
    
    # Select features for this action
    feature_names = _get_feature_names(imu_columns)
    feature_indices, selected_names = _select_top_k_features(
        per_action_importance, action, top_k, feature_names
    )
    print(f"[INFO] Selected {len(feature_indices)} features: {selected_names[:5]}...")
    
    all_results = []
    for test_subject in subjects:
        if test_subject not in action_subjects:
            print(f"[Skip] {test_subject} has no {action} data")
            continue
        
        # Check resume
        fold_file = output_dir / action / f"fold_{test_subject}.json"
        if fold_file.exists():
            print(f"[Resume] Loading {fold_file}")
            with open(fold_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue
        
        # Split streams for this action
        train_streams = [(sid, df) for sid, df in action_streams 
                        if _extract_subject_from_stream_id(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in action_streams 
                       if _extract_subject_from_stream_id(sid) == test_subject]
        
        if not test_streams:
            print(f"[Skip] No test streams for {test_subject}/{action}")
            continue
        
        print(f"\n[Fold] action={action} test={test_subject} train={len(train_streams)} test={len(test_streams)}")
        
        # Compute z-score stats
        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
        
        # Train causal RF with feature subset
        t0 = time.time()
        clf = train_causal_rf_with_feature_subset(
            train_z, imu_columns, feature_indices,
            window_size=int(window_size), stride=int(train_stride),
            n_estimators=int(n_estimators), max_depth=int(max_depth), max_samples=float(max_samples),
        )
        train_time = time.time() - t0
        
        # Predict and smooth
        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
            probs = predict_causal_rf_with_feature_subset(
                clf, df, imu_columns, feature_indices, window_size=int(window_size), stride=1
            )
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 10 == 0 or stream_idx == len(test_z):
                print(f"  [CausalRF] predicted {stream_idx}/{len(test_z)} test streams", flush=True)
        
        # Apply smoothing
        smooth_w = smoothing_window
        smoothed_streams = []
        for stream_id, df, probs in raw_prob_cache:
            cur = probs
            if int(smooth_w) > 1:
                smoothed = np.zeros_like(probs)
                csum = np.cumsum(probs, axis=0)
                for i in range(len(probs)):
                    start = max(0, i - int(smooth_w) + 1)
                    total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
                    smoothed[i] = total / float(i - start + 1)
                cur = smoothed
            smoothed_streams.append((stream_id, df, cur))
        
        def predict_fn(df, _cache_iter=iter(smoothed_streams)):
            _, _, cached_probs = next(_cache_iter)
            return cached_probs, None
        
        macro_classes = [cb.OTHER_LABEL] + [a for a in all_actions if a != cb.OTHER_LABEL]
        results = cb.evaluate_all_streams(predict_fn, test_z, macro_classes, mm_cfg)
        results["model_name"] = f"Causal RF + Feature Subset (Top-{top_k})"
        results["evaluation_protocol"] = "loso_per_action"
        results["test_subject"] = test_subject
        results["action"] = action
        results["train_time_s"] = train_time
        results["smoothing_window"] = smooth_w
        results["selected_features"] = selected_names
        results["config"] = {
            "window_size": int(window_size), "train_stride": int(train_stride),
            "smoothing_window": int(smooth_w), "n_estimators": int(n_estimators),
            "max_depth": int(max_depth), "max_samples": float(max_samples),
            "top_k": int(top_k),
        }
        results.pop("stream_rows", None)
        
        # Save immediately
        fold_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved: {fold_file} | Rep F1={results.get('rep_f1', 0):.4f}")
        all_results.append(results)
    
    # Aggregate across all folds for this action
    if not all_results:
        return {"action": action, "error": "No results"}
    
    total_tp = sum(r["tp"] for r in all_results)
    total_fp = sum(r["fp"] for r in all_results)
    total_fn = sum(r["fn"] for r in all_results)
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    
    overall = {
        "n_folds": len(all_results), "subjects": subjects,
        "streams": sum(r.get("stream_count", 0) for r in all_results),
        "n_true": sum(r["n_true"] for r in all_results), "n_pred": sum(r["n_pred"] for r in all_results),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": p, "recall": r, "rep_f1": f1,
        "start_mae_ms": float(np.mean([r.get("start_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("start_mae_ms", float("nan")))])),
        "end_mae_ms": float(np.mean([r.get("end_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("end_mae_ms", float("nan")))])),
        "transition_mae_ms": float(np.mean([r.get("transition_mae_ms", float("nan")) for r in all_results if np.isfinite(r.get("transition_mae_ms", float("nan")))])),
    }
    
    summary = {
        "action": action, "overall": overall,
        "fold_results": all_results,
    }
    
    # Save action summary
    with open(output_dir / action / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n[Action {action} Summary] Rep F1={f1:.4f}, Precision={p:.4f}, Recall={r:.4f}")
    
    return summary


def run_all_actions(
    config_path: Path,
    output_dir: Path,
    subjects: List[str] | None = None,
    actions: List[str] | None = None,
    window_size: int = 100,
    top_k: int = 30,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
    quick: bool = False,
) -> Dict:
    """Run per-action LOSO with feature subset selection for all actions."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    
    # Load all data
    print("[INFO] Loading all streams...")
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)
    
    if subjects is None:
        subjects = sorted(set(all_subjects))
    if actions is None:
        actions = available_actions
    
    if quick:
        subjects = subjects[:3]
        print(f"[QUICK MODE] Using first 3 subjects: {subjects}")
    
    print(f"[INFO] Subjects: {subjects}")
    print(f"[INFO] Actions: {actions}")
    print(f"[INFO] Total streams: {len(all_streams)}")
    
    # Load per-action feature importance
    importance_path = Path("artifacts/baseline_comparison/per_action_feature_importance/per_action_importance.json")
    if not importance_path.exists():
        raise FileNotFoundError(f"Per-action importance not found: {importance_path}. Run extract_per_action_feature_importance.py first.")
    with open(importance_path, "r") as f:
        per_action_importance = json.load(f)
    print(f"[INFO] Loaded per-action importance for {len(per_action_importance)} actions")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run per-action LOSO
    action_results = {}
    for action in actions:
        action_summary = run_action_loso(
            action=action,
            all_streams=all_streams,
            output_dir=output_dir,
            subjects=subjects,
            imu_columns=imu_columns,
            mm_cfg=mm_cfg,
            per_action_importance=per_action_importance,
            all_actions=actions,
            window_size=window_size,
            top_k=top_k,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            train_stride=train_stride,
            smoothing_window=smoothing_window,
        )
        action_results[action] = action_summary
        
        # Print progress
        completed = sum(1 for a in actions if a in action_results)
        f1s = [r["overall"]["rep_f1"] for r in action_results.values() if "overall" in r and "rep_f1" in r["overall"]]
        if f1s:
            print(f"\n[Progress] {completed}/{len(actions)} actions completed")
            print(f"  Mean Rep F1 across actions: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # Aggregate across all actions
    all_folds = []
    for action, summary in action_results.items():
        if "fold_results" in summary:
            all_folds.extend(summary["fold_results"])
    
    if all_folds:
        total_tp = sum(r["tp"] for r in all_folds)
        total_fp = sum(r["fp"] for r in all_folds)
        total_fn = sum(r["fn"] for r in all_folds)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        
        grand_summary = {
            "model": f"Causal RF + Feature Subset (Top-{top_k}, Per-Action LOSO)",
            "overall": {
                "n_folds": len(all_folds), "actions": actions, "subjects": subjects,
                "streams": sum(r.get("stream_count", 0) for r in all_folds),
                "n_true": sum(r["n_true"] for r in all_folds), "n_pred": sum(r["n_pred"] for r in all_folds),
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "precision": p, "recall": r, "rep_f1": f1,
            },
            "by_action": {action: res.get("overall", {}) for action, res in action_results.items()},
        }
        
        with open(output_dir / "grand_summary.json", "w", encoding="utf-8") as f:
            json.dump(grand_summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("GRAND SUMMARY (All Actions)")
        print(f"{'='*60}")
        print(json.dumps(grand_summary["overall"], indent=2))
        print(f"\n[OK] Results saved to {output_dir}")
        return grand_summary
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Per-Action LOSO with feature subset selection")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/per_action_feature_subset")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--quick", action="store_true", help="Run on first 3 subjects only")
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    actions = [a.strip() for a in args.actions.split(",") if a.strip()] if args.actions else None
    
    run_all_actions(
        Path(args.config), Path(args.output),
        subjects=subjects, actions=actions,
        window_size=args.window_size, top_k=args.top_k,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        max_samples=args.max_samples, train_stride=args.train_stride,
        smoothing_window=args.smoothing_window, quick=args.quick,
    )


if __name__ == "__main__":
    main()
