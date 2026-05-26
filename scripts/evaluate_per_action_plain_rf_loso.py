"""Per-Action LOSO evaluation of plain Causal RF (no feature subset selection).

Trains one model per action using ALL features. This serves as the baseline
for evaluating the benefit of per-action feature subset selection.

Usage:
    python scripts/evaluate_per_action_plain_rf_loso.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/per_action_plain_rf \
        --window-size 100
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


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) >= 3:
        return parts[-2]
    return "unknown"


def _extract_subject_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[0] if parts else "unknown"


def run_action_loso(
    action: str,
    all_streams: List[Tuple[str, pd.DataFrame]],
    output_dir: Path,
    subjects: List[str],
    imu_columns: Sequence[str],
    mm_cfg,
    all_actions: List[str],
    window_size: int = 100,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
) -> Dict:
    """Run LOSO for a single action using all features."""
    print(f"\n{'='*60}")
    print(f"ACTION: {action} | All features")
    print(f"{'='*60}")
    
    action_streams = [(sid, df) for sid, df in all_streams if _extract_action_from_stream_id(sid) == action]
    print(f"[INFO] Total {action} streams: {len(action_streams)}")
    
    if not action_streams:
        return {"action": action, "error": "No streams found"}
    
    action_subjects = sorted({_extract_subject_from_stream_id(sid) for sid, _ in action_streams})
    print(f"[INFO] Subjects with {action}: {action_subjects}")
    
    all_results = []
    for test_subject in subjects:
        if test_subject not in action_subjects:
            print(f"[Skip] {test_subject} has no {action} data")
            continue
        
        fold_file = output_dir / action / f"fold_{test_subject}.json"
        if fold_file.exists():
            print(f"[Resume] Loading {fold_file}")
            with open(fold_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue
        
        train_streams = [(sid, df) for sid, df in action_streams 
                        if _extract_subject_from_stream_id(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in action_streams 
                       if _extract_subject_from_stream_id(sid) == test_subject]
        
        if not test_streams:
            print(f"[Skip] No test streams for {test_subject}/{action}")
            continue
        
        print(f"\n[Fold] action={action} test={test_subject} train={len(train_streams)} test={len(test_streams)}")
        
        stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
        test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
        
        # Train causal RF with ALL features (default behavior)
        t0 = time.time()
        clf = crf.train_causal_rf(
            train_z, imu_columns,
            window_size=int(window_size), stride=int(train_stride),
            n_estimators=int(n_estimators), max_depth=int(max_depth), max_samples=float(max_samples),
        )
        train_time = time.time() - t0
        
        # Predict and smooth
        raw_prob_cache = []
        for stream_idx, (stream_id, df) in enumerate(test_z, start=1):
            probs = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
            raw_prob_cache.append((stream_id, df, probs))
            if stream_idx % 10 == 0 or stream_idx == len(test_z):
                print(f"  [CausalRF] predicted {stream_idx}/{len(test_z)} test streams", flush=True)
        
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
        results["model_name"] = "Causal RF (Per-Action, All Features)"
        results["evaluation_protocol"] = "loso_per_action"
        results["test_subject"] = test_subject
        results["action"] = action
        results["train_time_s"] = train_time
        results["smoothing_window"] = smooth_w
        results["config"] = {
            "window_size": int(window_size), "train_stride": int(train_stride),
            "smoothing_window": int(smooth_w), "n_estimators": int(n_estimators),
            "max_depth": int(max_depth), "max_samples": float(max_samples),
        }
        results.pop("stream_rows", None)
        
        fold_file.parent.mkdir(parents=True, exist_ok=True)
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved: {fold_file} | Rep F1={results.get('rep_f1', 0):.4f}")
        all_results.append(results)
    
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
    
    summary = {"action": action, "overall": overall, "fold_results": all_results}
    
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
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.7,
    train_stride: int = 10,
    smoothing_window: int = 15,
    quick: bool = False,
) -> Dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    data_cfg = raw.get("data", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    
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
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    action_results = {}
    for action in actions:
        action_summary = run_action_loso(
            action=action,
            all_streams=all_streams,
            output_dir=output_dir,
            subjects=subjects,
            imu_columns=imu_columns,
            mm_cfg=mm_cfg,
            all_actions=actions,
            window_size=window_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
            train_stride=train_stride,
            smoothing_window=smoothing_window,
        )
        action_results[action] = action_summary
        
        completed = sum(1 for a in actions if a in action_results)
        f1s = [r["overall"]["rep_f1"] for r in action_results.values() if "overall" in r and "rep_f1" in r["overall"]]
        if f1s:
            print(f"\n[Progress] {completed}/{len(actions)} actions completed")
            print(f"  Mean Rep F1 across actions: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
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
            "model": "Causal RF (Per-Action, All Features)",
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
    parser = argparse.ArgumentParser(description="Per-Action LOSO with ALL features (baseline)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/per_action_plain_rf")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--smoothing-window", type=int, default=15)
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
        window_size=args.window_size,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        max_samples=args.max_samples, train_stride=args.train_stride,
        smoothing_window=args.smoothing_window, quick=args.quick,
    )


if __name__ == "__main__":
    main()
