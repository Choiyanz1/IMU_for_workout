#!/usr/bin/env python3
"""Compute per-phase confusion matrix from a subset of streams."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

def _load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
from preprocessing.sdtw_rep_segmentation import infer_sample_rate_hz


def evaluate_phase_confusion(df, probs, sample_rate, smoothing_window=15):
    """Compute per-phase confusion matrix."""
    gt_labels = df["phase"].to_numpy()
    
    # Smooth probabilities
    if smoothing_window > 1:
        smoothed = np.zeros_like(probs)
        csum = np.cumsum(probs, axis=0)
        for i in range(len(probs)):
            start = max(0, i - smoothing_window + 1)
            total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
            count = i - start + 1
            smoothed[i] = total / float(count)
        probs = smoothed
    
    pred_labels = np.argmax(probs, axis=1)
    
    # Phase mapping
    phase_names = {0: "other", 1: "concentric", 2: "eccentric"}
    classes = [0, 1, 2]
    
    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for gt, pred in zip(gt_labels, pred_labels):
        cm[gt, pred] += 1
    
    # Per-class metrics
    results = {}
    for c in classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = cm[c, :].sum()
        results[phase_names[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        }
    
    overall_acc = np.mean(gt_labels == pred_labels)
    macro_f1 = np.mean([results[phase_names[c]]["f1"] for c in classes])
    
    return {
        "overall_accuracy": float(overall_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "per_class": results,
    }


def main():
    import yaml
    
    config_path = ROOT / "config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    
    print("[INFO] Loading streams...")
    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    all_streams, all_subjects, available_actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_streams = cb._resample_streams_to_rate(all_streams, imu_columns, time_column, target_sample_rate)
    
    # Pick a few representative subjects and actions
    test_subjects = ["haoyu", "kevin", "yanz"]
    test_actions = ["db_bench_press", "db_biceps_curl", "db_squat", "db_weighted_crunch"]
    
    all_confusions = []
    
    for action in test_actions:
        action_streams = [(sid, df) for sid, df in all_streams 
                         if sid.split("/")[-2] == action]
        
        for test_subject in test_subjects:
            print(f"\n[Eval] action={action}, test={test_subject}")
            
            train_streams = [(sid, df) for sid, df in action_streams
                            if sid.split("/")[0] != test_subject]
            test_streams = [(sid, df) for sid, df in action_streams
                           if sid.split("/")[0] == test_subject]
            
            if not test_streams:
                continue
            
            stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
            train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
            test_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
            
            # Train
            import evaluate_per_action_plain_rf_loso as paf
            clf = paf.train_per_action_causal_rf(
                train_z, imu_columns,
                window_size=100, stride=10,
                n_estimators=100, max_depth=15, max_samples=0.7,
            )
            
            # Evaluate a few test streams
            for stream_id, df in test_z[:3]:  # first 3 streams
                probs = paf.predict_per_action_causal_rf(clf, df, imu_columns, window_size=100, stride=1)
                sample_rate = infer_sample_rate_hz(df)
                result = evaluate_phase_confusion(df, probs, sample_rate)
                result["action"] = action
                result["subject"] = test_subject
                result["stream_id"] = stream_id
                all_confusions.append(result)
                
                print(f"  Stream: {stream_id}")
                print(f"    Accuracy: {result['overall_accuracy']:.3f}, Macro F1: {result['macro_f1']:.3f}")
                for phase, metrics in result["per_class"].items():
                    print(f"    {phase:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, N={metrics['support']}")
    
    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE PER-PHASE METRICS (across sampled streams)")
    print("=" * 80)
    
    phases = ["other", "concentric", "eccentric"]
    for phase in phases:
        precisions = [r["per_class"][phase]["precision"] for r in all_confusions]
        recalls = [r["per_class"][phase]["recall"] for r in all_confusions]
        f1s = [r["per_class"][phase]["f1"] for r in all_confusions]
        print(f"{phase:12s}: Precision={np.mean(precisions):.3f}±{np.std(precisions):.3f}, "
              f"Recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}, "
              f"F1={np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    
    all_accs = [r["overall_accuracy"] for r in all_confusions]
    all_f1s = [r["macro_f1"] for r in all_confusions]
    print(f"\nOverall: Accuracy={np.mean(all_accs):.3f}±{np.std(all_accs):.3f}, "
          f"Macro F1={np.mean(all_f1s):.3f}±{np.std(all_f1s):.3f}")


if __name__ == "__main__":
    main()
