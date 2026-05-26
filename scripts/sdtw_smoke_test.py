"""Quick smoke test for SDTW LOSO (3 subjects only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import write_report
from preprocessing.sdtw_rep_segmentation import (
    SDTWConfig,
    active_segments_from_phase,
    detect_reps_sdtw_templates,
    fit_sdtw_templates,
    infer_sample_rate_hz,
    match_segments,
    summarize_detection_metrics,
)
from preprocessing.micro_macro_segments import truth_reps_from_labels


def load_all_streams(config_path: Path, data_dir: Path, exclude_patterns: list[str] = None) -> List[Tuple[str, pd.DataFrame]]:
    """Load all CSV streams from data_dir."""
    import fnmatch
    if exclude_patterns is None:
        exclude_patterns = ["*whole_session*", "*_w", "*rest_after*", "*big_rest*"]
    
    streams = []
    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for action_dir in sorted(session_dir.iterdir()):
                if not action_dir.is_dir():
                    continue
                for set_dir in sorted(action_dir.iterdir()):
                    if not set_dir.is_dir():
                        continue
                    # Skip excluded directories
                    set_name = set_dir.name
                    if any(fnmatch.fnmatch(set_name, pat) for pat in exclude_patterns):
                        continue
                    for csv_file in sorted(set_dir.glob("*.csv")):
                        try:
                            df = pd.read_csv(csv_file)
                            if len(df) < 10:
                                continue
                            stream_id = f"{subject_dir.name}/{session_dir.name}/{action_dir.name}/{set_dir.name}/{csv_file.stem}"
                            streams.append((stream_id, df))
                        except Exception:
                            continue
    return streams


def run_sdtw_smoke(
    config_path: Path,
    output_dir: Path,
    test_subjects: List[str],
    actions: List[str] | None = None,
) -> Dict:
    """Run SDTW smoke test on selected subjects."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = Path(cfg.get("data", {}).get("data_dir", "./datasets/raw_data"))

    all_streams = load_all_streams(config_path, data_dir)
    print(f"[INFO] Loaded {len(all_streams)} total streams")

    all_results = []
    for test_subject in test_subjects:
        print(f"\n[Fold] test={test_subject}")
        
        train_streams = [(sid, df) for sid, df in all_streams 
                        if sid.split("/")[0] != test_subject]
        test_streams = [(sid, df) for sid, df in all_streams 
                       if sid.split("/")[0] == test_subject]
        
        if not test_streams:
            print(f"  [Skip] No test streams for {test_subject}")
            continue
        
        print(f"  Train: {len(train_streams)} streams, Test: {len(test_streams)} streams")
        
        # Fit templates per action
        sdtw_cfg = SDTWConfig()
        imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
        
        # Group train streams by action
        from collections import defaultdict
        action_streams = defaultdict(list)
        for sid, df in train_streams:
            action = sid.split("/")[2] if len(sid.split("/")) > 2 else "unknown"
            action_streams[action].append(df)
        
        templates = []
        for action, dfs in action_streams.items():
            try:
                action_templates = fit_sdtw_templates(action, dfs, imu_cols, sdtw_cfg)
                templates.extend(action_templates)
                print(f"  Action {action}: {len(action_templates)} templates")
            except Exception as e:
                print(f"  Action {action}: skip ({e})")
        
        print(f"  Total templates: {len(templates)}")
        
        # Evaluate test streams
        all_metrics = []
        for stream_id, stream_df in test_streams:
            sample_rate = infer_sample_rate_hz(stream_df)
            
            # Detect
            detections = detect_reps_sdtw_templates(stream_df, templates, imu_cols, sdtw_cfg)
            
            # Ground truth
            gt_reps = truth_reps_from_labels(
                stream_df["phase"].to_numpy(),
                actions=stream_df["action_type"].astype(str).to_numpy() if "action_type" in stream_df.columns else None,
                min_phase_samples=1,
            )
            truth = [(int(r.start_idx), int(r.end_idx)) for r in gt_reps]
            
            # Metrics
            metrics = summarize_detection_metrics(detections, truth, sample_rate)
            metrics["stream_id"] = stream_id
            all_metrics.append(metrics)
            print(f"  {stream_id}: F1={metrics['f1']:.3f}, n_true={metrics['n_true']:.0f}, n_pred={metrics['n_pred']:.0f}")
        
        # Aggregate
        total_tp = sum(m["tp"] for m in all_metrics)
        total_fp = sum(m["fp"] for m in all_metrics)
        total_fn = sum(m["fn"] for m in all_metrics)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        
        results = {
            "test_subject": test_subject,
            "n_train_streams": len(train_streams),
            "n_test_streams": len(test_streams),
            "precision": p,
            "recall": r,
            "rep_f1": f1,
            "stream_metrics": all_metrics,
        }
        
        fold_file = output_dir / f"fold_{test_subject}.json"
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  [OK] F1={f1:.3f}, P={p:.3f}, R={r:.3f} -> {fold_file}")
        all_results.append(results)
    
    # Grand summary
    if all_results:
        total_tp = sum(sum(m["tp"] for m in r["stream_metrics"]) for r in all_results)
        total_fp = sum(sum(m["fp"] for m in r["stream_metrics"]) for r in all_results)
        total_fn = sum(sum(m["fn"] for m in r["stream_metrics"]) for r in all_results)
        p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        
        summary = {
            "n_folds": len(all_results),
            "test_subjects": test_subjects,
            "precision": p,
            "recall": r,
            "rep_f1": f1,
            "n_true": sum(sum(m["n_true"] for m in r["stream_metrics"]) for r in all_results),
            "n_pred": sum(sum(m["n_pred"] for m in r["stream_metrics"]) for r in all_results),
        }
        
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print(f"SMOKE TEST SUMMARY: Rep F1={f1:.3f}, P={p:.3f}, R={r:.3f}")
        print(f"{'='*60}")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/sdtw_smoke")
    parser.add_argument("--subjects", default="haoyu,kevin,yoru")
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    run_sdtw_smoke(Path(args.config), Path(args.output), subjects)
