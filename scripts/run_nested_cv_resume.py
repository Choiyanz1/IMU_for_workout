"""
Resume-capable 9-fold nested CV for per-action RF + Boundary Refiner.

Usage:
    python scripts/run_nested_cv_resume.py

This script:
1. Scans existing artifacts/baseline_comparison/ for already-completed outer folds.
2. For missing subjects, runs benchmark_per_action_rf_refiner.py with medium-fidelity settings.
3. After each subject finishes, merges into a unified results file.
4. Supports --resume: re-run the script and it will skip completed folds.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ALL_SUBJECTS = ["haoyu", "hsianshun", "kevin", "thomas", "tsenyu", "yanz", "yoru", "yushuan", "ziho"]

# Unified output directory
UNIFIED_OUTPUT = Path("artifacts/baseline_comparison/nested_cv_9fold_unified")

# Configuration for the nested benchmark runs
# Using medium-fidelity settings (relaxed from aggressive smoke, but not full budget)
BENCHMARK_CONFIG = {
    "script": "scripts/benchmark_per_action_rf_refiner.py",
    "config": "configs/micro_macro_recognition_8act_test_yushuan.yaml",
    # Modality-only search: faster, less overfitting than full window search
    "modality_only_search": True,
    "selection_window_size": 50,
    "selection_edge_window": 20,
    # Search all 7 modality combinations (but modality-only means window is fixed)
    "modalities": "acc,gyro,mag,acc+gyro,acc+mag,gyro+mag,acc+gyro+mag",
    # Use all 8 actions from the config (config already has 8)
    # "include_actions": "",  # empty = use all from config
    # Medium-fidelity RF / refiner budget
    "n_estimators": 50,
    "max_depth": 15,
    "max_samples": 0.7,
    "refiner_n_estimators": 150,
    "refiner_max_depth": 16,
    "refiner_min_samples_leaf": 2,
    "target_matched_reps": 300,
    "max_refiner_train_streams": 30,
    "match_iou_train": 0.3,
    "max_shift": 20,
    "train_stride": 10,
    # Inner validation: use all train subjects except outer
    "max_inner_subjects": 0,  # 0 = use all available inner subjects
    # Default modality guardrail (prevent collapsing to single modality if it hurts recall)
    "disable_default_modality_guardrail": False,
    "default_modality_min_improvement": 0.01,
    "default_modality_max_recall_drop": 0.02,
    # Parallelism
    "parallel_action_jobs": 1,
    "parallel_outer_jobs": 1,
}


def find_existing_folds() -> dict[str, Path]:
    """Scan artifacts/baseline_comparison/ for nested CV results with fold data."""
    existing = {}
    base = Path("artifacts/baseline_comparison")
    if not base.exists():
        return existing
    
    for result_file in base.rglob("*/results.json"):
        try:
            if not result_file.exists():
                continue
            data = json.loads(result_file.read_text(encoding="utf-8"))
            if "folds" not in data or "tuned_overall" not in data:
                continue
            # This is a nested CV result
            for fold_name in data.get("folds", {}).keys():
                if fold_name in ALL_SUBJECTS:
                    if fold_name not in existing:
                        existing[fold_name] = result_file.parent
        except Exception:
            continue
    return existing


def run_nested_benchmark_for_subject(subject: str, output_dir: Path) -> bool:
    """Run benchmark_per_action_rf_refiner.py for a single outer subject."""
    cfg = BENCHMARK_CONFIG
    
    cmd = [
        sys.executable, cfg["script"],
        "--config", cfg["config"],
        "--output", str(output_dir),
        "--outer-subjects", subject,
        "--modalities", cfg["modalities"],
        "--n-estimators", str(cfg["n_estimators"]),
        "--max-depth", str(cfg["max_depth"]),
        "--max-samples", str(cfg["max_samples"]),
        "--refiner-n-estimators", str(cfg["refiner_n_estimators"]),
        "--refiner-max-depth", str(cfg["refiner_max_depth"]),
        "--refiner-min-samples-leaf", str(cfg["refiner_min_samples_leaf"]),
        "--target-matched-reps", str(cfg["target_matched_reps"]),
        "--max-refiner-train-streams", str(cfg["max_refiner_train_streams"]),
        "--match-iou-train", str(cfg["match_iou_train"]),
        "--max-shift", str(cfg["max_shift"]),
        "--train-stride", str(cfg["train_stride"]),
        "--selection-metric", "rep_f1",
        "--parallel-action-jobs", str(cfg["parallel_action_jobs"]),
        "--parallel-outer-jobs", str(cfg["parallel_outer_jobs"]),
        "--default-modality-min-improvement", str(cfg["default_modality_min_improvement"]),
        "--default-modality-max-recall-drop", str(cfg["default_modality_max_recall_drop"]),
    ]
    
    if cfg["modality_only_search"]:
        cmd.append("--modality-only-search")
        cmd.extend(["--selection-window-size", str(cfg["selection_window_size"])])
        cmd.extend(["--selection-edge-window", str(cfg["selection_edge_window"])])
    
    if cfg.get("max_inner_subjects", 0) > 0:
        cmd.extend(["--max-inner-subjects", str(cfg["max_inner_subjects"])])
    
    if cfg["disable_default_modality_guardrail"]:
        cmd.append("--disable-default-modality-guardrail")
    
    print(f"\n{'='*60}")
    print(f"Running nested CV for outer subject: {subject}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed = time.time() - start
        print(f"\n[OK] Subject {subject} completed in {elapsed/60:.1f} minutes.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n[ERROR] Subject {subject} failed after {elapsed/60:.1f} minutes.")
        print(f"Return code: {e.returncode}")
        return False


def merge_into_unified(existing_folds: dict[str, Path]) -> None:
    """Merge all existing nested CV results into a unified 9-fold report."""
    UNIFIED_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    unified = {
        "benchmark": "per_action_rf_nested_cv_9fold_unified",
        "config": BENCHMARK_CONFIG["config"],
        "description": "Resume-capable 9-fold LOSO with inner tuning (modality-only search)",
        "subjects_completed": [],
        "subjects_missing": [],
        "folds": {},
    }
    
    # Collect tuned/baseline overalls for mean calculation
    tuned_overalls = []
    baseline_overalls = []
    
    for subject in ALL_SUBJECTS:
        if subject in existing_folds:
            result_path = existing_folds[subject] / "results.json"
            if result_path.exists():
                try:
                    data = json.load(open(result_path))
                    if "folds" in data and subject in data["folds"]:
                        unified["folds"][subject] = data["folds"][subject]
                        unified["subjects_completed"].append(subject)
                        
                        # Also try to get per-fold tuned/baseline metrics
                        fold_data = data["folds"][subject]
                        if "tuned_overall" in fold_data:
                            tuned_overalls.append(fold_data["tuned_overall"])
                        if "baseline_overall" in fold_data:
                            baseline_overalls.append(fold_data["baseline_overall"])
                except Exception as e:
                    print(f"[WARN] Could not read {result_path}: {e}")
                    unified["subjects_missing"].append(subject)
        else:
            unified["subjects_missing"].append(subject)
    
    # Compute aggregate across completed folds
    def _avg_metrics(metrics_list):
        if not metrics_list:
            return {}
        keys = ["precision", "recall", "rep_f1", "micro_f1_at_50", "start_mae_ms", "end_mae_ms", "transition_mae_ms"]
        out = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if k in m and m[k] is not None]
            if vals:
                out[f"{k}_mean"] = sum(vals) / len(vals)
                out[f"{k}_std"] = (sum((v - out[f"{k}_mean"])**2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0.0
        return out
    
    unified["tuned_aggregate"] = _avg_metrics(tuned_overalls)
    unified["baseline_aggregate"] = _avg_metrics(baseline_overalls)
    unified["completion_rate"] = len(unified["subjects_completed"]) / len(ALL_SUBJECTS)
    
    out_path = UNIFIED_OUTPUT / "unified_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Unified report written to: {out_path}")
    print(f"Completed: {len(unified['subjects_completed'])}/9 subjects")
    print(f"Missing: {unified['subjects_missing']}")
    if unified["tuned_aggregate"]:
        print(f"Tuned Rep F1 (completed folds): {unified['tuned_aggregate'].get('rep_f1_mean', 0):.4f} ± {unified['tuned_aggregate'].get('rep_f1_std', 0):.4f}")
    if unified["baseline_aggregate"]:
        print(f"Baseline Rep F1 (completed folds): {unified['baseline_aggregate'].get('rep_f1_mean', 0):.4f} ± {unified['baseline_aggregate'].get('rep_f1_std', 0):.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Resume-capable 9-fold nested CV for per-action RF + refiner.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and report status, do not run.")
    parser.add_argument("--only-subjects", default="", help="Comma-separated list of specific subjects to run (skip others).")
    parser.add_argument("--merge-only", action="store_true", help="Only merge existing results into unified report.")
    args = parser.parse_args()
    
    # 1. Scan existing folds
    print("Scanning for existing nested CV results...")
    existing = find_existing_folds()
    print(f"Found completed folds for: {sorted(existing.keys())}")
    
    # 2. Determine which subjects to run
    subjects_to_run = []
    if args.only_subjects:
        subjects_to_run = [s.strip() for s in args.only_subjects.split(",") if s.strip() in ALL_SUBJECTS]
    else:
        subjects_to_run = [s for s in ALL_SUBJECTS if s not in existing]
    
    print(f"Subjects to run: {subjects_to_run}")
    print(f"Subjects already done: {sorted(existing.keys())}")
    
    if args.merge_only:
        merge_into_unified(existing)
        return
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for subject in subjects_to_run:
            out_dir = UNIFIED_OUTPUT / f"fold_{subject}"
            print(f"  - {subject} -> {out_dir}")
        return
    
    # 3. Run missing subjects one by one
    for subject in subjects_to_run:
        out_dir = UNIFIED_OUTPUT / f"fold_{subject}"
        success = run_nested_benchmark_for_subject(subject, out_dir)
        
        if success:
            # Add to existing map so next merge includes it
            existing[subject] = out_dir
            # Merge after each successful fold
            merge_into_unified(existing)
        else:
            print(f"[WARN] Stopping after failure on {subject}. Re-run to resume.")
            break
    
    # 4. Final merge
    merge_into_unified(existing)
    print("\nDone.")


if __name__ == "__main__":
    main()
