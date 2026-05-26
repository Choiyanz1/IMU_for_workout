"""Aggregate Phase 1a results and generate LaTeX-ready tables."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_summary(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fold_metrics(summary: Dict, metric: str = "rep_f1") -> List[float]:
    """Extract per-fold values from various JSON structures."""
    vals = []
    
    # Case 1: Direct fold_results list
    if "fold_results" in summary:
        for fold in summary["fold_results"]:
            if metric in fold and fold[metric] is not None:
                v = float(fold[metric])
                if np.isfinite(v):
                    vals.append(v)
    
    # Case 2: fold_summaries (Sliding-window RF style)
    if "fold_summaries" in summary:
        for fold in summary["fold_summaries"]:
            if metric in fold and fold[metric] is not None:
                v = float(fold[metric])
                if np.isfinite(v):
                    vals.append(v)
    
    # Case 3: Per-action nested structure (grand_summary)
    if "by_action" in summary:
        for action, action_summary in summary["by_action"].items():
            if isinstance(action_summary, dict) and metric in action_summary:
                v = float(action_summary[metric])
                if np.isfinite(v):
                    vals.append(v)
    
    # Case 4: stream_metrics (Peak Detection style) - aggregate by test_subject
    if not vals and "stream_metrics" in summary:
        from collections import defaultdict
        by_subject = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "values": []})
        for stream in summary["stream_metrics"]:
            subj = stream.get("test_subject", "unknown")
            if metric in ["precision", "recall", "rep_f1"]:
                # For rep-level metrics, we need to aggregate TP/FP/FN across streams
                by_subject[subj]["tp"] += stream.get("tp", 0)
                by_subject[subj]["fp"] += stream.get("fp", 0)
                by_subject[subj]["fn"] += stream.get("fn", 0)
            elif metric in stream:
                by_subject[subj]["values"].append(float(stream[metric]))
        
        for subj, data in by_subject.items():
            if metric == "precision":
                tp, fp = data["tp"], data["fp"]
                vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            elif metric == "recall":
                tp, fn = data["tp"], data["fn"]
                vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            elif metric == "rep_f1":
                tp, fp, fn = data["tp"], data["fp"], data["fn"]
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                vals.append(f1)
            elif data["values"]:
                vals.append(np.mean(data["values"]))
    
    # Case 5: classifier comparison results (quick_compare_3fold.json)
    if not vals and "results" in summary:
        method_map = {
            "Random Forest": "Causal RF",
            "XGBoost": "XGBoost",
            "CatBoost": "CatBoost (3-fold)*",
        }
        # Find which method this summary belongs to by matching in parent results dict
        for method_name, method_vals in summary["results"].items():
            if method_vals:  # List of per-fold F1s
                vals = [float(v) for v in method_vals if np.isfinite(float(v))]
                break
    
    # Case 6: Simple overall (fallback)
    if not vals and "overall" in summary and metric in summary["overall"]:
        v = float(summary["overall"][metric])
        if np.isfinite(v):
            vals.append(v)
    
    return vals


def format_mean_std(vals: List[float], decimals: int = 3) -> str:
    if not vals:
        return "N/A"
    mean = np.mean(vals)
    std = np.std(vals)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main():
    base = Path("artifacts/baseline_comparison")
    
    results = {}
    
    # Load Peak Detection (7 subjects cleaned, fixed bug)
    pd_paths = [
        base / "peak_detection_7subjects_cleaned_fixed/results.json",
        base / "peak_detection_7subjects_cleaned_fixed/summary.json",
        base / "full_loso_peak_detection/results.json",
    ]
    for pd_path in pd_paths:
        if pd_path.exists():
            results["Peak Detection"] = load_summary(pd_path)
            break
    
    # Load Causal RF (7 subjects cleaned, best config: w=100, n=100)
    crf_paths = [
        base / "causal_rf_n100_w100/summary.json",
        base / "loso_7subjects_plain_rf_cleaned/summary.json",
        base / "full_loso_causal_rf/summary.json",
    ]
    for crf_path in crf_paths:
        if crf_path.exists():
            results["Causal RF"] = load_summary(crf_path)
            break
    
    # Load Sliding-window RF (7 subjects cleaned)
    swrf_path = base / "sliding_window_rf_7subjects_cleaned/summary.json"
    if swrf_path.exists():
        results["Sliding-window RF"] = load_summary(swrf_path)
    
    # Load CatBoost (3-fold quick compare)
    cb_path = base / "classifier_comparison/quick_compare_3fold.json"
    if cb_path.exists():
        cb_summary = load_summary(cb_path)
        results["CatBoost (3-fold)*"] = cb_summary
    
    # Load Per-action RF Refiner (corrected)
    rfr_path = base / "full_loso_per_action_rf_refiner_corrected/grand_summary.json"
    if rfr_path.exists():
        results["Per-action RF+Refiner"] = load_summary(rfr_path)
    else:
        rfr_path = base / "full_loso_per_action_rf_refiner/grand_summary.json"
        if rfr_path.exists():
            results["Per-action RF+Refiner"] = load_summary(rfr_path)
    
    # Build comparison table
    print("="*80)
    print("PHASE 1a: REP SEGMENTATION BASELINE COMPARISON")
    print("Protocol: 7-fold Strict LOSO (Leave-One-Subject-Out)")
    print("="*80)
    print()
    
    rows = []
    for method, summary in results.items():
        overall = summary.get("overall", {})
        f1s = extract_fold_metrics(summary, "rep_f1")
        precisions = extract_fold_metrics(summary, "precision")
        recalls = extract_fold_metrics(summary, "recall")
        iou_f1s = extract_fold_metrics(summary, "micro_f1_at_50")
        
        row = {
            "Method": method,
            "Rep F1": format_mean_std(f1s),
            "Precision": format_mean_std(precisions),
            "Recall": format_mean_std(recalls),
            "IoU-F1@50": format_mean_std(iou_f1s),
            "n_true": overall.get("n_true", "N/A"),
            "n_pred": overall.get("n_pred", "N/A"),
            "tp": overall.get("tp", "N/A"),
        }
        rows.append(row)
    
    # Print Markdown table
    print("## Markdown Table")
    print()
    print("| Method | Rep F1 | Precision | Recall | IoU-F1@50 | n_true | n_pred | tp |")
    print("|--------|--------|-----------|--------|-----------|--------|--------|-----|")
    for row in rows:
        print(f"| {row['Method']} | {row['Rep F1']} | {row['Precision']} | {row['Recall']} | {row['IoU-F1@50']} | {row['n_true']} | {row['n_pred']} | {row['tp']} |")
    print()
    
    # Print LaTeX table
    print("## LaTeX Table")
    print()
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Rep Segmentation Baseline Comparison (7-fold LOSO)}")
    print("\\label{tab:rep-baselines}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Rep F1 & Precision & Recall & IoU-F1@50 \\\\")
    print("\\midrule")
    for row in rows:
        print(f"{row['Method']} & {row['Rep F1']} & {row['Precision']} & {row['Recall']} & {row['IoU-F1@50']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Save to file
    out_dir = Path("artifacts/baseline_comparison/phase1a_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# Phase 1a: Rep Segmentation Baseline Comparison\n\n")
        f.write("Protocol: 7-fold Strict LOSO (Leave-One-Subject-Out)\n\n")
        f.write("| Method | Rep F1 | Precision | Recall | IoU-F1@50 | n_true | n_pred | tp |\n")
        f.write("|--------|--------|-----------|--------|-----------|--------|--------|-----|\n")
        for row in rows:
            f.write(f"| {row['Method']} | {row['Rep F1']} | {row['Precision']} | {row['Recall']} | {row['IoU-F1@50']} | {row['n_true']} | {row['n_pred']} | {row['tp']} |\n")
        f.write("\n")
        f.write("## LaTeX\n\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Rep Segmentation Baseline Comparison (7-fold LOSO)}\n")
        f.write("\\label{tab:rep-baselines}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Rep F1 & Precision & Recall & IoU-F1@50 \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(f"{row['Method']} & {row['Rep F1']} & {row['Precision']} & {row['Recall']} & {row['IoU-F1@50']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"[OK] Saved results to {out_dir / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
