#!/usr/bin/env python3
"""Analyze EXISTING results for rep count metrics across all baselines."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

print("=" * 90)
print("REP COUNT METRICS ANALYSIS (Existing Results Only)")
print("=" * 90)

# ---------------------------------------------------------------------------
# 1. Per-Action Plain RF (7-fold LOSO) — from existing artifacts
# ---------------------------------------------------------------------------
print("\n### 1. Per-Action Plain RF (7-fold LOSO, 226 streams) ###\n")

base_dir = ROOT / "artifacts" / "baseline_comparison" / "per_action_plain_rf_7fold"
grand_summary = json.loads((base_dir / "grand_summary.json").read_text())

overall = grand_summary["overall"]
print(f"  Rep F1:            {overall['rep_f1']:.4f}")
print(f"  Precision:         {overall['precision']:.4f}")
print(f"  Recall:            {overall['recall']:.4f}")
print(f"  IoU-F1@50:         0.706 (from 02-phase1-rep-segmentation.md)")

# Extract rep count metrics from per-action summaries
all_folds = []
for action_dir in sorted(base_dir.iterdir()):
    if not action_dir.is_dir():
        continue
    summary_path = action_dir / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text())
    for fold in data.get("fold_results", []):
        all_folds.append({
            "action": data["action"],
            "test_subject": fold.get("test_subject", "unknown"),
            "stream_count": fold.get("stream_count", 0),
            "n_pred": fold.get("n_pred", 0),
            "n_true": fold.get("n_true", 0),
            "exact_count": fold.get("exact_count_streams", 0),
            "over": fold.get("over_segmented_streams", 0),
            "under": fold.get("under_segmented_streams", 0),
            "zero_tp": fold.get("zero_tp_streams", 0),
        })

if all_folds:
    total_streams = sum(f["stream_count"] for f in all_folds)
    total_exact = sum(f["exact_count"] for f in all_folds)
    total_over = sum(f["over"] for f in all_folds)
    total_under = sum(f["under"] for f in all_folds)
    total_zero_tp = sum(f["zero_tp"] for f in all_folds)
    exact_ratio = total_exact / total_streams if total_streams > 0 else 0
    
    # Per-action breakdown
    action_groups = {}
    for f in all_folds:
        action = f["action"]
        if action not in action_groups:
            action_groups[action] = []
        action_groups[action].append(f)
    
    print(f"\n  Per-Action Rep Count Breakdown:")
    print(f"  {'Action':<20} {'Streams':>8} {'Exact%':>8} {'Over':>6} {'Under':>6} {'ZeroTP':>6} {'MADiff':>8}")
    print(f"  {'-' * 70}")
    
    for action, folds in sorted(action_groups.items()):
        streams = sum(f["stream_count"] for f in folds)
        exact = sum(f["exact_count"] for f in folds)
        over = sum(f["over"] for f in folds)
        under = sum(f["under"] for f in folds)
        zero_tp = sum(f["zero_tp"] for f in folds)
        ratio = exact / streams if streams > 0 else 0
        # Mean abs count diff per stream
        count_diffs = []
        for f in folds:
            count_diffs.extend([abs(f["n_pred"] - f["n_true"])] * f["stream_count"])
        mad = np.mean(count_diffs) if count_diffs else 0
        print(f"  {action:<20} {streams:>8} {ratio:>7.1%} {over:>6} {under:>6} {zero_tp:>6} {mad:>8.2f}")
    
    print(f"  {'-' * 70}")
    total_pred = sum(f["n_pred"] for f in all_folds)
    total_true = sum(f["n_true"] for f in all_folds)
    all_count_diffs = []
    for f in all_folds:
        all_count_diffs.extend([abs(f["n_pred"] - f["n_true"])] * f["stream_count"])
    total_mad = np.mean(all_count_diffs) if all_count_diffs else 0
    print(f"  {'OVERALL':<20} {total_streams:>8} {exact_ratio:>7.1%} {total_over:>6} {total_under:>6} {total_zero_tp:>6} {total_mad:>8.2f}")
    
    print(f"\n  Key Rep Count Metrics:")
    print(f"    Exact Count Ratio:   {exact_ratio:.1%} ({total_exact}/{total_streams})")
    print(f"    Over-segmented:      {total_over} folds (not streams)")
    print(f"    Under-segmented:     {total_under} folds")
    print(f"    Zero TP folds:       {total_zero_tp}")
    print(f"    Mean Abs Diff:       {total_mad:.2f} reps/fold")

# ---------------------------------------------------------------------------
# 2. modality_count_guardrail_yoru_v1 (yoru subject)
# ---------------------------------------------------------------------------
print("\n\n### 2. modality_count_guardrail_yoru_v1 (yoru held-out) ###\n")

yoru_path = ROOT / "artifacts" / "baseline_comparison" / "modality_count_guardrail_yoru_v1" / "yoru" / "results.json"
if yoru_path.exists():
    yoru_data = json.loads(yoru_path.read_text())
    
    # baseline_overall = guardrail selected baseline_reference
    baseline = yoru_data.get("baseline_overall", {})
    print(f"  (Guardrail selected: baseline_reference — no duration prior, no refiner)")
    print(f"  Streams:             {baseline.get('stream_count', 0)}")
    print(f"  Rep F1:              {baseline.get('rep_f1', 0):.4f}")
    print(f"  Precision:           {baseline.get('precision', 0):.4f}")
    print(f"  Recall:              {baseline.get('recall', 0):.4f}")
    print(f"  IoU-F1@50:           {baseline.get('micro_f1_at_50', 0):.4f}")
    print(f"  Exact Count Ratio:   {baseline.get('exact_count_streams', 0)}/{baseline.get('stream_count', 0)} = "
          f"{baseline.get('exact_count_streams', 0)/baseline.get('stream_count', 1):.1%}")
    print(f"  Over-segmented:      {baseline.get('over_segmented_streams', 0)}")
    print(f"  Under-segmented:     {baseline.get('under_segmented_streams', 0)}")
    print(f"  Zero TP:             {baseline.get('zero_tp_streams', 0)}")

# ---------------------------------------------------------------------------
# 3. Comparison Table
# ---------------------------------------------------------------------------
print("\n\n### 3. SIDE-BY-SIDE COMPARISON ###\n")
print(f"  {'Metric':<25} {'Per-Action RF':>18} {'yoru_v1 baseline':>18} {'Delta':>12}")
print(f"  {'-' * 75}")

# Per-Action RF metrics
pa_rep_f1 = overall['rep_f1']
pa_prec = overall['precision']
pa_rec = overall['recall']
pa_exact = exact_ratio
pa_mad = total_mad

# yoru_v1 metrics
y_rep_f1 = baseline.get('rep_f1', 0)
y_prec = baseline.get('precision', 0)
y_rec = baseline.get('recall', 0)
y_exact = baseline.get('exact_count_streams', 0) / baseline.get('stream_count', 1) if baseline.get('stream_count', 0) > 0 else 0
y_iou = baseline.get('micro_f1_at_50', 0)

print(f"  {'Rep F1':<25} {pa_rep_f1:>18.4f} {y_rep_f1:>18.4f} {y_rep_f1 - pa_rep_f1:>+12.4f}")
print(f"  {'Precision':<25} {pa_prec:>18.4f} {y_prec:>18.4f} {y_prec - pa_prec:>+12.4f}")
print(f"  {'Recall':<25} {pa_rec:>18.4f} {y_rec:>18.4f} {y_rec - pa_rec:>+12.4f}")
print(f"  {'IoU-F1@50':<25} {'0.706':>18} {y_iou:>18.4f} {y_iou - 0.706:>+12.4f}")
print(f"  {'Exact Count Ratio':<25} {pa_exact:>17.1%} {y_exact:>17.1%} {(y_exact - pa_exact):>+11.1%}")
print(f"  {'Mean Abs Count Diff':<25} {pa_mad:>18.2f} {'N/A':>18} {'N/A':>12}")

print(f"\n  ! IMPORTANT CAVEAT:")
print(f"     • Per-Action RF: 7-fold LOSO (all 7 subjects, 226 streams)")
print(f"     • yoru_v1:       single-subject test (yoru only, 25 streams)")
print(f"     • Different test sets → NOT directly comparable!")
print(f"     • yoru_v1's baseline_reference is the SAME as Per-Action RF but on yoru only")

# ---------------------------------------------------------------------------
# 4. Key Insights
# ---------------------------------------------------------------------------
print("\n\n### 4. KEY INSIGHTS ###\n")
print(f"  1. Rep Count is ALREADY computed in existing results:")
print(f"     • Per-Action Plain RF: {exact_ratio:.1%} exact count (149/226 streams)")
print(f"     • This is the THIRD critical metric alongside Rep F1 and IoU-F1@50")
print(f"")
print(f"  2. Weakness by action:")
print(f"     • db_weighted_crunch: 40.6% exact count (worst)")
print(f"     • db_shoulder_press:  52.2% exact count")
print(f"     • db_biceps_curl:     96.0% exact count (best)")
print(f"")
print(f"  3. For deployment, Rep Count accuracy is as important as Rep F1:")
print(f"     • Users care about 'how many reps did I do?'")
print(f"     • A model with F1=0.85 but only 65% exact count is problematic")
print(f"     • Target: >90% exact count for user acceptance")
print(f"")
print(f"  4. Current gap to target:")
print(f"     • Need +24.1% improvement (from 65.9% → 90%)")
print(f"     • Potential improvements:")
print(f"       - Duration Prior (may help over/under segmentation)")
print(f"       - Boundary Refiner (may improve count accuracy via better edges)")
print(f"       - Post-processing (merge/split heuristics)")

print("\n" + "=" * 90)
print("Next: Run evaluate_browse_model_rf.py to see if Duration Prior + Refiner improves exact count")
print("=" * 90)
