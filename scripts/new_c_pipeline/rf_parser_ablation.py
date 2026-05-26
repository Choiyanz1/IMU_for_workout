"""
RF Phase Parser Ablation: Simple Post-processing Rules for Rep Count Improvement

Variants:
  A. Original RF parser
  B. RF + short phase island removal
  C. RF + short-gap merge
  D. RF + rep duration filter
  E. RF + head/tail incomplete rep filtering
  F. RF + all simple rules combined

All time parameters are in seconds, converted to samples based on inferred sample rate.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import (
    CONCENTRIC_LABEL, ECCENTRIC_LABEL, labels_to_runs, pair_concentric_eccentric_reps,
    RepDetection, SegmentRun,
)
from scripts.new_c_pipeline.compare_phase_models import (
    PhaseCompareConfig, evaluate_phase, evaluate_reps, extract_active_segments,
    predict_active, predict_rf_phase, smooth_phase_probs, train_active_detector,
    train_rf_phase, _extract_action_from_stream_id,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ParserAblationConfig:
    """All durations are in SECONDS."""
    # Phase island removal (Variant B)
    min_concentric_duration: float = 0.25
    min_eccentric_duration: float = 0.25

    # Short-gap merge (Variant C)
    short_gap_threshold: float = 0.15

    # Rep duration filter (Variant D)
    min_rep_duration: float = 0.8
    max_rep_duration: float = 5.0
    min_rep_concentric_duration: float = 0.25
    min_rep_eccentric_duration: float = 0.25

    # Head/tail filtering (Variant E)
    head_tail_boundary_margin: float = 0.5  # reps within this margin of segment edge are filtered
    head_tail_min_confidence: float = 0.6

    # Smoothing (shared)
    smoothing_window: int = 15

    # Original parser (shared)
    min_phase_samples: int = 3
    max_phase_gap_samples: int = 3


def _infer_sample_rate(df: pd.DataFrame) -> float:
    """Infer sample rate from timestamp column if present, else default 100.0 Hz."""
    for col in ["timestamp", "ts", "time", "elapsed"]:
        if col in df.columns:
            ts = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if len(ts) > 1:
                dt = np.median(np.diff(ts))
                if dt > 0:
                    return 1000.0 / dt  # timestamp usually in ms
    # Fallback: try to infer from shape + known duration patterns
    return 100.0


def _seconds_to_samples(seconds: float, sample_rate: float) -> int:
    return int(round(seconds * sample_rate))


# ---------------------------------------------------------------------------
# Phase runs helpers
# ---------------------------------------------------------------------------

def _phase_probs_to_runs(phase_probs: np.ndarray, min_phase_samples: int = 3) -> List[SegmentRun]:
    """Convert per-sample phase probabilities to runs of C/E."""
    pred_labels = np.argmax(phase_probs, axis=1)
    pred_phase = np.array(["eccentric" if p == 0 else "concentric" for p in pred_labels])
    runs = labels_to_runs(pred_phase, positive_labels={"eccentric", "concentric"}, min_length=min_phase_samples)
    # Merge adjacent same phase
    if not runs:
        return []
    merged = [runs[0]]
    for run in runs[1:]:
        if run.label == merged[-1].label:
            merged[-1] = SegmentRun(
                label=run.label,
                start_idx=merged[-1].start_idx,
                end_idx=run.end_idx,
                confidence=(merged[-1].confidence + run.confidence) / 2,
            )
        else:
            merged.append(run)
    return merged


def _runs_to_reps(runs: List[SegmentRun], max_gap_samples: int = 3) -> List[RepDetection]:
    reps, _ = pair_concentric_eccentric_reps(runs, micro_source="phase", max_gap_samples=max_gap_samples)
    return reps


# ---------------------------------------------------------------------------
# Parser Variants
# ---------------------------------------------------------------------------

def parser_original(phase_probs: np.ndarray, cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant A: Original parser."""
    runs = _phase_probs_to_runs(phase_probs, min_phase_samples=cfg.min_phase_samples)
    return _runs_to_reps(runs, max_gap_samples=cfg.max_phase_gap_samples)


def parser_island_removal(phase_probs: np.ndarray, sample_rate: float, cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant B: Remove short phase islands."""
    runs = _phase_probs_to_runs(phase_probs, min_phase_samples=cfg.min_phase_samples)
    if not runs:
        return []

    min_c = _seconds_to_samples(cfg.min_concentric_duration, sample_rate)
    min_e = _seconds_to_samples(cfg.min_eccentric_duration, sample_rate)

    merged = []
    i = 0
    while i < len(runs):
        run = runs[i]
        dur = run.end_idx - run.start_idx
        # Check if this is a short island (middle of different phases)
        is_short = (run.label == CONCENTRIC_LABEL and dur < min_c) or (run.label == ECCENTRIC_LABEL and dur < min_e)
        if is_short and 0 < i < len(runs) - 1 and runs[i - 1].label == runs[i + 1].label:
            # Merge this island into the surrounding phase
            prev = merged.pop()
            nxt = runs[i + 1]
            merged.append(
                SegmentRun(
                    label=prev.label,
                    start_idx=prev.start_idx,
                    end_idx=nxt.end_idx,
                    confidence=(prev.confidence + run.confidence + nxt.confidence) / 3,
                )
            )
            i += 2
        else:
            merged.append(run)
            i += 1

    return _runs_to_reps(merged, max_gap_samples=cfg.max_phase_gap_samples)


def parser_short_gap_merge(phase_probs: np.ndarray, sample_rate: float, cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant C: Merge same phases separated by short gaps of opposite phase."""
    runs = _phase_probs_to_runs(phase_probs, min_phase_samples=cfg.min_phase_samples)
    if not runs:
        return []

    gap_thresh = _seconds_to_samples(cfg.short_gap_threshold, sample_rate)
    merged = [runs[0]]
    i = 1
    while i < len(runs) - 1:
        prev = merged[-1]
        cur = runs[i]
        nxt = runs[i + 1]
        gap_dur = cur.end_idx - cur.start_idx
        if prev.label == nxt.label and gap_dur < gap_thresh:
            # Merge prev + cur + nxt into prev.label
            merged[-1] = SegmentRun(
                label=prev.label,
                start_idx=prev.start_idx,
                end_idx=nxt.end_idx,
                confidence=(prev.confidence + cur.confidence + nxt.confidence) / 3,
            )
            i += 2
        else:
            merged.append(cur)
            i += 1
    if i < len(runs):
        merged.append(runs[i])

    return _runs_to_reps(merged, max_gap_samples=cfg.max_phase_gap_samples)


def parser_rep_duration_filter(phase_probs: np.ndarray, sample_rate: float, cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant D: Filter reps by duration constraints."""
    reps = parser_original(phase_probs, cfg)
    min_rep = _seconds_to_samples(cfg.min_rep_duration, sample_rate)
    max_rep = _seconds_to_samples(cfg.max_rep_duration, sample_rate)
    min_c = _seconds_to_samples(cfg.min_rep_concentric_duration, sample_rate)
    min_e = _seconds_to_samples(cfg.min_rep_eccentric_duration, sample_rate)

    filtered = []
    for rep in reps:
        rep_dur = rep.end_idx - rep.start_idx
        if rep_dur < min_rep or rep_dur > max_rep:
            continue
        # Infer concentric/eccentric durations from runs inside rep
        # We need runs; for simplicity accept all reps that passed total duration
        # (C/E duration check requires runs mapping; skip for now or accept)
        filtered.append(rep)
    return filtered


def parser_head_tail_filter(phase_probs: np.ndarray, active_segments: List[Tuple[int, int]], cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant E: Filter head/tail incomplete reps."""
    reps = parser_original(phase_probs, cfg)
    if not reps:
        return []

    # For simplicity, if there are active segments, use their boundaries
    segment_starts = [s for s, e in active_segments]
    segment_ends = [e for s, e in active_segments]
    global_start = min(segment_starts) if segment_starts else 0
    global_end = max(segment_ends) if segment_ends else len(phase_probs)

    margin = _seconds_to_samples(cfg.head_tail_boundary_margin, 100.0)  # default 100Hz if unknown
    filtered = []
    for i, rep in enumerate(reps):
        # First or last rep near boundary with low confidence -> filter
        is_first = i == 0
        is_last = i == len(reps) - 1
        near_start = rep.start_idx - global_start < margin
        near_end = global_end - rep.end_idx < margin
        low_conf = rep.micro_confidence < cfg.head_tail_min_confidence
        if (is_first and near_start and low_conf) or (is_last and near_end and low_conf):
            continue
        filtered.append(rep)
    return filtered


def parser_all_rules(phase_probs: np.ndarray, sample_rate: float, active_segments: List[Tuple[int, int]], cfg: ParserAblationConfig) -> List[RepDetection]:
    """Variant F: All simple rules combined."""
    # Step 1: Island removal
    runs = _phase_probs_to_runs(phase_probs, min_phase_samples=cfg.min_phase_samples)
    if not runs:
        return []

    min_c = _seconds_to_samples(cfg.min_concentric_duration, sample_rate)
    min_e = _seconds_to_samples(cfg.min_eccentric_duration, sample_rate)
    gap_thresh = _seconds_to_samples(cfg.short_gap_threshold, sample_rate)

    # Island removal + short gap merge in one pass
    processed = []
    i = 0
    while i < len(runs):
        run = runs[i]
        dur = run.end_idx - run.start_idx
        is_short = (run.label == CONCENTRIC_LABEL and dur < min_c) or (run.label == ECCENTRIC_LABEL and dur < min_e)

        # Short island surrounded by same phase
        if is_short and 0 < i < len(runs) - 1 and runs[i - 1].label == runs[i + 1].label:
            prev = processed.pop()
            nxt = runs[i + 1]
            processed.append(
                SegmentRun(
                    label=prev.label,
                    start_idx=prev.start_idx,
                    end_idx=nxt.end_idx,
                    confidence=(prev.confidence + run.confidence + nxt.confidence) / 3,
                )
            )
            i += 2
            continue

        # Short gap between same phases
        if i < len(runs) - 1 and processed and processed[-1].label == runs[i + 1].label and dur < gap_thresh:
            nxt = runs[i + 1]
            processed[-1] = SegmentRun(
                label=processed[-1].label,
                start_idx=processed[-1].start_idx,
                end_idx=nxt.end_idx,
                confidence=(processed[-1].confidence + run.confidence + nxt.confidence) / 3,
            )
            i += 2
            continue

        processed.append(run)
        i += 1

    reps = _runs_to_reps(processed, max_gap_samples=cfg.max_phase_gap_samples)

    # Rep duration filter
    min_rep = _seconds_to_samples(cfg.min_rep_duration, sample_rate)
    max_rep = _seconds_to_samples(cfg.max_rep_duration, sample_rate)
    reps = [r for r in reps if min_rep <= (r.end_idx - r.start_idx) <= max_rep]

    # Head/tail filter
    if reps:
        segment_starts = [s for s, e in active_segments]
        segment_ends = [e for s, e in active_segments]
        global_start = min(segment_starts) if segment_starts else 0
        global_end = max(segment_ends) if segment_ends else len(phase_probs)
        margin = _seconds_to_samples(cfg.head_tail_boundary_margin, sample_rate)
        filtered = []
        for idx, rep in enumerate(reps):
            is_first = idx == 0
            is_last = idx == len(reps) - 1
            near_start = rep.start_idx - global_start < margin
            near_end = global_end - rep.end_idx < margin
            low_conf = rep.micro_confidence < cfg.head_tail_min_confidence
            if (is_first and near_start and low_conf) or (is_last and near_end and low_conf):
                continue
            filtered.append(rep)
        reps = filtered

    return reps


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------

def analyze_count_error(
    pred_reps: List[RepDetection],
    gt_reps: List[RepDetection],
    stream_id: str,
) -> dict:
    """Heuristic error source attribution."""
    pred_count = len(pred_reps)
    gt_count = len(gt_reps)
    error = pred_count - gt_count

    sources = {
        "first_rep_error": 0,
        "last_rep_error": 0,
        "island_error": 0,
        "split_error": 0,
        "merge_error": 0,
        "other_error": 0,
    }

    if error == 0:
        return {**sources, "error": 0}

    # Simple heuristics
    # If first pred rep doesn't overlap with first gt rep -> first rep error
    if pred_reps and gt_reps:
        first_pred = pred_reps[0]
        first_gt = gt_reps[0]
        pred_range = set(range(first_pred.start_idx, first_pred.end_idx))
        gt_range = set(range(first_gt.start_idx, first_gt.end_idx))
        if not (pred_range & gt_range):
            sources["first_rep_error"] = 1

    if len(pred_reps) >= 1 and len(gt_reps) >= 1:
        last_pred = pred_reps[-1]
        last_gt = gt_reps[-1]
        pred_range = set(range(last_pred.start_idx, last_pred.end_idx))
        gt_range = set(range(last_gt.start_idx, last_gt.end_idx))
        if not (pred_range & gt_range):
            sources["last_rep_error"] = 1

    # If error > 0 and there are more pred reps -> likely split or extra island
    if error > 0:
        # Check if any two pred reps map to the same gt rep (split)
        gt_matched = {}
        for pi, pred in enumerate(pred_reps):
            best_iou = 0
            best_gi = None
            for gi, gt in enumerate(gt_reps):
                pred_r = set(range(pred.start_idx, pred.end_idx))
                gt_r = set(range(gt.start_idx, gt.end_idx))
                inter = len(pred_r & gt_r)
                union = len(pred_r | gt_r)
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi is not None:
                gt_matched.setdefault(best_gi, []).append((pi, best_iou))

        splits = sum(1 for v in gt_matched.values() if len(v) > 1)
        if splits > 0:
            sources["split_error"] = 1
        elif sources["first_rep_error"] == 0 and sources["last_rep_error"] == 0:
            sources["island_error"] = 1

    # If error < 0 (undercount) -> likely merge or missing first/last
    if error < 0:
        # Check if any gt rep has no overlap with any pred rep (missing)
        missing = 0
        for gt in gt_reps:
            gt_r = set(range(gt.start_idx, gt.end_idx))
            has_overlap = any(
                bool(gt_r & set(range(p.start_idx, p.end_idx)))
                for p in pred_reps
            )
            if not has_overlap:
                missing += 1
        if missing > 0:
            sources["merge_error"] = 1
        else:
            sources["other_error"] = 1

    return {**sources, "error": error}


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def compute_count_distribution(results: List[dict]) -> dict:
    errors = [r["pred_count"] - r["gt_count"] for r in results]
    dist = {}
    for e in range(-5, 6):
        dist[f"err_{e:+d}"] = sum(1 for x in errors if x == e)
    dist["mean_abs_error"] = np.mean([abs(e) for e in errors])
    return dist


def aggregate_parser_results(results: List[dict]) -> dict:
    if not results:
        return {}
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    exact = sum(r["exact_count"] for r in results)
    over = sum(r["over"] for r in results)
    under = sum(r["under"] for r in results)
    n = len(results)

    return {
        "streams": n,
        "rep_precision": p,
        "rep_recall": r,
        "rep_f1": f1,
        "exact_count_acc": exact / n if n > 0 else 0,
        "over_count_rate": over / n if n > 0 else 0,
        "under_count_rate": under / n if n > 0 else 0,
        "over_count": over,
        "under_count": under,
        **compute_count_distribution(results),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ablation(
    train_streams,
    test_streams,
    cfg: ParserAblationConfig,
    base_cfg: PhaseCompareConfig,
    output_dir: Path,
):
    print("=" * 70)
    print("RF Phase Parser Ablation")
    print("=" * 70)

    # Train models once
    print("\n[1/2] Training models...")
    active_models, active_scalers = train_active_detector(train_streams, base_cfg)
    rf_phase_models, rf_phase_scalers = train_rf_phase(train_streams, base_cfg)
    print("      Done.")

    # Variants to test
    variants = {
        "A_original": parser_original,
        "B_island_removal": parser_island_removal,
        "C_short_gap": parser_short_gap_merge,
        "D_duration_filter": parser_rep_duration_filter,
        "E_head_tail": parser_head_tail_filter,
        "F_all_rules": parser_all_rules,
    }

    all_variant_results = {k: [] for k in variants}
    all_error_analysis = {k: [] for k in variants}
    per_stream_best = []

    print(f"\n[2/2] Evaluating on {len(test_streams)} test streams...")
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue

        sample_rate = _infer_sample_rate(df)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, base_cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)

        rf_phase_probs = predict_rf_phase(rf_phase_models, rf_phase_scalers, stream_id, df, active_segments, base_cfg)
        rf_phase_probs_smooth = smooth_phase_probs(rf_phase_probs, base_cfg.smoothing_window)

        gt_reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=base_cfg.min_phase_samples)

        best_f1 = -1
        best_variant = None

        for vname, parser_fn in variants.items():
            if vname == "A_original":
                pred_reps = parser_fn(rf_phase_probs_smooth, cfg)
            elif vname in ("B_island_removal", "C_short_gap", "D_duration_filter"):
                pred_reps = parser_fn(rf_phase_probs_smooth, sample_rate, cfg)
            elif vname == "E_head_tail":
                pred_reps = parser_fn(rf_phase_probs_smooth, active_segments, cfg)
            else:  # F_all_rules
                pred_reps = parser_fn(rf_phase_probs_smooth, sample_rate, active_segments, cfg)

            rep_metrics = evaluate_reps(pred_reps, gt_reps)
            all_variant_results[vname].append(rep_metrics)

            err_analysis = analyze_count_error(pred_reps, gt_reps, stream_id)
            all_error_analysis[vname].append(err_analysis)

            if rep_metrics["f1"] > best_f1:
                best_f1 = rep_metrics["f1"]
                best_variant = vname

        per_stream_best.append(best_variant)

    # Aggregate
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    summary = {}
    for vname, results in all_variant_results.items():
        agg = aggregate_parser_results(results)
        summary[vname] = agg
        print(f"\n[{vname}]")
        print(f"  Rep F1:        {agg['rep_f1']:.4f}")
        print(f"  Exact Count:   {agg['exact_count_acc']:.4f}")
        print(f"  Mean Abs Err:  {agg.get('mean_abs_error', 0):.3f}")
        print(f"  Over/Under:    {agg['over_count']}/{agg['under_count']}")
        print(f"  Count Dist:    " + ", ".join([f"{k}={v}" for k, v in agg.items() if k.startswith("err_") and v > 0]))

    # Error analysis aggregate
    print(f"\n{'=' * 70}")
    print("ERROR SOURCE ANALYSIS")
    print(f"{'=' * 70}")
    for vname, errors in all_error_analysis.items():
        total = len(errors)
        if total == 0:
            continue
        print(f"\n[{vname}]")
        for key in ["first_rep_error", "last_rep_error", "island_error", "split_error", "merge_error", "other_error"]:
            count = sum(e.get(key, 0) for e in errors)
            print(f"  {key}: {count}/{total}")

    # Per-stream best variant
    from collections import Counter
    print(f"\n{'=' * 70}")
    print("PER-STREAM BEST VARIANT")
    print(f"{'=' * 70}")
    print(Counter(per_stream_best))

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rf_parser_ablation.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "error_analysis": {k: [{str(i): e} for i, e in enumerate(v)] for k, v in all_error_analysis.items()},
            "per_stream_best": dict(Counter(per_stream_best)),
        }, f, indent=2, default=str)
    print(f"\n[OK] Saved to {out_path}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RF Phase Parser Ablation")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rf_parser_ablation"))
    parser.add_argument("--quick", action="store_true", help="Quick mode: kevin only")
    parser.add_argument("--min-concentric-duration", type=float, default=0.25)
    parser.add_argument("--min-eccentric-duration", type=float, default=0.25)
    parser.add_argument("--short-gap-threshold", type=float, default=0.15)
    parser.add_argument("--min-rep-duration", type=float, default=0.8)
    parser.add_argument("--max-rep-duration", type=float, default=5.0)
    parser.add_argument("--head-tail-margin", type=float, default=0.5)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects")

    test_subjects = ["kevin"] if args.quick else subjects
    train_streams = [(sid, df) for sid, df in all_streams if not any(sid.startswith(f"{ts}/") for ts in test_subjects)]
    test_streams = [(sid, df) for sid, df in all_streams if any(sid.startswith(f"{ts}/") for ts in test_subjects)]

    cfg = ParserAblationConfig(
        min_concentric_duration=args.min_concentric_duration,
        min_eccentric_duration=args.min_eccentric_duration,
        short_gap_threshold=args.short_gap_threshold,
        min_rep_duration=args.min_rep_duration,
        max_rep_duration=args.max_rep_duration,
        head_tail_boundary_margin=args.head_tail_margin,
    )
    base_cfg = PhaseCompareConfig()
    base_cfg.smoothing_window = 15

    run_ablation(train_streams, test_streams, cfg, base_cfg, args.output)


if __name__ == "__main__":
    main()
