"""
Data-driven per-action post-processing threshold analysis.

Computes from training data per fold:
  - Concentric phase duration distribution (samples)
  - Eccentric phase duration distribution (samples)
  - Total rep duration distribution (samples)
  - C/E ratio distribution

Outputs data-driven thresholds for:
  min_C_duration, min_E_duration, min_rep_duration, max_rep_duration, max_gap_samples
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.micro_macro_segments import labels_to_runs, pair_concentric_eccentric_reps, SegmentRun
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels


def _extract_action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def analyze_action_durations(streams):
    """Analyze ground truth durations per action from a list of (stream_id, df) streams."""
    action_stats = {}

    for stream_id, df in streams:
        if "phase" not in df.columns:
            continue
        action = _extract_action_from_stream_id(stream_id)
        if action not in action_stats:
            action_stats[action] = {
                "concentric_durations": [],
                "eccentric_durations": [],
                "rep_durations": [],
                "ce_ratios": [],
                "n_reps": 0,
                "n_streams": 0,
            }

        action_stats[action]["n_streams"] += 1
        phase_arr = df["phase"].to_numpy()

        # Extract concentric/eccentric segments
        c_segs = []
        e_segs = []
        in_seg = False
        seg_start = 0
        current_label = ""

        for i, p in enumerate(phase_arr):
            label = str(p)
            if label in {"concentric", "eccentric"}:
                if not in_seg or label != current_label:
                    if in_seg:
                        # Save previous
                        dur = i - seg_start
                        if current_label == "concentric":
                            c_segs.append(dur)
                        elif current_label == "eccentric":
                            e_segs.append(dur)
                    seg_start = i
                    current_label = label
                    in_seg = True
            else:
                if in_seg:
                    dur = i - seg_start
                    if current_label == "concentric":
                        c_segs.append(dur)
                    elif current_label == "eccentric":
                        e_segs.append(dur)
                    in_seg = False

        if in_seg:
            dur = len(phase_arr) - seg_start
            if current_label == "concentric":
                c_segs.append(dur)
            elif current_label == "eccentric":
                e_segs.append(dur)

        action_stats[action]["concentric_durations"].extend(c_segs)
        action_stats[action]["eccentric_durations"].extend(e_segs)

        # Extract reps
        gt_reps = truth_reps_from_labels(phase_arr, min_phase_samples=1)
        action_stats[action]["n_reps"] += len(gt_reps)
        for rep in gt_reps:
            rep_dur = rep.end_idx - rep.start_idx
            action_stats[action]["rep_durations"].append(rep_dur)
            seg = phase_arr[rep.start_idx:rep.end_idx]
            c_count = np.sum(seg == "concentric")
            e_count = np.sum(seg == "eccentric")
            if e_count > 0:
                action_stats[action]["ce_ratios"].append(c_count / e_count)

    return action_stats


def compute_thresholds(action_stats: dict) -> dict:
    """Compute data-driven thresholds from action statistics."""
    thresholds = {}

    for action, stats in action_stats.items():
        c_durs = np.array(stats["concentric_durations"])
        e_durs = np.array(stats["eccentric_durations"])
        rep_durs = np.array(stats["rep_durations"])

        def _percentile(arr, p):
            return int(np.percentile(arr, p)) if len(arr) > 0 else 0

        def _median(arr):
            return int(np.median(arr)) if len(arr) > 0 else 0

        # Concentric: use 10th percentile as min, but at least 3
        min_c = max(3, _percentile(c_durs, 10)) if len(c_durs) > 0 else 3

        # Eccentric: use 10th percentile as min, but at least 3
        min_e = max(3, _percentile(e_durs, 10)) if len(e_durs) > 0 else 3

        # Rep: min at 10th percentile, max at 95th percentile
        min_rep = max(10, _percentile(rep_durs, 10)) if len(rep_durs) > 0 else 10
        max_rep = _percentile(rep_durs, 95) if len(rep_durs) > 0 else 9999

        # Max gap: use 50th percentile of min(C_med, E_med) / 3, but at least 3
        c_med = _median(c_durs) if len(c_durs) > 0 else 10
        e_med = _median(e_durs) if len(e_durs) > 0 else 10
        max_gap = max(3, min(c_med, e_med) // 3)

        thresholds[action] = {
            "min_C_duration": min_c,
            "min_E_duration": min_e,
            "min_rep_duration": min_rep,
            "max_rep_duration": max_rep,
            "max_gap_samples": max_gap,
            "_stats": {
                "n_streams": stats["n_streams"],
                "n_reps": stats["n_reps"],
                "concentric_median": _median(c_durs),
                "concentric_10th": _percentile(c_durs, 10),
                "concentric_90th": _percentile(c_durs, 90),
                "eccentric_median": _median(e_durs),
                "eccentric_10th": _percentile(e_durs, 10),
                "eccentric_90th": _percentile(e_durs, 90),
                "rep_median": _median(rep_durs),
                "rep_10th": _percentile(rep_durs, 10),
                "rep_95th": _percentile(rep_durs, 95),
                "ce_ratio_median": float(np.median(stats["ce_ratios"])) if stats["ce_ratios"] else None,
            }
        }

    return thresholds


def main():
    raw = yaml.safe_load(open("config.yaml"))
    all_streams, subjects, actions = _load_streams(raw, ["sets"])
    print(f"Loaded {len(all_streams)} streams from {len(subjects)} subjects")
    print(f"Actions: {actions}")

    # For each fold, compute thresholds from training data
    all_fold_thresholds = {}

    for fold_idx, test_subject in enumerate(subjects):
        print(f"\nFold {fold_idx + 1}/9: test={test_subject}")
        train_streams = [(sid, df) for sid, df in all_streams if not sid.startswith(f"{test_subject}/")]
        action_stats = analyze_action_durations(train_streams)
        thresholds = compute_thresholds(action_stats)
        all_fold_thresholds[test_subject] = thresholds

        print("  Per-action thresholds from training data:")
        for action, t in sorted(thresholds.items()):
            s = t["_stats"]
            print(f"    {action:20s}: min_C={t['min_C_duration']:3d} min_E={t['min_E_duration']:3d} "
                  f"min_rep={t['min_rep_duration']:4d} max_rep={t['max_rep_duration']:4d} gap={t['max_gap_samples']:2d} "
                  f"(C_med={s['concentric_median']:3d}, E_med={s['eccentric_median']:3d}, "
                  f"rep_med={s['rep_median']:4d}, n_reps={s['n_reps']:4d})")

    # Also compute global thresholds from ALL data
    print(f"\n{'='*60}")
    print("GLOBAL thresholds (from all data):")
    global_stats = analyze_action_durations(all_streams)
    global_thresholds = compute_thresholds(global_stats)

    # For global, take median across actions
    all_min_c = [t["min_C_duration"] for t in global_thresholds.values()]
    all_min_e = [t["min_E_duration"] for t in global_thresholds.values()]
    all_min_rep = [t["min_rep_duration"] for t in global_thresholds.values()]
    all_max_rep = [t["max_rep_duration"] for t in global_thresholds.values()]
    all_gap = [t["max_gap_samples"] for t in global_thresholds.values()]

    global_pp = {
        "min_C_duration": int(np.median(all_min_c)),
        "min_E_duration": int(np.median(all_min_e)),
        "min_rep_duration": int(np.median(all_min_rep)),
        "max_rep_duration": int(np.median(all_max_rep)),
        "max_gap_samples": int(np.median(all_gap)),
    }
    print(f"  Global: min_C={global_pp['min_C_duration']} min_E={global_pp['min_E_duration']} "
          f"min_rep={global_pp['min_rep_duration']} max_rep={global_pp['max_rep_duration']} gap={global_pp['max_gap_samples']}")

    # Save
    output = {
        "global_pp": global_pp,
        "per_fold_per_action": all_fold_thresholds,
        "action_stats_all": {a: {k: v for k, v in s.items() if k != "_stats"}
                             for a, s in global_thresholds.items()},
    }

    out_dir = Path("artifacts/cnn_ma25viterbi_9fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_driven_pp_thresholds.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Saved thresholds to {out_path}")


if __name__ == "__main__":
    main()
