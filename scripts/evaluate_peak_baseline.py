"""Magnitude Peak Detection baseline for Rep Segmentation.

This script implements the simplest possible baseline:
1. Compute acc_mag = sqrt(ax^2 + ay^2 + az^2)
2. Smooth with uniform filter (window=9)
3. Use scipy.signal.find_peaks to detect local maxima
4. Peaks define rep boundaries

Usage:
    python scripts/evaluate_peak_baseline.py \
        --config config.yaml \
        --output artifacts/baseline_comparison/peak_detection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.sdtw_rep_segmentation import (
    infer_sample_rate_hz,
    match_segments,
    summarize_detection_metrics,
)


def compute_acc_mag(df: pd.DataFrame) -> np.ndarray:
    """Compute acceleration magnitude from ax, ay, az."""
    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    if not acc_cols:
        raise ValueError("No accelerometer columns found")
    acc = df[acc_cols].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(acc ** 2, axis=1))


def compute_gyro_mag(df: pd.DataFrame) -> np.ndarray:
    """Compute gyroscope magnitude from gx, gy, gz."""
    gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]
    if not gyro_cols:
        raise ValueError("No gyroscope columns found")
    gyro = df[gyro_cols].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(gyro ** 2, axis=1))


def compute_6axis_mag(df: pd.DataFrame) -> np.ndarray:
    """Compute 6-axis magnitude from ax, ay, az, gx, gy, gz."""
    cols = [c for c in ["ax", "ay", "az", "gx", "gy", "gz"] if c in df.columns]
    if not cols:
        raise ValueError("No IMU columns found")
    data = df[cols].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(data ** 2, axis=1))


def smooth_signal(x: np.ndarray, window: int = 9) -> np.ndarray:
    """Apply uniform smoothing filter."""
    if window <= 1:
        return x.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def estimate_duration_prior(train_streams: Sequence[Tuple[str, pd.DataFrame]]) -> Dict[str, float]:
    """Estimate rep duration statistics from train subjects using ground truth reps."""
    durations = []
    for _, df in train_streams:
        if "phase" not in df.columns:
            continue
        from preprocessing.micro_macro_segments import truth_reps_from_labels
        try:
            if "action_type" in df.columns:
                reps = truth_reps_from_labels(
                    df["phase"].to_numpy(),
                    actions=df["action_type"].astype(str).to_numpy(),
                    min_phase_samples=1,
                )
            else:
                reps = truth_reps_from_labels(df["phase"].to_numpy(), min_phase_samples=1)
            for rep in reps:
                durations.append(int(rep.end_idx) - int(rep.start_idx))
        except Exception:
            continue
    
    if not durations:
        # Fallback: assume ~200 samples @ 100Hz = 2 seconds
        return {"median": 200.0, "min": 100.0, "max": 400.0, "sample_rate": 100.0}
    
    durations = np.asarray(durations, dtype=np.float64)
    return {
        "median": float(np.median(durations)),
        "min": float(np.percentile(durations, 10)),
        "max": float(np.percentile(durations, 90)),
        "sample_rate": 100.0,  # Assume 100Hz
    }


def detect_peaks(
    acc_mag: np.ndarray,
    duration_prior: Dict[str, float],
    height_percentile: float = 25.0,
    prominence_percentile: float = 50.0,
) -> List[int]:
    """Detect peak indices in smoothed acc_mag signal.
    
    Returns indices where peaks occur (one per rep).
    """
    # Dynamic thresholds based on signal statistics
    height = float(np.percentile(acc_mag, height_percentile))
    
    # Prominence: relative to local minima
    baseline = float(np.percentile(acc_mag, 10))
    prominence = max(0.1, float(np.percentile(acc_mag, prominence_percentile)) - baseline)
    
    # Distance: minimum samples between reps (from duration prior)
    min_distance = int(duration_prior["min"] * 0.8)
    
    peaks, _ = signal.find_peaks(
        acc_mag,
        height=height,
        distance=min_distance,
        prominence=prominence,
    )
    return peaks.tolist()


def peaks_to_reps(
    peaks: List[int],
    acc_mag: np.ndarray,
    duration_prior: Dict[str, float],
) -> List[Tuple[int, int]]:
    """Convert peak indices to rep [start, end] intervals.
    
    Each rep is the interval between consecutive peaks, centered on the peak.
    """
    if not peaks:
        return []
    
    median_duration = int(duration_prior["median"])
    half_dur = median_duration // 2
    
    reps = []
    for i, peak in enumerate(peaks):
        if i == 0:
            # First rep: start from beginning or before first peak
            start = max(0, peak - half_dur)
        else:
            # Midpoint between previous peak and this peak
            start = (peaks[i-1] + peak) // 2
        
        if i == len(peaks) - 1:
            # Last rep: end at signal end or after last peak
            end = min(len(acc_mag), peak + half_dur)
        else:
            # Midpoint between this peak and next peak
            end = (peak + peaks[i+1]) // 2
        
        reps.append((int(start), int(end)))
    
    return reps


def evaluate_stream(
    stream_df: pd.DataFrame,
    duration_prior: Dict[str, float],
    mag_mode: str = "acc",
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    """Evaluate Peak Detection on a single stream.
    
    mag_mode: "acc" | "gyro" | "6axis"
    Returns (predicted_reps, metrics).
    """
    if mag_mode == "acc":
        mag = compute_acc_mag(stream_df)
        feature_name = "acc_mag_peak"
    elif mag_mode == "gyro":
        mag = compute_gyro_mag(stream_df)
        feature_name = "gyro_mag_peak"
    elif mag_mode == "6axis":
        mag = compute_6axis_mag(stream_df)
        feature_name = "6axis_mag_peak"
    else:
        raise ValueError(f"Unknown mag_mode: {mag_mode}")
    
    smoothed = smooth_signal(mag)
    
    # Detect peaks
    peaks = detect_peaks(smoothed, duration_prior)
    pred_reps = peaks_to_reps(peaks, smoothed, duration_prior)
    
    # Get ground truth reps using standard pairing
    from preprocessing.micro_macro_segments import truth_reps_from_labels
    truth_reps_list = truth_reps_from_labels(stream_df["phase"].to_numpy(), min_phase_samples=1)
    truth = [(int(r.start_idx), int(r.end_idx)) for r in truth_reps_list]
    
    sample_rate = infer_sample_rate_hz(stream_df)
    
    # Summarize metrics
    from preprocessing.sdtw_rep_segmentation import SegmentDetection
    detections = [
        SegmentDetection(
            start_idx=start,
            end_idx=end,
            cost=0.0,
            feature=feature_name,
            action_type="unknown",
            template_id="peak_baseline",
            exemplar_source="",
            normalized_cost=0.0,
        )
        for start, end in pred_reps
    ]
    
    metrics = summarize_detection_metrics(detections, truth, sample_rate)
    return pred_reps, metrics


def run_peak_baseline(
    config_path: Path,
    output_dir: Path,
    subjects: List[str] | None = None,
    actions: List[str] | None = None,
    mag_mode: str = "acc",
) -> Dict:
    """Run Peak Detection baseline with strict LOSO.
    
    For each fold:
    1. Use 8 subjects as train to estimate duration prior
    2. Use 1 subject as test
    3. Report metrics
    """
    # Load config
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {})
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    exclude_patterns = data_cfg.get("exclude_patterns", ["*whole_session*", "*_w", "*rest_after*"])
    
    if actions is None:
        actions = data_cfg.get("include_actions", [])
    if not actions:
        # Auto-detect actions
        actions = sorted({p.name for subject in data_dir.iterdir() if subject.is_dir()
                         for sess in subject.iterdir() if sess.is_dir()
                         for p in sess.iterdir() if p.is_dir() and "rest" not in p.name})
    
    if subjects is None:
        subjects = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    
    print(f"[INFO] Peak Detection Baseline")
    print(f"[INFO] subjects={subjects}, actions={actions}")
    
    # Helper to load streams
    def load_subject_streams(subject: str) -> List[Tuple[str, pd.DataFrame]]:
        streams = []
        subject_dir = data_dir / subject
        if not subject_dir.exists():
            return []
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for action in actions:
                action_dir = session_dir / action
                if not action_dir.exists():
                    continue
                for set_dir in sorted(action_dir.iterdir()):
                    if not set_dir.is_dir() or not set_dir.name.startswith("set"):
                        continue
                    # Check exclude patterns
                    skip = False
                    for pattern in exclude_patterns:
                        import fnmatch
                        if any(fnmatch.fnmatch(part, pattern) for part in set_dir.parts):
                            skip = True
                            break
                    if skip:
                        continue
                    csvs = sorted(set_dir.glob("*.csv"))
                    frames = []
                    for csv_path in csvs:
                        if "whole_session" in csv_path.name:
                            continue
                        try:
                            df = pd.read_csv(csv_path)
                            if "phase" in df.columns:
                                frames.append(df)
                        except Exception:
                            continue
                    if frames:
                        stream_df = pd.concat(frames, ignore_index=True)
                        streams.append((f"{subject}/{session_dir.name}/{action}/{set_dir.name}", stream_df))
        return streams
    
    # Run LOSO
    all_metrics = []
    all_rows = []
    
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        print(f"\n[Fold] test={test_subject}, train={train_subjects}")
        
        # Load train streams
        train_streams = []
        for subject in train_subjects:
            train_streams.extend(load_subject_streams(subject))
        
        # Estimate duration prior from train
        duration_prior = estimate_duration_prior(train_streams)
        print(f"  Duration prior: median={duration_prior['median']:.1f}, "
              f"min={duration_prior['min']:.1f}, max={duration_prior['max']:.1f}")
        
        # Evaluate test streams
        test_streams = load_subject_streams(test_subject)
        for stream_id, stream_df in test_streams:
            pred_reps, metrics = evaluate_stream(stream_df, duration_prior, mag_mode=mag_mode)
            metrics["stream_id"] = stream_id
            metrics["test_subject"] = test_subject
            all_metrics.append(metrics)
            all_rows.append({
                "test_subject": test_subject,
                "stream_id": stream_id,
                "n_true": metrics["n_true"],
                "n_pred": metrics["n_pred"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "rep_f1": metrics["f1"],
                "start_mae_ms": metrics.get("start_mae_ms", float("nan")),
                "end_mae_ms": metrics.get("end_mae_ms", float("nan")),
            })
            print(f"  {stream_id}: Rep F1={metrics['f1']:.3f}, "
                  f"n_true={metrics['n_true']:.0f}, n_pred={metrics['n_pred']:.0f}")
    
    # Aggregate
    total_tp = sum(m["tp"] for m in all_metrics)
    total_fp = sum(m["fp"] for m in all_metrics)
    total_fn = sum(m["fn"] for m in all_metrics)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    overall = {
        "streams": len(all_metrics),
        "n_true": sum(m["n_true"] for m in all_metrics),
        "n_pred": sum(m["n_pred"] for m in all_metrics),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "rep_f1": f1,
        "start_mae_ms": float(np.mean([m.get("start_mae_ms", float("nan")) for m in all_metrics if np.isfinite(m.get("start_mae_ms", float("nan")))])),
        "end_mae_ms": float(np.mean([m.get("end_mae_ms", float("nan")) for m in all_metrics if np.isfinite(m.get("end_mae_ms", float("nan")))])),
    }
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": "Magnitude Peak Detection",
        "overall": overall,
        "stream_metrics": all_rows,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(output_dir / "stream_metrics.csv", index=False)
    
    print(f"\n{'='*60}")
    print("PEAK DETECTION BASELINE SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(overall, indent=2))
    print(f"\n[OK] Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Peak Detection baseline for Rep Segmentation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/peak_detection")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--mag-mode", default="acc", choices=["acc", "gyro", "6axis"],
                        help="Magnitude mode for peak detection (default: acc)")
    args = parser.parse_args()
    
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    actions = [a.strip() for a in args.actions.split(",") if a.strip()] if args.actions else None
    
    run_peak_baseline(
        Path(args.config),
        Path(args.output),
        subjects=subjects,
        actions=actions,
        mag_mode=args.mag_mode,
    )


if __name__ == "__main__":
    main()
