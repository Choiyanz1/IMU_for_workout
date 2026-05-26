from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.reporting import primary_metric_table, write_run_manifest
from preprocessing.window_pipeline import ZScoreStats, apply_zscore
from scripts.grid_micro_macro_postprocess import _load_model_for_grid
from train.micro_macro_recognition import (
    DTWMicroConfig,
    MicroMacroConfig,
    TrainConfig,
    _evaluate_streams,
    _filter_subjects,
    _load_config,
    _load_streams,
    _median_sample_rate,
    _resample_streams_to_rate,
    _resolve_device,
    _write_report_md,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate a trained micro/macro run with overridden postprocessing/eval settings.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=["sets", "whole", "both"], default=None)
    parser.add_argument("--micro-smoothing-window", type=int, default=None)
    parser.add_argument("--min-phase-samples", type=int, default=None)
    parser.add_argument("--max-phase-gap-samples", type=int, default=None)
    parser.add_argument("--min-rep-duration-seconds", type=float, default=None)
    parser.add_argument("--min-rep-confidence", type=float, default=None)
    parser.add_argument("--micro-decoder", choices=["greedy", "viterbi"], default=None)
    parser.add_argument("--micro-decoder-switch-penalty", type=float, default=None)
    parser.add_argument("--micro-decoder-invalid-transition-penalty", type=float, default=None)
    parser.add_argument("--micro-decoder-min-run-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config_path = args.config or (run_dir / "metadata" / "config_snapshot.yaml")
    raw = _load_config(config_path)

    feature_cfg = raw.get("feature", {}) or {}
    train_cfg = TrainConfig(**(raw.get("train", {}) or {}))
    mm_raw = dict(raw.get("micro_macro", {}) or {})
    dtw_raw = dict(mm_raw.pop("dtw", {}) or {})
    configured_micro_source = str(mm_raw.pop("micro_source", "both"))
    dtw_cfg = DTWMicroConfig(**dtw_raw)
    mm_cfg = MicroMacroConfig(**mm_raw)

    if args.micro_smoothing_window is not None:
        mm_cfg.micro_smoothing_window = int(args.micro_smoothing_window)
    if args.min_phase_samples is not None:
        mm_cfg.min_phase_samples = int(args.min_phase_samples)
    if args.max_phase_gap_samples is not None:
        mm_cfg.max_phase_gap_samples = int(args.max_phase_gap_samples)
    if args.min_rep_duration_seconds is not None:
        mm_cfg.min_rep_duration_seconds = float(args.min_rep_duration_seconds)
    if args.min_rep_confidence is not None:
        mm_cfg.min_rep_confidence = float(args.min_rep_confidence)
    if args.micro_decoder is not None:
        mm_cfg.micro_decoder = str(args.micro_decoder)
    if args.micro_decoder_switch_penalty is not None:
        mm_cfg.micro_decoder_switch_penalty = float(args.micro_decoder_switch_penalty)
    if args.micro_decoder_invalid_transition_penalty is not None:
        mm_cfg.micro_decoder_invalid_transition_penalty = float(args.micro_decoder_invalid_transition_penalty)
    if args.micro_decoder_min_run_samples is not None:
        mm_cfg.micro_decoder_min_run_samples = int(args.micro_decoder_min_run_samples)

    device_setting = args.device or train_cfg.device
    device = _resolve_device(device_setting)
    train_cfg.device = str(device)

    stats = ZScoreStats.load(run_dir / "metadata" / "zscore_stats.json")
    model, macro_classes, micro_classes, semantic_micro_classes, imu_columns = _load_model_for_grid(run_dir, device, "true" if mm_cfg.causal else "false")

    modes = [str(args.mode)] if args.mode else list(mm_cfg.train_on_modes)
    streams, subjects, actions = _load_streams(raw, modes)
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int((raw.get("window", {}) or {}).get("sample_rate_hz", 50))
    if bool(mm_cfg.resample_to_window_rate):
        streams = _resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    test_subject = str(train_cfg.test_subject) if train_cfg.test_subject else sorted(set(subjects))[-1]
    train_subjects = [s for s in sorted(set(subjects)) if s != test_subject]
    train_streams = _filter_subjects(streams, train_subjects, subject_column)
    test_streams = _filter_subjects(streams, [test_subject], subject_column)
    train_sequences = [df for _, df in train_streams]

    train_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    train_sequences = [df for _, df in train_streams]

    out_dir = args.output_dir or (run_dir / f"reeval_smooth{mm_cfg.micro_smoothing_window}")
    for sub in ("models", "metrics", "detections", "plots", "metadata"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    summary = _evaluate_streams(
        model=model,
        streams=test_streams,
        train_sequences_for_dtw=train_sequences,
        imu_columns=imu_columns,
        macro_classes=macro_classes,
        micro_classes=micro_classes,
        semantic_micro_classes=semantic_micro_classes,
        micro_source="tcn",
        mm_cfg=mm_cfg,
        dtw_cfg=dtw_cfg,
        output_dir=out_dir,
        device=device,
    )
    summary.update(
        {
            "micro_source": "tcn",
            "task": "micro_macro_recognition_reeval",
            "model_name": "ds_ms_tcn_tcn_reeval",
            "configured_micro_source": configured_micro_source,
            "resolved_micro_source": "tcn",
            "modes": modes,
            "train_subjects": train_subjects,
            "test_subject": test_subject,
            "macro_classes": macro_classes,
            "micro_classes": micro_classes,
            "semantic_micro_classes": semantic_micro_classes,
            "sample_rate_hz_for_training_slices": _median_sample_rate(train_streams, float(target_sample_rate)),
            "resample_to_window_rate": bool(mm_cfg.resample_to_window_rate),
            "min_rep_duration_seconds": float(mm_cfg.min_rep_duration_seconds),
            "min_rep_confidence": float(mm_cfg.min_rep_confidence),
            "micro_smoothing_window": int(mm_cfg.micro_smoothing_window),
            "micro_decoder": str(mm_cfg.micro_decoder),
            "micro_decoder_switch_penalty": float(mm_cfg.micro_decoder_switch_penalty),
            "micro_decoder_invalid_transition_penalty": float(mm_cfg.micro_decoder_invalid_transition_penalty),
            "micro_decoder_min_run_samples": int(mm_cfg.micro_decoder_min_run_samples),
            "primary_metrics": primary_metric_table(summary.get("overall", {}) or {}),
        }
    )
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_run_manifest(
        out_dir,
        task="micro_macro_recognition_reeval",
        model_name="ds_ms_tcn_tcn_reeval",
        config_path=config_path,
        extras={
            "source_run_dir": str(run_dir),
            "micro_smoothing_window": int(mm_cfg.micro_smoothing_window),
            "min_phase_samples": int(mm_cfg.min_phase_samples),
            "max_phase_gap_samples": int(mm_cfg.max_phase_gap_samples),
            "min_rep_duration_seconds": float(mm_cfg.min_rep_duration_seconds),
            "min_rep_confidence": float(mm_cfg.min_rep_confidence),
            "micro_decoder": str(mm_cfg.micro_decoder),
            "micro_decoder_switch_penalty": float(mm_cfg.micro_decoder_switch_penalty),
            "micro_decoder_invalid_transition_penalty": float(mm_cfg.micro_decoder_invalid_transition_penalty),
            "micro_decoder_min_run_samples": int(mm_cfg.micro_decoder_min_run_samples),
        },
    )
    _write_report_md(out_dir, summary, config_path, train_cfg, mm_cfg, dtw_cfg)
    print(json.dumps(summary["overall"], indent=2))
    print(f"[OK] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
