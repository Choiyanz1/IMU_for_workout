"""Export the current full automatic realtime workout pipeline bundle.

The bundle contains:
- raw6 causal CNN phase model (PyTorch and optional ONNX);
- periodic active/rest RF gate;
- dual-head RF action branch;
- normalization stats, duration priors, and decoder/runtime config.

It is trained on all valid set streams plus available rest streams for deployment
engineering. Do not use this artifact for headline held-out metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dual_head_rf_action_loso import ACTIONS, load_non_action_streams  # noqa: E402
from scripts.evaluate_periodic_active_gate_loso import train_gate as train_periodic_active_gate  # noqa: E402
from scripts.evaluate_predicted_action_top5_pipeline import train_action_rf  # noqa: E402
from scripts.evaluate_realtime_soft_top5_pipeline import compute_action_norm  # noqa: E402
from scripts.export_raw6_cnn_deploy import _write_onnx  # noqa: E402
from scripts.new_c_pipeline.duration_merge_decoder_9fold import build_duration_priors  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import stream_action, train_raw6_model  # noqa: E402
from scripts.new_c_pipeline.selective_duration_merge_decoder_9fold import ACTION_SETS  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, set_seed, should_exclude  # noqa: E402
from train.micro_macro_recognition import _load_streams  # noqa: E402


def _json_dump(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _rf_to_json_dict(model) -> dict:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :].astype(float)
        denom = np.sum(values, axis=1, keepdims=True)
        denom = np.where(denom < 1e-12, 1.0, denom)
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "proba": (values / denom).tolist(),
            }
        )
    return {
        "format": "sklearn_random_forest_classifier_v1",
        "classes": [cls.item() if hasattr(cls, "item") else cls for cls in model.classes_],
        "n_features_in": int(model.n_features_in_),
        "trees": trees,
    }


def _scaler_to_json_dict(scaler) -> dict:
    return {
        "format": "standard_scaler_v1",
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
    }


def _build_runtime_args(args: argparse.Namespace, sample_rate_hz: float) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(args.seed),
        sample_rate_hz=float(sample_rate_hz),
        window_samples=int(args.active_gate_window_samples),
        stride_samples=int(args.active_gate_stride_samples),
        window_active_fraction=float(args.window_active_threshold),
        n_estimators=int(args.active_n_estimators),
        max_depth=int(args.active_max_depth),
        min_samples_leaf=int(args.active_min_samples_leaf),
        label_mode="binary",
        transition_energy_quantile=0.7,
        enter_threshold=float(args.active_enter_threshold),
        exit_threshold=float(args.active_exit_threshold),
        enter_hold_samples=max(1, int(args.phase_step_samples)),
        exit_hold_samples=int(args.active_exit_hold_samples),
        min_active_samples=int(args.min_active_segment_samples),
        bridge_gap_samples=int(args.active_mask_bridge_samples),
        cooldown_samples=0,
        action_window_samples=int(args.action_window_samples),
        action_stride_samples=int(args.action_stride_samples),
        window_active_threshold=float(args.window_active_threshold),
        action_n_estimators=int(args.action_n_estimators),
        action_max_depth=int(args.action_max_depth),
        action_min_samples_leaf=int(args.action_min_samples_leaf),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full automatic realtime workout bundle.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/deploy/full_auto_realtime_current")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--active-gate-features", choices=["basic", "periodic"], default="periodic")
    parser.add_argument("--active-gate-window-samples", type=int, default=200)
    parser.add_argument("--active-gate-stride-samples", type=int, default=50)
    parser.add_argument("--active-enter-threshold", type=float, default=0.7)
    parser.add_argument("--active-exit-threshold", type=float, default=0.7)
    parser.add_argument("--active-exit-hold-samples", type=int, default=0)
    parser.add_argument("--min-active-segment-samples", type=int, default=0)
    parser.add_argument("--active-mask-bridge-samples", type=int, default=0)
    parser.add_argument("--active-n-estimators", type=int, default=50)
    parser.add_argument("--active-max-depth", type=int, default=12)
    parser.add_argument("--active-min-samples-leaf", type=int, default=2)
    parser.add_argument("--action-window-samples", type=int, default=200)
    parser.add_argument("--action-stride-samples", type=int, default=100)
    parser.add_argument("--action-n-estimators", type=int, default=50)
    parser.add_argument("--action-max-depth", type=int, default=12)
    parser.add_argument("--action-min-samples-leaf", type=int, default=2)
    parser.add_argument("--window-active-threshold", type=float, default=0.5)
    parser.add_argument("--phase-window-samples", type=int, default=300)
    parser.add_argument("--phase-step-samples", type=int, default=10)
    parser.add_argument("--phase-smoothing-window", type=int, default=25)
    parser.add_argument("--fixed-lag-samples", type=int, default=100)
    parser.add_argument("--viterbi-penalty", type=float, default=0.3)
    parser.add_argument("--merge-threshold-scale", type=float, default=0.8)
    parser.add_argument("--soft-top5-mass-threshold", type=float, default=0.65)
    parser.add_argument("--soft-action-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--soft-margin-threshold", type=float, default=0.05)
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--event-confirm-min-reps", type=int, default=2)
    parser.add_argument("--event-confirm-gap-samples", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in set_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    imu_columns = list(raw_cfg.get("feature", {}).get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    sample_rate_hz = float(raw_cfg.get("window", {}).get("sample_rate_hz", 100))

    runtime_args = _build_runtime_args(args, sample_rate_hz)

    print(f"[EXPORT] valid set streams={len(set_streams)} rest streams={len(rest_streams)}")
    print(f"[EXPORT] actions={sorted({stream_action(sid) for sid, _ in set_streams})}")
    print(f"[EXPORT] train phase CNN hidden={args.hidden} epochs={args.epochs} device={device}")
    phase_model, phase_mean, phase_std, n_segments = train_raw6_model(set_streams, imu_columns, args.hidden, args.epochs, device)
    duration_priors = build_duration_priors(set_streams, [5])

    print("[EXPORT] train periodic active gate")
    active_clf, active_scaler, active_info = train_periodic_active_gate(
        [*set_streams, *rest_streams], imu_columns, runtime_args, args.active_gate_features
    )

    print("[EXPORT] train dual-head action RF")
    action_active_rf, action_rf, train_action_streams = train_action_rf(set_streams, rest_streams, imu_columns, runtime_args)
    action_mean, action_std = compute_action_norm(train_action_streams, imu_columns)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_model = phase_model.cpu().eval()
    torch.save(
        {
            "model_state_dict": phase_model.state_dict(),
            "model_class": "CausalCNN_PhaseOnly",
            "input_channels": len(imu_columns),
            "hidden": int(args.hidden),
            "num_classes": 2,
            "dropout": 0.2,
        },
        out_dir / "phase_model.pt",
    )
    if not args.skip_onnx:
        _write_onnx(phase_model, out_dir / "phase_model.onnx", len(imu_columns), int(args.phase_window_samples))

    joblib.dump(active_clf, out_dir / "active_gate_rf.joblib")
    joblib.dump(active_scaler, out_dir / "active_gate_scaler.joblib")
    joblib.dump(action_active_rf, out_dir / "action_active_rf.joblib")
    joblib.dump(action_rf, out_dir / "action_rf.joblib")
    _json_dump(out_dir / "active_gate_rf.json", _rf_to_json_dict(active_clf))
    _json_dump(out_dir / "active_gate_scaler.json", _scaler_to_json_dict(active_scaler))
    _json_dump(out_dir / "action_active_rf.json", _rf_to_json_dict(action_active_rf))
    _json_dump(out_dir / "action_rf.json", _rf_to_json_dict(action_rf))

    _json_dump(
        out_dir / "normalization.json",
        {
            "imu_columns": imu_columns,
            "phase_mean": np.asarray(phase_mean, dtype=float).tolist(),
            "phase_std": np.asarray(phase_std, dtype=float).tolist(),
            "action_mean": np.asarray(action_mean, dtype=float).tolist(),
            "action_std": np.asarray(action_std, dtype=float).tolist(),
        },
    )
    _json_dump(
        out_dir / "pipeline_config.json",
        {
            "sample_rate_hz": sample_rate_hz,
            "actions": ACTIONS,
            "active_gate": {
                "feature_mode": args.active_gate_features,
                "window_samples": int(args.active_gate_window_samples),
                "stride_samples": int(args.active_gate_stride_samples),
                "enter_threshold": float(args.active_enter_threshold),
                "exit_threshold": float(args.active_exit_threshold),
                "enter_hold_samples": max(1, int(args.phase_step_samples)),
                "exit_hold_samples": int(args.active_exit_hold_samples),
                "min_active_samples": int(args.min_active_segment_samples),
                "bridge_gap_samples": int(args.active_mask_bridge_samples),
                "window_active_fraction": float(args.window_active_threshold),
            },
            "action_branch": {
                "window_samples": int(args.action_window_samples),
                "stride_samples": int(args.action_stride_samples),
                "posterior_weight": "action_active_probability",
            },
            "phase_decoder": {
                "window_samples": int(args.phase_window_samples),
                "step_samples": int(args.phase_step_samples),
                "smoothing_window": int(args.phase_smoothing_window),
                "fixed_lag_samples": int(args.fixed_lag_samples),
                "viterbi_penalty": float(args.viterbi_penalty),
                "fixed_lag_active_mask": True,
                "min_phase_samples": 3,
                "max_phase_gap_samples": 3,
            },
            "soft_top5": {
                "enabled": True,
                "threshold_scale": float(args.merge_threshold_scale),
                "top5_actions": ACTION_SETS["top5"],
                "top5_mass_threshold": float(args.soft_top5_mass_threshold),
                "action_confidence_threshold": float(args.soft_action_confidence_threshold),
                "margin_threshold": float(args.soft_margin_threshold),
                "max_gap_samples": int(args.max_gap_samples),
                "duration_priors": duration_priors,
            },
            "event_confirmation": {
                "enabled": True,
                "min_reps": int(args.event_confirm_min_reps),
                "gap_samples": int(args.event_confirm_gap_samples),
            },
        },
    )
    _json_dump(
        out_dir / "metadata.json",
        {
            "artifact_name": out_dir.name,
            "model_family": "full automatic IMU workout pipeline",
            "source_config": str(args.config),
            "sample_rate_hz": sample_rate_hz,
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "seed": int(args.seed),
            "train_set_streams": len(set_streams),
            "train_rest_streams": len(rest_streams),
            "train_phase_segments": int(n_segments),
            "active_gate_train_info": active_info,
            "excluded_sessions": EXCLUDED_SESSIONS,
            "files": {
                "phase_torch": "phase_model.pt",
                "phase_onnx": None if args.skip_onnx else "phase_model.onnx",
                "active_gate_rf": "active_gate_rf.joblib",
                "active_gate_scaler": "active_gate_scaler.joblib",
                "active_gate_rf_json": "active_gate_rf.json",
                "active_gate_scaler_json": "active_gate_scaler.json",
                "action_active_rf": "action_active_rf.joblib",
                "action_rf": "action_rf.joblib",
                "action_active_rf_json": "action_active_rf.json",
                "action_rf_json": "action_rf.json",
                "normalization": "normalization.json",
                "pipeline_config": "pipeline_config.json",
            },
            "caveat": "Deployment-engineering artifact trained on all valid data. Full-session quality is still gated by active/rest detection; tune active thresholds on-device if needed.",
        },
    )

    print(f"[OK] exported full automatic realtime bundle to {out_dir}")


if __name__ == "__main__":
    main()
