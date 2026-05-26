"""Export deployable per-action plain causal RF models.

This trains one RandomForestClassifier per action on all valid training streams and
writes the model, normalization stats, label map, and metadata needed for later
inference. It intentionally keeps the deployable artifact small and separate from
raw datasets and experiment outputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_mod(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
crf = _load_mod(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")


EXCLUDED_SESSION_SUBSTRINGS = (
    "yanz/1000",
    "thomas/thomas",
    "thomas/thomas_2",
    "kevin/kevin",
    "_tsenyu_temp",
    "_ziho_temp",
)


def _extract_action(stream_id: str) -> str:
    parts = [p for p in str(stream_id).replace("\\", "/").split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def _is_valid_stream(stream_id: str) -> bool:
    normalized = str(stream_id).replace("\\", "/")
    return not any(excluded in normalized for excluded in EXCLUDED_SESSION_SUBSTRINGS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployable per-action plain RF models.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/deploy/per_action_plain_rf_current")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mm_raw = raw.get("micro_macro", {}) or {}
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, _subjects, actions = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

    streams = [(sid, df) for sid, df in streams if _is_valid_stream(sid)]
    out_dir = Path(args.output)
    models_dir = out_dir / "models"
    stats_dir = out_dir / "normalization"
    models_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    action_summaries = {}
    for action in actions:
        action_streams = [(sid, df) for sid, df in streams if _extract_action(sid) == action]
        if not action_streams:
            continue

        print(f"[EXPORT] action={action} streams={len(action_streams)}")
        stats = cb.compute_train_stats([df for _, df in action_streams], imu_columns)
        train_z = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in action_streams]
        clf = crf.train_causal_rf(
            train_z,
            imu_columns,
            window_size=int(args.window_size),
            stride=int(args.train_stride),
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            max_samples=float(args.max_samples),
        )

        joblib.dump(clf, models_dir / f"{action}.joblib", compress=3)
        stats.save(stats_dir / f"{action}.json")
        action_summaries[action] = {
            "streams": len(action_streams),
            "model": f"models/{action}.joblib",
            "normalization": f"normalization/{action}.json",
        }

    label_map = {
        "micro_labels": list(cb.MICRO_LABELS),
        "other_label": cb.OTHER_LABEL,
        "eccentric_label": cb.ECCENTRIC_LABEL,
        "concentric_label": cb.CONCENTRIC_LABEL,
    }
    metadata = {
        "artifact_name": "per_action_plain_rf_current",
        "model_family": "per-action causal RandomForestClassifier",
        "source_config": str(config_path),
        "actions": sorted(action_summaries),
        "imu_columns": imu_columns,
        "sample_rate_hz": target_sample_rate,
        "excluded_session_substrings": list(EXCLUDED_SESSION_SUBSTRINGS),
        "hyperparameters": {
            "window_size": int(args.window_size),
            "train_stride": int(args.train_stride),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "max_samples": float(args.max_samples),
        },
        "action_artifacts": action_summaries,
        "inference_note": "Apply the per-action z-score stats, extract trailing 1.0s RF features, then call the corresponding action model.",
    }

    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] exported {len(action_summaries)} action models to {out_dir}")


if __name__ == "__main__":
    main()
