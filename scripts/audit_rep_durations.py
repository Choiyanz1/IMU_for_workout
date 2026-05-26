from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cb = _load_module(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
from preprocessing.micro_macro_segments import truth_reps_from_labels, CONCENTRIC_LABEL, ECCENTRIC_LABEL


config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/micro_macro_recognition_8act_test_yushuan.yaml"
min_phase = int(sys.argv[2]) if len(sys.argv) > 2 else 3

raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
mm_raw = raw.get("micro_macro", {}) or {}
mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})

feature_cfg = raw.get("feature", {}) or {}
imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
time_column = str(feature_cfg.get("time_column", "sensor_ts"))
target_sample_rate = int(raw.get("window", {}).get("sample_rate_hz", 100))
subject_column = str(feature_cfg.get("subject_column", "subject_id"))

modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
streams, subjects, _ = cb._load_streams(raw, modes)
if mm_cfg.resample_to_window_rate:
    streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)

def action_from_path(sid):
    parts = [p for p in str(sid).split("/") if p]
    return parts[-2] if len(parts) >= 2 else "unknown"

def subject_from_path(sid):
    parts = [p for p in str(sid).split("/") if p]
    return parts[0] if parts else "unknown"

rows = []
for sid, df in streams:
    truth = truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=mm_cfg.min_phase_samples,
    )
    action = action_from_path(sid)
    subject = subject_from_path(sid)
    for rep in truth:
        total = int(rep.end_idx) - int(rep.start_idx)
        con = int(rep.transition_idx) - int(rep.start_idx)
        ecc = int(rep.end_idx) - int(rep.transition_idx)
        rows.append({
            "stream": sid, "action": action, "subject": subject,
            "total": total, "concentric": con, "eccentric": ecc,
            "ratio": round(con / max(1, total), 3),
        })

per_action = defaultdict(list)
for r in rows:
    per_action[r["action"]].append(r)

print(f"{'Action':25s} {'Count':>6s} {'Min':>6s} {'P10':>6s} {'P50':>6s} {'P90':>6s} {'Max':>7s} {'Mean':>8s} {'Anomalies':>10s}")
print("=" * 90)
anomalies = []
for action in sorted(per_action):
    vals = np.array([r["total"] for r in per_action[action]])
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    lo = p10 - 1.5 * (p90 - p10)
    hi = p90 + 1.5 * (p90 - p10)
    low_thresh = max(10, int(p10 - 2.0 * (p90 - p10)))
    high_thresh = int(p90 + 2.0 * (p90 - p10))
    n_anom = sum(1 for v in vals if v < low_thresh or v > high_thresh)
    print(f"{action:25s} {len(vals):>6d} {int(vals.min()):>6d} {int(p10):>6d} {int(p50):>6d} {int(p90):>6d} {int(vals.max()):>7d} {vals.mean():>8.1f} {n_anom:>10d}")
    for r in per_action[action]:
        if r["total"] < low_thresh or r["total"] > high_thresh:
            anomalies.append(r)

print(f"\n=== {len(anomalies)} 個異常 rep ===")
print(f"{'Subject':12s} {'Action':22s} {'Set':50s} {'Samples':>8s} {'Con':>6s} {'Ecc':>6s}")
print("=" * 110)
for r in sorted(anomalies, key=lambda x: x["total"], reverse=True):
    set_name = "/".join(r["stream"].split("/")[-3:])
    print(f"{r['subject']:12s} {r['action']:22s} {set_name:50s} {r['total']:>8d} {r['concentric']:>6d} {r['eccentric']:>6d}")
