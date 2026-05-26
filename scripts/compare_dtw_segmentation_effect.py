from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.rep_segmentation import _load_rep_csvs, _truth_segments_for_stream
from preprocessing.sdtw_rep_segmentation import SDTWConfig, detect_reps_sdtw_templates, fit_sdtw_templates, infer_sample_rate_hz, summarize_detection_metrics


def _natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _load_set_dir(set_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(set_dir.glob("rep*.csv"), key=_natural_key):
        df = pd.read_csv(csv_path)
        df = df.copy()
        df["_source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No rep CSVs in {set_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SDTW rep segmentation against online decoder results on the same streams.")
    parser.add_argument("--config", type=Path, default=Path("configs/rep_segmentation.yaml"))
    parser.add_argument("--set-dir", action="append", required=True, help="Set directory to evaluate. Repeatable.")
    parser.add_argument("--online-summary", action="append", default=[], help="Optional matching streaming_summary.json paths for side-by-side comparison.")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/dtw_vs_online_compare.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = cfg.get("data", {}) or {}
    seg_cfg = cfg.get("segmentation", {}) or {}
    feature_cfg = cfg.get("feature", {}) or {}
    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])
    raw_sdtw_cfg = dict(seg_cfg.get("sdtw", {}) or {})
    motion_columns = list(raw_sdtw_cfg.pop("motion_columns", feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"])))
    sdtw_cfg = SDTWConfig(**raw_sdtw_cfg)
    subjects = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    online_summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.online_summary]
    results = []
    for idx, set_dir_value in enumerate(args.set_dir):
        set_dir = Path(set_dir_value)
        subject = set_dir.parent.parent.name
        action = set_dir.parent.name
        train_subjects = [s for s in subjects if s != subject]
        train_reps = []
        for train_subject in train_subjects:
            train_reps.extend(_load_rep_csvs(data_dir, train_subject, action, exclude_patterns, sdtw_cfg))
        templates = fit_sdtw_templates(action, train_reps, motion_columns, sdtw_cfg)
        stream_df = _load_set_dir(set_dir)
        detections = detect_reps_sdtw_templates(stream_df, templates, motion_columns, sdtw_cfg)
        truth = _truth_segments_for_stream(stream_df)
        sample_rate = infer_sample_rate_hz(stream_df)
        dtw_metrics = summarize_detection_metrics(detections, truth, sample_rate_hz=sample_rate)

        row = {
            "set_dir": set_dir.as_posix(),
            "subject": subject,
            "action": action,
            "dtw": dtw_metrics,
        }
        if idx < len(online_summaries):
            online = online_summaries[idx]
            row["online"] = {
                k: online.get(k)
                for k in [
                    "precision",
                    "recall",
                    "f1",
                    "start_mae_ms",
                    "end_mae_ms",
                    "transition_mae_ms",
                    "rep_action_accuracy",
                    "online_rep_emit_delay_ms",
                    "online_rep_emit_delay_ms_p95",
                ]
            }
        results.append(row)

    out = {"results": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
