from __future__ import annotations

import argparse
import concurrent.futures
import html
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cb = _load_module(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")
base = _load_module(ROOT / "scripts" / "train_rf_boundary_refiner.py", "rf_boundary_refiner_mod")
crf = _load_module(ROOT / "scripts" / "evaluate_causal_rf.py", "evaluate_causal_rf_mod")


MODALITY_SPECS = [
    ("acc", ("ax", "ay", "az")),
    ("gyro", ("gx", "gy", "gz")),
    ("mag", ("mx", "my", "mz")),
    ("acc+gyro", ("ax", "ay", "az", "gx", "gy", "gz")),
    ("acc+mag", ("ax", "ay", "az", "mx", "my", "mz")),
    ("gyro+mag", ("gx", "gy", "gz", "mx", "my", "mz")),
    ("acc+gyro+mag", ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz")),
]
HIGHER_IS_BETTER = {"rep_f1", "precision", "recall", "micro_f1_at_10", "micro_f1_at_25", "micro_f1_at_50"}
LOWER_IS_BETTER = {"start_mae_ms", "end_mae_ms", "transition_mae_ms"}
METRIC_KEYS = [
    "precision",
    "recall",
    "rep_f1",
    "start_mae_ms",
    "end_mae_ms",
    "transition_mae_ms",
    "micro_f1_at_10",
    "micro_f1_at_25",
    "micro_f1_at_50",
    "exact_count_streams",
    "over_segmented_streams",
    "under_segmented_streams",
    "zero_tp_streams",
    "stream_count",
    "n_true",
    "n_pred",
    "tp",
    "fp",
    "fn",
]


def _action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    if len(parts) < 2:
        return "unknown"
    return parts[-2]


def _slugify(value: str) -> str:
    text = str(value).strip().lower()
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"+", "/", " ", "-"}:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out or "value"


def _parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in _parse_csv_list(value)]


def _modality_group_count(name: str) -> int:
    text = str(name).strip()
    if not text:
        return 0
    return text.count("+") + 1


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _compact_summary(summary: dict) -> dict:
    out = dict(summary)
    out.pop("stream_rows", None)
    return out


def _stream_ids_key(streams: Sequence[tuple[str, pd.DataFrame]]) -> tuple[str, ...]:
    return tuple(str(stream_id) for stream_id, _ in streams)


def _subject_of_df(df: pd.DataFrame) -> str:
    if "_split_subject" in df.columns:
        return str(df.iloc[0]["_split_subject"])
    if "subject_id" in df.columns:
        return str(df.iloc[0]["subject_id"])
    return "unknown"


def _diff_metrics(tuned: dict, baseline: dict) -> dict:
    out = {}
    for key in METRIC_KEYS:
        tuned_val = _safe_float(tuned.get(key))
        base_val = _safe_float(baseline.get(key))
        if tuned_val is None or base_val is None:
            out[key] = None
            continue
        if key in HIGHER_IS_BETTER:
            out[key] = tuned_val - base_val
        elif key in LOWER_IS_BETTER:
            out[key] = base_val - tuned_val
        else:
            out[key] = tuned_val - base_val
    return out


def _augment_count_consistency(summary: dict, rows: Sequence[dict]) -> dict:
    out = dict(summary)
    stream_count = max(1, int(out.get("stream_count", 0)))
    exact_count_streams = int(out.get("exact_count_streams", 0))
    abs_count_diffs = [abs(float(row.get("count_diff", 0.0))) for row in rows]
    out["exact_count_ratio"] = float(exact_count_streams) / float(stream_count)
    out["mean_abs_count_diff"] = float(np.mean(abs_count_diffs)) if abs_count_diffs else None
    return out


def _selection_sort_key(summary: dict, selection_metric: str) -> tuple:
    metric_val = _safe_float(summary.get(selection_metric), default=-1e12)
    if selection_metric in LOWER_IS_BETTER:
        metric_score = -metric_val
    else:
        metric_score = metric_val
    rep_f1 = _safe_float(summary.get("rep_f1"), default=-1e12)
    exact_count_ratio = _safe_float(summary.get("exact_count_ratio"), default=-1e12)
    mean_abs_count_diff = _safe_float(summary.get("mean_abs_count_diff"), default=1e12)
    recall = _safe_float(summary.get("recall"), default=-1e12)
    micro50 = _safe_float(summary.get("micro_f1_at_50"), default=-1e12)
    under_segmented = _safe_float(summary.get("under_segmented_streams"), default=1e12)
    precision = _safe_float(summary.get("precision"), default=-1e12)
    transition_mae = _safe_float(summary.get("transition_mae_ms"), default=1e12)
    return (
        metric_score,
        exact_count_ratio,
        -mean_abs_count_diff,
        rep_f1,
        recall,
        micro50,
        -under_segmented,
        precision,
        -transition_mae,
    )


def _available_modalities(
    streams: Sequence[tuple[str, pd.DataFrame]],
    requested: Sequence[str],
    *,
    min_groups: int = 1,
) -> list[tuple[str, tuple[str, ...]]]:
    available_columns = set()
    for _, df in streams:
        available_columns.update(str(col) for col in df.columns)
    requested_set = {name.strip() for name in requested if name.strip()}
    out = []
    for name, columns in MODALITY_SPECS:
        if requested_set and name not in requested_set:
            continue
        if _modality_group_count(name) < max(1, int(min_groups)):
            continue
        if all(col in available_columns for col in columns):
            out.append((name, columns))
    return out


def _find_default_modality_candidate(action_candidates: Sequence[dict], default_imu_columns: Sequence[str]) -> dict | None:
    target = ",".join(str(col) for col in default_imu_columns)
    baseline_rows = [
        row
        for row in action_candidates
        if str(row.get("modality", "")) == "baseline_reference" and str(row.get("imu_columns", "")) == target
    ]
    if baseline_rows:
        return max(baseline_rows, key=lambda row: _selection_sort_key(row, str(row.get("selection_metric", "rep_f1"))))
    default_rows = [row for row in action_candidates if str(row.get("imu_columns", "")) == target]
    if not default_rows:
        return None
    return max(default_rows, key=lambda row: _selection_sort_key(row, str(row.get("selection_metric", "rep_f1"))))


def _normalize_streams(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    test_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
) -> tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, pd.DataFrame]]]:
    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_norm = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_norm = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]
    return train_norm, test_norm


def _truth_reps(df: pd.DataFrame, min_phase_samples: int):
    return base.truth_reps_from_labels(
        df["phase"].to_numpy(),
        actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
        min_phase_samples=max(1, int(min_phase_samples)),
    )


def _quantile_fields(values: Sequence[float], prefix: str) -> dict[str, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    out: dict[str, float | None] = {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": None,
        f"{prefix}_std": None,
        f"{prefix}_min": None,
        f"{prefix}_max": None,
        f"{prefix}_p10": None,
        f"{prefix}_p25": None,
        f"{prefix}_median": None,
        f"{prefix}_p75": None,
        f"{prefix}_p90": None,
    }
    if not arr.size:
        return out
    out[f"{prefix}_mean"] = float(np.mean(arr))
    out[f"{prefix}_std"] = float(np.std(arr))
    out[f"{prefix}_min"] = float(np.min(arr))
    out[f"{prefix}_max"] = float(np.max(arr))
    out[f"{prefix}_p10"] = float(np.quantile(arr, 0.10))
    out[f"{prefix}_p25"] = float(np.quantile(arr, 0.25))
    out[f"{prefix}_median"] = float(np.quantile(arr, 0.50))
    out[f"{prefix}_p75"] = float(np.quantile(arr, 0.75))
    out[f"{prefix}_p90"] = float(np.quantile(arr, 0.90))
    return out


def _selection_mode_name(modality_only_search: bool) -> str:
    return "modality_only" if modality_only_search else "full"


def _apply_fixed_search_candidates(
    duration_stats: dict[str, dict],
    *,
    window_size: int,
    edge_window: int,
) -> dict[str, dict]:
    fixed = {}
    for action, stats in duration_stats.items():
        cur = dict(stats)
        cur["trailing_window_candidates"] = [int(window_size)]
        cur["edge_window_candidates"] = [int(edge_window)]
        fixed[action] = cur
    return fixed


def _clamped_unique(values: Sequence[float], lower: int, upper: int) -> list[int]:
    out = []
    seen = set()
    for value in values:
        cur = int(round(float(value)))
        cur = max(int(lower), min(int(upper), cur))
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
    return sorted(out)


def _apply_duration_prior(reps, min_samples: int, max_samples: int):
    out = []
    for rep in reps:
        duration = int(rep.end_idx) - int(rep.start_idx)
        if int(min_samples) > 0 and duration < int(min_samples):
            continue
        if int(max_samples) > 0 and duration > int(max_samples):
            continue
        out.append(rep)
    return out


def _summarize_train_fold_durations(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    min_phase_samples: int,
    trailing_multipliers: Sequence[float],
    edge_multipliers: Sequence[float],
    trailing_min_samples: int,
    trailing_max_samples: int,
    edge_min_samples: int,
    edge_max_samples: int,
    duration_low_quantile: float,
    duration_high_quantile: float,
) -> dict[str, dict]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    stream_counts = Counter()
    subjects_by_action: dict[str, set[str]] = defaultdict(set)

    for stream_id, df in train_streams:
        action = _action_from_stream_id(stream_id)
        if action == "unknown":
            continue
        stream_counts[action] += 1
        subject = str(df.iloc[0]["_split_subject"]) if "_split_subject" in df.columns else str(df.iloc[0]["subject_id"])
        subjects_by_action[action].add(subject)
        sample_rate_hz = float(base.infer_sample_rate_hz(df))
        truth = _truth_reps(df, min_phase_samples)
        for rep in truth:
            rep_samples = int(rep.end_idx) - int(rep.start_idx)
            concentric_samples = int(rep.transition_idx) - int(rep.start_idx)
            eccentric_samples = int(rep.end_idx) - int(rep.transition_idx)
            if rep_samples <= 0:
                continue
            grouped_rows[action].append(
                {
                    "sample_rate_hz": sample_rate_hz,
                    "rep_duration_samples": rep_samples,
                    "concentric_duration_samples": max(0, concentric_samples),
                    "eccentric_duration_samples": max(0, eccentric_samples),
                    "transition_ratio": float(concentric_samples) / float(rep_samples),
                }
            )

    out: dict[str, dict] = {}
    for action in sorted(stream_counts):
        rows = grouped_rows.get(action, [])
        rep_samples = [row["rep_duration_samples"] for row in rows]
        concentric_samples = [row["concentric_duration_samples"] for row in rows]
        eccentric_samples = [row["eccentric_duration_samples"] for row in rows]
        transition_ratio = [row["transition_ratio"] for row in rows]
        sample_rates = [row["sample_rate_hz"] for row in rows]
        rate = float(np.median(sample_rates)) if sample_rates else 100.0
        stats = {
            "action": action,
            "train_stream_count": int(stream_counts[action]),
            "train_subject_count": int(len(subjects_by_action[action])),
            "rep_count": int(len(rows)),
            "sample_rate_hz": rate,
        }
        stats.update(_quantile_fields(rep_samples, "rep_duration_samples"))
        stats.update(_quantile_fields([float(v) / max(rate, 1e-9) for v in rep_samples], "rep_duration_seconds"))
        stats.update(_quantile_fields(concentric_samples, "concentric_duration_samples"))
        stats.update(_quantile_fields([float(v) / max(rate, 1e-9) for v in concentric_samples], "concentric_duration_seconds"))
        stats.update(_quantile_fields(eccentric_samples, "eccentric_duration_samples"))
        stats.update(_quantile_fields([float(v) / max(rate, 1e-9) for v in eccentric_samples], "eccentric_duration_seconds"))
        stats.update(_quantile_fields(transition_ratio, "transition_ratio"))

        median_rep = _safe_float(stats.get("rep_duration_samples_median"), default=float(trailing_min_samples))
        rep_low = _safe_float(np.quantile(rep_samples, duration_low_quantile) if rep_samples else None, default=float(trailing_min_samples))
        rep_high = _safe_float(np.quantile(rep_samples, duration_high_quantile) if rep_samples else None, default=float(trailing_max_samples))
        min_prior = max(1, int(round(rep_low)))
        max_prior = max(min_prior + 1, int(round(rep_high)))
        trailing_candidates = _clamped_unique(
            [median_rep * float(mult) for mult in trailing_multipliers],
            trailing_min_samples,
            trailing_max_samples,
        )
        edge_candidates = _clamped_unique(
            [median_rep * float(mult) for mult in edge_multipliers],
            edge_min_samples,
            edge_max_samples,
        )
        if not trailing_candidates:
            trailing_candidates = [int(trailing_min_samples)]
        if not edge_candidates:
            edge_candidates = [int(edge_min_samples)]
        stats["duration_low_quantile"] = float(duration_low_quantile)
        stats["duration_high_quantile"] = float(duration_high_quantile)
        stats["min_rep_duration_samples"] = int(min_prior)
        stats["max_rep_duration_samples"] = int(max_prior)
        stats["min_rep_duration_seconds"] = float(min_prior) / max(rate, 1e-9)
        stats["max_rep_duration_seconds"] = float(max_prior) / max(rate, 1e-9)
        stats["trailing_window_candidates"] = [int(v) for v in trailing_candidates]
        stats["edge_window_candidates"] = [int(v) for v in edge_candidates]
        out[action] = stats
    return out


def _duration_stats_rows(duration_stats: dict[str, dict]) -> list[dict]:
    rows = []
    for action, stats in sorted(duration_stats.items()):
        row = dict(stats)
        row["trailing_window_candidates"] = ",".join(str(v) for v in stats.get("trailing_window_candidates", []))
        row["edge_window_candidates"] = ",".join(str(v) for v in stats.get("edge_window_candidates", []))
        rows.append(row)
    return rows


def _write_duration_report(
    path: Path,
    outer_subject: str,
    train_subjects: Sequence[str],
    duration_stats: dict[str, dict],
    *,
    selection_mode: str,
    fixed_window_size: int | None = None,
    fixed_edge_window: int | None = None,
) -> None:
    lines = [
        f"# Train-Fold Duration Stats ({outer_subject})",
        "",
        "## Protocol",
        "",
        f"- Held-out outer subject: `{outer_subject}`",
        f"- Train subjects only: `{', '.join(train_subjects)}`",
        "- Candidate windows are generated from train-fold median rep duration only.",
        "- Duration priors use train-fold rep duration quantiles only.",
        "",
        "## Per-Action Summary",
        "",
        "| Action | Train Streams | Reps | Median Rep (samples) | Rep p10-p90 (samples) | Transition Ratio Median | Trailing Candidates | Edge Candidates | Duration Prior (samples) |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    if str(selection_mode) == "modality_only":
        lines[6] = (
            f"- Candidate windows are fixed during selection: trailing=`{int(fixed_window_size or 0)}`, "
            f"edge=`{int(fixed_edge_window or 0)}`."
        )
    for action, stats in sorted(duration_stats.items()):
        lines.append(
            "| "
            f"`{action}` | {int(stats.get('train_stream_count', 0))} | {int(stats.get('rep_count', 0))} | "
            f"{int(round(_safe_float(stats.get('rep_duration_samples_median'), 0.0) or 0.0))} | "
            f"{int(round(_safe_float(stats.get('rep_duration_samples_p10'), 0.0) or 0.0))}-"
            f"{int(round(_safe_float(stats.get('rep_duration_samples_p90'), 0.0) or 0.0))} | "
            f"{(_safe_float(stats.get('transition_ratio_median'), 0.0) or 0.0):.3f} | "
            f"`{', '.join(str(v) for v in stats.get('trailing_window_candidates', []))}` | "
            f"`{', '.join(str(v) for v in stats.get('edge_window_candidates', []))}` | "
            f"{int(stats.get('min_rep_duration_samples', 0))}-"
            f"{int(stats.get('max_rep_duration_samples', 0))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coverage_rows(streams: Sequence[tuple[str, pd.DataFrame]]) -> list[dict]:
    counter = Counter()
    for stream_id, df in streams:
        action = _action_from_stream_id(stream_id)
        subject = str(df.iloc[0]["_split_subject"]) if "_split_subject" in df.columns else str(df.iloc[0]["subject_id"])
        counter[(subject, action)] += 1
    return [
        {"subject": subject, "action": action, "stream_count": count}
        for (subject, action), count in sorted(counter.items())
    ]


def _render_summary_html(path: Path, title: str, metric_rows: list[dict], extra_sections: list[str]) -> None:
    trs = []
    for row in metric_rows:
        link = html.escape(str(row.get("link", "")))
        label = html.escape(str(row.get("label", "")))
        value_cells = []
        for key in ["rep_f1", "precision", "recall", "micro_f1_at_50", "start_mae_ms", "end_mae_ms", "transition_mae_ms"]:
            val = _safe_float(row.get(key))
            if val is None:
                value_cells.append("<td></td>")
            elif "mae" in key:
                value_cells.append(f"<td>{val:.1f}</td>")
            else:
                value_cells.append(f"<td>{val:.4f}</td>")
        label_html = f'<a href="{link}">{label}</a>' if link else label
        trs.append(f"<tr><td>{label_html}</td>{''.join(value_cells)}</tr>")
    html_text = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:24px;line-height:1.4}} table{{border-collapse:collapse;width:100%;margin:16px 0}} td,th{{border:1px solid #ddd;padding:6px 8px;text-align:left}} th{{background:#f3f4f6}} code{{background:#f4f4f5;padding:1px 4px;border-radius:4px}} .section{{margin-top:24px}}</style>
</head><body><h1>{html.escape(title)}</h1><table><thead><tr><th>Item</th><th>Rep F1</th><th>Precision</th><th>Recall</th><th>micro_f1@50</th><th>Start MAE</th><th>End MAE</th><th>Transition MAE</th></tr></thead><tbody>{''.join(trs)}</tbody></table>{''.join(extra_sections)}</body></html>"""
    path.write_text(html_text, encoding="utf-8")


def _predict_prob_cache(clf, streams: Sequence[tuple[str, pd.DataFrame]], imu_columns: Sequence[str], window_size: int) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for stream_id, df in streams:
        cache[str(stream_id)] = crf.predict_causal_rf(clf, df, imu_columns, window_size=int(window_size), stride=1)
    return cache


def _build_eval_stream_cache(
    streams: Sequence[tuple[str, pd.DataFrame]],
    prob_cache: dict[str, np.ndarray],
    mm_cfg,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for stream_id, df in streams:
        stream_key = str(stream_id)
        probs = prob_cache[stream_key]
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        out[stream_key] = {
            "df": df,
            "probs": probs,
            "coarse_reps": list(coarse_reps),
            "truth": _truth_reps(df, 1),
            "sample_rate_hz": float(base.infer_sample_rate_hz(df)),
        }
    return out


def _collect_matched_examples(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    train_prob_cache: dict[str, np.ndarray],
    mm_cfg,
    match_iou_train: float,
    max_shift: int,
    target_matched_reps: int,
    max_refiner_train_streams: int,
    max_refiner_train_streams_per_subject: int,
    max_matched_reps_per_stream: int,
    max_matched_reps_per_subject: int,
) -> list[dict]:
    matched_examples: list[dict] = []
    train_subset = []
    stream_count_by_subject: Counter[str] = Counter()
    for stream_id, df in train_streams:
        subject = _subject_of_df(df)
        if int(max_refiner_train_streams_per_subject) > 0 and stream_count_by_subject[subject] >= int(max_refiner_train_streams_per_subject):
            continue
        train_subset.append((stream_id, df))
        stream_count_by_subject[subject] += 1
        if int(max_refiner_train_streams) > 0 and len(train_subset) >= int(max_refiner_train_streams):
            break

    matched_by_subject: Counter[str] = Counter()
    for stream_id, df in train_subset:
        probs = train_prob_cache[str(stream_id)]
        coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
        truth = _truth_reps(df, int(mm_cfg.min_phase_samples))
        matches = base.match_segments(
            [(r.start_idx, r.end_idx) for r in coarse_reps],
            [(r.start_idx, r.end_idx) for r in truth],
            iou_threshold=float(match_iou_train),
        )
        subject = _subject_of_df(df)
        matched_this_stream = 0
        for pred_idx, true_idx, _ in matches:
            if int(max_matched_reps_per_stream) > 0 and matched_this_stream >= int(max_matched_reps_per_stream):
                break
            if int(max_matched_reps_per_subject) > 0 and matched_by_subject[subject] >= int(max_matched_reps_per_subject):
                break
            pred_rep = coarse_reps[pred_idx]
            true_rep = truth[true_idx]
            matched_examples.append(
                {
                    "df": df,
                    "subject": subject,
                    "probs": probs,
                    "pred_rep": pred_rep,
                    "true_rep": true_rep,
                    "start_shift": int(np.clip(true_rep.start_idx - pred_rep.start_idx, -max_shift, max_shift)),
                    "transition_shift": int(np.clip(true_rep.transition_idx - pred_rep.transition_idx, -max_shift, max_shift)),
                    "end_shift": int(np.clip(true_rep.end_idx - pred_rep.end_idx, -max_shift, max_shift)),
                }
            )
            matched_this_stream += 1
            matched_by_subject[subject] += 1
        if int(target_matched_reps) > 0 and len(matched_examples) >= int(target_matched_reps):
            break
    return matched_examples


def _fit_refiner_regressor(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x, y)
    return model


def _fit_refiner_from_examples(
    matched_examples: Sequence[dict],
    imu_columns: Sequence[str],
    edge_window: int,
    max_shift: int,
    feature_cache: dict | None = None,
    min_examples: int = 1,
    n_estimators: int = 300,
    max_depth: int = 18,
    min_samples_leaf: int = 2,
):
    if not matched_examples or len(matched_examples) < int(max(1, min_examples)):
        return None
    cache_key = None
    if feature_cache is not None:
        cache_key = (int(edge_window), tuple(str(col) for col in imu_columns))
    cached = feature_cache.get(cache_key) if feature_cache is not None and cache_key in feature_cache else None
    if cached is None:
        start_rows = []
        trans_rows = []
        end_rows = []
        y_start = []
        y_trans = []
        y_end = []
        for example in matched_examples:
            df = example["df"]
            probs = example["probs"]
            pred_rep = example["pred_rep"]
            start_rows.append(base._build_edge_features(df, probs, pred_rep, "start", int(edge_window), imu_columns))
            trans_rows.append(base._build_edge_features(df, probs, pred_rep, "transition", int(edge_window), imu_columns))
            end_rows.append(base._build_edge_features(df, probs, pred_rep, "end", int(edge_window), imu_columns))
            y_start.append(float(example["start_shift"]))
            y_trans.append(float(example["transition_shift"]))
            y_end.append(float(example["end_shift"]))
        x_start, feature_keys = base._rows_to_matrix(start_rows)
        x_trans, _ = base._rows_to_matrix(trans_rows)
        x_end, _ = base._rows_to_matrix(end_rows)
        cached = {
            "x_start": x_start,
            "x_trans": x_trans,
            "x_end": x_end,
            "y_start": np.asarray(y_start, dtype=np.float32),
            "y_trans": np.asarray(y_trans, dtype=np.float32),
            "y_end": np.asarray(y_end, dtype=np.float32),
            "feature_keys": feature_keys,
        }
        if feature_cache is not None and cache_key is not None:
            feature_cache[cache_key] = cached
    return {
        "start": _fit_refiner_regressor(cached["x_start"], cached["y_start"], n_estimators, max_depth, min_samples_leaf),
        "transition": _fit_refiner_regressor(cached["x_trans"], cached["y_trans"], n_estimators, max_depth, min_samples_leaf),
        "end": _fit_refiner_regressor(cached["x_end"], cached["y_end"], n_estimators, max_depth, min_samples_leaf),
        "feature_keys": cached["feature_keys"],
        "matched_rep_count": int(len(matched_examples)),
        "edge_window": int(edge_window),
        "max_shift": int(max_shift),
    }


def _evaluate_prob_cache(
    *,
    action: str,
    streams: Sequence[tuple[str, pd.DataFrame]],
    prob_cache: dict[str, np.ndarray],
    imu_columns: Sequence[str],
    modality_name: str,
    mm_cfg,
    window_size: int,
    edge_window: int,
    min_rep_duration_samples: int,
    max_rep_duration_samples: int,
    max_shift: int,
    refiner,
    stream_eval_cache: dict[str, dict] | None,
    output_root: Path | None,
    output_tag: str,
    outer_subject: str,
) -> list[dict]:
    rows = []
    for stream_id, df in streams:
        stream_key = str(stream_id)
        cached = stream_eval_cache.get(stream_key) if stream_eval_cache is not None else None
        if cached is None:
            probs = prob_cache[stream_key]
            coarse_reps, _ = base._coarse_predict_reps(df, probs, mm_cfg)
            truth = _truth_reps(df, 1)
            sample_rate = float(base.infer_sample_rate_hz(df))
        else:
            probs = cached["probs"]
            coarse_reps = list(cached["coarse_reps"])
            truth = cached["truth"]
            sample_rate = float(cached["sample_rate_hz"])
        coarse_reps = _apply_duration_prior(coarse_reps, min_rep_duration_samples, max_rep_duration_samples)
        if refiner is not None:
            pred_reps = base._refine_reps(
                df,
                probs,
                coarse_reps,
                refiner,
                imu_columns,
                edge_window=int(edge_window),
                max_shift=int(max_shift),
            )
        else:
            pred_reps = list(coarse_reps)
        pred_reps = _apply_duration_prior(pred_reps, min_rep_duration_samples, max_rep_duration_samples)
        metrics = base.rep_metrics(pred_reps, truth, sample_rate)
        pred_labels = base._phase_labels_from_reps(len(df), pred_reps)
        gt_labels = cb.micro_labels_from_phase(df["phase"].to_numpy())
        gt_runs = base.labels_to_runs(gt_labels, positive_labels=(base.CONCENTRIC_LABEL, base.ECCENTRIC_LABEL), min_length=1)
        pred_runs = base.labels_to_runs(pred_labels, positive_labels=(base.CONCENTRIC_LABEL, base.ECCENTRIC_LABEL), min_length=1)
        seg_metrics = base.segment_iou_f1(gt_runs, pred_runs)
        metrics.update({
            "micro_f1_at_10": seg_metrics["f1_at_10"],
            "micro_f1_at_25": seg_metrics["f1_at_25"],
            "micro_f1_at_50": seg_metrics["f1_at_50"],
        })
        row = {
            **metrics,
            "rep_f1": float(metrics.get("f1", 0.0)),
            "stream_id": stream_id,
            "action": action,
            "outer_test_subject": outer_subject,
            "modality": modality_name,
            "imu_columns": ",".join(str(col) for col in imu_columns),
            "window_size": int(window_size),
            "edge_window": int(edge_window),
            "min_rep_duration_samples": int(min_rep_duration_samples),
            "max_rep_duration_samples": int(max_rep_duration_samples),
            "used_refiner": bool(refiner is not None),
            "count_diff": float(metrics.get("n_pred", 0.0) - metrics.get("n_true", 0.0)),
        }
        if output_root is not None:
            rel_parts = [p for p in str(stream_id).split("/") if p]
            svg_path = output_root.joinpath("stream_replays", output_tag, *rel_parts).with_suffix(".svg")
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            detections = [
                base.SegmentDetection(
                    start_idx=int(rep.start_idx),
                    end_idx=int(rep.end_idx),
                    cost=0.0,
                    feature="rf_per_action_benchmark",
                    action_type=action,
                    template_id=f"{action}_{_slugify(modality_name)}_{window_size}_{edge_window}",
                    exemplar_source=stream_id,
                    normalized_cost=0.0,
                )
                for rep in pred_reps
            ]
            base._write_segmentation_svg(
                svg_path,
                stream_id,
                df,
                [(r.start_idx, r.end_idx) for r in truth],
                detections,
                {
                    "f1": float(metrics.get("f1", 0.0)),
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "n_true": float(metrics.get("n_true", 0.0)),
                    "n_pred": float(metrics.get("n_pred", 0.0)),
                },
                sample_rate,
            )
            row["svg_rel"] = svg_path.relative_to(output_root).as_posix()
        rows.append(row)
    return rows


def _evaluate_action_run(
    *,
    action: str,
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    test_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    modality_name: str,
    mm_cfg,
    window_size: int,
    edge_window: int,
    min_rep_duration_samples: int,
    max_rep_duration_samples: int,
    train_stride: int,
    match_iou_train: float,
    max_shift: int,
    n_estimators: int,
    max_depth: int,
    max_samples: float,
    target_matched_reps: int,
    max_refiner_train_streams: int,
    max_refiner_train_streams_per_subject: int,
    max_matched_reps_per_stream: int,
    max_matched_reps_per_subject: int,
    min_matched_reps_for_refiner: int,
    refiner_n_estimators: int,
    refiner_max_depth: int,
    refiner_min_samples_leaf: int,
    output_root: Path | None,
    output_tag: str,
    outer_subject: str,
) -> dict:
    if not train_streams or not test_streams:
        return {
            "summary": {"action": action, "modality": modality_name, "stream_count": 0},
            "rows": [],
            "rf_train_time_s": None,
            "refiner_train_time_s": None,
            "used_refiner": False,
        }

    train_norm, test_norm = _normalize_streams(train_streams, test_streams, imu_columns)

    t0 = time.time()
    clf = crf.train_causal_rf(
        train_norm,
        imu_columns,
        window_size=int(window_size),
        stride=int(train_stride),
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_samples=float(max_samples),
    )
    rf_train_time = time.time() - t0

    train_prob_cache = _predict_prob_cache(clf, train_norm, imu_columns, int(window_size))
    test_prob_cache = _predict_prob_cache(clf, test_norm, imu_columns, int(window_size))
    test_eval_cache = _build_eval_stream_cache(test_norm, test_prob_cache, mm_cfg)

    t0 = time.time()
    matched_examples = _collect_matched_examples(
        train_norm,
        train_prob_cache,
        mm_cfg,
        float(match_iou_train),
        int(max_shift),
        int(target_matched_reps),
        int(max_refiner_train_streams),
        int(max_refiner_train_streams_per_subject),
        int(max_matched_reps_per_stream),
        int(max_matched_reps_per_subject),
    )
    refiner = _fit_refiner_from_examples(
        matched_examples,
        imu_columns,
        int(edge_window),
        int(max_shift),
        feature_cache={},
        min_examples=int(min_matched_reps_for_refiner),
        n_estimators=int(refiner_n_estimators),
        max_depth=int(refiner_max_depth),
        min_samples_leaf=int(refiner_min_samples_leaf),
    )
    used_refiner = refiner is not None
    refiner_train_time = time.time() - t0

    rows = _evaluate_prob_cache(
        action=action,
        streams=test_norm,
        prob_cache=test_prob_cache,
        imu_columns=imu_columns,
        modality_name=modality_name,
        mm_cfg=mm_cfg,
        window_size=int(window_size),
        edge_window=int(edge_window),
        min_rep_duration_samples=int(min_rep_duration_samples),
        max_rep_duration_samples=int(max_rep_duration_samples),
        max_shift=int(max_shift),
        refiner=refiner,
        stream_eval_cache=test_eval_cache,
        output_root=output_root,
        output_tag=output_tag,
        outer_subject=outer_subject,
    )

    summary = _compact_summary(base._aggregate_rows(rows))
    summary.update(
        {
            "action": action,
            "modality": modality_name,
            "imu_columns": list(imu_columns),
            "window_size": int(window_size),
            "edge_window": int(edge_window),
            "min_rep_duration_samples": int(min_rep_duration_samples),
            "max_rep_duration_samples": int(max_rep_duration_samples),
            "rf_train_time_s": float(rf_train_time),
            "refiner_train_time_s": float(refiner_train_time),
            "used_refiner": bool(used_refiner),
        }
    )
    return {
        "summary": summary,
        "rows": rows,
        "rf_train_time_s": float(rf_train_time),
        "refiner_train_time_s": float(refiner_train_time),
        "used_refiner": bool(used_refiner),
    }


def _write_fold_outputs(
    fold_dir: Path,
    duration_stats: dict[str, dict],
    selection_rows: list[dict],
    best_rows: list[dict],
    tuned_rows: list[dict],
    tuned_summary: dict,
    baseline_rows: list[dict],
    baseline_summary: dict,
    fold_results: dict,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_duration_stats_rows(duration_stats)).to_csv(fold_dir / "duration_stats.csv", index=False)
    (fold_dir / "duration_stats.json").write_text(json.dumps(duration_stats, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(selection_rows).to_csv(fold_dir / "selection_summary.csv", index=False)
    pd.DataFrame(best_rows).to_csv(fold_dir / "best_config_per_action.csv", index=False)
    pd.DataFrame(tuned_rows).to_csv(fold_dir / "stream_metrics.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(fold_dir / "baseline_stream_metrics.csv", index=False)
    (fold_dir / "results.json").write_text(json.dumps(fold_results, indent=2, default=str), encoding="utf-8")
    base._write_index_html(fold_dir / "index.html", tuned_rows)
    base._write_index_html(fold_dir / "baseline_index.html", baseline_rows)

    summary_rows = [
        {
            "label": "tuned_outer_eval",
            "link": "index.html",
            **tuned_summary,
        },
        {
            "label": "baseline_outer_eval",
            "link": "baseline_index.html",
            **baseline_summary,
        },
    ]
    best_table = [
        "<div class=\"section\"><h2>Best Config Per Action</h2><table><thead><tr><th>Action</th><th>Modality</th><th>Window</th><th>Edge</th><th>Duration Prior</th><th>Inner Score</th></tr></thead><tbody>"
    ]
    for row in best_rows:
        best_table.append(
            "<tr>"
            f"<td><code>{html.escape(str(row.get('action', '')))}</code></td>"
            f"<td><code>{html.escape(str(row.get('modality', '')))}</code></td>"
            f"<td>{int(row.get('window_size', 0))}</td>"
            f"<td>{int(row.get('edge_window', 0))}</td>"
            f"<td>{int(row.get('min_rep_duration_samples', 0))}-{int(row.get('max_rep_duration_samples', 0))}</td>"
            f"<td>{(_safe_float(row.get('selection_metric_value'), 0.0) or 0.0):.4f}</td>"
            "</tr>"
        )
    best_table.append("</tbody></table></div>")
    _render_summary_html(fold_dir / "summary.html", f"Outer Fold: {fold_dir.name}", summary_rows, best_table)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _compute_stability_rows(all_best_rows: list[dict]) -> list[dict]:
    stability_rows = []
    grouped_best: dict[str, list[dict]] = defaultdict(list)
    for row in all_best_rows:
        grouped_best[str(row.get("action", "unknown"))].append(row)
    for action, rows in sorted(grouped_best.items()):
        winners = Counter(
            (
                str(row.get("modality", "")),
                int(row.get("window_size", 0)),
                int(row.get("edge_window", 0)),
                int(row.get("min_rep_duration_samples", 0)),
                int(row.get("max_rep_duration_samples", 0)),
            )
            for row in rows
        )
        if not winners:
            continue
        (modality, window_size, edge_window, min_prior, max_prior), win_count = winners.most_common(1)[0]
        stability_rows.append(
            {
                "action": action,
                "top_modality": modality,
                "top_window_size": int(window_size),
                "top_edge_window": int(edge_window),
                "top_min_rep_duration_samples": int(min_prior),
                "top_max_rep_duration_samples": int(max_prior),
                "win_count": int(win_count),
                "outer_fold_count": int(len(rows)),
                "win_ratio": float(win_count) / max(1.0, float(len(rows))),
                "distinct_winner_count": int(len(winners)),
            }
        )
    return stability_rows


def _write_root_outputs(
    *,
    out_dir: Path,
    config_path: str,
    outer_subjects: Sequence[str],
    available_modalities: Sequence[tuple[str, Sequence[str]]],
    default_imu_columns: Sequence[str],
    baseline_window_size: int,
    baseline_edge_window: int,
    selection_metric: str,
    selection_mode: str,
    selection_window_size: int,
    selection_edge_window: int,
    min_modality_groups: int,
    default_modality_guardrail: bool,
    default_modality_min_improvement: float,
    default_modality_max_recall_drop: float,
    default_modality_max_exact_count_ratio_drop: float,
    default_modality_max_mean_abs_count_diff_increase: float,
    all_outer_rows: list[dict],
    all_baseline_rows: list[dict],
    all_selection_rows: list[dict],
    all_best_rows: list[dict],
    fold_summary_rows: list[dict],
    fold_results_lookup: dict,
) -> None:
    overall_tuned = _compact_summary(base._aggregate_rows(all_outer_rows))
    overall_baseline = _compact_summary(base._aggregate_rows(all_baseline_rows))
    overall_delta = _diff_metrics(overall_tuned, overall_baseline)
    stability_rows = _compute_stability_rows(all_best_rows)

    results = {
        "benchmark": "per_action_rf_nested_cv",
        "config": str(config_path),
        "outer_subjects": list(outer_subjects),
        "selection_metric": str(selection_metric),
        "selection_mode": str(selection_mode),
        "selection_config": {
            "window_size": int(selection_window_size),
            "edge_window": int(selection_edge_window),
            "min_modality_groups": int(min_modality_groups),
            "default_modality_guardrail": bool(default_modality_guardrail),
            "default_modality_min_improvement": float(default_modality_min_improvement),
            "default_modality_max_recall_drop": float(default_modality_max_recall_drop),
            "default_modality_max_exact_count_ratio_drop": float(default_modality_max_exact_count_ratio_drop),
            "default_modality_max_mean_abs_count_diff_increase": float(default_modality_max_mean_abs_count_diff_increase),
        },
        "modalities": [name for name, _ in available_modalities],
        "baseline_config": {
            "imu_columns": list(default_imu_columns),
            "window_size": int(baseline_window_size),
            "edge_window": int(baseline_edge_window),
        },
        "tuned_overall": overall_tuned,
        "baseline_overall": overall_baseline,
        "delta_vs_baseline": overall_delta,
        "folds": fold_results_lookup,
        "action_stability": stability_rows,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(all_outer_rows).to_csv(out_dir / "stream_metrics.csv", index=False)
    pd.DataFrame(all_baseline_rows).to_csv(out_dir / "baseline_stream_metrics.csv", index=False)
    pd.DataFrame(all_selection_rows).to_csv(out_dir / "selection_summary.csv", index=False)
    pd.DataFrame(all_best_rows).to_csv(out_dir / "best_config_per_action.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(out_dir / "action_stability.csv", index=False)
    pd.DataFrame(fold_summary_rows).to_csv(out_dir / "outer_fold_summaries.csv", index=False)
    base._write_index_html(out_dir / "index.html", all_outer_rows)
    base._write_index_html(out_dir / "baseline_index.html", all_baseline_rows)

    extra_sections = [
        "<div class=\"section\"><h2>Fold Links</h2><ul>"
        + "".join(
            f'<li><a href="{html.escape(subject)}/summary.html"><code>{html.escape(subject)}</code></a></li>'
            for subject in outer_subjects
            if (out_dir / subject / "summary.html").exists()
        )
        + "</ul></div>",
        "<div class=\"section\"><h2>Action Stability</h2><table><thead><tr><th>Action</th><th>Top Modality</th><th>Window</th><th>Edge</th><th>Win Ratio</th><th>Distinct Winners</th></tr></thead><tbody>"
        + "".join(
            "<tr>"
            f"<td><code>{html.escape(str(row.get('action', '')))}</code></td>"
            f"<td><code>{html.escape(str(row.get('top_modality', '')))}</code></td>"
            f"<td>{int(row.get('top_window_size', 0))}</td>"
            f"<td>{int(row.get('top_edge_window', 0))}</td>"
            f"<td>{(_safe_float(row.get('win_ratio'), 0.0) or 0.0):.3f}</td>"
            f"<td>{int(row.get('distinct_winner_count', 0))}</td>"
            "</tr>"
            for row in stability_rows
        )
        + "</tbody></table></div>",
    ]
    _render_summary_html(
        out_dir / "summary.html",
        "Per-Action RF Nested Benchmark",
        [
            {"label": "tuned_nested_cv", "link": "index.html", **overall_tuned},
            {"label": "baseline_default", "link": "baseline_index.html", **overall_baseline},
        ],
        extra_sections,
    )


def _args_to_cli_list(args: argparse.Namespace, overrides: dict[str, object] | None = None, skip_keys: set[str] | None = None) -> list[str]:
    overrides = overrides or {}
    skip_keys = skip_keys or set()
    cli: list[str] = []
    for key, value in vars(args).items():
        if key in skip_keys or key in overrides:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        cli.extend([flag, str(value)])
    return cli


def _copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _run_subprocess_command(command: list[str], workdir: Path) -> None:
    subprocess.run(command, cwd=str(workdir), check=True)


def _run_parallel_outer_workers(
    *,
    script_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
    outer_subjects: Sequence[str],
    available_modalities: Sequence[tuple[str, Sequence[str]]],
    default_imu_columns: Sequence[str],
    selection_mode: str,
    selection_window_size: int,
    selection_edge_window: int,
    min_modality_groups: int,
    default_modality_guardrail: bool,
    default_modality_min_improvement: float,
    default_modality_max_recall_drop: float,
    default_modality_max_exact_count_ratio_drop: float,
    default_modality_max_mean_abs_count_diff_increase: float,
) -> None:
    worker_base = out_dir / "_outer_workers"
    worker_base.mkdir(parents=True, exist_ok=True)
    commands = []
    worker_outputs = []
    for subject in outer_subjects:
        worker_out = worker_base / subject
        commands.append(
            [
                sys.executable,
                str(script_path),
                *_args_to_cli_list(
                    args,
                    overrides={
                        "output": worker_out,
                        "outer_subjects": subject,
                        "parallel_outer_jobs": 1,
                        "parallel_action_jobs": 1,
                        "parallel_worker": True,
                    },
                ),
            ]
        )
        worker_outputs.append((subject, worker_out))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallel_outer_jobs))) as executor:
        futures = [executor.submit(_run_subprocess_command, cmd, ROOT) for cmd in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    all_outer_rows: list[dict] = []
    all_baseline_rows: list[dict] = []
    all_selection_rows: list[dict] = []
    all_best_rows: list[dict] = []
    fold_summary_rows: list[dict] = []
    fold_results_lookup: dict = {}
    coverage_written = False
    for subject, worker_out in worker_outputs:
        worker_results = _load_json(worker_out / "results.json")
        fold_results_lookup[subject] = worker_results["folds"][subject]
        all_outer_rows.extend(_load_csv_rows(worker_out / "stream_metrics.csv"))
        all_baseline_rows.extend(_load_csv_rows(worker_out / "baseline_stream_metrics.csv"))
        all_selection_rows.extend(_load_csv_rows(worker_out / "selection_summary.csv"))
        all_best_rows.extend(_load_csv_rows(worker_out / "best_config_per_action.csv"))
        fold_summary_rows.extend(_load_csv_rows(worker_out / "outer_fold_summaries.csv"))
        if not coverage_written and (worker_out / "subject_action_coverage.csv").exists():
            shutil.copy2(worker_out / "subject_action_coverage.csv", out_dir / "subject_action_coverage.csv")
            coverage_written = True
        if (worker_out / subject).exists():
            shutil.copytree(worker_out / subject, out_dir / subject, dirs_exist_ok=True)

    _write_root_outputs(
        out_dir=out_dir,
        config_path=args.config,
        outer_subjects=outer_subjects,
        available_modalities=available_modalities,
        default_imu_columns=default_imu_columns,
        baseline_window_size=int(args.baseline_window_size),
        baseline_edge_window=int(args.baseline_edge_window),
        selection_metric=args.selection_metric,
        selection_mode=selection_mode,
        selection_window_size=int(selection_window_size),
        selection_edge_window=int(selection_edge_window),
        min_modality_groups=int(min_modality_groups),
        default_modality_guardrail=bool(default_modality_guardrail),
        default_modality_min_improvement=float(default_modality_min_improvement),
        default_modality_max_recall_drop=float(default_modality_max_recall_drop),
        default_modality_max_exact_count_ratio_drop=float(default_modality_max_exact_count_ratio_drop),
        default_modality_max_mean_abs_count_diff_increase=float(default_modality_max_mean_abs_count_diff_increase),
        all_outer_rows=all_outer_rows,
        all_baseline_rows=all_baseline_rows,
        all_selection_rows=all_selection_rows,
        all_best_rows=all_best_rows,
        fold_summary_rows=fold_summary_rows,
        fold_results_lookup=fold_results_lookup,
    )


def _merge_action_worker_outputs(
    *,
    worker_outputs: Sequence[tuple[str, Path]],
    final_out_dir: Path,
    outer_subject: str,
    available_modalities: Sequence[tuple[str, Sequence[str]]],
    default_imu_columns: Sequence[str],
    baseline_window_size: int,
    baseline_edge_window: int,
    selection_metric: str,
    config_path: str,
    selection_mode: str,
    selection_window_size: int,
    selection_edge_window: int,
    min_modality_groups: int,
    default_modality_guardrail: bool,
    default_modality_min_improvement: float,
    default_modality_max_recall_drop: float,
    default_modality_max_exact_count_ratio_drop: float,
    default_modality_max_mean_abs_count_diff_increase: float,
) -> None:
    all_outer_rows: list[dict] = []
    all_baseline_rows: list[dict] = []
    all_selection_rows: list[dict] = []
    all_best_rows: list[dict] = []
    fold_summary_rows: list[dict] = []
    fold_results_lookup: dict = {}
    duration_stats = None
    train_subjects = []
    available_modalities_fold = [name for name, _ in available_modalities]
    for _action, worker_out in worker_outputs:
        worker_results = _load_json(worker_out / "results.json")
        fold = worker_results["folds"][outer_subject]
        duration_stats = duration_stats or fold.get("duration_stats")
        train_subjects = train_subjects or fold.get("train_subjects", [])
        all_outer_rows.extend(_load_csv_rows(worker_out / "stream_metrics.csv"))
        all_baseline_rows.extend(_load_csv_rows(worker_out / "baseline_stream_metrics.csv"))
        all_selection_rows.extend(_load_csv_rows(worker_out / "selection_summary.csv"))
        all_best_rows.extend(_load_csv_rows(worker_out / "best_config_per_action.csv"))
        fold_dir = worker_out / outer_subject
        if (fold_dir / "stream_replays").exists():
            _copy_tree_contents(fold_dir / "stream_replays", final_out_dir / outer_subject / "stream_replays")
        if duration_stats is None and (fold_dir / "duration_stats.json").exists():
            duration_stats = _load_json(fold_dir / "duration_stats.json")
        if not (final_out_dir / "subject_action_coverage.csv").exists() and (worker_out / "subject_action_coverage.csv").exists():
            shutil.copy2(worker_out / "subject_action_coverage.csv", final_out_dir / "subject_action_coverage.csv")

    tuned_summary = _compact_summary(base._aggregate_rows(all_outer_rows))
    baseline_summary = _compact_summary(base._aggregate_rows(all_baseline_rows))
    delta_summary = _diff_metrics(tuned_summary, baseline_summary)
    best_map = {str(row["action"]): dict(row) for row in all_best_rows}
    fold_results_lookup[outer_subject] = {
        "outer_test_subject": outer_subject,
        "train_subjects": train_subjects,
        "selection_metric": selection_metric,
        "available_modalities": available_modalities_fold,
        "duration_stats": duration_stats or {},
        "best_config_per_action": best_map,
        "tuned_overall": tuned_summary,
        "baseline_overall": baseline_summary,
        "delta_vs_baseline": delta_summary,
    }
    fold_summary_rows.append(
        {
            "outer_test_subject": outer_subject,
            **{f"tuned_{k}": tuned_summary.get(k) for k in METRIC_KEYS},
            **{f"baseline_{k}": baseline_summary.get(k) for k in METRIC_KEYS},
            **{f"delta_{k}": delta_summary.get(k) for k in METRIC_KEYS},
        }
    )
    _write_fold_outputs(
        fold_dir=final_out_dir / outer_subject,
        duration_stats=duration_stats or {},
        selection_rows=all_selection_rows,
        best_rows=all_best_rows,
        tuned_rows=all_outer_rows,
        tuned_summary=tuned_summary,
        baseline_rows=all_baseline_rows,
        baseline_summary=baseline_summary,
        fold_results=fold_results_lookup[outer_subject],
    )
    _write_root_outputs(
        out_dir=final_out_dir,
        config_path=config_path,
        outer_subjects=[outer_subject],
        available_modalities=available_modalities,
        default_imu_columns=default_imu_columns,
        baseline_window_size=int(baseline_window_size),
        baseline_edge_window=int(baseline_edge_window),
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        selection_window_size=int(selection_window_size),
        selection_edge_window=int(selection_edge_window),
        min_modality_groups=int(min_modality_groups),
        default_modality_guardrail=bool(default_modality_guardrail),
        default_modality_min_improvement=float(default_modality_min_improvement),
        default_modality_max_recall_drop=float(default_modality_max_recall_drop),
        default_modality_max_exact_count_ratio_drop=float(default_modality_max_exact_count_ratio_drop),
        default_modality_max_mean_abs_count_diff_increase=float(default_modality_max_mean_abs_count_diff_increase),
        all_outer_rows=all_outer_rows,
        all_baseline_rows=all_baseline_rows,
        all_selection_rows=all_selection_rows,
        all_best_rows=all_best_rows,
        fold_summary_rows=fold_summary_rows,
        fold_results_lookup=fold_results_lookup,
    )


def _run_parallel_action_workers(
    *,
    script_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
    outer_subject: str,
    action_names: Sequence[str],
    available_modalities: Sequence[tuple[str, Sequence[str]]],
    default_imu_columns: Sequence[str],
    selection_mode: str,
    selection_window_size: int,
    selection_edge_window: int,
    min_modality_groups: int,
    default_modality_guardrail: bool,
    default_modality_min_improvement: float,
    default_modality_max_recall_drop: float,
    default_modality_max_exact_count_ratio_drop: float,
    default_modality_max_mean_abs_count_diff_increase: float,
) -> None:
    worker_base = out_dir / "_action_workers"
    worker_base.mkdir(parents=True, exist_ok=True)
    commands = []
    worker_outputs = []
    for action in action_names:
        worker_out = worker_base / action
        commands.append(
            [
                sys.executable,
                str(script_path),
                *_args_to_cli_list(
                    args,
                    overrides={
                        "output": worker_out,
                        "outer_subjects": outer_subject,
                        "include_actions": action,
                        "parallel_outer_jobs": 1,
                        "parallel_action_jobs": 1,
                        "parallel_worker": True,
                    },
                ),
            ]
        )
        worker_outputs.append((action, worker_out))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.parallel_action_jobs))) as executor:
        futures = [executor.submit(_run_subprocess_command, cmd, ROOT) for cmd in commands]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    _merge_action_worker_outputs(
        worker_outputs=worker_outputs,
        final_out_dir=out_dir,
        outer_subject=outer_subject,
        available_modalities=available_modalities,
        default_imu_columns=default_imu_columns,
        baseline_window_size=int(args.baseline_window_size),
        baseline_edge_window=int(args.baseline_edge_window),
        selection_metric=args.selection_metric,
        config_path=args.config,
        selection_mode=selection_mode,
        selection_window_size=int(selection_window_size),
        selection_edge_window=int(selection_edge_window),
        min_modality_groups=int(min_modality_groups),
        default_modality_guardrail=bool(default_modality_guardrail),
        default_modality_min_improvement=float(default_modality_min_improvement),
        default_modality_max_recall_drop=float(default_modality_max_recall_drop),
        default_modality_max_exact_count_ratio_drop=float(default_modality_max_exact_count_ratio_drop),
        default_modality_max_mean_abs_count_diff_increase=float(default_modality_max_mean_abs_count_diff_increase),
    )


def main():
    parser = argparse.ArgumentParser(description="Nested cross-subject benchmark for per-action causal RF boundary refinement.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/per_action_rf_nested_benchmark")
    parser.add_argument("--outer-subjects", default="")
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--modalities", default="")
    parser.add_argument("--max-outer-subjects", type=int, default=0)
    parser.add_argument("--max-inner-subjects", type=int, default=0)
    parser.add_argument("--baseline-window-size", type=int, default=50)
    parser.add_argument("--baseline-edge-window", type=int, default=20)
    parser.add_argument("--trailing-multipliers", default="0.25,0.5,0.75")
    parser.add_argument("--edge-multipliers", default="0.10,0.15,0.20")
    parser.add_argument("--trailing-min-samples", type=int, default=12)
    parser.add_argument("--trailing-max-samples", type=int, default=160)
    parser.add_argument("--edge-min-samples", type=int, default=6)
    parser.add_argument("--edge-max-samples", type=int, default=64)
    parser.add_argument("--duration-low-quantile", type=float, default=0.10)
    parser.add_argument("--duration-high-quantile", type=float, default=0.90)
    parser.add_argument("--selection-metric", default="rep_f1")
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--match-iou-train", type=float, default=0.3)
    parser.add_argument("--max-shift", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--max-samples", type=float, default=0.7)
    parser.add_argument("--target-matched-reps", type=int, default=500)
    parser.add_argument("--max-refiner-train-streams", type=int, default=40)
    parser.add_argument("--max-refiner-train-streams-per-subject", type=int, default=0)
    parser.add_argument("--max-matched-reps-per-stream", type=int, default=0)
    parser.add_argument("--max-matched-reps-per-subject", type=int, default=0)
    parser.add_argument("--min-matched-reps-for-refiner", type=int, default=24)
    parser.add_argument("--refiner-n-estimators", type=int, default=300)
    parser.add_argument("--refiner-max-depth", type=int, default=18)
    parser.add_argument("--refiner-min-samples-leaf", type=int, default=2)
    parser.add_argument("--parallel-outer-jobs", type=int, default=1)
    parser.add_argument("--parallel-action-jobs", type=int, default=1)
    parser.add_argument("--modality-only-search", action="store_true")
    parser.add_argument("--selection-window-size", type=int, default=0)
    parser.add_argument("--selection-edge-window", type=int, default=0)
    parser.add_argument("--min-modality-groups", type=int, default=1)
    parser.add_argument("--disable-default-modality-guardrail", action="store_true")
    parser.add_argument("--default-modality-min-improvement", type=float, default=0.01)
    parser.add_argument("--default-modality-max-recall-drop", type=float, default=0.02)
    parser.add_argument("--default-modality-max-exact-count-ratio-drop", type=float, default=0.0)
    parser.add_argument("--default-modality-max-mean-abs-count-diff-increase", type=float, default=0.0)
    parser.add_argument("--parallel-worker", action="store_true")
    args = parser.parse_args()

    if args.selection_metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
        raise ValueError(
            f"Unsupported selection metric: {args.selection_metric}. "
            f"Choose one of {sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)}"
        )
    if int(args.parallel_outer_jobs) > 1 and int(args.parallel_action_jobs) > 1:
        raise ValueError("Use either parallel outer jobs or parallel action jobs in one invocation, not both.")

    selection_mode = _selection_mode_name(bool(args.modality_only_search))
    selection_window_size = int(args.selection_window_size) if int(args.selection_window_size) > 0 else int(args.baseline_window_size)
    selection_edge_window = int(args.selection_edge_window) if int(args.selection_edge_window) > 0 else int(args.baseline_edge_window)
    min_modality_groups = max(1, int(args.min_modality_groups))
    if args.modality_only_search and min_modality_groups < 2:
        min_modality_groups = 2
    use_default_modality_guardrail = bool(args.modality_only_search and not args.disable_default_modality_guardrail)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    train_cfg = cb.TrainConfig(**{k: v for k, v in train_raw.items() if k in cb.TrainConfig.__dataclass_fields__})
    cb.set_seed(train_cfg.seed)

    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    default_imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    requested_actions = set(_parse_csv_list(args.include_actions))
    requested_subjects = _parse_csv_list(args.outer_subjects)
    requested_modalities = _parse_csv_list(args.modalities)

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        all_sensor_columns = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
        streams = cb._resample_streams_to_rate(streams, all_sensor_columns, time_column, target_sample_rate)

    if requested_actions:
        streams = [(sid, df) for sid, df in streams if _action_from_stream_id(sid) in requested_actions]
    if not streams:
        raise ValueError("No streams left after filtering actions.")

    available_modalities = _available_modalities(streams, requested_modalities, min_groups=min_modality_groups)
    if not available_modalities:
        raise ValueError("No requested modality subsets are available in the loaded streams.")

    subject_coverage = _coverage_rows(streams)
    subject_names = sorted({str(row["subject"]) for row in subject_coverage})
    outer_subjects = requested_subjects or subject_names
    outer_subjects = [subject for subject in outer_subjects if subject in subject_names]
    if int(args.max_outer_subjects) > 0:
        outer_subjects = outer_subjects[: int(args.max_outer_subjects)]
    if not outer_subjects:
        raise ValueError("No valid outer subjects selected.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(subject_coverage).to_csv(out_dir / "subject_action_coverage.csv", index=False)

    print(
        f"[INFO] benchmark outer_subjects={outer_subjects} actions={sorted({_action_from_stream_id(sid) for sid, _ in streams})} "
        f"modalities={[name for name, _ in available_modalities]} selection_mode={selection_mode} "
        f"selection_window={selection_window_size} selection_edge={selection_edge_window} "
        f"min_modality_groups={min_modality_groups} default_guardrail={use_default_modality_guardrail}",
        flush=True,
    )

    script_path = Path(__file__).resolve()
    if not args.parallel_worker and int(args.parallel_outer_jobs) > 1 and len(outer_subjects) > 1:
        _run_parallel_outer_workers(
            script_path=script_path,
            args=args,
            out_dir=out_dir,
            outer_subjects=outer_subjects,
            available_modalities=available_modalities,
            default_imu_columns=default_imu_columns,
            selection_mode=selection_mode,
            selection_window_size=selection_window_size,
            selection_edge_window=selection_edge_window,
            min_modality_groups=min_modality_groups,
            default_modality_guardrail=use_default_modality_guardrail,
            default_modality_min_improvement=float(args.default_modality_min_improvement),
            default_modality_max_recall_drop=float(args.default_modality_max_recall_drop),
            default_modality_max_exact_count_ratio_drop=float(args.default_modality_max_exact_count_ratio_drop),
            default_modality_max_mean_abs_count_diff_increase=float(args.default_modality_max_mean_abs_count_diff_increase),
        )
        print(f"[OK] wrote {out_dir / 'results.json'}")
        print(f"[OK] open {out_dir / 'summary.html'} for benchmark summary")
        return

    if not args.parallel_worker and int(args.parallel_action_jobs) > 1 and len(outer_subjects) == 1:
        outer_subject = outer_subjects[0]
        outer_train_subjects = [subject for subject in subject_names if subject != outer_subject]
        outer_train_streams = cb._filter_subjects(streams, outer_train_subjects, subject_column)
        outer_test_streams = cb._filter_subjects(streams, [outer_subject], subject_column)
        duration_stats = _summarize_train_fold_durations(
            outer_train_streams,
            min_phase_samples=int(mm_cfg.min_phase_samples),
            trailing_multipliers=_parse_float_list(args.trailing_multipliers),
            edge_multipliers=_parse_float_list(args.edge_multipliers),
            trailing_min_samples=int(args.trailing_min_samples),
            trailing_max_samples=int(args.trailing_max_samples),
            edge_min_samples=int(args.edge_min_samples),
            edge_max_samples=int(args.edge_max_samples),
            duration_low_quantile=float(args.duration_low_quantile),
            duration_high_quantile=float(args.duration_high_quantile),
        )
        if args.modality_only_search:
            duration_stats = _apply_fixed_search_candidates(
                duration_stats,
                window_size=selection_window_size,
                edge_window=selection_edge_window,
            )
        action_names = sorted(
            set(_action_from_stream_id(sid) for sid, _ in outer_test_streams)
            & set(_action_from_stream_id(sid) for sid, _ in outer_train_streams)
            & set(duration_stats.keys())
        )
        if len(action_names) > 1:
            _run_parallel_action_workers(
                script_path=script_path,
                args=args,
                out_dir=out_dir,
                outer_subject=outer_subject,
                action_names=action_names,
                available_modalities=available_modalities,
                default_imu_columns=default_imu_columns,
                selection_mode=selection_mode,
                selection_window_size=selection_window_size,
                selection_edge_window=selection_edge_window,
                min_modality_groups=min_modality_groups,
                default_modality_guardrail=use_default_modality_guardrail,
                default_modality_min_improvement=float(args.default_modality_min_improvement),
                default_modality_max_recall_drop=float(args.default_modality_max_recall_drop),
                default_modality_max_exact_count_ratio_drop=float(args.default_modality_max_exact_count_ratio_drop),
                default_modality_max_mean_abs_count_diff_increase=float(args.default_modality_max_mean_abs_count_diff_increase),
            )
            print(f"[OK] wrote {out_dir / 'results.json'}")
            print(f"[OK] open {out_dir / 'summary.html'} for benchmark summary")
            return

    trailing_multipliers = _parse_float_list(args.trailing_multipliers)
    edge_multipliers = _parse_float_list(args.edge_multipliers)

    all_outer_rows = []
    all_baseline_rows = []
    all_selection_rows = []
    all_best_rows = []
    fold_summary_rows = []
    fold_results_lookup = {}

    for outer_subject in outer_subjects:
        fold_dir = out_dir / outer_subject
        fold_dir.mkdir(parents=True, exist_ok=True)
        outer_train_subjects = [subject for subject in subject_names if subject != outer_subject]
        outer_train_streams = cb._filter_subjects(streams, outer_train_subjects, subject_column)
        outer_test_streams = cb._filter_subjects(streams, [outer_subject], subject_column)
        if not outer_train_streams or not outer_test_streams:
            print(f"[WARN] skip outer_subject={outer_subject} train={len(outer_train_streams)} test={len(outer_test_streams)}", flush=True)
            continue

        duration_stats = _summarize_train_fold_durations(
            outer_train_streams,
            min_phase_samples=int(mm_cfg.min_phase_samples),
            trailing_multipliers=trailing_multipliers,
            edge_multipliers=edge_multipliers,
            trailing_min_samples=int(args.trailing_min_samples),
            trailing_max_samples=int(args.trailing_max_samples),
            edge_min_samples=int(args.edge_min_samples),
            edge_max_samples=int(args.edge_max_samples),
            duration_low_quantile=float(args.duration_low_quantile),
            duration_high_quantile=float(args.duration_high_quantile),
        )
        if args.modality_only_search:
            duration_stats = _apply_fixed_search_candidates(
                duration_stats,
                window_size=selection_window_size,
                edge_window=selection_edge_window,
            )
        _write_duration_report(
            fold_dir / "duration_report.md",
            outer_subject,
            outer_train_subjects,
            duration_stats,
            selection_mode=selection_mode,
            fixed_window_size=selection_window_size,
            fixed_edge_window=selection_edge_window,
        )

        action_names = sorted(
            set(_action_from_stream_id(sid) for sid, _ in outer_test_streams)
            & set(_action_from_stream_id(sid) for sid, _ in outer_train_streams)
            & set(duration_stats.keys())
        )
        outer_train_by_action_subject: dict[str, dict[str, list[tuple[str, pd.DataFrame]]]] = defaultdict(lambda: defaultdict(list))
        outer_test_by_action: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
        for stream_id, df in outer_train_streams:
            action = _action_from_stream_id(stream_id)
            subject = str(df.iloc[0]["_split_subject"]) if "_split_subject" in df.columns else str(df.iloc[0][subject_column])
            outer_train_by_action_subject[action][subject].append((stream_id, df))
        for stream_id, df in outer_test_streams:
            outer_test_by_action[_action_from_stream_id(stream_id)].append((stream_id, df))

        inner_subjects = list(outer_train_subjects)
        if int(args.max_inner_subjects) > 0:
            inner_subjects = inner_subjects[: int(args.max_inner_subjects)]
        normalize_cache: dict[tuple, tuple[list[tuple[str, pd.DataFrame]], list[tuple[str, pd.DataFrame]]]] = {}
        rf_cache: dict[tuple, dict] = {}

        print(
            f"[INFO] outer_subject={outer_subject} train_subjects={outer_train_subjects} "
            f"inner_subjects={inner_subjects} actions={action_names}",
            flush=True,
        )

        selection_rows = []
        best_rows = []
        tuned_rows = []
        baseline_rows = []

        for action in action_names:
            action_duration = duration_stats.get(action)
            if not action_duration:
                continue
            print(f"[INFO] outer_subject={outer_subject} action={action} selection_start", flush=True)
            action_subject_map = outer_train_by_action_subject.get(action, {})
            action_outer_train = [stream for subject in outer_train_subjects for stream in action_subject_map.get(subject, [])]
            action_outer_test = list(outer_test_by_action.get(action, []))
            candidate_rows_by_key: dict[tuple, list[dict]] = defaultdict(list)
            candidate_subjects_by_key: dict[tuple, set[str]] = defaultdict(set)
            for modality_name, modality_columns in available_modalities:
                for val_subject in inner_subjects:
                    inner_train = [stream for subject in outer_train_subjects if subject != val_subject for stream in action_subject_map.get(subject, [])]
                    inner_val = list(action_subject_map.get(val_subject, []))
                    if not inner_train or not inner_val:
                        continue
                    norm_key = (
                        action,
                        modality_name,
                        tuple(modality_columns),
                        _stream_ids_key(inner_train),
                        _stream_ids_key(inner_val),
                    )
                    if norm_key not in normalize_cache:
                        normalize_cache[norm_key] = _normalize_streams(inner_train, inner_val, modality_columns)
                    inner_train_norm, inner_val_norm = normalize_cache[norm_key]
                    for window_size in action_duration.get("trailing_window_candidates", []):
                        rf_key = norm_key + (int(window_size),)
                        if rf_key not in rf_cache:
                            t0 = time.time()
                            clf = crf.train_causal_rf(
                                inner_train_norm,
                                modality_columns,
                                window_size=int(window_size),
                                stride=int(args.train_stride),
                                n_estimators=int(args.n_estimators),
                                max_depth=int(args.max_depth),
                                max_samples=float(args.max_samples),
                            )
                            rf_train_time = time.time() - t0
                            train_prob_cache = _predict_prob_cache(clf, inner_train_norm, modality_columns, int(window_size))
                            val_prob_cache = _predict_prob_cache(clf, inner_val_norm, modality_columns, int(window_size))
                            val_eval_cache = _build_eval_stream_cache(inner_val_norm, val_prob_cache, mm_cfg)
                            matched_examples = _collect_matched_examples(
                                inner_train_norm,
                                train_prob_cache,
                                mm_cfg,
                                float(args.match_iou_train),
                                int(args.max_shift),
                                int(args.target_matched_reps),
                                int(args.max_refiner_train_streams),
                                int(args.max_refiner_train_streams_per_subject),
                                int(args.max_matched_reps_per_stream),
                                int(args.max_matched_reps_per_subject),
                            )
                            rf_cache[rf_key] = {
                                "train_streams": inner_train_norm,
                                "val_streams": inner_val_norm,
                                "train_prob_cache": train_prob_cache,
                                "val_prob_cache": val_prob_cache,
                                "val_eval_cache": val_eval_cache,
                                "matched_examples": matched_examples,
                                "refiner_feature_cache": {},
                                "rf_train_time_s": float(rf_train_time),
                            }
                        artifacts = rf_cache[rf_key]
                        for edge_window in action_duration.get("edge_window_candidates", []):
                            candidate_key = (modality_name, tuple(modality_columns), int(window_size), int(edge_window))
                            refiner = _fit_refiner_from_examples(
                                artifacts["matched_examples"],
                                modality_columns,
                                int(edge_window),
                                int(args.max_shift),
                                feature_cache=artifacts["refiner_feature_cache"],
                                min_examples=int(args.min_matched_reps_for_refiner),
                                n_estimators=int(args.refiner_n_estimators),
                                max_depth=int(args.refiner_max_depth),
                                min_samples_leaf=int(args.refiner_min_samples_leaf),
                            )
                            rows = _evaluate_prob_cache(
                                action=action,
                                streams=artifacts["val_streams"],
                                prob_cache=artifacts["val_prob_cache"],
                                imu_columns=modality_columns,
                                modality_name=modality_name,
                                mm_cfg=mm_cfg,
                                window_size=int(window_size),
                                edge_window=int(edge_window),
                                min_rep_duration_samples=int(action_duration.get("min_rep_duration_samples", 0)),
                                max_rep_duration_samples=int(action_duration.get("max_rep_duration_samples", 0)),
                                max_shift=int(args.max_shift),
                                refiner=refiner,
                                stream_eval_cache=artifacts["val_eval_cache"],
                                output_root=None,
                                output_tag="",
                                outer_subject=outer_subject,
                            )
                            for row in rows:
                                tagged = dict(row)
                                tagged["inner_val_subject"] = val_subject
                                candidate_rows_by_key[candidate_key].append(tagged)
                            candidate_subjects_by_key[candidate_key].add(val_subject)
                    if use_default_modality_guardrail and all(col in set(str(c) for c in inner_train[0][1].columns) for col in default_imu_columns):
                        baseline_modality_name = "baseline_reference"
                        baseline_columns = tuple(default_imu_columns)
                        baseline_norm_key = (
                            action,
                            baseline_modality_name,
                            baseline_columns,
                            _stream_ids_key(inner_train),
                            _stream_ids_key(inner_val),
                        )
                        if baseline_norm_key not in normalize_cache:
                            normalize_cache[baseline_norm_key] = _normalize_streams(inner_train, inner_val, baseline_columns)
                        baseline_train_norm, baseline_val_norm = normalize_cache[baseline_norm_key]
                        baseline_rf_key = baseline_norm_key + (int(selection_window_size),)
                        if baseline_rf_key not in rf_cache:
                            t0 = time.time()
                            clf = crf.train_causal_rf(
                                baseline_train_norm,
                                baseline_columns,
                                window_size=int(selection_window_size),
                                stride=int(args.train_stride),
                                n_estimators=int(args.n_estimators),
                                max_depth=int(args.max_depth),
                                max_samples=float(args.max_samples),
                            )
                            rf_train_time = time.time() - t0
                            train_prob_cache = _predict_prob_cache(clf, baseline_train_norm, baseline_columns, int(selection_window_size))
                            val_prob_cache = _predict_prob_cache(clf, baseline_val_norm, baseline_columns, int(selection_window_size))
                            val_eval_cache = _build_eval_stream_cache(baseline_val_norm, val_prob_cache, mm_cfg)
                            matched_examples = _collect_matched_examples(
                                baseline_train_norm,
                                train_prob_cache,
                                mm_cfg,
                                float(args.match_iou_train),
                                int(args.max_shift),
                                int(args.target_matched_reps),
                                int(args.max_refiner_train_streams),
                                int(args.max_refiner_train_streams_per_subject),
                                int(args.max_matched_reps_per_stream),
                                int(args.max_matched_reps_per_subject),
                            )
                            rf_cache[baseline_rf_key] = {
                                "train_streams": baseline_train_norm,
                                "val_streams": baseline_val_norm,
                                "train_prob_cache": train_prob_cache,
                                "val_prob_cache": val_prob_cache,
                                "val_eval_cache": val_eval_cache,
                                "matched_examples": matched_examples,
                                "refiner_feature_cache": {},
                                "rf_train_time_s": float(rf_train_time),
                            }
                        baseline_artifacts = rf_cache[baseline_rf_key]
                        baseline_refiner = _fit_refiner_from_examples(
                            baseline_artifacts["matched_examples"],
                            baseline_columns,
                            int(selection_edge_window),
                            int(args.max_shift),
                            feature_cache=baseline_artifacts["refiner_feature_cache"],
                            min_examples=int(args.min_matched_reps_for_refiner),
                            n_estimators=int(args.refiner_n_estimators),
                            max_depth=int(args.refiner_max_depth),
                            min_samples_leaf=int(args.refiner_min_samples_leaf),
                        )
                        baseline_reference_rows = _evaluate_prob_cache(
                            action=action,
                            streams=baseline_artifacts["val_streams"],
                            prob_cache=baseline_artifacts["val_prob_cache"],
                            imu_columns=baseline_columns,
                            modality_name=baseline_modality_name,
                            mm_cfg=mm_cfg,
                            window_size=int(selection_window_size),
                            edge_window=int(selection_edge_window),
                            min_rep_duration_samples=0,
                            max_rep_duration_samples=0,
                            max_shift=int(args.max_shift),
                            refiner=baseline_refiner,
                            stream_eval_cache=baseline_artifacts["val_eval_cache"],
                            output_root=None,
                            output_tag="",
                            outer_subject=outer_subject,
                        )
                        baseline_candidate_key = (baseline_modality_name, baseline_columns, int(selection_window_size), int(selection_edge_window))
                        for row in baseline_reference_rows:
                            tagged = dict(row)
                            tagged["inner_val_subject"] = val_subject
                            candidate_rows_by_key[baseline_candidate_key].append(tagged)
                        candidate_subjects_by_key[baseline_candidate_key].add(val_subject)
            for (modality_name, modality_columns, window_size, edge_window), candidate_val_rows in candidate_rows_by_key.items():
                if not candidate_val_rows:
                    continue
                first_row = candidate_val_rows[0]
                candidate_summary = _compact_summary(base._aggregate_rows(candidate_val_rows))
                candidate_summary = _augment_count_consistency(candidate_summary, candidate_val_rows)
                candidate_summary.update(
                    {
                        "outer_test_subject": outer_subject,
                        "action": action,
                        "modality": modality_name,
                        "imu_columns": ",".join(modality_columns),
                        "window_size": int(window_size),
                        "edge_window": int(edge_window),
                        "min_rep_duration_samples": int(first_row.get("min_rep_duration_samples", action_duration.get("min_rep_duration_samples", 0))),
                        "max_rep_duration_samples": int(first_row.get("max_rep_duration_samples", action_duration.get("max_rep_duration_samples", 0))),
                        "inner_fold_count": int(len(candidate_subjects_by_key[(modality_name, modality_columns, window_size, edge_window)])),
                        "selection_metric": args.selection_metric,
                        "selection_metric_value": _safe_float(candidate_summary.get(args.selection_metric)),
                    }
                )
                selection_rows.append(candidate_summary)
            action_candidates = [row for row in selection_rows if row.get("action") == action]
            if not action_candidates:
                print(f"[WARN] outer_subject={outer_subject} action={action} no_valid_candidates", flush=True)
                continue
            best = max(action_candidates, key=lambda row: _selection_sort_key(row, args.selection_metric))
            selection_source = "nested_best"
            default_candidate = _find_default_modality_candidate(action_candidates, default_imu_columns)
            if use_default_modality_guardrail and default_candidate is not None:
                best_metric = _safe_float(best.get(args.selection_metric), default=-1e12)
                default_metric = _safe_float(default_candidate.get(args.selection_metric), default=-1e12)
                best_recall = _safe_float(best.get("recall"), default=-1e12)
                default_recall = _safe_float(default_candidate.get("recall"), default=-1e12)
                best_exact_count_ratio = _safe_float(best.get("exact_count_ratio"), default=-1e12)
                default_exact_count_ratio = _safe_float(default_candidate.get("exact_count_ratio"), default=-1e12)
                best_mean_abs_count_diff = _safe_float(best.get("mean_abs_count_diff"), default=1e12)
                default_mean_abs_count_diff = _safe_float(default_candidate.get("mean_abs_count_diff"), default=1e12)
                same_default = str(best.get("imu_columns", "")) == str(default_candidate.get("imu_columns", ""))
                metric_gap = best_metric - default_metric
                recall_gap = best_recall - default_recall
                exact_count_ratio_gap = best_exact_count_ratio - default_exact_count_ratio
                mean_abs_count_diff_gap = best_mean_abs_count_diff - default_mean_abs_count_diff
                if (not same_default) and (
                    metric_gap < float(args.default_modality_min_improvement)
                    or recall_gap < -float(args.default_modality_max_recall_drop)
                    or exact_count_ratio_gap < -float(args.default_modality_max_exact_count_ratio_drop)
                    or mean_abs_count_diff_gap > float(args.default_modality_max_mean_abs_count_diff_increase)
                ):
                    best = dict(default_candidate)
                    best["guardrail_metric_gap"] = float(metric_gap)
                    best["guardrail_recall_gap"] = float(recall_gap)
                    best["guardrail_exact_count_ratio_gap"] = float(exact_count_ratio_gap)
                    best["guardrail_mean_abs_count_diff_gap"] = float(mean_abs_count_diff_gap)
                    selection_source = "default_modality_guardrail"
                else:
                    best = dict(best)
                    best["guardrail_metric_gap"] = float(metric_gap)
                    best["guardrail_recall_gap"] = float(recall_gap)
                    best["guardrail_exact_count_ratio_gap"] = float(exact_count_ratio_gap)
                    best["guardrail_mean_abs_count_diff_gap"] = float(mean_abs_count_diff_gap)
            else:
                best = dict(best)
            best["selection_source"] = selection_source
            best_rows.append(dict(best))

            tuned = _evaluate_action_run(
                action=action,
                train_streams=action_outer_train,
                test_streams=action_outer_test,
                imu_columns=tuple(_parse_csv_list(str(best.get("imu_columns", "")))) or tuple(col for col in best.get("imu_columns", [])),
                modality_name=str(best.get("modality", "unknown")),
                mm_cfg=mm_cfg,
                window_size=int(best.get("window_size", args.baseline_window_size)),
                edge_window=int(best.get("edge_window", args.baseline_edge_window)),
                min_rep_duration_samples=int(best.get("min_rep_duration_samples", 0)),
                max_rep_duration_samples=int(best.get("max_rep_duration_samples", 0)),
                train_stride=int(args.train_stride),
                match_iou_train=float(args.match_iou_train),
                max_shift=int(args.max_shift),
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                max_samples=float(args.max_samples),
                target_matched_reps=int(args.target_matched_reps),
                max_refiner_train_streams=int(args.max_refiner_train_streams),
                max_refiner_train_streams_per_subject=int(args.max_refiner_train_streams_per_subject),
                max_matched_reps_per_stream=int(args.max_matched_reps_per_stream),
                max_matched_reps_per_subject=int(args.max_matched_reps_per_subject),
                min_matched_reps_for_refiner=int(args.min_matched_reps_for_refiner),
                refiner_n_estimators=int(args.refiner_n_estimators),
                refiner_max_depth=int(args.refiner_max_depth),
                refiner_min_samples_leaf=int(args.refiner_min_samples_leaf),
                output_root=fold_dir,
                output_tag=f"tuned/{action}",
                outer_subject=outer_subject,
            )
            for row in tuned["rows"]:
                row["selection_source"] = selection_source
                tuned_rows.append(row)

            baseline = _evaluate_action_run(
                action=action,
                train_streams=action_outer_train,
                test_streams=action_outer_test,
                imu_columns=tuple(default_imu_columns),
                modality_name="baseline_config",
                mm_cfg=mm_cfg,
                window_size=int(args.baseline_window_size),
                edge_window=int(args.baseline_edge_window),
                min_rep_duration_samples=0,
                max_rep_duration_samples=0,
                train_stride=int(args.train_stride),
                match_iou_train=float(args.match_iou_train),
                max_shift=int(args.max_shift),
                n_estimators=int(args.n_estimators),
                max_depth=int(args.max_depth),
                max_samples=float(args.max_samples),
                target_matched_reps=int(args.target_matched_reps),
                max_refiner_train_streams=int(args.max_refiner_train_streams),
                max_refiner_train_streams_per_subject=int(args.max_refiner_train_streams_per_subject),
                max_matched_reps_per_stream=int(args.max_matched_reps_per_stream),
                max_matched_reps_per_subject=int(args.max_matched_reps_per_subject),
                min_matched_reps_for_refiner=int(args.min_matched_reps_for_refiner),
                refiner_n_estimators=int(args.refiner_n_estimators),
                refiner_max_depth=int(args.refiner_max_depth),
                refiner_min_samples_leaf=int(args.refiner_min_samples_leaf),
                output_root=fold_dir,
                output_tag=f"baseline/{action}",
                outer_subject=outer_subject,
            )
            for row in baseline["rows"]:
                row["selection_source"] = "baseline_config"
                baseline_rows.append(row)

        tuned_summary = _compact_summary(base._aggregate_rows(tuned_rows))
        baseline_summary = _compact_summary(base._aggregate_rows(baseline_rows))
        delta_summary = _diff_metrics(tuned_summary, baseline_summary)
        fold_results = {
            "outer_test_subject": outer_subject,
            "train_subjects": outer_train_subjects,
            "selection_metric": args.selection_metric,
            "available_modalities": [name for name, _ in available_modalities],
            "duration_stats": duration_stats,
            "best_config_per_action": {row["action"]: dict(row) for row in best_rows},
            "tuned_overall": tuned_summary,
            "baseline_overall": baseline_summary,
            "delta_vs_baseline": delta_summary,
        }
        _write_fold_outputs(
            fold_dir=fold_dir,
            duration_stats=duration_stats,
            selection_rows=selection_rows,
            best_rows=best_rows,
            tuned_rows=tuned_rows,
            tuned_summary=tuned_summary,
            baseline_rows=baseline_rows,
            baseline_summary=baseline_summary,
            fold_results=fold_results,
        )

        fold_summary_rows.append(
            {
                "outer_test_subject": outer_subject,
                **{f"tuned_{k}": tuned_summary.get(k) for k in METRIC_KEYS},
                **{f"baseline_{k}": baseline_summary.get(k) for k in METRIC_KEYS},
                **{f"delta_{k}": delta_summary.get(k) for k in METRIC_KEYS},
            }
        )
        for row in selection_rows:
            all_selection_rows.append(dict(row))
        for row in best_rows:
            all_best_rows.append(dict(row))
        all_outer_rows.extend(tuned_rows)
        all_baseline_rows.extend(baseline_rows)
        fold_results_lookup[outer_subject] = fold_results

    _write_root_outputs(
        out_dir=out_dir,
        config_path=args.config,
        outer_subjects=outer_subjects,
        available_modalities=available_modalities,
        default_imu_columns=default_imu_columns,
        baseline_window_size=int(args.baseline_window_size),
        baseline_edge_window=int(args.baseline_edge_window),
        selection_metric=args.selection_metric,
        selection_mode=selection_mode,
        selection_window_size=int(selection_window_size),
        selection_edge_window=int(selection_edge_window),
        min_modality_groups=int(min_modality_groups),
        default_modality_guardrail=bool(use_default_modality_guardrail),
        default_modality_min_improvement=float(args.default_modality_min_improvement),
        default_modality_max_recall_drop=float(args.default_modality_max_recall_drop),
        default_modality_max_exact_count_ratio_drop=float(args.default_modality_max_exact_count_ratio_drop),
        default_modality_max_mean_abs_count_diff_increase=float(args.default_modality_max_mean_abs_count_diff_increase),
        all_outer_rows=all_outer_rows,
        all_baseline_rows=all_baseline_rows,
        all_selection_rows=all_selection_rows,
        all_best_rows=all_best_rows,
        fold_summary_rows=fold_summary_rows,
        fold_results_lookup=fold_results_lookup,
    )
    root_results = _load_json(out_dir / "results.json")
    print(json.dumps({
        "tuned": root_results["tuned_overall"],
        "baseline": root_results["baseline_overall"],
        "delta_vs_baseline": root_results["delta_vs_baseline"],
    }, indent=2))
    print(f"[OK] wrote {out_dir / 'results.json'}")
    print(f"[OK] open {out_dir / 'summary.html'} for benchmark summary")


if __name__ == "__main__":
    main()
