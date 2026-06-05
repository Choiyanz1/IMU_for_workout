"""Streaming-style replay for the global-active + soft top5 pipeline.

This script avoids offline active-segment extraction and full-sequence rep parsing.
It simulates causal updates from past samples only:
- global active RF on trailing IMU windows;
- action RF posterior on trailing IMU windows;
- raw6 CNN phase probabilities from trailing 300-sample windows;
- stateful C/E rep parser;
- delayed one-rep soft duration merge.

It also reports a bounded-latency path that applies causal MA smoothing and
fixed-lag Viterbi before parsing reps, so each finalized label uses at most the
configured lag of future samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.micro_macro_segments import RepDetection  # noqa: E402
from scripts.evaluate_dual_head_rf_action_loso import (  # noqa: E402
    ACTIONS,
    extract_features_batch,
    load_non_action_streams,
)
from scripts.evaluate_predicted_action_top5_pipeline import (  # noqa: E402
    _probs_for_actions,
    train_action_rf,
    train_global_active_detector,
)
from scripts.evaluate_periodic_active_gate_loso import (  # noqa: E402
    predict_active_prob as predict_periodic_active_prob,
    state_machine as periodic_active_state_machine,
    train_gate as train_periodic_active_gate,
)
from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    _extract_window_features_batch,
)
from scripts.new_c_pipeline.duration_merge_decoder_9fold import (  # noqa: E402
    build_duration_priors,
    evaluate_with_reps,
    threshold_for_action,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    aggregate_rich,
    stream_subject,
    train_raw6_model,
)
from scripts.new_c_pipeline.selective_duration_merge_decoder_9fold import ACTION_SETS  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import EXCLUDED_SESSIONS, set_seed, should_exclude, smooth_ma  # noqa: E402
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def softmax_np(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-12)


def compute_action_norm(train_streams, imu_columns):
    values = [df[list(imu_columns)].to_numpy(dtype=np.float32) for _, df in train_streams if len(df)]
    stacked = np.concatenate(values, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def trailing_window(values: np.ndarray, end_exclusive: int, size: int) -> np.ndarray:
    start = max(0, int(end_exclusive) - int(size))
    window = values[start:end_exclusive]
    if len(window) == 0:
        window = values[:1]
    if len(window) < size:
        window = np.pad(window, ((size - len(window), 0), (0, 0)), mode="edge")
    return window.astype(np.float32, copy=False)


def fixed_lag_viterbi_decode(phase_probs: np.ndarray, penalty: float, lag_samples: int) -> np.ndarray:
    n = len(phase_probs)
    if n == 0:
        return phase_probs.copy()
    lag = max(0, int(lag_samples))
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2), dtype=np.float64)
    back = np.zeros((n, 2), dtype=np.int64)
    labels = np.full(n, -1, dtype=np.int64)
    dp[0] = log_probs[0]
    for t in range(1, n):
        for state in range(2):
            stay = dp[t - 1, state]
            switch = dp[t - 1, 1 - state] - penalty
            if stay >= switch:
                dp[t, state] = log_probs[t, state] + stay
                back[t, state] = state
            else:
                dp[t, state] = log_probs[t, state] + switch
                back[t, state] = 1 - state
        finalize_idx = t - lag
        if finalize_idx >= 0:
            state = int(np.argmax(dp[t]))
            for k in range(t, finalize_idx, -1):
                state = int(back[k, state])
            labels[finalize_idx] = state
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    labels[labels < 0] = path[labels < 0]
    result = np.zeros((n, 2), dtype=np.float32)
    result[labels == 0, 0] = 1.0
    result[labels == 1, 1] = 1.0
    return result


class OnlineRepParser:
    def __init__(self, min_phase_samples: int = 3, max_gap_samples: int = 3) -> None:
        self.min_phase_samples = int(min_phase_samples)
        self.max_gap_samples = int(max_gap_samples)
        self.current_label: str | None = None
        self.current_start: int | None = None
        self.last_concentric: tuple[int, int] | None = None

    def update(self, idx: int, label: str | None) -> list[RepDetection]:
        emitted: list[RepDetection] = []
        if label is None:
            emitted.extend(self._close_run(idx))
            self.current_label = None
            self.current_start = None
            return emitted
        if self.current_label is None:
            self.current_label = label
            self.current_start = idx
            return emitted
        if label != self.current_label:
            emitted.extend(self._close_run(idx))
            self.current_label = label
            self.current_start = idx
        return emitted

    def finish(self, n_samples: int) -> list[RepDetection]:
        return self._close_run(n_samples)

    def _close_run(self, end_idx: int) -> list[RepDetection]:
        if self.current_label is None or self.current_start is None:
            return []
        start = int(self.current_start)
        end = int(end_idx)
        label = self.current_label
        self.current_label = None
        self.current_start = None
        if end - start < self.min_phase_samples:
            return []
        if label == "concentric":
            self.last_concentric = (start, end)
            return []
        if label == "eccentric" and self.last_concentric is not None:
            c_start, c_end = self.last_concentric
            if start - c_end <= self.max_gap_samples:
                self.last_concentric = None
                return [
                    RepDetection(
                        start_idx=int(c_start),
                        transition_idx=int(start),
                        end_idx=int(end),
                        micro_source="online_phase",
                        micro_confidence=1.0,
                    )
                ]
        return []


class OnlineSoftMerger:
    def __init__(self, max_gap_samples: int) -> None:
        self.max_gap_samples = int(max_gap_samples)
        self.pending: RepDetection | None = None
        self.final_reps: list[RepDetection] = []

    def add(self, rep: RepDetection, threshold: float | None) -> None:
        if threshold is None or threshold <= 0:
            self._flush_pending()
            self.final_reps.append(rep)
            return

        duration = int(rep.end_idx) - int(rep.start_idx)
        if self.pending is not None:
            gap = int(rep.start_idx) - int(self.pending.end_idx)
            if gap <= self.max_gap_samples:
                self.final_reps.append(self._merge(self.pending, rep))
                self.pending = None
                return
            self._flush_pending()

        if duration < threshold:
            self.pending = rep
        else:
            self.final_reps.append(rep)

    def finish(self) -> list[RepDetection]:
        self._flush_pending()
        return self.final_reps

    def _flush_pending(self) -> None:
        if self.pending is not None:
            self.final_reps.append(self.pending)
            self.pending = None

    @staticmethod
    def _merge(a: RepDetection, b: RepDetection) -> RepDetection:
        return RepDetection(
            start_idx=int(a.start_idx),
            transition_idx=int(a.transition_idx),
            end_idx=int(b.end_idx),
            micro_source="online_soft_merge",
            micro_confidence=float(a.micro_confidence),
        )


def soft_threshold_from_context(posterior: np.ndarray | None, duration_priors, args) -> tuple[float | None, dict[str, object]]:
    if posterior is None:
        return None, {"soft_enabled": False}
    order = np.argsort(posterior)[::-1]
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    top5_indices = [ACTIONS.index(action) for action in ACTION_SETS["top5"] if action in ACTIONS]
    top5_mass = float(np.sum(posterior[top5_indices]))
    top_conf = float(posterior[top])
    margin = float(posterior[top] - posterior[second])
    meta = {
        "soft_enabled": False,
        "top_action": ACTIONS[top],
        "top_confidence": top_conf,
        "top5_mass": top5_mass,
        "margin": margin,
    }
    if top5_mass < args.soft_top5_mass_threshold or top_conf < args.soft_action_confidence_threshold or margin < args.soft_margin_threshold:
        return None, meta
    weights = np.asarray([posterior[ACTIONS.index(action)] for action in ACTION_SETS["top5"]], dtype=np.float32)
    thresholds = np.asarray([threshold_for_action(duration_priors, action, 5) for action in ACTION_SETS["top5"]], dtype=np.float32)
    threshold = float(np.dot(weights, thresholds) / max(float(weights.sum()), 1e-8))
    meta.update({"soft_enabled": True, "threshold": threshold})
    return threshold, meta


def apply_online_soft_merge(reps, posterior_by_sample, duration_priors, args, scale: float = 1.0):
    merger = OnlineSoftMerger(args.max_gap_samples)
    last_meta: dict[str, object] = {"soft_enabled": False}
    n = len(posterior_by_sample)
    for rep in reps:
        idx = min(max(int(rep.end_idx) + int(args.fixed_lag_samples), 0), max(0, n - 1))
        posterior = posterior_by_sample[idx]
        threshold, last_meta = soft_threshold_from_context(posterior, duration_priors, args)
        if threshold is not None:
            threshold *= float(scale)
            last_meta = {**last_meta, "threshold_scale": float(scale), "scaled_threshold": float(threshold)}
        merger.add(rep, threshold)
    merged = merger.finish()
    merged = filter_confirmed_rep_groups(merged, args.min_confirmed_reps, args.confirmed_set_gap_samples)
    return merged, last_meta


def append_rest_tail(stream_id: str, df: pd.DataFrame, data_dir: Path, seconds: float, sample_rate_hz: float, imu_columns) -> tuple[pd.DataFrame, int, int]:
    if seconds <= 0:
        return df, len(df), 0
    parts = [part for part in str(stream_id).split("/") if part]
    if len(parts) < 4:
        return df, len(df), 0
    subject, session, action, set_name = parts[0], parts[1], parts[2], parts[3]
    rest_dir = data_dir / subject / session / action / f"rest_after_{set_name}"
    if not rest_dir.exists():
        return df, len(df), 0

    max_rows = int(round(float(seconds) * float(sample_rate_hz)))
    for csv_path in sorted(rest_dir.glob("*.csv")):
        try:
            rest_df = pd.read_csv(csv_path)
        except Exception:
            continue
        if rest_df.empty or not set(imu_columns).issubset(rest_df.columns):
            continue
        if max_rows > 0:
            rest_df = rest_df.iloc[:max_rows].copy()
        else:
            rest_df = rest_df.copy()
        rest_df["subject_id"] = subject
        rest_df["action_type"] = "non_action"
        rest_df["phase"] = "non_action"
        set_len = len(df)
        combined = pd.concat([df.copy(), rest_df], ignore_index=True, sort=False)
        return combined, set_len, len(rest_df)
    return df, len(df), 0


def reps_overlapping_rest(reps: list[RepDetection], rest_start_idx: int) -> int:
    return int(sum(1 for rep in reps if int(rep.end_idx) > int(rest_start_idx)))


def reps_starting_in_rest(reps: list[RepDetection], rest_start_idx: int) -> int:
    return int(sum(1 for rep in reps if int(rep.start_idx) >= int(rest_start_idx)))


def reps_starting_after_rest_grace(reps: list[RepDetection], rest_start_idx: int, grace_samples: int) -> int:
    cutoff = int(rest_start_idx) + max(0, int(grace_samples))
    return int(sum(1 for rep in reps if int(rep.start_idx) >= cutoff))


def reps_crossing_rest_boundary(reps: list[RepDetection], rest_start_idx: int) -> int:
    return int(sum(1 for rep in reps if int(rep.start_idx) < int(rest_start_idx) < int(rep.end_idx)))


def rep_debug_rows(reps: list[RepDetection], offset: int = 0) -> list[dict[str, int | float | str]]:
    rows = []
    for rep in reps:
        rows.append(
            {
                "start_idx": int(rep.start_idx) - int(offset),
                "transition_idx": int(rep.transition_idx) - int(offset),
                "end_idx": int(rep.end_idx) - int(offset),
                "duration_samples": int(rep.end_idx) - int(rep.start_idx),
                "source": str(rep.micro_source),
                "confidence": float(rep.micro_confidence),
            }
        )
    return rows


def segment_debug_rows(mask: np.ndarray, offset: int = 0) -> list[dict[str, int]]:
    return [
        {"start_idx": int(start) - int(offset), "end_idx": int(end) - int(offset), "duration_samples": int(end) - int(start)}
        for start, end in mask_segments(mask)
    ]


def active_mask_summary(mask: np.ndarray, probs: np.ndarray, gt_phases: np.ndarray | None = None) -> dict[str, object]:
    mask = np.asarray(mask, dtype=bool)
    probs = np.asarray(probs, dtype=np.float32)
    n = int(len(mask))
    segments = mask_segments(mask)
    summary: dict[str, object] = {
        "samples": n,
        "active_samples": int(np.sum(mask)),
        "active_rate": float(np.mean(mask)) if n else 0.0,
        "active_segments": int(len(segments)),
        "longest_active_segment_samples": int(max((end - start for start, end in segments), default=0)),
        "mean_active_probability": float(np.mean(probs)) if len(probs) else 0.0,
        "max_active_probability": float(np.max(probs)) if len(probs) else 0.0,
    }
    if gt_phases is not None:
        gt = np.asarray(gt_phases)
        valid = gt[:n]
        active_gt = np.isin(valid, ["concentric", "eccentric"])
        rest_gt = ~active_gt
        summary.update(
            {
                "gt_active_samples": int(np.sum(active_gt)),
                "active_coverage": float(np.mean(mask[: len(valid)][active_gt])) if np.any(active_gt) else 0.0,
                "gt_rest_active_rate": float(np.mean(mask[: len(valid)][rest_gt])) if np.any(rest_gt) else 0.0,
            }
        )
    return summary


def compact_event_confirmation(meta: dict[str, object]) -> dict[str, object]:
    if not bool(meta.get("enabled", False)):
        return dict(meta)
    events = list(meta.get("events", []))
    return {
        "enabled": True,
        "events_total": int(len(events)),
        "confirmed_events": int(meta.get("confirmed_events", 0)),
        "rejected_events": int(meta.get("rejected_events", 0)),
        "input_reps": int(meta.get("input_reps", 0)),
        "kept_reps": int(meta.get("kept_reps", 0)),
        "dropped_reps": int(meta.get("dropped_reps", 0)),
        "rejected_reps": int(sum(int(event.get("reps", 0)) for event in events if not bool(event.get("confirmed", False)))),
    }


def filter_confirmed_rep_groups(reps: list[RepDetection], min_reps: int, max_gap_samples: int) -> list[RepDetection]:
    if min_reps <= 1 or not reps:
        return reps
    ordered = sorted(reps, key=lambda rep: int(rep.start_idx))
    groups: list[list[RepDetection]] = []
    current: list[RepDetection] = [ordered[0]]
    for rep in ordered[1:]:
        gap = int(rep.start_idx) - int(current[-1].end_idx)
        if gap <= int(max_gap_samples):
            current.append(rep)
        else:
            groups.append(current)
            current = [rep]
    groups.append(current)

    kept: list[RepDetection] = []
    for group in groups:
        if len(group) >= int(min_reps):
            kept.extend(group)
    return kept


def mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        start = i
        while i < n and bool(mask[i]):
            i += 1
        segments.append((start, i))
    return segments


def longest_false_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return int(best)


def _posterior_event_stats(posterior_by_sample, start: int, end: int) -> dict[str, float | str | None]:
    rows = [posterior_by_sample[i] for i in range(max(0, start), max(0, end)) if posterior_by_sample[i] is not None]
    if not rows:
        return {
            "top_action": None,
            "top_confidence": 0.0,
            "margin": 0.0,
            "top5_mass": 0.0,
        }
    mean_posterior = np.mean(np.stack(rows).astype(np.float32), axis=0)
    order = np.argsort(mean_posterior)[::-1]
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    top5_indices = [ACTIONS.index(action) for action in ACTION_SETS["top5"] if action in ACTIONS]
    return {
        "top_action": ACTIONS[top],
        "top_confidence": float(mean_posterior[top]),
        "margin": float(mean_posterior[top] - mean_posterior[second]),
        "top5_mass": float(np.sum(mean_posterior[top5_indices])),
    }


def filter_event_confirmed_reps(
    reps: list[RepDetection],
    active_mask: np.ndarray,
    posterior_by_sample,
    action_active_prob_by_sample: np.ndarray,
    args,
) -> tuple[list[RepDetection], dict[str, object]]:
    if args.event_confirm_min_reps <= 0 or not reps:
        return reps, {"enabled": False, "input_reps": int(len(reps)), "kept_reps": int(len(reps)), "dropped_reps": 0}

    kept: list[RepDetection] = []
    event_rows: list[dict[str, object]] = []
    cooldown_until = -1
    ordered = sorted(reps, key=lambda rep: int(rep.start_idx))
    rep_groups: list[list[RepDetection]] = []
    current: list[RepDetection] = [ordered[0]]
    max_gap = int(args.event_confirm_gap_samples)
    split_inactive = int(args.event_confirm_split_inactive_samples)
    for rep in ordered[1:]:
        gap = int(rep.start_idx) - int(current[-1].end_idx)
        quiet_split = False
        if split_inactive > 0 and gap > 0:
            gap_mask = active_mask[int(current[-1].end_idx) : int(rep.start_idx)]
            quiet_split = longest_false_run(gap_mask) >= split_inactive
        if gap <= max_gap and not quiet_split:
            current.append(rep)
        else:
            rep_groups.append(current)
            current = [rep]
    rep_groups.append(current)

    for event_reps in rep_groups:
        rep_start = min(int(rep.start_idx) for rep in event_reps)
        rep_end = max(int(rep.end_idx) for rep in event_reps)
        active_overlaps = [(start, end) for start, end in mask_segments(active_mask) if end > rep_start and start < rep_end]
        start = min([rep_start, *[segment[0] for segment in active_overlaps]])
        end = max([rep_end, *[segment[1] for segment in active_overlaps]])
        action_active = action_active_prob_by_sample[start:end]
        if len(action_active) == 0:
            action_active = np.zeros((1,), dtype=np.float32)
        posterior_stats = _posterior_event_stats(posterior_by_sample, start, end)
        rep_rows = []
        evidence_reps: list[RepDetection] = []
        for rep in event_reps:
            r_start = max(0, int(rep.start_idx))
            r_end = max(r_start + 1, min(len(action_active_prob_by_sample), int(rep.end_idx)))
            rep_action_active = action_active_prob_by_sample[r_start:r_end]
            duration = int(rep.end_idx) - int(rep.start_idx)
            rep_action_max = float(np.max(rep_action_active)) if len(rep_action_active) else 0.0
            rep_action_mean = float(np.mean(rep_action_active)) if len(rep_action_active) else 0.0
            evidence_ok = True
            if args.event_confirm_rep_action_active_min_max > 0:
                evidence_ok = evidence_ok and rep_action_max >= float(args.event_confirm_rep_action_active_min_max)
            if args.event_confirm_rep_min_duration_samples > 0:
                evidence_ok = evidence_ok and duration >= int(args.event_confirm_rep_min_duration_samples)
            if args.event_confirm_rep_max_duration_samples > 0:
                evidence_ok = evidence_ok and duration <= int(args.event_confirm_rep_max_duration_samples)
            rep_rows.append(
                {
                    "start_idx": int(rep.start_idx),
                    "end_idx": int(rep.end_idx),
                    "duration_samples": int(duration),
                    "action_active_mean": rep_action_mean,
                    "action_active_max": rep_action_max,
                    "evidence_ok": bool(evidence_ok),
                }
            )
            if evidence_ok:
                evidence_reps.append(rep)
        active_threshold = float(args.event_confirm_action_active_threshold)
        event = {
            "start_idx": int(start),
            "end_idx": int(end),
            "reps": int(len(event_reps)),
            "evidence_reps": int(len(evidence_reps)),
            "action_active_mean": float(np.mean(action_active)),
            "action_active_max": float(np.max(action_active)),
            "action_active_fraction": float(np.mean(action_active >= active_threshold)),
            "rep_evidence": rep_rows,
            **posterior_stats,
        }
        confirmation_count = len(evidence_reps) if args.event_confirm_use_rep_evidence else len(event_reps)
        confirmed = confirmation_count >= int(args.event_confirm_min_reps)
        confirmed = confirmed and event["action_active_mean"] >= float(args.event_confirm_action_active_min_mean)
        confirmed = confirmed and event["action_active_max"] >= float(args.event_confirm_action_active_min_max)
        confirmed = confirmed and event["action_active_fraction"] >= float(args.event_confirm_action_active_min_fraction)
        confirmed = confirmed and event["top_confidence"] >= float(args.event_confirm_top_confidence_threshold)
        confirmed = confirmed and event["margin"] >= float(args.event_confirm_margin_threshold)
        confirmed = confirmed and event["top5_mass"] >= float(args.event_confirm_top5_mass_threshold)
        in_post_event_cooldown = int(start) < cooldown_until
        if in_post_event_cooldown:
            confirmed = False
        event["confirmed"] = bool(confirmed)
        event["in_post_event_cooldown"] = bool(in_post_event_cooldown)
        event_rows.append(event)
        if confirmed:
            kept.extend(evidence_reps if args.event_confirm_drop_low_evidence_reps else event_reps)
            if args.event_confirm_post_event_cooldown_samples > 0:
                cooldown_until = int(end) + int(args.event_confirm_post_event_cooldown_samples)

    return kept, {
        "enabled": True,
        "events": event_rows,
        "confirmed_events": int(sum(1 for event in event_rows if event["confirmed"])),
        "rejected_events": int(sum(1 for event in event_rows if not event["confirmed"])),
        "input_reps": int(len(reps)),
        "kept_reps": int(len(kept)),
        "dropped_reps": int(len(reps) - len(kept)),
    }


def parse_reps_masked(hard_labels: np.ndarray, active_mask: np.ndarray) -> list[RepDetection]:
    parser = OnlineRepParser()
    reps: list[RepDetection] = []
    for idx, label_idx in enumerate(hard_labels):
        label = None
        if bool(active_mask[idx]):
            label = "eccentric" if int(label_idx) == 0 else "concentric"
        reps.extend(parser.update(idx, label))
    reps.extend(parser.finish(len(hard_labels)))
    return reps


def clean_active_mask(mask: np.ndarray, min_active_samples: int, bridge_gap_samples: int) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool).copy()
    n = len(cleaned)
    bridge_gap = int(max(0, bridge_gap_samples))
    if bridge_gap > 0 and n > 0:
        i = 0
        while i < n:
            if cleaned[i]:
                i += 1
                continue
            start = i
            while i < n and not cleaned[i]:
                i += 1
            end = i
            if start > 0 and end < n and end - start <= bridge_gap:
                cleaned[start:end] = True

    min_active = int(max(0, min_active_samples))
    if min_active > 1 and n > 0:
        i = 0
        while i < n:
            if not cleaned[i]:
                i += 1
                continue
            start = i
            while i < n and cleaned[i]:
                i += 1
            end = i
            if end - start < min_active:
                cleaned[start:end] = False
    return cleaned


def aggregate_false_positive_rows(rows: list[dict[str, object]], variant_names: list[str], sample_rate_hz: float) -> dict[str, object]:
    total_samples = int(sum(int(row.get("samples", 0)) for row in rows))
    total_minutes = float(total_samples / max(float(sample_rate_hz), 1e-8) / 60.0)
    active_samples = int(sum(int(row.get("active_samples", 0)) for row in rows))
    out: dict[str, object] = {
        "streams": len(rows),
        "samples": total_samples,
        "duration_min": total_minutes,
        "active_samples": active_samples,
        "active_sample_rate": float(active_samples / max(1, total_samples)),
        "mean_active_probability": float(np.mean([float(row.get("mean_active_probability", 0.0)) for row in rows])) if rows else 0.0,
        "max_active_probability": float(max([float(row.get("max_active_probability", 0.0)) for row in rows], default=0.0)),
        "variants": {},
    }
    variants = out["variants"]
    assert isinstance(variants, dict)
    for name in variant_names:
        counts = [int(row.get("counts", {}).get(name, 0)) for row in rows]
        total = int(sum(counts))
        variants[name] = {
            "false_reps_total": total,
            "streams_with_false_reps": int(sum(1 for value in counts if value > 0)),
            "stream_false_rep_rate": float(sum(1 for value in counts if value > 0) / max(1, len(counts))),
            "false_reps_per_min": float(total / max(total_minutes, 1e-8)),
        }
    return out


def aggregate_appended_rest_rows(rows: list[dict[str, object]], variant_names: list[str], sample_rate_hz: float) -> dict[str, object]:
    total_rest_samples = int(sum(int(row.get("rest_samples", 0)) for row in rows))
    total_rest_minutes = float(total_rest_samples / max(float(sample_rate_hz), 1e-8) / 60.0)
    active_rest_samples = int(sum(int(row.get("active_rest_samples", 0)) for row in rows))
    out: dict[str, object] = {
        "streams": len(rows),
        "rest_samples": total_rest_samples,
        "rest_duration_min": total_rest_minutes,
        "active_rest_samples": active_rest_samples,
        "active_rest_sample_rate": float(active_rest_samples / max(1, total_rest_samples)),
        "variants": {},
    }
    variants = out["variants"]
    assert isinstance(variants, dict)
    for name in variant_names:
        overlaps = [int(row.get("rest_overlap_counts", {}).get(name, 0)) for row in rows]
        new_rest = [int(row.get("new_rest_rep_counts", {}).get(name, 0)) for row in rows]
        grace_rest = [int(row.get("post_grace_rest_rep_counts", {}).get(name, 0)) for row in rows]
        boundary = [int(row.get("boundary_crossing_rep_counts", {}).get(name, 0)) for row in rows]
        total = int(sum(overlaps))
        new_total = int(sum(new_rest))
        grace_total = int(sum(grace_rest))
        boundary_total = int(sum(boundary))
        variants[name] = {
            "rest_overlap_reps_total": total,
            "streams_with_rest_overlap_reps": int(sum(1 for value in overlaps if value > 0)),
            "stream_rest_overlap_rate": float(sum(1 for value in overlaps if value > 0) / max(1, len(overlaps))),
            "rest_overlap_reps_per_min": float(total / max(total_rest_minutes, 1e-8)),
            "new_rest_reps_total": new_total,
            "streams_with_new_rest_reps": int(sum(1 for value in new_rest if value > 0)),
            "new_rest_reps_per_min": float(new_total / max(total_rest_minutes, 1e-8)),
            "post_grace_rest_reps_total": grace_total,
            "streams_with_post_grace_rest_reps": int(sum(1 for value in grace_rest if value > 0)),
            "post_grace_rest_reps_per_min": float(grace_total / max(total_rest_minutes, 1e-8)),
            "boundary_crossing_reps_total": boundary_total,
            "streams_with_boundary_crossing_reps": int(sum(1 for value in boundary if value > 0)),
        }
    return out


def simulate_stream(
    stream_id,
    df,
    cfg,
    model,
    phase_mean,
    phase_std,
    active_model,
    active_scaler,
    action_active_rf,
    action_rf,
    action_mean,
    action_std,
    duration_priors,
    args,
    device,
    periodic_active_bundle=None,
):
    values = df[list(cfg.imu_columns)].to_numpy(dtype=np.float32)
    n = len(values)
    phase_probs = np.ones((n, 2), dtype=np.float32) * 0.5
    phase_raw_probs = np.ones((n, 2), dtype=np.float32) * 0.5
    posterior_by_sample: list[np.ndarray | None] = [None] * n
    active_prob_by_sample = np.zeros(n, dtype=np.float32)
    action_active_prob_by_sample = np.zeros(n, dtype=np.float32)
    active_state_by_sample = np.zeros(n, dtype=bool)
    raw_parser = OnlineRepParser()
    soft_merger = OnlineSoftMerger(args.max_gap_samples)
    raw_reps: list[RepDetection] = []
    smooth_buffer: list[np.ndarray] = []
    action_running = np.zeros((len(ACTIONS),), dtype=np.float64)
    action_weight = 0.0
    posterior: np.ndarray | None = None
    last_soft_meta: dict[str, object] = {"soft_enabled": False}
    events: list[dict[str, object]] = []
    phase_windows: list[np.ndarray] = []

    model.eval()
    if periodic_active_bundle is None:
        active_ends = sorted(set([1, n, *range(cfg.active_stride, n + 1, cfg.active_stride)]))
        active_windows = np.stack([trailing_window(values, end, cfg.active_window_size) for end in active_ends]).astype(np.float32)
        active_features = _extract_window_features_batch(active_windows)
        active_raw_probs = active_model.predict_proba(active_scaler.transform(active_features))
        active_idx = list(active_model.classes_).index(1) if 1 in active_model.classes_ else 0
        active_prob_by_model = np.zeros(n, dtype=np.float32)
        prev_end = 0
        for i, end in enumerate(active_ends):
            active_prob_by_model[prev_end:int(end)] = float(active_raw_probs[i, active_idx])
            prev_end = int(end)
        if prev_end < n:
            active_prob_by_model[prev_end:] = active_prob_by_model[prev_end - 1] if prev_end > 0 else 0.0
        active_state_by_model = None
    else:
        periodic_clf, periodic_scaler, periodic_mode = periodic_active_bundle
        active_prob_by_model = predict_periodic_active_prob(df, cfg.imu_columns, periodic_clf, periodic_scaler, args, periodic_mode)
        active_state_by_model = periodic_active_state_machine(active_prob_by_model, args)

    action_ends = list(range(args.action_window_samples, n + 1, args.action_stride_samples))
    if n >= args.action_window_samples and (not action_ends or action_ends[-1] != n):
        action_ends.append(n)
    action_active_probs = []
    action_prob_rows = []
    if action_ends:
        action_windows = np.stack([trailing_window(values, end, args.action_window_samples) for end in action_ends]).astype(np.float32)
        action_windows = (action_windows - action_mean) / action_std
        action_features = extract_features_batch(action_windows)
        action_active_raw = action_active_rf.predict_proba(action_features)
        active_class_to_col = {int(cls): idx for idx, cls in enumerate(action_active_rf.classes_)}
        active_col = active_class_to_col.get(1, 0)
        action_active_probs = [float(p) for p in action_active_raw[:, active_col]]
        action_prob_rows = [row for row in _probs_for_actions(action_rf, action_rf.predict_proba(action_features))]

    phase_ends = sorted(set([1, n, *range(args.phase_step_samples, n + 1, args.phase_step_samples)]))
    action_cursor = 0
    active_prob = float(active_prob_by_model[0]) if n else 0.0
    action_active_prob = 0.0
    active_enter_threshold = float(args.active_enter_threshold if args.active_enter_threshold is not None else args.active_threshold)
    active_exit_threshold = float(args.active_exit_threshold if args.active_exit_threshold is not None else args.active_threshold)
    active_exit_hold_events = max(1, int(np.ceil(float(args.active_exit_hold_samples) / max(1, int(args.phase_step_samples)))))
    active_state = False
    active_below_exit = 0
    for end in phase_ends:
        t = end - 1
        active_prob = float(active_prob_by_model[t]) if n else 0.0
        while action_cursor < len(action_ends) and action_ends[action_cursor] <= end:
            action_active_prob = float(action_active_probs[action_cursor])
            weight = max(action_active_prob, 1e-6)
            action_running += np.asarray(action_prob_rows[action_cursor], dtype=np.float64) * weight
            action_weight += weight
            posterior = action_running / max(action_weight, 1e-8)
            action_cursor += 1
        gate_ok = args.action_active_gate_threshold <= 0 or action_active_prob >= float(args.action_active_gate_threshold)
        if not active_state and active_prob >= active_enter_threshold:
            active_state = True
            active_below_exit = 0
        elif active_state:
            if active_prob < active_exit_threshold:
                active_below_exit += 1
                if active_below_exit >= active_exit_hold_events:
                    active_state = False
                    active_below_exit = 0
            else:
                active_below_exit = 0
        if active_state_by_model is not None:
            active_state = bool(active_state_by_model[t])
        event = {
            "t": t,
            "active": bool(active_state and gate_ok),
            "active_prob": float(active_prob),
            "action_active_prob": float(action_active_prob),
            "posterior": None if posterior is None else posterior.copy(),
            "phase_index": None,
        }
        if event["active"]:
            phase_window = trailing_window((values - phase_mean) / phase_std, end, args.phase_window_samples)
            event["phase_index"] = len(phase_windows)
            phase_windows.append(phase_window)
        events.append(event)

    phase_event_probs: list[np.ndarray] = []
    if phase_windows:
        batch = np.stack(phase_windows).astype(np.float32)
        with torch.no_grad():
            for start in range(0, len(batch), args.phase_batch_size):
                x_np = batch[start : start + args.phase_batch_size]
                x = torch.from_numpy(x_np).float().transpose(1, 2).to(device)
                probs = F.softmax(model(x), dim=1).cpu().numpy()[:, :, -1].astype(np.float32)
                phase_event_probs.extend([p for p in probs])

    prev_t = -1
    for event in events:
        t = int(event["t"])
        if event["active"]:
            probs = phase_event_probs[int(event["phase_index"])]
            smooth_buffer.append(probs)
            smooth_buffer = smooth_buffer[-args.phase_smoothing_window :]
            smoothed = np.mean(np.stack(smooth_buffer), axis=0).astype(np.float32)
            label_idx = int(np.argmax(smoothed))
            label = "eccentric" if label_idx == 0 else "concentric"
            fill_prob = smoothed
        else:
            smooth_buffer.clear()
            label = None
            fill_prob = np.asarray([0.5, 0.5], dtype=np.float32)

        start_fill = max(0, prev_t + 1)
        if event["active"]:
            raw_fill_prob = phase_event_probs[int(event["phase_index"])]
        else:
            raw_fill_prob = np.asarray([0.5, 0.5], dtype=np.float32)
        phase_raw_probs[start_fill : t + 1] = raw_fill_prob
        phase_probs[start_fill : t + 1] = fill_prob
        active_prob_by_sample[start_fill : t + 1] = float(active_prob)
        action_active_prob_by_sample[start_fill : t + 1] = float(action_active_prob)
        active_state_by_sample[start_fill : t + 1] = bool(event["active"])
        for idx in range(start_fill, t + 1):
            posterior_by_sample[idx] = event["posterior"]
        for idx in range(start_fill, t + 1):
            for rep in raw_parser.update(idx, label):
                raw_reps.append(rep)
                threshold, last_soft_meta = soft_threshold_from_context(event["posterior"], duration_priors, args)
                soft_merger.add(rep, threshold)
        prev_t = t

    for rep in raw_parser.finish(n):
        raw_reps.append(rep)
        threshold, last_soft_meta = soft_threshold_from_context(posterior, duration_priors, args)
        soft_merger.add(rep, threshold)
    soft_reps = soft_merger.finish()

    fixed_input = smooth_ma(phase_raw_probs, args.phase_smoothing_window)
    fixed_probs = fixed_lag_viterbi_decode(fixed_input, args.viterbi_penalty, args.fixed_lag_samples)
    active_mask = clean_active_mask(active_state_by_sample, args.min_active_segment_samples, args.active_mask_bridge_samples)
    if args.fixed_lag_active_mask:
        fixed_probs[~active_mask] = 0.5
        fixed_reps = parse_reps_masked(np.argmax(fixed_probs, axis=1), active_mask)
    else:
        fixed_reps = parse_reps_masked(np.argmax(fixed_probs, axis=1), np.ones(len(fixed_probs), dtype=bool))
    fixed_reps_before_event = list(fixed_reps)
    fixed_reps, event_confirm_meta = filter_event_confirmed_reps(
        fixed_reps,
        active_mask,
        posterior_by_sample,
        action_active_prob_by_sample,
        args,
    )
    fixed_reps = filter_confirmed_rep_groups(fixed_reps, args.min_confirmed_reps, args.confirmed_set_gap_samples)
    fixed_reps_after_event = list(fixed_reps)
    fixed_soft: dict[str, tuple[list[RepDetection], dict[str, object]]] = {}
    for scale in args.merge_threshold_scales:
        fixed_soft[str(scale)] = apply_online_soft_merge(fixed_reps, posterior_by_sample, duration_priors, args, scale)
    diagnostics = {
        "active_samples": int(np.sum(active_mask)),
        "mean_active_probability": float(np.mean(active_prob_by_sample)) if n else 0.0,
        "max_active_probability": float(np.max(active_prob_by_sample)) if n else 0.0,
        "mean_action_active_probability": float(np.mean(action_active_prob_by_sample)) if n else 0.0,
        "max_action_active_probability": float(np.max(action_active_prob_by_sample)) if n else 0.0,
        "active_prob_by_sample": active_prob_by_sample,
        "action_active_prob_by_sample": action_active_prob_by_sample,
        "active_state_by_sample": active_mask,
        "fixed_reps_before_event": fixed_reps_before_event,
        "fixed_reps_after_event": fixed_reps_after_event,
        "event_confirmation": event_confirm_meta,
    }
    return phase_probs, raw_reps, soft_reps, fixed_probs, fixed_reps, fixed_soft, posterior, last_soft_meta, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="True streaming-style global-active + soft top5 replay.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/action_recognition/realtime_soft_top5/summary_e5_h64.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--action-window-samples", type=int, default=200)
    parser.add_argument("--action-stride-samples", type=int, default=100)
    parser.add_argument("--window-active-threshold", type=float, default=0.5)
    parser.add_argument("--action-n-estimators", type=int, default=50)
    parser.add_argument("--action-max-depth", type=int, default=12)
    parser.add_argument("--action-min-samples-leaf", type=int, default=2)
    parser.add_argument("--action-active-gate-threshold", type=float, default=0.0, help="Optional second active gate from the action RF active head; <=0 disables it.")
    parser.add_argument("--active-gate-features", choices=["legacy", "basic", "periodic"], default="legacy")
    parser.add_argument("--label-mode", choices=["binary", "tri_motion"], default="binary")
    parser.add_argument("--transition-energy-quantile", type=float, default=0.7)
    parser.add_argument("--active-gate-window-samples", type=int, default=200)
    parser.add_argument("--active-gate-stride-samples", type=int, default=50)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    parser.add_argument("--active-enter-threshold", type=float, default=None)
    parser.add_argument("--active-exit-threshold", type=float, default=None)
    parser.add_argument("--active-exit-hold-samples", type=int, default=0)
    parser.add_argument("--min-active-segment-samples", type=int, default=0)
    parser.add_argument("--active-mask-bridge-samples", type=int, default=0)
    parser.add_argument("--phase-window-samples", type=int, default=300)
    parser.add_argument("--phase-step-samples", type=int, default=10)
    parser.add_argument("--phase-batch-size", type=int, default=256)
    parser.add_argument("--phase-smoothing-window", type=int, default=25)
    parser.add_argument("--fixed-lag-samples", type=int, default=100)
    parser.add_argument("--fixed-lag-active-mask", action="store_true", help="Use active state as no-phase mask for fixed-lag parsing; needed for full-session rest checks.")
    parser.add_argument("--viterbi-penalty", type=float, default=0.3)
    parser.add_argument("--merge-threshold-scales", default="0.8,1.0,1.2")
    parser.add_argument("--soft-top5-mass-threshold", type=float, default=0.65)
    parser.add_argument("--soft-action-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--soft-margin-threshold", type=float, default=0.05)
    parser.add_argument("--max-gap-samples", type=int, default=50)
    parser.add_argument("--min-confirmed-reps", type=int, default=0)
    parser.add_argument("--confirmed-set-gap-samples", type=int, default=300)
    parser.add_argument("--event-confirm-min-reps", type=int, default=0, help="Enable event-level confirmation when >0; buffered fixed-lag reps are released only for confirmed active events.")
    parser.add_argument("--event-confirm-gap-samples", type=int, default=300)
    parser.add_argument("--event-confirm-split-inactive-samples", type=int, default=0)
    parser.add_argument("--event-confirm-post-event-cooldown-samples", type=int, default=0)
    parser.add_argument("--event-confirm-action-active-threshold", type=float, default=0.5)
    parser.add_argument("--event-confirm-action-active-min-fraction", type=float, default=0.0)
    parser.add_argument("--event-confirm-action-active-min-mean", type=float, default=0.0)
    parser.add_argument("--event-confirm-action-active-min-max", type=float, default=0.0)
    parser.add_argument("--event-confirm-top-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--event-confirm-margin-threshold", type=float, default=0.0)
    parser.add_argument("--event-confirm-top5-mass-threshold", type=float, default=0.0)
    parser.add_argument("--event-confirm-use-rep-evidence", action="store_true")
    parser.add_argument("--event-confirm-drop-low-evidence-reps", action="store_true")
    parser.add_argument("--event-confirm-rep-action-active-min-max", type=float, default=0.0)
    parser.add_argument("--event-confirm-rep-min-duration-samples", type=int, default=0)
    parser.add_argument("--event-confirm-rep-max-duration-samples", type=int, default=0)
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument("--rest-tail-seconds", type=float, default=20.0)
    parser.add_argument("--rest-tail-grace-samples", type=int, default=200, help="Evaluation-only grace after appended set end before counting new rest reps as post-grace false reps.")
    parser.add_argument("--max-rest-streams-per-fold", type=int, default=0, help="0 means evaluate all held-out rest streams.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--only-subjects", default="", help="Comma-separated held-out subjects to run; empty means all subjects/max-folds.")
    parser.add_argument("--store-debug-reps", action="store_true")
    args = parser.parse_args()
    args.merge_threshold_scales = [float(x.strip()) for x in str(args.merge_threshold_scales).split(",") if x.strip()]
    args.window_samples = int(args.active_gate_window_samples)
    args.stride_samples = int(args.active_gate_stride_samples)
    args.window_active_fraction = float(args.window_active_threshold)
    args.n_estimators = int(args.action_n_estimators)
    args.max_depth = int(args.action_max_depth)
    args.min_samples_leaf = int(args.action_min_samples_leaf)
    args.enter_threshold = float(args.active_enter_threshold if args.active_enter_threshold is not None else args.active_threshold)
    args.exit_threshold = float(args.active_exit_threshold if args.active_exit_threshold is not None else args.active_threshold)
    args.enter_hold_samples = max(1, int(args.phase_step_samples))
    args.exit_hold_samples = int(args.active_exit_hold_samples)
    args.min_active_samples = int(args.min_active_segment_samples)
    args.bridge_gap_samples = int(args.active_mask_bridge_samples)
    args.cooldown_samples = 0

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    all_streams, _subjects, _actions = _load_streams(raw_cfg, ["sets"])
    set_streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    rest_streams = load_non_action_streams(raw_cfg)
    subjects = sorted({stream_subject(sid) for sid, _ in set_streams})
    only_subjects = [subject.strip() for subject in str(args.only_subjects).split(",") if subject.strip()]
    eval_subjects = [subject for subject in subjects if subject in set(only_subjects)] if only_subjects else subjects
    eval_subjects = eval_subjects[: args.max_folds] if args.max_folds and args.max_folds > 0 else eval_subjects

    print(f"streams={len(set_streams)} rest={len(rest_streams)} subjects={subjects} device={device}", flush=True)
    print(f"realtime epochs={args.epochs} hidden={args.hidden} phase_step={args.phase_step_samples}", flush=True)

    raw_results = []
    soft_results = []
    fixed_results = []
    fixed_soft_results = {str(scale): [] for scale in args.merge_threshold_scales}
    fixed_streams = []
    fixed_soft_streams = {str(scale): [] for scale in args.merge_threshold_scales}
    rest_false_rows = []
    appended_rest_rows = []
    variant_names = ["raw_online", "soft_online", "fixed_lag_raw", *[f"fixed_lag_soft_x{scale}" for scale in args.merge_threshold_scales]]
    folds = []
    data_dir = Path(raw_cfg.get("data", {}).get("data_dir", "datasets/raw_data"))
    for fold_idx, test_subject in enumerate(eval_subjects, start=1):
        print(f"\nFold {fold_idx}/{len(eval_subjects)} test={test_subject}", flush=True)
        train_set_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in set_streams if stream_subject(sid) == test_subject]
        train_rest_streams = [(sid, df) for sid, df in rest_streams if stream_subject(sid) != test_subject]
        test_rest_streams = [(sid, df) for sid, df in rest_streams if stream_subject(sid) == test_subject]
        if args.max_rest_streams_per_fold and args.max_rest_streams_per_fold > 0:
            test_rest_streams = test_rest_streams[: args.max_rest_streams_per_fold]

        action_active_rf, action_rf, train_action_streams = train_action_rf(train_set_streams, train_rest_streams, cfg.imu_columns, args)
        action_mean, action_std = compute_action_norm(train_action_streams, cfg.imu_columns)
        if args.active_gate_features == "legacy":
            active_model, active_scaler = train_global_active_detector(train_set_streams, train_rest_streams, cfg, args)
            periodic_active_bundle = None
        else:
            active_model, active_scaler = None, None
            periodic_clf, periodic_scaler, _periodic_info = train_periodic_active_gate(
                [*train_set_streams, *train_rest_streams], cfg.imu_columns, args, args.active_gate_features
            )
            periodic_active_bundle = (periodic_clf, periodic_scaler, args.active_gate_features)
        duration_priors = build_duration_priors(train_set_streams, [5])
        model, phase_mean, phase_std, n_segments = train_raw6_model(train_set_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"  trained phase segments={n_segments}", flush=True)

        fold_raw = []
        fold_soft = []
        fold_fixed = []
        fold_fixed_soft = {str(scale): [] for scale in args.merge_threshold_scales}
        fold_rest_false_rows = []
        fold_appended_rest_rows = []
        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            phase_probs, raw_reps, soft_reps, fixed_probs, fixed_reps, fixed_soft, posterior, soft_meta, diagnostics = simulate_stream(
                stream_id,
                df,
                cfg,
                model,
                phase_mean,
                phase_std,
                active_model,
                active_scaler,
                action_active_rf,
                action_rf,
                action_mean,
                action_std,
                duration_priors,
                args,
                device,
                periodic_active_bundle,
            )
            raw = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, raw_reps)
            soft = evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, soft_reps)
            fixed = evaluate_with_reps(stream_id, fixed_probs, gt_reps, gt_phases, fixed_reps)
            active_summary = active_mask_summary(
                diagnostics["active_state_by_sample"], diagnostics["active_prob_by_sample"], gt_phases
            )
            event_summary = compact_event_confirmation(diagnostics["event_confirmation"])
            row_diagnostics = {
                "active_mask": active_summary,
                "event_confirmation": event_summary,
                "fixed_reps_before_event": int(len(diagnostics["fixed_reps_before_event"])),
                "fixed_reps_after_event": int(len(diagnostics["fixed_reps_after_event"])),
            }
            soft.update({"soft_meta": soft_meta, "posterior": posterior.tolist() if posterior is not None else None})
            fixed.update({"diagnostics": row_diagnostics})
            raw_results.append(raw)
            soft_results.append(soft)
            fixed_results.append(fixed)
            fixed_streams.append(fixed)
            fold_raw.append(raw)
            fold_soft.append(soft)
            fold_fixed.append(fixed)
            for scale_key, (scaled_reps, scaled_meta) in fixed_soft.items():
                scaled = evaluate_with_reps(stream_id, fixed_probs, gt_reps, gt_phases, scaled_reps)
                scaled.update(
                    {
                        "soft_meta": scaled_meta,
                        "posterior": posterior.tolist() if posterior is not None else None,
                        "diagnostics": row_diagnostics,
                    }
                )
                if args.store_debug_reps:
                    scaled["debug"] = {
                        "active_segments": segment_debug_rows(np.asarray(diagnostics["active_state_by_sample"], dtype=bool)),
                        "fixed_reps_before_event": rep_debug_rows(diagnostics["fixed_reps_before_event"]),
                        "fixed_reps_after_event": rep_debug_rows(diagnostics["fixed_reps_after_event"]),
                        "fixed_lag_soft_reps": rep_debug_rows(scaled_reps),
                        "event_confirmation": diagnostics["event_confirmation"],
                    }
                fixed_soft_results[scale_key].append(scaled)
                fixed_soft_streams[scale_key].append(scaled)
                fold_fixed_soft[scale_key].append(scaled)

            combined_df, set_len, rest_len = append_rest_tail(stream_id, df, data_dir, args.rest_tail_seconds, args.sample_rate_hz, cfg.imu_columns)
            if rest_len > 0:
                (
                    combined_phase_probs,
                    combined_raw_reps,
                    combined_soft_reps,
                    combined_fixed_probs,
                    combined_fixed_reps,
                    combined_fixed_soft,
                    _combined_posterior,
                    _combined_soft_meta,
                    combined_diagnostics,
                ) = simulate_stream(
                    f"{stream_id}+rest_tail",
                    combined_df,
                    cfg,
                    model,
                    phase_mean,
                    phase_std,
                    active_model,
                    active_scaler,
                    action_active_rf,
                    action_rf,
                    action_mean,
                    action_std,
                    duration_priors,
                    args,
                    device,
                    periodic_active_bundle,
                )
                appended_counts = {
                    "raw_online": reps_overlapping_rest(combined_raw_reps, set_len),
                    "soft_online": reps_overlapping_rest(combined_soft_reps, set_len),
                    "fixed_lag_raw": reps_overlapping_rest(combined_fixed_reps, set_len),
                }
                appended_new_rest_counts = {
                    "raw_online": reps_starting_in_rest(combined_raw_reps, set_len),
                    "soft_online": reps_starting_in_rest(combined_soft_reps, set_len),
                    "fixed_lag_raw": reps_starting_in_rest(combined_fixed_reps, set_len),
                }
                appended_post_grace_counts = {
                    "raw_online": reps_starting_after_rest_grace(combined_raw_reps, set_len, args.rest_tail_grace_samples),
                    "soft_online": reps_starting_after_rest_grace(combined_soft_reps, set_len, args.rest_tail_grace_samples),
                    "fixed_lag_raw": reps_starting_after_rest_grace(combined_fixed_reps, set_len, args.rest_tail_grace_samples),
                }
                appended_boundary_counts = {
                    "raw_online": reps_crossing_rest_boundary(combined_raw_reps, set_len),
                    "soft_online": reps_crossing_rest_boundary(combined_soft_reps, set_len),
                    "fixed_lag_raw": reps_crossing_rest_boundary(combined_fixed_reps, set_len),
                }
                for scale_key, (scaled_reps, _scaled_meta) in combined_fixed_soft.items():
                    appended_counts[f"fixed_lag_soft_x{scale_key}"] = reps_overlapping_rest(scaled_reps, set_len)
                    appended_new_rest_counts[f"fixed_lag_soft_x{scale_key}"] = reps_starting_in_rest(scaled_reps, set_len)
                    appended_post_grace_counts[f"fixed_lag_soft_x{scale_key}"] = reps_starting_after_rest_grace(
                        scaled_reps, set_len, args.rest_tail_grace_samples
                    )
                    appended_boundary_counts[f"fixed_lag_soft_x{scale_key}"] = reps_crossing_rest_boundary(scaled_reps, set_len)
                active_states = np.asarray(combined_diagnostics["active_state_by_sample"], dtype=bool)
                active_rest = int(np.sum(active_states[set_len:]))
                appended_row = {
                    "stream_id": stream_id,
                    "samples": int(len(combined_df)),
                    "set_samples": int(set_len),
                    "rest_samples": int(rest_len),
                    "active_rest_samples": int(active_rest),
                    "rest_overlap_counts": appended_counts,
                    "new_rest_rep_counts": appended_new_rest_counts,
                    "post_grace_rest_rep_counts": appended_post_grace_counts,
                    "boundary_crossing_rep_counts": appended_boundary_counts,
                }
                if args.store_debug_reps:
                    appended_row["debug"] = {
                        "active_rest_segments": segment_debug_rows(active_states[set_len:]),
                        "fixed_lag_raw_rest_reps": rep_debug_rows(
                            [rep for rep in combined_fixed_reps if int(rep.end_idx) > int(set_len)], set_len
                        ),
                        "fixed_lag_soft_x0.8_rest_reps": rep_debug_rows(
                            [rep for rep in combined_fixed_soft.get("0.8", ([], {}))[0] if int(rep.end_idx) > int(set_len)], set_len
                        ),
                    }
                appended_rest_rows.append(appended_row)
                fold_appended_rest_rows.append(appended_row)

        for stream_id, df in test_rest_streams:
            phase_probs, raw_reps, soft_reps, fixed_probs, fixed_reps, fixed_soft, posterior, soft_meta, diagnostics = simulate_stream(
                stream_id,
                df,
                cfg,
                model,
                phase_mean,
                phase_std,
                active_model,
                active_scaler,
                action_active_rf,
                action_rf,
                action_mean,
                action_std,
                duration_priors,
                args,
                device,
                periodic_active_bundle,
            )
            counts = {
                "raw_online": len(raw_reps),
                "soft_online": len(soft_reps),
                "fixed_lag_raw": len(fixed_reps),
            }
            for scale_key, (scaled_reps, _scaled_meta) in fixed_soft.items():
                counts[f"fixed_lag_soft_x{scale_key}"] = len(scaled_reps)
            rest_row = {
                "stream_id": stream_id,
                "samples": int(len(df)),
                "active_samples": int(diagnostics["active_samples"]),
                "mean_active_probability": float(diagnostics["mean_active_probability"]),
                "max_active_probability": float(diagnostics["max_active_probability"]),
                "counts": counts,
            }
            if args.store_debug_reps:
                active_states = np.asarray(diagnostics["active_state_by_sample"], dtype=bool)
                rest_row["debug"] = {
                    "active_segments": segment_debug_rows(active_states),
                    "fixed_lag_raw_reps": rep_debug_rows(fixed_reps),
                    "fixed_lag_soft_x0.8_reps": rep_debug_rows(fixed_soft.get("0.8", ([], {}))[0]),
                }
            rest_false_rows.append(rest_row)
            fold_rest_false_rows.append(rest_row)

        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "raw_online": aggregate_rich(fold_raw),
            "soft_online": aggregate_rich(fold_soft),
            "fixed_lag_raw": aggregate_rich(fold_fixed),
            "fixed_lag_soft": {scale: aggregate_rich(rows) for scale, rows in fold_fixed_soft.items()},
            "rest_false_positive_summary": aggregate_false_positive_rows(fold_rest_false_rows, variant_names, args.sample_rate_hz),
            "appended_rest_tail_summary": aggregate_appended_rest_rows(fold_appended_rest_rows, variant_names, args.sample_rate_hz),
        }
        folds.append(fold_summary)
        for name in ["raw_online", "soft_online", "fixed_lag_raw"]:
            agg = fold_summary[name]
            print(
                f"  {name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
                flush=True,
            )
        for scale, agg in fold_summary["fixed_lag_soft"].items():
            print(
                f"  fixed_lag_soft_x{scale}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
                flush=True,
            )
        rest_summary = fold_summary["rest_false_positive_summary"]
        appended_summary = fold_summary["appended_rest_tail_summary"]
        print(
            f"  rest-only active_rate={rest_summary['active_sample_rate']:.3f} fixed_lag_soft_x0.8_false_reps={rest_summary['variants'].get('fixed_lag_soft_x0.8', {}).get('false_reps_total', 0)}",
            flush=True,
        )
        print(
            f"  appended-rest active_rate={appended_summary['active_rest_sample_rate']:.3f} fixed_lag_soft_x0.8_rest_overlap_reps={appended_summary['variants'].get('fixed_lag_soft_x0.8', {}).get('rest_overlap_reps_total', 0)}",
            flush=True,
        )

    output = {
        "settings": vars(args),
        "excluded_sessions": EXCLUDED_SESSIONS,
        "raw_online_total": aggregate_rich(raw_results),
        "soft_online_total": aggregate_rich(soft_results),
        "fixed_lag_raw_total": aggregate_rich(fixed_results),
        "fixed_lag_soft_total": {scale: aggregate_rich(rows) for scale, rows in fixed_soft_results.items()},
        "rest_false_positive_summary": aggregate_false_positive_rows(rest_false_rows, variant_names, args.sample_rate_hz),
        "appended_rest_tail_summary": aggregate_appended_rest_rows(appended_rest_rows, variant_names, args.sample_rate_hz),
        "folds": folds,
        "soft_streams": soft_results,
        "fixed_lag_streams": fixed_streams,
        "fixed_lag_soft_streams": fixed_soft_streams,
        "rest_false_positive_rows": rest_false_rows,
        "appended_rest_tail_rows": appended_rest_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nTOTAL", flush=True)
    for name in ["raw_online_total", "soft_online_total", "fixed_lag_raw_total"]:
        agg = output[name]
        print(
            f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
            flush=True,
        )
    for scale, agg in output["fixed_lag_soft_total"].items():
        print(
            f"fixed_lag_soft_x{scale}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} MAE={agg['mean_abs_count_error']:.2f} PhaseIoU={agg['phase_seg_iou_f1_50_avg']:.4f} CE={agg['ce_ratio_mae']:.3f}",
            flush=True,
        )
    rest_total = output["rest_false_positive_summary"]
    appended_total = output["appended_rest_tail_summary"]
    print(
        f"rest_false_positive: streams={rest_total['streams']} active_rate={rest_total['active_sample_rate']:.3f} fixed_lag_soft_x0.8_false_reps={rest_total['variants'].get('fixed_lag_soft_x0.8', {}).get('false_reps_total', 0)}",
        flush=True,
    )
    print(
        f"appended_rest_tail: streams={appended_total['streams']} active_rate={appended_total['active_rest_sample_rate']:.3f} fixed_lag_soft_x0.8_rest_overlap_reps={appended_total['variants'].get('fixed_lag_soft_x0.8', {}).get('rest_overlap_reps_total', 0)}",
        flush=True,
    )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
