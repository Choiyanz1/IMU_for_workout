from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


MICRO_LABELS = ("other", "concentric", "eccentric")
OTHER_LABEL = "other"
CONCENTRIC_LABEL = "concentric"
ECCENTRIC_LABEL = "eccentric"


@dataclass
class SegmentRun:
    label: str
    start_idx: int
    end_idx: int
    confidence: float = 1.0


@dataclass
class RepDetection:
    start_idx: int
    transition_idx: int
    end_idx: int
    micro_source: str
    micro_confidence: float
    pred_action_type: str = "unknown"
    action_confidence: float = float("nan")


@dataclass
class PairingDiagnostic:
    reason: str
    label: str
    start_idx: int
    end_idx: int


def micro_labels_from_phase(phases: Sequence[object]) -> np.ndarray:
    labels = []
    for phase in phases:
        value = str(phase)
        if value == CONCENTRIC_LABEL:
            labels.append(CONCENTRIC_LABEL)
        elif value == ECCENTRIC_LABEL:
            labels.append(ECCENTRIC_LABEL)
        else:
            labels.append(OTHER_LABEL)
    return np.asarray(labels, dtype=object)


def macro_labels_from_action(actions: Sequence[object], active_micro: Sequence[object]) -> np.ndarray:
    out = []
    for action, micro in zip(actions, active_micro):
        out.append(str(action) if str(micro) != OTHER_LABEL else OTHER_LABEL)
    return np.asarray(out, dtype=object)


def labels_to_runs(
    labels: Sequence[object],
    positive_labels: Iterable[str] | None = None,
    probabilities: np.ndarray | None = None,
    min_length: int = 1,
) -> List[SegmentRun]:
    if positive_labels is None:
        positive = None
    else:
        positive = set(str(x) for x in positive_labels)
    values = [str(x) for x in labels]
    runs: List[SegmentRun] = []
    if not values:
        return runs

    start = 0
    cur = values[0]
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != cur:
            if (positive is None or cur in positive) and i - start >= int(min_length):
                conf = 1.0
                if probabilities is not None and cur in MICRO_LABELS:
                    cls_idx = MICRO_LABELS.index(cur)
                    conf = float(np.mean(probabilities[start:i, cls_idx]))
                runs.append(SegmentRun(cur, start, i, conf))
            if i < len(values):
                start = i
                cur = values[i]
    return runs


def pair_concentric_eccentric_reps(
    micro_runs: Sequence[SegmentRun],
    micro_source: str,
    max_gap_samples: int = 0,
) -> Tuple[List[RepDetection], List[PairingDiagnostic]]:
    """Pair fixed `concentric -> eccentric` runs into reps.

    The phase order is intentionally not configurable.
    """
    reps: List[RepDetection] = []
    diagnostics: List[PairingDiagnostic] = []
    active = [run for run in micro_runs if run.label in {CONCENTRIC_LABEL, ECCENTRIC_LABEL}]
    i = 0
    while i < len(active):
        run = active[i]
        if run.label != CONCENTRIC_LABEL:
            diagnostics.append(PairingDiagnostic("unexpected_phase_before_concentric", run.label, run.start_idx, run.end_idx))
            i += 1
            continue
        if i + 1 >= len(active):
            diagnostics.append(PairingDiagnostic("missing_eccentric_after_concentric", run.label, run.start_idx, run.end_idx))
            i += 1
            continue
        nxt = active[i + 1]
        gap = int(nxt.start_idx) - int(run.end_idx)
        if nxt.label != ECCENTRIC_LABEL:
            diagnostics.append(PairingDiagnostic("missing_eccentric_after_concentric", run.label, run.start_idx, run.end_idx))
            i += 1
            continue
        if gap > int(max_gap_samples):
            diagnostics.append(PairingDiagnostic("phase_gap_too_large", run.label, run.start_idx, nxt.end_idx))
            i += 1
            continue
        reps.append(
            RepDetection(
                start_idx=int(run.start_idx),
                transition_idx=int(nxt.start_idx),
                end_idx=int(nxt.end_idx),
                micro_source=micro_source,
                micro_confidence=float((run.confidence + nxt.confidence) / 2.0),
            )
        )
        i += 2
    return reps, diagnostics


def aggregate_action_for_reps(
    reps: Sequence[RepDetection],
    macro_probs: np.ndarray,
    macro_classes: Sequence[str],
) -> List[RepDetection]:
    out: List[RepDetection] = []
    classes = [str(c) for c in macro_classes]
    for rep in reps:
        start = max(0, int(rep.start_idx))
        end = min(len(macro_probs), int(rep.end_idx))
        if end <= start:
            out.append(rep)
            continue
        mean_prob = np.mean(macro_probs[start:end], axis=0)
        best_idx = int(np.argmax(mean_prob))
        label = classes[best_idx]
        if label == OTHER_LABEL:
            label = "uncertain"
        out.append(
            RepDetection(
                start_idx=rep.start_idx,
                transition_idx=rep.transition_idx,
                end_idx=rep.end_idx,
                micro_source=rep.micro_source,
                micro_confidence=rep.micro_confidence,
                pred_action_type=label,
                action_confidence=float(mean_prob[best_idx]),
            )
        )
    return out


def reps_to_rows(stream_id: str, reps: Sequence[RepDetection]) -> List[Dict[str, object]]:
    return [
        {
            "stream_id": stream_id,
            "rep_idx": idx,
            "start_idx": rep.start_idx,
            "transition_idx": rep.transition_idx,
            "end_idx": rep.end_idx,
            "micro_source": rep.micro_source,
            "micro_confidence": rep.micro_confidence,
            "pred_action_type": rep.pred_action_type,
            "action_confidence": rep.action_confidence,
        }
        for idx, rep in enumerate(reps)
    ]


def diagnostics_to_rows(stream_id: str, diagnostics: Sequence[PairingDiagnostic]) -> List[Dict[str, object]]:
    return [
        {
            "stream_id": stream_id,
            "reason": item.reason,
            "label": item.label,
            "start_idx": item.start_idx,
            "end_idx": item.end_idx,
        }
        for item in diagnostics
    ]


def segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter) / float(union) if union > 0 else 0.0


def match_segments(
    predicted: Sequence[Tuple[int, int]],
    truth: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    pairs = []
    for pi, pred in enumerate(predicted):
        for ti, true in enumerate(truth):
            iou = segment_iou(pred, true)
            if iou >= iou_threshold:
                pairs.append((pi, ti, iou))
    out = []
    used_p, used_t = set(), set()
    for pi, ti, iou in sorted(pairs, key=lambda x: x[2], reverse=True):
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        out.append((pi, ti, iou))
    return out


def rep_metrics(
    predicted: Sequence[RepDetection],
    truth_reps: Sequence[RepDetection],
    sample_rate_hz: float,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    pred_seg = [(r.start_idx, r.end_idx) for r in predicted]
    true_seg = [(r.start_idx, r.end_idx) for r in truth_reps]
    matches = match_segments(pred_seg, true_seg, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = max(0, len(pred_seg) - tp)
    fn = max(0, len(true_seg) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    start_err, end_err, transition_err, action_ok = [], [], [], []
    for pi, ti, _ in matches:
        p = predicted[pi]
        t = truth_reps[ti]
        start_err.append(abs(p.start_idx - t.start_idx) / sample_rate_hz * 1000.0)
        end_err.append(abs(p.end_idx - t.end_idx) / sample_rate_hz * 1000.0)
        transition_err.append(abs(p.transition_idx - t.transition_idx) / sample_rate_hz * 1000.0)
        if t.pred_action_type not in {"unknown", "uncertain"}:
            action_ok.append(1.0 if p.pred_action_type == t.pred_action_type else 0.0)

    def mean_or_nan(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_pred": float(len(pred_seg)),
        "n_true": float(len(true_seg)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "start_mae_ms": mean_or_nan(start_err),
        "end_mae_ms": mean_or_nan(end_err),
        "transition_mae_ms": mean_or_nan(transition_err),
        "rep_action_accuracy": mean_or_nan(action_ok),
    }


def truth_reps_from_labels(
    phases: Sequence[object],
    actions: Sequence[object] | None = None,
    min_phase_samples: int = 1,
) -> List[RepDetection]:
    micro = micro_labels_from_phase(phases)
    runs = labels_to_runs(
        micro,
        positive_labels=(CONCENTRIC_LABEL, ECCENTRIC_LABEL),
        min_length=min_phase_samples,
    )
    reps, _ = pair_concentric_eccentric_reps(runs, micro_source="ground_truth")
    if actions is None:
        return reps
    action_values = np.asarray([str(x) for x in actions], dtype=object)
    out = []
    for rep in reps:
        segment = action_values[rep.start_idx:rep.end_idx]
        if len(segment) == 0:
            label = "unknown"
        else:
            labels, counts = np.unique(segment, return_counts=True)
            label = str(labels[int(np.argmax(counts))])
        rep.pred_action_type = label
        out.append(rep)
    return out


def _magnitude(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.zeros(len(df), dtype=np.float64)
    x = df.loc[:, available].to_numpy(dtype=np.float64)
    return np.sqrt(np.sum(x * x, axis=1))


def _polyline(values: np.ndarray, x0: float, y0: float, width: float, height: float) -> str:
    if len(values) == 0:
        return ""
    xs = x0 + np.arange(len(values)) / max(1, len(values) - 1) * width
    lo = float(np.nanpercentile(values, 2))
    hi = float(np.nanpercentile(values, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1.0)
    clipped = np.clip(values, lo, hi)
    ys = y0 + height - ((clipped - lo) / max(hi - lo, 1e-8) * height)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _rects(segments: Sequence[Tuple[int, int, str]], n: int, x0: float, width: float, y: float, h: float) -> str:
    colors = {
        CONCENTRIC_LABEL: "#22c55e",
        ECCENTRIC_LABEL: "#f97316",
        OTHER_LABEL: "#94a3b8",
    }
    parts = []
    for start, end, label in segments:
        x = x0 + start / max(1, n) * width
        w = max(1.0, (end - start) / max(1, n) * width)
        fill = colors.get(label, "#38bdf8")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="0.65"/>')
    return "\n".join(parts)


def write_micro_macro_svg(
    path: Path,
    stream_id: str,
    df: pd.DataFrame,
    gt_micro_runs: Sequence[SegmentRun],
    pred_micro_runs: Sequence[SegmentRun],
    gt_reps: Sequence[RepDetection],
    pred_reps: Sequence[RepDetection],
    macro_runs: Sequence[SegmentRun],
    sample_rate_hz: float,
) -> None:
    width, height = 1280, 760
    x0, plot_w = 80, 1120
    n = len(df)
    acc = _magnitude(df, ["ax", "ay", "az"])
    gyro = _magnitude(df, ["gx", "gy", "gz"])
    acc_pts = _polyline(acc, x0, 230, plot_w, 170)
    gyro_pts = _polyline(gyro, x0, 480, plot_w, 170)

    gt_micro = _rects([(r.start_idx, r.end_idx, r.label) for r in gt_micro_runs], n, x0, plot_w, 95, 18)
    pred_micro = _rects([(r.start_idx, r.end_idx, r.label) for r in pred_micro_runs], n, x0, plot_w, 125, 18)
    macro_rects = _rects([(r.start_idx, r.end_idx, r.label) for r in macro_runs if r.label != OTHER_LABEL], n, x0, plot_w, 160, 18)

    def lines(reps: Sequence[RepDetection], color: str, dash: str) -> str:
        out = []
        for rep in reps:
            for idx, sw in ((rep.start_idx, 1.3), (rep.transition_idx, 2.0), (rep.end_idx, 1.3)):
                x = x0 + idx / max(1, n) * plot_w
                out.append(f'<line x1="{x:.1f}" y1="205" x2="{x:.1f}" y2="675" stroke="{color}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')
        return "\n".join(out)

    title = html.escape(stream_id)
    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .small {{ font-size: 12px; fill: #475569; }}
    .label {{ font-size: 14px; font-weight: 700; }}
  </style>
  <text x="40" y="36" font-size="20" font-weight="700">{title}</text>
  <text x="40" y="62" class="small">duration={duration_s:.1f}s, green=concentric, orange=eccentric, blue-ish=macro action</text>
  <text x="40" y="109" class="label">GT micro</text>
  <rect x="{x0}" y="95" width="{plot_w}" height="18" fill="#f8fafc" stroke="#cbd5e1"/>{gt_micro}
  <text x="40" y="139" class="label">Pred micro</text>
  <rect x="{x0}" y="125" width="{plot_w}" height="18" fill="#f8fafc" stroke="#cbd5e1"/>{pred_micro}
  <text x="40" y="174" class="label">Macro</text>
  <rect x="{x0}" y="160" width="{plot_w}" height="18" fill="#f8fafc" stroke="#cbd5e1"/>{macro_rects}
  {lines(gt_reps, "#16a34a", "5 4")}
  {lines(pred_reps, "#dc2626", "none")}
  <text x="40" y="250" class="label">acc_mag</text>
  <rect x="{x0}" y="230" width="{plot_w}" height="170" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{acc_pts}" fill="none" stroke="#2563eb" stroke-width="1.4"/>
  <text x="40" y="500" class="label">gyro_mag</text>
  <rect x="{x0}" y="480" width="{plot_w}" height="170" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{gyro_pts}" fill="none" stroke="#7c3aed" stroke-width="1.4"/>
  <text x="{x0}" y="710" class="small">GT rep lines dashed green; predicted rep lines red; thicker middle line is phase transition.</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
