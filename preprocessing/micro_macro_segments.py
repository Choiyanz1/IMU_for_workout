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


def sample_classification_metrics(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    labels: Sequence[str],
) -> Dict[str, float]:
    """Return sample-wise accuracy plus macro-F1 without requiring sklearn."""
    true_values = [str(x) for x in y_true]
    pred_values = [str(x) for x in y_pred]
    label_values = [str(x) for x in labels]
    n = min(len(true_values), len(pred_values))
    if n == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    true_values = true_values[:n]
    pred_values = pred_values[:n]
    accuracy = sum(t == p for t, p in zip(true_values, pred_values)) / float(n)
    f1s = []
    for label in label_values:
        tp = sum(t == label and p == label for t, p in zip(true_values, pred_values))
        fp = sum(t != label and p == label for t, p in zip(true_values, pred_values))
        fn = sum(t == label and p != label for t, p in zip(true_values, pred_values))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(accuracy), "macro_f1": float(np.mean(f1s) if f1s else float("nan"))}


def segment_iou_f1(
    true_runs: Sequence[SegmentRun],
    pred_runs: Sequence[SegmentRun],
    iou_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
) -> Dict[str, float]:
    """Segment-level IoU F1@k for temporal segmentation labels.

    A prediction can match a truth segment only when both label and IoU pass the
    threshold. Matching is greedy by IoU, matching the common MS-TCN evaluation.
    """
    out: Dict[str, float] = {}
    pairs = []
    for pi, pred in enumerate(pred_runs):
        for ti, true in enumerate(true_runs):
            if pred.label != true.label:
                continue
            iou = segment_iou((pred.start_idx, pred.end_idx), (true.start_idx, true.end_idx))
            pairs.append((pi, ti, iou))
    for threshold in iou_thresholds:
        used_p, used_t = set(), set()
        tp = 0
        for pi, ti, iou in sorted(pairs, key=lambda x: x[2], reverse=True):
            if iou < threshold or pi in used_p or ti in used_t:
                continue
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
        fp = max(0, len(pred_runs) - tp)
        fn = max(0, len(true_runs) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"f1_at_{int(round(threshold * 100)):02d}"] = float(f1)
    return out


def edit_score(true_runs: Sequence[SegmentRun], pred_runs: Sequence[SegmentRun]) -> float:
    """Normalized segmental edit score after collapsing repeated labels."""
    true_labels = [run.label for run in true_runs]
    pred_labels = [run.label for run in pred_runs]
    if not true_labels and not pred_labels:
        return 100.0
    if not true_labels or not pred_labels:
        return 0.0
    rows = len(true_labels) + 1
    cols = len(pred_labels) + 1
    dp = np.zeros((rows, cols), dtype=np.int32)
    dp[:, 0] = np.arange(rows)
    dp[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if true_labels[i - 1] == pred_labels[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    distance = int(dp[-1, -1])
    return float((1.0 - distance / max(len(true_labels), len(pred_labels))) * 100.0)


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


def _label_color(label: str) -> str:
    colors = {
        CONCENTRIC_LABEL: "#22c55e",
        ECCENTRIC_LABEL: "#f97316",
        OTHER_LABEL: "#94a3b8",
        "uncertain": "#64748b",
    }
    palette = (
        "#2563eb",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#ca8a04",
        "#db2777",
        "#16a34a",
        "#9333ea",
    )
    if label in colors:
        return colors[label]
    idx = sum(ord(ch) for ch in str(label)) % len(palette)
    return palette[idx]


def _rects(
    segments: Sequence[Tuple[int, int, str]],
    n: int,
    x0: float,
    width: float,
    y: float,
    h: float,
    opacity: float = 0.65,
) -> str:
    parts = []
    for start, end, label in segments:
        x = x0 + start / max(1, n) * width
        w = max(1.0, (end - start) / max(1, n) * width)
        fill = _label_color(str(label))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" fill-opacity="{opacity:.2f}"/>')
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
    .tiny {{ font-size: 10px; fill: #111827; }}
    .label {{ font-size: 14px; font-weight: 700; }}
    .action-label {{ paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }}
  </style>
  <text x="40" y="36" font-size="20" font-weight="700">{title}</text>
  <text x="40" y="62" class="small">duration={duration_s:.1f}s, green=concentric, orange=eccentric</text>
  <text x="40" y="109" class="label">GT micro</text>
  <rect x="{x0}" y="95" width="{plot_w}" height="18" fill="#f8fafc" stroke="#cbd5e1"/>{gt_micro}
  <text x="40" y="139" class="label">Pred micro</text>
  <rect x="{x0}" y="125" width="{plot_w}" height="18" fill="#f8fafc" stroke="#cbd5e1"/>{pred_micro}
  <text x="40" y="174" class="label">Stage 4 macro</text>
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


def write_action_prediction_svg(
    path: Path,
    stream_id: str,
    df: pd.DataFrame,
    gt_macro_runs: Sequence[SegmentRun],
    pred_macro_runs: Sequence[SegmentRun],
    pred_reps: Sequence[RepDetection],
    sample_rate_hz: float,
) -> None:
    width, height = 1280, 560
    x0, plot_w = 110, 1080
    n = len(df)
    acc = _magnitude(df, ["ax", "ay", "az"])
    gyro = _magnitude(df, ["gx", "gy", "gz"])
    acc_pts = _polyline(acc, x0, 230, plot_w, 110)
    gyro_pts = _polyline(gyro, x0, 370, plot_w, 110)
    gt_rects = _rects([(r.start_idx, r.end_idx, r.label) for r in gt_macro_runs if r.label != OTHER_LABEL], n, x0, plot_w, 96, 28, opacity=0.78)
    pred_rects = _rects([(r.start_idx, r.end_idx, r.label) for r in pred_macro_runs if r.label != OTHER_LABEL], n, x0, plot_w, 142, 28, opacity=0.78)

    def rep_action_blocks(reps: Sequence[RepDetection]) -> str:
        out = []
        for rep in reps:
            label = str(rep.pred_action_type)
            if label in {"unknown", ""}:
                continue
            start = max(0, int(rep.start_idx))
            end = min(n, int(rep.end_idx))
            if end <= start:
                continue
            x = x0 + start / max(1, n) * plot_w
            w = max(1.0, (end - start) / max(1, n) * plot_w)
            mid = x + w / 2.0
            color = _label_color(label)
            conf = "" if not np.isfinite(rep.action_confidence) else f" {rep.action_confidence:.2f}"
            text = html.escape(f"{label}{conf}")
            out.append(f'<rect x="{x:.1f}" y="184" width="{w:.1f}" height="26" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="1"/>')
            out.append(f'<text x="{mid:.1f}" y="202" text-anchor="middle" class="tiny action-label">{text}</text>')
        return "\n".join(out)

    labels = sorted(
        {
            str(r.label)
            for r in list(gt_macro_runs) + list(pred_macro_runs)
            if str(r.label) != OTHER_LABEL
        }
        | {str(r.pred_action_type) for r in pred_reps if str(r.pred_action_type) not in {"unknown", ""}}
    )
    legend_parts = []
    lx, ly = x0, 522
    for idx, label in enumerate(labels[:8]):
        x = lx + idx * 135
        legend_parts.append(f'<rect x="{x:.1f}" y="{ly - 11:.1f}" width="12" height="12" fill="{_label_color(label)}" fill-opacity="0.80"/>')
        legend_parts.append(f'<text x="{x + 18:.1f}" y="{ly:.1f}" class="tiny">{html.escape(label)}</text>')
    legend = "\n".join(legend_parts)

    title = html.escape(stream_id)
    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .small {{ font-size: 12px; fill: #475569; }}
    .tiny {{ font-size: 10px; fill: #111827; }}
    .label {{ font-size: 14px; font-weight: 700; }}
    .action-label {{ paint-order: stroke; stroke: #ffffff; stroke-width: 3px; stroke-linejoin: round; }}
  </style>
  <text x="40" y="36" font-size="20" font-weight="700">{title}</text>
  <text x="40" y="62" class="small">duration={duration_s:.1f}s, colored bands show action type</text>
  <text x="40" y="115" class="label">GT action</text>
  <rect x="{x0}" y="96" width="{plot_w}" height="28" fill="#f8fafc" stroke="#cbd5e1"/>{gt_rects}
  <text x="40" y="161" class="label">Pred action</text>
  <rect x="{x0}" y="142" width="{plot_w}" height="28" fill="#f8fafc" stroke="#cbd5e1"/>{pred_rects}
  <text x="40" y="202" class="label">Rep label</text>
  <rect x="{x0}" y="184" width="{plot_w}" height="26" fill="#f8fafc" stroke="#cbd5e1"/>{rep_action_blocks(pred_reps)}
  <text x="40" y="250" class="label">acc_mag</text>
  <rect x="{x0}" y="230" width="{plot_w}" height="110" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{acc_pts}" fill="none" stroke="#2563eb" stroke-width="1.3"/>
  <text x="40" y="390" class="label">gyro_mag</text>
  <rect x="{x0}" y="370" width="{plot_w}" height="110" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{gyro_pts}" fill="none" stroke="#7c3aed" stroke-width="1.3"/>
  {legend}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_streaming_prediction_svg(
    path: Path,
    stream_id: str,
    df: pd.DataFrame,
    gt_micro_runs: Sequence[SegmentRun],
    online_micro_runs: Sequence[SegmentRun],
    gt_macro_runs: Sequence[SegmentRun],
    online_macro_runs: Sequence[SegmentRun],
    sample_rate_hz: float,
    buffer_size: int,
) -> None:
    width, height = 1280, 680
    x0, plot_w = 110, 1080
    n = len(df)
    acc = _magnitude(df, ["ax", "ay", "az"])
    gyro = _magnitude(df, ["gx", "gy", "gz"])
    acc_pts = _polyline(acc, x0, 300, plot_w, 120)
    gyro_pts = _polyline(gyro, x0, 455, plot_w, 120)
    gt_micro = _rects([(r.start_idx, r.end_idx, r.label) for r in gt_micro_runs], n, x0, plot_w, 92, 24, opacity=0.72)
    pred_micro = _rects([(r.start_idx, r.end_idx, r.label) for r in online_micro_runs], n, x0, plot_w, 130, 24, opacity=0.72)
    gt_macro = _rects([(r.start_idx, r.end_idx, r.label) for r in gt_macro_runs if r.label != OTHER_LABEL], n, x0, plot_w, 185, 28, opacity=0.78)
    pred_macro = _rects([(r.start_idx, r.end_idx, r.label) for r in online_macro_runs if r.label != OTHER_LABEL], n, x0, plot_w, 230, 28, opacity=0.78)

    labels = sorted(
        {r.label for r in list(gt_micro_runs) + list(online_micro_runs)}
        | {r.label for r in list(gt_macro_runs) + list(online_macro_runs) if r.label != OTHER_LABEL}
    )
    legend_parts = []
    lx, ly = x0, 625
    for idx, label in enumerate(labels[:8]):
        x = lx + idx * 135
        legend_parts.append(f'<rect x="{x:.1f}" y="{ly - 11:.1f}" width="12" height="12" fill="{_label_color(label)}" fill-opacity="0.82"/>')
        legend_parts.append(f'<text x="{x + 18:.1f}" y="{ly:.1f}" class="tiny">{html.escape(str(label))}</text>')
    legend = "\n".join(legend_parts)

    title = html.escape(stream_id)
    duration_s = n / sample_rate_hz if sample_rate_hz > 0 else 0.0
    buffer_s = buffer_size / sample_rate_hz if sample_rate_hz > 0 else 0.0
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
    .small {{ font-size: 12px; fill: #475569; }}
    .tiny {{ font-size: 10px; fill: #111827; }}
    .label {{ font-size: 14px; font-weight: 700; }}
  </style>
  <text x="40" y="36" font-size="20" font-weight="700">Streaming prediction: {title}</text>
  <text x="40" y="62" class="small">duration={duration_s:.1f}s, rolling buffer={buffer_size} samples ({buffer_s:.1f}s); predictions are emitted one sample at a time.</text>
  <text x="40" y="110" class="label">GT micro</text>
  <rect x="{x0}" y="92" width="{plot_w}" height="24" fill="#f8fafc" stroke="#cbd5e1"/>{gt_micro}
  <text x="40" y="148" class="label">Online micro</text>
  <rect x="{x0}" y="130" width="{plot_w}" height="24" fill="#f8fafc" stroke="#cbd5e1"/>{pred_micro}
  <text x="40" y="204" class="label">GT action</text>
  <rect x="{x0}" y="185" width="{plot_w}" height="28" fill="#f8fafc" stroke="#cbd5e1"/>{gt_macro}
  <text x="40" y="249" class="label">Online action</text>
  <rect x="{x0}" y="230" width="{plot_w}" height="28" fill="#f8fafc" stroke="#cbd5e1"/>{pred_macro}
  <text x="40" y="320" class="label">acc_mag</text>
  <rect x="{x0}" y="300" width="{plot_w}" height="120" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{acc_pts}" fill="none" stroke="#2563eb" stroke-width="1.3"/>
  <text x="40" y="475" class="label">gyro_mag</text>
  <rect x="{x0}" y="455" width="{plot_w}" height="120" fill="#f8fafc" stroke="#cbd5e1"/>
  <polyline points="{gyro_pts}" fill="none" stroke="#7c3aed" stroke-width="1.3"/>
  {legend}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
