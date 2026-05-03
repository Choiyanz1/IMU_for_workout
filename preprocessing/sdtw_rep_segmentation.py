from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ACTIVE_PHASES = {"eccentric", "concentric"}


@dataclass
class SDTWConfig:
    smoothing_window: int = 9
    min_point_gap: int = 8
    point_value_epsilon: float = 0.08
    min_scale_percent: float = 0.30
    threshold_percentile: float = 90.0
    threshold_margin: float = 0.10
    duration_min_ratio: float = 0.55
    duration_max_ratio: float = 1.75
    max_overlap_iou: float = 0.20
    max_candidates_per_endpoint: int = 30
    template_middle_fraction: float = 0.50
    template_min_edge_reps: int = 1
    max_templates: int = 3


@dataclass
class SDTWTemplate:
    template_id: str
    action_type: str
    feature_names: List[str]
    ranked_features: List[str]
    dtw_feature: str
    query_indices: np.ndarray
    query_values: np.ndarray
    active_duration: int
    min_duration: int
    max_duration: int
    min_scale: float
    cost_threshold: float
    calibration_costs: List[float]
    exemplar_source: str
    exemplar_rep_position: float


@dataclass
class SegmentDetection:
    start_idx: int
    end_idx: int
    cost: float
    feature: str
    action_type: str
    template_id: str
    exemplar_source: str
    normalized_cost: float


def infer_sample_rate_hz(df: pd.DataFrame, time_column: str = "sensor_ts", default: float = 100.0) -> float:
    if time_column not in df.columns or len(df) < 2:
        return default
    values = pd.to_numeric(df[time_column], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(values) < 2:
        return default
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default
    median_diff = float(np.median(diffs))
    for divisor in (1.0, 1e3, 1e6):
        rate = divisor / median_diff
        if 1.0 <= rate <= 2000.0:
            return float(rate)
    return default


def active_segments_from_phase(df: pd.DataFrame, phase_column: str = "phase") -> List[Tuple[int, int]]:
    if phase_column not in df.columns:
        return []
    active = df[phase_column].astype(str).isin(ACTIVE_PHASES).to_numpy()
    if len(active) == 0:
        return []

    segments: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, is_active in enumerate(active):
        if is_active and not in_run:
            start = i
            in_run = True
        elif not is_active and in_run:
            segments.append((start, i))
            in_run = False
    if in_run:
        segments.append((start, len(active)))
    return segments


def active_motion_slice(df: pd.DataFrame, phase_column: str = "phase") -> pd.DataFrame:
    segments = active_segments_from_phase(df, phase_column)
    if not segments:
        out = df.copy().reset_index(drop=True)
        out.attrs.update(df.attrs)
        return out
    start, end = max(segments, key=lambda seg: seg[1] - seg[0])
    out = df.iloc[start:end].copy().reset_index(drop=True)
    out.attrs.update(df.attrs)
    return out


def build_motion_features(df: pd.DataFrame, imu_columns: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    available_columns = [col for col in imu_columns if col in df.columns]
    if not available_columns:
        raise ValueError("No requested motion columns are present in the dataframe")
    base = df.loc[:, available_columns].to_numpy(dtype=np.float64)
    names = list(available_columns)
    extras: List[np.ndarray] = []
    extra_names: List[str] = []

    acc_idx = [i for i, name in enumerate(names) if name.startswith("a")]
    gyro_idx = [i for i, name in enumerate(names) if name.startswith("g")]
    mag_idx = [i for i, name in enumerate(names) if name.startswith("m")]
    if len(acc_idx) >= 2:
        extras.append(np.sqrt(np.sum(base[:, acc_idx] ** 2, axis=1, keepdims=True)))
        extra_names.append("acc_mag")
    if len(gyro_idx) >= 2:
        extras.append(np.sqrt(np.sum(base[:, gyro_idx] ** 2, axis=1, keepdims=True)))
        extra_names.append("gyro_mag")
    if len(mag_idx) >= 2:
        extras.append(np.sqrt(np.sum(base[:, mag_idx] ** 2, axis=1, keepdims=True)))
        extra_names.append("mag_mag")

    if extras:
        matrix = np.concatenate([base] + extras, axis=1)
        names = names + extra_names
    else:
        matrix = base
    return matrix, names


def smooth_matrix(matrix: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return matrix.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.empty_like(matrix, dtype=np.float64)
    pad = window // 2
    for col in range(matrix.shape[1]):
        padded = np.pad(matrix[:, col], (pad, pad), mode="edge")
        out[:, col] = np.convolve(padded, kernel, mode="valid")
    return out


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        std = 1.0
    return (values - mean) / std


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.empty_like(matrix, dtype=np.float64)
    for col in range(matrix.shape[1]):
        out[:, col] = robust_zscore(matrix[:, col])
    return out


def rank_features(matrix: np.ndarray, feature_names: Sequence[str]) -> List[str]:
    change = np.sum(np.abs(np.diff(matrix, axis=0)), axis=0)
    order = np.argsort(change)[::-1]
    return [feature_names[i] for i in order]


def extract_segment_points(
    signal: np.ndarray,
    min_gap: int,
    value_epsilon: float,
    include_start_crossings: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    if len(signal) == 1:
        return np.asarray([0], dtype=np.int64), signal.copy()

    velocity = np.diff(signal)
    candidates = {0, len(signal) - 1}
    for i in range(1, len(velocity)):
        if velocity[i - 1] == 0 or velocity[i] == 0 or np.sign(velocity[i - 1]) != np.sign(velocity[i]):
            candidates.add(i)

    if include_start_crossings:
        start_value = signal[0]
        centered = signal - start_value
        for i in range(1, len(centered)):
            if centered[i - 1] == 0 or centered[i] == 0 or np.sign(centered[i - 1]) != np.sign(centered[i]):
                candidates.add(i)

    ordered = sorted(candidates)
    kept_idx: List[int] = []
    kept_val: List[float] = []
    for idx in ordered:
        value = float(signal[idx])
        if kept_idx and (idx - kept_idx[-1] < min_gap or abs(value - kept_val[-1]) < value_epsilon):
            kept_idx[-1] = idx
            kept_val[-1] = value
        else:
            kept_idx.append(idx)
            kept_val.append(value)

    if kept_idx[0] != 0:
        kept_idx.insert(0, 0)
        kept_val.insert(0, float(signal[0]))
    if kept_idx[-1] != len(signal) - 1:
        kept_idx.append(len(signal) - 1)
        kept_val.append(float(signal[-1]))

    return np.asarray(kept_idx, dtype=np.int64), np.asarray(kept_val, dtype=np.float64)


def dtw_path_cost(query: np.ndarray, candidate: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    query = robust_zscore(query)
    candidate = robust_zscore(candidate)
    m, n = len(query), len(candidate)
    if m == 0 or n == 0:
        return float("inf"), []

    costs = np.abs(query[:, None] - candidate[None, :])
    dp = np.empty((m, n), dtype=np.float64)
    dp[0, 0] = costs[0, 0]
    for i in range(1, m):
        dp[i, 0] = costs[i, 0] + dp[i - 1, 0]
    for j in range(1, n):
        dp[0, j] = costs[0, j] + dp[0, j - 1]
    for i in range(1, m):
        for j in range(1, n):
            dp[i, j] = costs[i, j] + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    i, j = m - 1, n - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            options = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            step = int(np.argmin(options))
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()

    per_query = []
    for qi in range(m):
        aligned = [costs[i, j] for i, j in path if i == qi]
        if aligned:
            per_query.append(float(np.mean(aligned)))
    if not per_query:
        return float("inf"), path
    return float(np.mean(per_query)), path


def _candidate_cost(query_values: np.ndarray, candidate_values: np.ndarray, min_scale: float) -> float:
    if len(candidate_values) < 2:
        return float("inf")
    if float(np.max(candidate_values) - np.min(candidate_values)) < min_scale:
        return float("inf")
    cost, _ = dtw_path_cost(query_values, candidate_values)
    return cost


def _active_reps_with_candidates(
    rep_dfs: Sequence[pd.DataFrame],
    phase_column: str,
) -> List[pd.DataFrame]:
    active_reps = [active_motion_slice(df, phase_column) for df in rep_dfs]
    return [df for df in active_reps if len(df) >= 8]


def _candidate_reps_for_templates(active_reps: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    candidate_reps = [
        df for df in active_reps
        if bool(df.attrs.get("template_candidate", False))
        or ("_template_candidate" in df.columns and bool(df["_template_candidate"].iloc[0]))
    ]
    return candidate_reps or list(active_reps)


def _source_group(df: pd.DataFrame) -> str:
    if "source_path" in df.attrs:
        parts = str(df.attrs["source_path"]).split("/")
        if len(parts) >= 3:
            return "/".join(parts[:3])
    return str(df.attrs.get("source_id", "unknown"))


def _select_exemplars(
    active_reps: Sequence[pd.DataFrame],
    median_duration: float,
    max_templates: int,
) -> List[pd.DataFrame]:
    candidates = sorted(
        _candidate_reps_for_templates(active_reps),
        key=lambda df: (
            abs(len(df) - median_duration),
            abs(float(df.attrs.get("rep_position", 0.5)) - 0.5),
            str(df.attrs.get("source_path", df.attrs.get("source_id", ""))),
        ),
    )
    if max_templates <= 1:
        return candidates[:1]

    selected: List[pd.DataFrame] = []
    used_groups = set()
    for candidate in candidates:
        group = _source_group(candidate)
        if group in used_groups:
            continue
        selected.append(candidate)
        used_groups.add(group)
        if len(selected) >= max_templates:
            return selected

    selected_ids = {id(df) for df in selected}
    for candidate in candidates:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if len(selected) >= max_templates:
            break
    return selected


def _build_template_from_exemplar(
    action_type: str,
    template_idx: int,
    exemplar_df: pd.DataFrame,
    active_reps: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    config: SDTWConfig,
    median_duration: float,
) -> SDTWTemplate:
    matrix, feature_names = build_motion_features(exemplar_df, imu_columns)
    matrix = normalize_matrix(smooth_matrix(matrix, config.smoothing_window))
    ranked = rank_features(matrix, feature_names)
    dtw_feature = ranked[0]
    feature_idx = feature_names.index(dtw_feature)
    query_indices, query_values = extract_segment_points(
        matrix[:, feature_idx],
        config.min_point_gap,
        config.point_value_epsilon,
        include_start_crossings=False,
    )
    if len(query_values) < 2:
        query_indices = np.asarray([0, len(exemplar_df) - 1], dtype=np.int64)
        query_values = matrix[query_indices, feature_idx]

    query_scale = float(np.max(query_values) - np.min(query_values))
    min_scale = query_scale * float(config.min_scale_percent)
    min_duration = max(2, int(round(median_duration * config.duration_min_ratio)))
    max_duration = max(min_duration + 1, int(round(median_duration * config.duration_max_ratio)))

    calibration_costs: List[float] = []
    for rep_df in active_reps:
        rep_matrix, rep_names = build_motion_features(rep_df, imu_columns)
        rep_matrix = normalize_matrix(smooth_matrix(rep_matrix, config.smoothing_window))
        if dtw_feature not in rep_names:
            continue
        rep_signal = rep_matrix[:, rep_names.index(dtw_feature)]
        _, rep_values = extract_segment_points(
            rep_signal,
            config.min_point_gap,
            config.point_value_epsilon,
            include_start_crossings=True,
        )
        cost = _candidate_cost(query_values, rep_values, min_scale)
        if np.isfinite(cost):
            calibration_costs.append(float(cost))

    if calibration_costs:
        threshold = float(np.percentile(calibration_costs, config.threshold_percentile))
        threshold += float(config.threshold_margin)
    else:
        threshold = 1.0 + float(config.threshold_margin)

    return SDTWTemplate(
        template_id=f"{action_type}_template{template_idx}",
        action_type=action_type,
        feature_names=list(feature_names),
        ranked_features=ranked,
        dtw_feature=dtw_feature,
        query_indices=query_indices,
        query_values=query_values,
        active_duration=int(round(median_duration)),
        min_duration=min_duration,
        max_duration=max_duration,
        min_scale=min_scale,
        cost_threshold=threshold,
        calibration_costs=calibration_costs,
        exemplar_source=str(exemplar_df.attrs.get("source_path", exemplar_df.attrs.get("source_id", "unknown"))),
        exemplar_rep_position=float(exemplar_df.attrs.get("rep_position", np.nan)),
    )


def fit_sdtw_templates(
    action_type: str,
    rep_dfs: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    config: SDTWConfig | None = None,
    phase_column: str = "phase",
) -> List[SDTWTemplate]:
    if config is None:
        config = SDTWConfig()
    active_reps = _active_reps_with_candidates(rep_dfs, phase_column)
    if not active_reps:
        raise ValueError(f"No active reps available for {action_type}")

    durations = np.asarray([len(df) for df in active_reps], dtype=np.float64)
    median_duration = float(np.median(durations))
    exemplars = _select_exemplars(active_reps, median_duration, int(config.max_templates))
    return [
        _build_template_from_exemplar(
            action_type=action_type,
            template_idx=i,
            exemplar_df=exemplar,
            active_reps=active_reps,
            imu_columns=imu_columns,
            config=config,
            median_duration=median_duration,
        )
        for i, exemplar in enumerate(exemplars)
    ]


def fit_sdtw_template(
    action_type: str,
    rep_dfs: Sequence[pd.DataFrame],
    imu_columns: Sequence[str],
    config: SDTWConfig | None = None,
    phase_column: str = "phase",
) -> SDTWTemplate:
    return fit_sdtw_templates(action_type, rep_dfs, imu_columns, config, phase_column)[0]


def _raw_template_candidates(
    df: pd.DataFrame,
    template: SDTWTemplate,
    imu_columns: Sequence[str],
    config: SDTWConfig,
    cost_threshold_scale: float = 1.0,
) -> List[SegmentDetection]:
    """Generate all valid SDTW candidates for a single template (pre-NMS).

    `cost_threshold_scale` widens the cost threshold during candidate generation;
    use values > 1.0 to admit more candidates (including likely false positives)
    that a downstream classifier can filter. The reported `normalized_cost` is
    still computed against the unscaled threshold for stable comparisons.
    """
    if len(df) < template.min_duration:
        return []

    matrix, feature_names = build_motion_features(df, imu_columns)
    matrix = normalize_matrix(smooth_matrix(matrix, config.smoothing_window))
    signal = matrix[:, feature_names.index(template.dtw_feature)]
    point_indices, point_values = extract_segment_points(
        signal,
        config.min_point_gap,
        config.point_value_epsilon,
        include_start_crossings=True,
    )
    if len(point_indices) < 2:
        return []

    threshold = template.cost_threshold * float(cost_threshold_scale)
    candidates: List[SegmentDetection] = []
    for end_pos in range(1, len(point_indices)):
        considered = 0
        for start_pos in range(end_pos - 1, -1, -1):
            start_idx = int(point_indices[start_pos])
            end_idx = int(point_indices[end_pos]) + 1
            duration = end_idx - start_idx
            if duration > template.max_duration:
                break
            if duration < template.min_duration:
                continue
            candidate_values = point_values[start_pos : end_pos + 1]
            cost = _candidate_cost(template.query_values, candidate_values, template.min_scale)
            if np.isfinite(cost) and cost <= threshold:
                candidates.append(
                    SegmentDetection(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        cost=float(cost),
                        feature=template.dtw_feature,
                        action_type=template.action_type,
                        template_id=template.template_id,
                        exemplar_source=template.exemplar_source,
                        normalized_cost=float(cost) / max(template.cost_threshold, 1e-8),
                    )
                )
            considered += 1
            if considered >= config.max_candidates_per_endpoint:
                break
    return candidates


def _nms_by_cost(
    candidates: Sequence[SegmentDetection],
    max_overlap_iou: float,
) -> List[SegmentDetection]:
    selected: List[SegmentDetection] = []
    for candidate in sorted(candidates, key=lambda item: item.normalized_cost):
        if all(
            _segment_iou((candidate.start_idx, candidate.end_idx), (s.start_idx, s.end_idx))
            <= max_overlap_iou
            for s in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.start_idx)


def detect_reps_sdtw(
    df: pd.DataFrame,
    template: SDTWTemplate,
    imu_columns: Sequence[str],
    config: SDTWConfig | None = None,
) -> List[SegmentDetection]:
    if config is None:
        config = SDTWConfig()
    candidates = _raw_template_candidates(df, template, imu_columns, config)
    return _nms_by_cost(candidates, config.max_overlap_iou)


def detect_reps_sdtw_templates(
    df: pd.DataFrame,
    templates: Sequence[SDTWTemplate],
    imu_columns: Sequence[str],
    config: SDTWConfig | None = None,
) -> List[SegmentDetection]:
    if config is None:
        config = SDTWConfig()
    candidates: List[SegmentDetection] = []
    for template in templates:
        candidates.extend(_raw_template_candidates(df, template, imu_columns, config))
    return _nms_by_cost(candidates, config.max_overlap_iou)


def generate_candidates_for_templates(
    df: pd.DataFrame,
    templates: Sequence[SDTWTemplate],
    imu_columns: Sequence[str],
    config: SDTWConfig | None = None,
    cost_threshold_scale: float = 1.0,
) -> List[SegmentDetection]:
    """Return all SDTW candidates from every template (no NMS).

    Use this when you want a downstream classifier to filter candidates.
    Setting `cost_threshold_scale > 1.0` admits more candidates (more recall
    upstream, more false positives for the classifier to learn from).
    """
    if config is None:
        config = SDTWConfig()
    out: List[SegmentDetection] = []
    for template in templates:
        out.extend(_raw_template_candidates(df, template, imu_columns, config, cost_threshold_scale))
    return out


def _segment_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    inter = max(0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:
        return 0.0
    return inter / union


def match_segments(
    predicted: Sequence[Tuple[int, int]],
    truth: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.50,
) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    for pred_idx, pred in enumerate(predicted):
        for true_idx, true in enumerate(truth):
            iou = _segment_iou(pred, true)
            if iou >= iou_threshold:
                pairs.append((pred_idx, true_idx, iou))
    matches: List[Tuple[int, int, float]] = []
    used_pred = set()
    used_true = set()
    for pred_idx, true_idx, iou in sorted(pairs, key=lambda item: item[2], reverse=True):
        if pred_idx in used_pred or true_idx in used_true:
            continue
        used_pred.add(pred_idx)
        used_true.add(true_idx)
        matches.append((pred_idx, true_idx, iou))
    return matches


def summarize_detection_metrics(
    detections: Sequence[SegmentDetection],
    truth_segments: Sequence[Tuple[int, int]],
    sample_rate_hz: float,
    iou_threshold: float = 0.50,
) -> Dict[str, float]:
    predicted = [(d.start_idx, d.end_idx) for d in detections]
    truth = list(truth_segments)
    matches = match_segments(predicted, truth, iou_threshold)

    tp = len(matches)
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(truth) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    start_errors_ms: List[float] = []
    end_errors_ms: List[float] = []
    duration_errors_ms: List[float] = []
    ious: List[float] = []
    for pred_idx, true_idx, iou in matches:
        pred = predicted[pred_idx]
        true = truth[true_idx]
        start_errors_ms.append(abs(pred[0] - true[0]) / sample_rate_hz * 1000.0)
        end_errors_ms.append(abs(pred[1] - true[1]) / sample_rate_hz * 1000.0)
        duration_errors_ms.append(abs((pred[1] - pred[0]) - (true[1] - true[0])) / sample_rate_hz * 1000.0)
        ious.append(iou)

    def mean_or_nan(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def median_or_nan(values: Sequence[float]) -> float:
        return float(np.median(values)) if values else float("nan")

    return {
        "n_pred": float(len(predicted)),
        "n_true": float(len(truth)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": mean_or_nan(ious),
        "start_mae_ms": mean_or_nan(start_errors_ms),
        "end_mae_ms": mean_or_nan(end_errors_ms),
        "duration_mae_ms": mean_or_nan(duration_errors_ms),
        "start_median_error_ms": median_or_nan(start_errors_ms),
        "end_median_error_ms": median_or_nan(end_errors_ms),
    }
