"""Replay or consume IMU CSV samples with the raw6 CNN + top5_p5 decoder.

Accepted input formats:
- Raw `zig_bt_client --stdout` lines:
  serial,type,ts,host_ts,ppg_a..j,ax,ay,az,gx,gy,gz,mx,my,mz
- Saved workout CSV rows containing `ax,ay,az,gx,gy,gz` headers.

The default deployment assumption is that the input is a workout set/active
interval and that action context is provided with `--action`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


RAW_COL_IMU_START = 14
IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


@dataclass
class RepEvent:
    start_idx: int
    transition_idx: int
    end_idx: int
    confidence: float = 1.0


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.sum(e, axis=axis, keepdims=True), 1e-12)


def _smooth_ma(phase_probs: np.ndarray, window: int) -> np.ndarray:
    n = len(phase_probs)
    smoothed = np.copy(phase_probs)
    if window <= 1:
        return smoothed
    for c in range(phase_probs.shape[1]):
        cumsum = np.cumsum(phase_probs[:, c])
        for i in range(n):
            start = max(0, i - window + 1)
            total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
            smoothed[i, c] = total / (i - start + 1)
    return smoothed


def _viterbi_decode(phase_probs: np.ndarray, penalty: float) -> np.ndarray:
    n = len(phase_probs)
    if n == 0:
        return phase_probs
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2), dtype=np.float64)
    dp[0] = log_probs[0]
    for i in range(1, n):
        for s in range(2):
            stay = dp[i - 1, s]
            switch = dp[i - 1, 1 - s] - penalty
            dp[i, s] = log_probs[i, s] + max(stay, switch)
    pred = np.zeros(n, dtype=np.int64)
    pred[-1] = int(np.argmax(dp[-1]))
    for i in range(n - 2, -1, -1):
        s = pred[i + 1]
        stay = dp[i, s]
        switch = dp[i, 1 - s] - penalty
        pred[i] = s if stay >= switch else 1 - s
    result = np.zeros((n, 2), dtype=np.float32)
    result[pred == 0, 0] = 1.0
    result[pred == 1, 1] = 1.0
    return result


def _parse_reps(labels: np.ndarray, min_phase: int, max_gap: int) -> list[RepEvent]:
    phase = np.asarray(["eccentric" if int(p) == 0 else "concentric" for p in labels], dtype=object)
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            if i - start >= min_phase:
                runs.append((str(phase[start]), start, i))
            start = i

    merged: list[tuple[str, int, int]] = []
    for label, s, e in runs:
        if merged and label == merged[-1][0]:
            merged[-1] = (label, merged[-1][1], e)
        else:
            merged.append((label, s, e))

    reps: list[RepEvent] = []
    i = 0
    while i < len(merged):
        label, s, e = merged[i]
        if label != "concentric" or i + 1 >= len(merged):
            i += 1
            continue
        next_label, ns, ne = merged[i + 1]
        if next_label == "eccentric" and ns - e <= max_gap:
            reps.append(RepEvent(s, ns, ne))
            i += 2
        else:
            i += 1
    return reps


def _merge_short_reps(reps: list[RepEvent], min_duration_samples: float, max_gap_samples: int) -> list[RepEvent]:
    if not reps:
        return []
    threshold = int(round(min_duration_samples))
    max_gap = int(max_gap_samples)
    merged: list[RepEvent] = []
    i = 0
    while i < len(reps):
        cur = reps[i]
        cur_duration = cur.end_idx - cur.start_idx
        if cur_duration < threshold and i + 1 < len(reps):
            nxt = reps[i + 1]
            if nxt.start_idx - cur.end_idx <= max_gap:
                merged.append(RepEvent(cur.start_idx, cur.transition_idx, nxt.end_idx, cur.confidence))
                i += 2
                continue
        if cur_duration < threshold and merged:
            prev = merged[-1]
            if cur.start_idx - prev.end_idx <= max_gap:
                merged[-1] = RepEvent(prev.start_idx, prev.transition_idx, cur.end_idx, prev.confidence)
                i += 1
                continue
        merged.append(cur)
        i += 1
    return merged


def _parse_raw_zig_line(line: str) -> tuple[float, float, float, float, float, float] | None:
    cols = [c.strip() for c in line.split(",")]
    if len(cols) < RAW_COL_IMU_START + 6 or (len(cols) > 1 and cols[1] != "IMU"):
        return None
    try:
        return tuple(float(cols[RAW_COL_IMU_START + i]) for i in range(6))  # type: ignore[return-value]
    except ValueError:
        return None


def _iter_samples_from_lines(lines: Iterable[str]) -> Iterable[tuple[float, float, float, float, float, float]]:
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("serial_num"):
            continue
        parsed = _parse_raw_zig_line(raw)
        if parsed is not None:
            yield parsed


def _load_samples(path: str) -> np.ndarray:
    if path == "-":
        return np.asarray(list(_iter_samples_from_lines(sys.stdin)), dtype=np.float32)

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        f.seek(0)
        if all(col in first for col in IMU_COLUMNS):
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                try:
                    rows.append(tuple(float(row[col]) for col in IMU_COLUMNS))
                except (TypeError, ValueError, KeyError):
                    continue
            return np.asarray(rows, dtype=np.float32)
        return np.asarray(list(_iter_samples_from_lines(f)), dtype=np.float32)


class TorchRunner:
    def __init__(self, artifact_dir: Path, device: str) -> None:
        import torch

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.new_c_pipeline.test_pca_input import CausalCNN_PhaseOnly

        checkpoint = torch.load(artifact_dir / "model.pt", map_location=device)
        self.device = torch.device(device)
        self.model = CausalCNN_PhaseOnly(
            checkpoint.get("input_channels", 6),
            checkpoint.get("hidden", 64),
            checkpoint.get("num_classes", 2),
            checkpoint.get("dropout", 0.2),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.torch = torch

    def predict_window(self, window: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(self.device)
            logits = self.model(x).cpu().numpy()[0]
        return _softmax(logits, axis=0).T.astype(np.float32)


class OnnxRunner:
    def __init__(self, artifact_dir: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(artifact_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_window(self, window: np.ndarray) -> np.ndarray:
        x = window.astype(np.float32).T[None, :, :]
        logits = self.session.run(None, {self.input_name: x})[0][0]
        return _softmax(logits, axis=0).T.astype(np.float32)


class RknnRunner:
    def __init__(self, artifact_dir: Path) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ModuleNotFoundError:
            from rknn.api import RKNN as RKNNLite

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(artifact_dir / "model.rknn"))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def predict_window(self, window: np.ndarray) -> np.ndarray:
        x = window.astype(np.float32).T[None, :, :]
        logits = self.rknn.inference(inputs=[x])[0][0]
        return _softmax(logits, axis=0).T.astype(np.float32)


def _predict_phase_probs(runner: TorchRunner | OnnxRunner | RknnRunner, samples: np.ndarray, mean: np.ndarray, std: np.ndarray, decoder: dict) -> np.ndarray:
    n = len(samples)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x_std = (samples - mean) / std
    slice_len = int(decoder.get("slice_len", 300))
    stride = int(decoder.get("overlap_stride", 150))
    probs_accum = np.ones((n, 2), dtype=np.float32) * 0.5
    counts = np.zeros(n, dtype=np.float32)

    if n <= slice_len:
        padded = np.pad(x_std, ((0, slice_len - n), (0, 0)), mode="edge")
        probs = runner.predict_window(padded)
        probs_accum[:n] += probs[:n]
        counts[:n] += 1.0
    else:
        starts = list(range(0, n - slice_len + 1, stride))
        if not starts or starts[-1] + slice_len < n:
            starts.append(n - slice_len)
        for start in starts:
            probs = runner.predict_window(x_std[start:start + slice_len])
            probs_accum[start:start + slice_len] += probs
            counts[start:start + slice_len] += 1.0

    valid = counts > 0
    probs_accum[valid] /= counts[valid][:, None]
    probs_accum = _smooth_ma(probs_accum, int(decoder.get("smoothing_window", 25)))
    return _viterbi_decode(probs_accum, float(decoder.get("viterbi_penalty", 0.3)))


def _duration_threshold(decoder: dict, action: str) -> float:
    priors = decoder.get("duration_priors", {}) or {}
    key = str(decoder.get("selective_merge_percentile", 5))
    if action in priors and key in priors[action]:
        return float(priors[action][key])
    return float(priors.get("__global__", {}).get(key, 0.0))


def _rep_to_dict(rep: RepEvent, sample_rate: float) -> dict:
    return {
        "start_idx": int(rep.start_idx),
        "transition_idx": int(rep.transition_idx),
        "end_idx": int(rep.end_idx),
        "start_sec": round(rep.start_idx / sample_rate, 3),
        "transition_sec": round(rep.transition_idx / sample_rate, 3),
        "end_sec": round(rep.end_idx / sample_rate, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay raw6 CNN + top5_p5 on client IMU CSV samples.")
    parser.add_argument("--artifact", default="artifacts/deploy/raw6_cnn_top5_p5_current")
    parser.add_argument("--input", default="-", help="Input CSV path, or '-' for stdin raw zig stream.")
    parser.add_argument("--action", required=True, help="Workout action context, e.g. db_rdl.")
    parser.add_argument("--runtime", choices=["torch", "onnx", "rknn"], default="torch")
    parser.add_argument("--device", default="cpu", help="PyTorch device for --runtime torch.")
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--emit-phases", action="store_true", help="Include per-sample hard phase labels in JSON output.")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    normalization = json.loads((artifact_dir / "normalization.json").read_text(encoding="utf-8"))
    decoder = json.loads((artifact_dir / "decoder_config.json").read_text(encoding="utf-8"))
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    sample_rate = float(args.sample_rate or metadata.get("sample_rate_hz", 100))

    samples = _load_samples(args.input)
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise RuntimeError(f"Expected [N, 6] IMU samples, got {samples.shape}")

    if args.runtime == "torch":
        runner = TorchRunner(artifact_dir, args.device)
    elif args.runtime == "onnx":
        runner = OnnxRunner(artifact_dir)
    else:
        runner = RknnRunner(artifact_dir)
    phase_probs = _predict_phase_probs(runner, samples, mean, std, decoder)
    hard_labels = np.argmax(phase_probs, axis=1)
    raw_reps = _parse_reps(hard_labels, int(decoder.get("min_phase_samples", 3)), int(decoder.get("max_phase_gap_samples", 3)))

    reps = raw_reps
    merge_applied = args.action in set(decoder.get("selective_merge_actions", []))
    threshold = None
    if merge_applied:
        threshold = _duration_threshold(decoder, args.action)
        reps = _merge_short_reps(raw_reps, threshold, int(decoder.get("selective_merge_max_gap_samples", 50)))

    phase_names = np.asarray(["eccentric" if int(v) == 0 else "concentric" for v in hard_labels], dtype=object)
    output = {
        "artifact": str(artifact_dir),
        "runtime": args.runtime,
        "action": args.action,
        "samples": int(len(samples)),
        "duration_sec": round(len(samples) / sample_rate, 3) if math.isfinite(sample_rate) and sample_rate > 0 else None,
        "raw_count_before_merge": int(len(raw_reps)),
        "display_count_top5_p5": int(len(reps)),
        "merge_applied": bool(merge_applied),
        "duration_threshold_samples": threshold,
        "reps": [_rep_to_dict(rep, sample_rate) for rep in reps],
    }
    if args.emit_phases:
        output["phase_labels"] = phase_names.tolist()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
