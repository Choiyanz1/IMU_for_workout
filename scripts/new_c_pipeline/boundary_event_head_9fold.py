"""Formal 9-fold boundary/event head probe for the raw6 causal CNN.

This experiment keeps the input, active detector, and base phase decoder family
fixed, then tests whether adding a lightweight transition-boundary head improves
rep grouping and C/E timing without sacrificing streaming feasibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from scripts.new_c_pipeline.duration_merge_decoder_9fold import (  # noqa: E402
    build_duration_priors,
    evaluate_with_reps,
    merge_short_reps,
    threshold_for_action,
)
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import (  # noqa: E402
    aggregate_rich,
    group_aggregate,
    stream_action,
    stream_subject,
    train_raw6_model,
)
from scripts.new_c_pipeline.selective_duration_merge_decoder_9fold import ACTION_SETS  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    CausalCNN_PhaseOnly,
    EXCLUDED_SESSIONS,
    PhaseDataset,
    SharedEncoder,
    extract_active_segments_data,
    normalize,
    parse_reps,
    predict_fast,
    set_seed,
    should_exclude,
    smooth_ma,
    train_fast,
    viterbi_decode,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


class CausalCNNBoundaryAware(nn.Module):
    def __init__(self, in_ch=6, hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = SharedEncoder(in_ch, hidden, dropout)
        self.phase_head = nn.Conv1d(hidden, 2, 1)
        self.boundary_head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.phase_head(features), self.boundary_head(features)


def extract_segments_with_boundary(streams, imu_columns, boundary_margin_samples):
    segments, labels, boundaries = [], [], []
    for _, df in streams:
        if "phase" not in df.columns:
            continue
        x = df[list(imu_columns)].to_numpy(dtype=np.float32)
        phase_arr = df["phase"].to_numpy()
        active_mask = np.array([str(p) in {"concentric", "eccentric"} for p in phase_arr])
        in_active = False
        seg_start = 0
        for i, is_active in enumerate(active_mask):
            if is_active and not in_active:
                seg_start = i
                in_active = True
            elif not is_active and in_active:
                if i - seg_start >= 10:
                    append_boundary_segment(x, phase_arr, seg_start, i, segments, labels, boundaries, boundary_margin_samples)
                in_active = False
        if in_active and len(active_mask) - seg_start >= 10:
            append_boundary_segment(x, phase_arr, seg_start, len(active_mask), segments, labels, boundaries, boundary_margin_samples)
    return segments, labels, boundaries


def append_boundary_segment(x, phase_arr, start, end, segments, labels, boundaries, boundary_margin_samples):
    seg_x = x[start:end]
    seg_phase = phase_arr[start:end]
    seg_labels = np.array([1 if str(p) == "concentric" else 0 for p in seg_phase], dtype=np.int64)
    seg_boundaries = np.zeros(len(seg_labels), dtype=np.float32)
    for change in np.where(np.diff(seg_labels) != 0)[0]:
        left = max(0, int(change) - boundary_margin_samples)
        right = min(len(seg_labels), int(change) + 1 + boundary_margin_samples)
        seg_boundaries[left:right] = 1.0
    segments.append(seg_x)
    labels.append(seg_labels)
    boundaries.append(seg_boundaries)


class BoundaryDataset(Dataset):
    def __init__(self, segments, labels, boundaries, slice_len=300):
        self.samples = []
        for seq, lab, bnd in zip(segments, labels, boundaries):
            n = len(seq)
            if n <= slice_len:
                pad_len = max(0, slice_len - n)
                seq_pad = np.pad(seq, ((0, pad_len), (0, 0)), mode="edge")
                lab_pad = np.pad(lab, (0, pad_len), constant_values=-1)
                bnd_pad = np.pad(bnd, (0, pad_len), constant_values=0.0)
                mask = np.concatenate([np.ones(n, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
                self.samples.append((seq_pad[:slice_len], lab_pad[:slice_len], bnd_pad[:slice_len], mask[:slice_len]))
            else:
                stride = slice_len // 2
                starts = list(range(0, n - slice_len + 1, stride))
                if not starts or starts[-1] + slice_len < n:
                    starts.append(n - slice_len)
                for start in starts:
                    self.samples.append((
                        seq[start:start + slice_len],
                        lab[start:start + slice_len],
                        bnd[start:start + slice_len],
                        np.ones(slice_len, dtype=np.float32),
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, lab, bnd, mask = self.samples[idx]
        return (
            torch.from_numpy(seq).float().transpose(0, 1),
            torch.from_numpy(lab).long(),
            torch.from_numpy(bnd).float(),
            torch.from_numpy(mask).float(),
        )


def train_boundary_model(train_streams, imu_columns, hidden, epochs, device, boundary_margin_samples, boundary_weight, pos_weight):
    segments, labels, boundaries = extract_segments_with_boundary(train_streams, imu_columns, boundary_margin_samples)
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    train_ds = BoundaryDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx], [boundaries[i] for i in train_idx])
    val_ds = BoundaryDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx], [boundaries[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = CausalCNNBoundaryAware(len(imu_columns), hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    boundary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y, b, m in train_loader:
            x, y, b, m = x.to(device), y.to(device), b.to(device), m.to(device)
            optimizer.zero_grad()
            phase_logits, boundary_logits = model(x)
            loss = boundary_training_loss(phase_logits, boundary_logits, y, b, m, phase_criterion, boundary_criterion, boundary_weight)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, y, b, m in val_loader:
                x, y, b, m = x.to(device), y.to(device), b.to(device), m.to(device)
                phase_logits, boundary_logits = model(x)
                loss = boundary_training_loss(phase_logits, boundary_logits, y, b, m, phase_criterion, boundary_criterion, boundary_weight)
                val_loss += float(loss.item())
                val_batches += 1
        avg_train = train_loss / n_batches if n_batches else float("inf")
        avg_val = val_loss / val_batches if val_batches else float("inf")
        print(f"  boundary epoch {epoch + 1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f}", flush=True)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean, std, len(segments)


def boundary_training_loss(phase_logits, boundary_logits, y, b, m, phase_criterion, boundary_criterion, boundary_weight):
    batch, classes, steps = phase_logits.shape
    phase_flat = phase_logits.permute(0, 2, 1).reshape(batch * steps, classes)
    labels_flat = y.reshape(batch * steps)
    mask_flat = m.reshape(batch * steps)
    valid = (labels_flat >= 0) & (mask_flat > 0)
    phase_loss = phase_criterion(phase_flat[valid], labels_flat[valid]) if valid.sum() else phase_logits.sum() * 0.0
    boundary_flat = boundary_logits.reshape(batch * steps)
    target_flat = b.reshape(batch * steps)
    boundary_loss = boundary_criterion(boundary_flat[valid], target_flat[valid]) if valid.sum() else boundary_logits.sum() * 0.0
    return phase_loss + boundary_weight * boundary_loss


def predict_boundary_model(model, df, active_segments, imu_columns, mean, std):
    x = df[list(imu_columns)].to_numpy(dtype=np.float32)
    x_std = (x - mean) / std
    n = len(x_std)
    phase_probs = np.ones((n, 2), dtype=float) * 0.5
    phase_counts = np.zeros(n, dtype=np.float32)
    boundary_probs = np.zeros(n, dtype=np.float32)
    boundary_counts = np.zeros(n, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for seg_start, seg_end in active_segments:
            if seg_start >= seg_end:
                continue
            seg_x = x_std[seg_start:seg_end]
            seg_len = len(seg_x)
            if seg_len <= 300:
                pad_len = 300 - seg_len
                padded = np.pad(seg_x, ((0, pad_len), (0, 0)), mode="edge")
                x_tensor = torch.from_numpy(padded).float().transpose(0, 1).unsqueeze(0).to(device)
                phase_logits, boundary_logits = model(x_tensor)
                probs = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                b_probs = torch.sigmoid(boundary_logits).cpu().numpy()[0, 0]
                phase_probs[seg_start:seg_end, :] += probs[:, :seg_len].T
                phase_counts[seg_start:seg_end] += 1.0
                boundary_probs[seg_start:seg_end] += b_probs[:seg_len]
                boundary_counts[seg_start:seg_end] += 1.0
            else:
                stride = 150
                starts = list(range(0, seg_len - 300 + 1, stride))
                if not starts or starts[-1] + 300 < seg_len:
                    starts.append(seg_len - 300)
                for start in starts:
                    window = seg_x[start:start + 300]
                    x_tensor = torch.from_numpy(window).float().transpose(0, 1).unsqueeze(0).to(device)
                    phase_logits, boundary_logits = model(x_tensor)
                    probs = F.softmax(phase_logits, dim=1).cpu().numpy()[0]
                    b_probs = torch.sigmoid(boundary_logits).cpu().numpy()[0, 0]
                    gs = seg_start + start
                    ge = gs + 300
                    phase_probs[gs:ge, :] += probs.T
                    phase_counts[gs:ge] += 1.0
                    boundary_probs[gs:ge] += b_probs
                    boundary_counts[gs:ge] += 1.0
    valid = phase_counts > 0
    phase_probs[valid] /= phase_counts[valid][:, None]
    boundary_probs[valid] /= boundary_counts[valid]
    return phase_probs, boundary_probs


def viterbi_decode_with_boundary(phase_probs, boundary_probs, penalty, boundary_weight):
    n = len(phase_probs)
    log_probs = np.log(np.clip(phase_probs, 1e-8, 1.0))
    dp = np.zeros((n, 2), dtype=float)
    back = np.zeros((n, 2), dtype=np.int64)
    dp[0] = log_probs[0]
    for i in range(1, n):
        switch_bonus = boundary_weight * float(boundary_probs[i])
        for state in range(2):
            stay_score = dp[i - 1, state]
            switch_score = dp[i - 1, 1 - state] - penalty + switch_bonus
            if switch_score > stay_score:
                dp[i, state] = log_probs[i, state] + switch_score
                back[i, state] = 1 - state
            else:
                dp[i, state] = log_probs[i, state] + stay_score
                back[i, state] = state
    labels = np.zeros(n, dtype=np.int64)
    labels[-1] = int(np.argmax(dp[-1]))
    for i in range(n - 2, -1, -1):
        labels[i] = back[i + 1, labels[i + 1]]
    return one_hot(labels)


def one_hot(labels):
    probs = np.zeros((len(labels), 2), dtype=float)
    probs[np.arange(len(labels)), labels.astype(int)] = 1.0
    return probs


def evaluate_phase_probs(stream_id, phase_probs, gt_reps, gt_phases):
    labels = np.argmax(phase_probs, axis=1)
    reps = parse_reps(labels)
    return evaluate_with_reps(stream_id, phase_probs, gt_reps, gt_phases, reps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--boundary-margin-samples", type=int, default=10)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--boundary-weights", default="0.1,0.2,0.4")
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/boundary_event_head_9fold_gpu_h64e20.json")
    args = parser.parse_args()

    boundary_weights = [float(x.strip()) for x in args.boundary_weights.split(",") if x.strip()]
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})

    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Remaining streams: {len(streams)}")
    print(f"Subjects: {subjects}")
    print(f"Settings: hidden={args.hidden}, epochs={args.epochs}, boundary_weights={boundary_weights}, device={device}")

    raw_results = []
    top5_results = []
    boundary_phase_results = []
    boundary_decoder_results = {f"boundary_b{w:g}": [] for w in boundary_weights}
    folds = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        rep_duration_priors = build_duration_priors(train_streams, [5])

        print("Training raw phase-only CNN...")
        raw_model, raw_mean, raw_std, raw_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"Raw train active segments={raw_segments}")

        print("Training boundary-aware CNN...")
        boundary_model, boundary_mean, boundary_std, boundary_segments = train_boundary_model(
            train_streams,
            cfg.imu_columns,
            args.hidden,
            args.epochs,
            device,
            args.boundary_margin_samples,
            args.boundary_loss_weight,
            args.pos_weight,
        )
        print(f"Boundary train active segments={boundary_segments}")

        print("Training active detector and evaluating variants...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)
        fold_raw = []
        fold_top5 = []
        fold_boundary_phase = []
        fold_boundary_decoder = {name: [] for name in boundary_decoder_results}

        for stream_id, df in test_streams:
            if "phase" not in df.columns:
                continue
            action = stream_action(stream_id)
            gt_phases = df["phase"].to_numpy()
            gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
            active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
            active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)

            raw_probs = predict_fast(raw_model, df, active_segments, cfg.imu_columns, raw_mean, raw_std, pca=None)
            raw_labels = raw_probs.argmax(axis=1)
            raw_reps = parse_reps(raw_labels)
            raw_result = evaluate_with_reps(stream_id, raw_probs, gt_reps, gt_phases, raw_reps)
            raw_results.append(raw_result)
            fold_raw.append(raw_result)

            top5_reps = raw_reps
            if action in ACTION_SETS["top5"]:
                threshold = threshold_for_action(rep_duration_priors, action, 5)
                top5_reps = merge_short_reps(raw_reps, threshold, max_gap_samples=50)
            top5_result = evaluate_with_reps(stream_id, raw_probs, gt_reps, gt_phases, top5_reps)
            top5_results.append(top5_result)
            fold_top5.append(top5_result)

            boundary_phase_probs, boundary_probs = predict_boundary_model(
                boundary_model, df, active_segments, cfg.imu_columns, boundary_mean, boundary_std
            )
            boundary_phase_decoded = viterbi_decode(smooth_ma(boundary_phase_probs, 25), 0.3)
            boundary_phase_result = evaluate_phase_probs(stream_id, boundary_phase_decoded, gt_reps, gt_phases)
            boundary_phase_results.append(boundary_phase_result)
            fold_boundary_phase.append(boundary_phase_result)

            smoothed = smooth_ma(boundary_phase_probs, 25)
            for w in boundary_weights:
                name = f"boundary_b{w:g}"
                decoded = viterbi_decode_with_boundary(smoothed, boundary_probs, penalty=0.3, boundary_weight=w)
                result = evaluate_phase_probs(stream_id, decoded, gt_reps, gt_phases)
                boundary_decoder_results[name].append(result)
                fold_boundary_decoder[name].append(result)

        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "raw": aggregate_rich(fold_raw),
            "top5_p5": aggregate_rich(fold_top5),
            "boundary_phase": aggregate_rich(fold_boundary_phase),
        }
        for name, results in fold_boundary_decoder.items():
            fold_summary[name] = aggregate_rich(results)
        folds.append(fold_summary)

        for name in ["raw", "top5_p5", "boundary_phase", *fold_boundary_decoder.keys()]:
            agg = fold_summary[name]
            print(
                f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
                f"MAE={agg['mean_abs_count_error']:.2f} CE={agg['ce_ratio_mae']:.3f} "
                f"Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}"
            )

    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn_plus_boundary_head",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "boundary_margin_samples": args.boundary_margin_samples,
            "boundary_loss_weight": args.boundary_loss_weight,
            "pos_weight": args.pos_weight,
            "boundary_decoder_weights": boundary_weights,
            "excluded_sessions": EXCLUDED_SESSIONS,
            "top5_reference_actions": ACTION_SETS["top5"],
        },
        "raw_total": aggregate_rich(raw_results),
        "top5_p5_total": aggregate_rich(top5_results),
        "boundary_phase_total": aggregate_rich(boundary_phase_results),
        "boundary_decoder_totals": {name: aggregate_rich(results) for name, results in boundary_decoder_results.items()},
        "raw_per_action": group_aggregate(raw_results, lambda item: item["action"]),
        "top5_p5_per_action": group_aggregate(top5_results, lambda item: item["action"]),
        "boundary_phase_per_action": group_aggregate(boundary_phase_results, lambda item: item["action"]),
        "boundary_decoder_per_action": {
            name: group_aggregate(results, lambda item: item["action"])
            for name, results in boundary_decoder_results.items()
        },
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    totals = {
        "raw": output["raw_total"],
        "top5_p5": output["top5_p5_total"],
        "boundary_phase": output["boundary_phase_total"],
    }
    totals.update(output["boundary_decoder_totals"])
    for name, agg in totals.items():
        print(
            f"{name}: RepF1={agg['rep_f1']:.4f} Exact={agg['exact_count_acc']:.3f} "
            f"MAE={agg['mean_abs_count_error']:.3f} CE={agg['ce_ratio_mae']:.3f} "
            f"Over={agg['over_rate']:.3f} Under={agg['under_rate']:.3f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
