"""9-fold LOSO probe for raw6 CNN plus streaming-safe derived IMU channels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.new_c_pipeline.compare_phase_models import (  # noqa: E402
    PhaseCompareConfig,
    extract_active_segments,
    predict_active,
    train_active_detector,
)
from scripts.new_c_pipeline.master_eval import evaluate_stream as evaluate_stream_rich  # noqa: E402
from scripts.new_c_pipeline.raw6_cnn_comprehensive_9fold import aggregate_rich  # noqa: E402
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    CausalCNN_PhaseOnly,
    EXCLUDED_SESSIONS,
    PhaseDataset,
    extract_active_segments_data,
    normalize,
    predict_fast,
    set_seed,
    should_exclude,
    train_fast,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


RAW_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


def stream_subject(stream_id: str) -> str:
    return stream_id.split("/")[0]


def stream_action(stream_id: str) -> str:
    parts = [p for p in stream_id.split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def add_derived_channels(df, mode: str):
    """Return a copied dataframe with causal derived channels added."""
    out = df.copy()
    x = out[list(RAW_COLUMNS)].to_numpy(dtype=np.float32)
    acc = x[:, :3]
    gyro = x[:, 3:]

    if mode in {"mag", "mag_delta", "mag_delta_jerk"}:
        out["acc_mag"] = np.linalg.norm(acc, axis=1).astype(np.float32)
        out["gyro_mag"] = np.linalg.norm(gyro, axis=1).astype(np.float32)

    if mode in {"delta", "mag_delta", "mag_delta_jerk"}:
        delta = np.diff(x, axis=0, prepend=x[:1])
        for idx, col in enumerate(RAW_COLUMNS):
            out[f"d_{col}"] = delta[:, idx].astype(np.float32)

    if mode == "mag_delta_jerk":
        delta = np.diff(x, axis=0, prepend=x[:1])
        jerk = np.diff(delta, axis=0, prepend=delta[:1])
        for idx, col in enumerate(RAW_COLUMNS):
            out[f"j_{col}"] = jerk[:, idx].astype(np.float32)

    return out


def columns_for_mode(mode: str):
    columns = list(RAW_COLUMNS)
    if mode in {"mag", "mag_delta", "mag_delta_jerk"}:
        columns.extend(["acc_mag", "gyro_mag"])
    if mode in {"delta", "mag_delta", "mag_delta_jerk"}:
        columns.extend([f"d_{col}" for col in RAW_COLUMNS])
    if mode == "mag_delta_jerk":
        columns.extend([f"j_{col}" for col in RAW_COLUMNS])
    return columns


def transform_streams(streams, mode: str):
    if mode == "raw6":
        return streams
    return [(sid, add_derived_channels(df, mode)) for sid, df in streams]


def train_model(train_streams, input_columns, hidden, epochs, device, train_idx=None, val_idx=None):
    segments, labels = extract_active_segments_data(train_streams, input_columns)
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments)
    if train_idx is None or val_idx is None:
        n_val = max(1, int(n_total * 0.15))
        indices = np.random.RandomState(42).permutation(n_total)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    train_ds = PhaseDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = CausalCNN_PhaseOnly(len(input_columns), hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=device)
    return model, mean, std, train_idx, val_idx, len(segments)


def evaluate_model(test_streams, cfg, active_models, active_scalers, pack, input_columns):
    model, mean, std = pack
    results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        phase_probs = predict_fast(model, df, active_segments, input_columns, mean, std, pca=None)
        result = evaluate_stream_rich(stream_id, df, phase_probs, gt_reps, gt_phases)
        result["subject"] = stream_subject(stream_id)
        result["action"] = stream_action(stream_id)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--mode", choices=["mag", "delta", "mag_delta", "mag_delta_jerk"], default="mag_delta")
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/raw6_plus_mag_delta_cnn_9fold_fast.json")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    subjects = sorted({stream_subject(sid) for sid, _ in streams})

    feature_columns = columns_for_mode(args.mode)
    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Remaining streams: {len(streams)}")
    print(f"Subjects: {subjects}")
    print(f"Settings: raw6 vs raw6_plus_{args.mode}, hidden={args.hidden}, epochs={args.epochs}, device={device}")
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")

    all_raw_results = []
    all_feature_results = []
    folds = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        train_feature_streams = transform_streams(train_streams, args.mode)
        test_feature_streams = transform_streams(test_streams, args.mode)

        print("Training raw6 CNN...")
        raw_model, raw_mean, raw_std, train_idx, val_idx, raw_segments = train_model(
            train_streams, list(RAW_COLUMNS), args.hidden, args.epochs, device
        )
        print(f"Raw train segments={raw_segments}")

        print(f"Training raw6_plus_{args.mode} CNN...")
        feature_model, feature_mean, feature_std, _, _, feature_segments = train_model(
            train_feature_streams, feature_columns, args.hidden, args.epochs, device, train_idx=train_idx, val_idx=val_idx
        )
        print(f"Feature train segments={feature_segments}")

        print("Training active detector on raw6 streams...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)

        raw_results = evaluate_model(test_streams, cfg, active_models, active_scalers, (raw_model, raw_mean, raw_std), list(RAW_COLUMNS))
        feature_results = evaluate_model(
            test_feature_streams,
            cfg,
            active_models,
            active_scalers,
            (feature_model, feature_mean, feature_std),
            feature_columns,
        )

        raw_agg = aggregate_rich(raw_results)
        feature_agg = aggregate_rich(feature_results)
        all_raw_results.extend(raw_results)
        all_feature_results.extend(feature_results)
        folds.append({"fold": fold_idx, "test_subject": test_subject, "raw6": raw_agg, f"raw6_plus_{args.mode}": feature_agg})

        print(
            f"RAW: RepF1={raw_agg['rep_f1']:.4f} Exact={raw_agg['exact_count_acc']:.3f} "
            f"MAE={raw_agg['mean_abs_count_error']:.2f} PhaseIoU={raw_agg['phase_seg_iou_f1_50_avg']:.4f} "
            f"CE={raw_agg['ce_ratio_mae']:.3f}"
        )
        print(
            f"{args.mode}: RepF1={feature_agg['rep_f1']:.4f} Exact={feature_agg['exact_count_acc']:.3f} "
            f"MAE={feature_agg['mean_abs_count_error']:.2f} PhaseIoU={feature_agg['phase_seg_iou_f1_50_avg']:.4f} "
            f"CE={feature_agg['ce_ratio_mae']:.3f}"
        )

    raw_total = aggregate_rich(all_raw_results)
    feature_total = aggregate_rich(all_feature_results)
    output = {
        "settings": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "mode": args.mode,
            "raw_columns": list(RAW_COLUMNS),
            "feature_columns": feature_columns,
            "decoder": "MA25 + Viterbi penalty=0.3",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw6_total": raw_total,
        f"raw6_plus_{args.mode}_total": feature_total,
        "folds": folds,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    print(
        f"RAW: RepF1={raw_total['rep_f1']:.4f} Exact={raw_total['exact_count_acc']:.3f} "
        f"MAE={raw_total['mean_abs_count_error']:.2f} PhaseIoU={raw_total['phase_seg_iou_f1_50_avg']:.4f} "
        f"CE={raw_total['ce_ratio_mae']:.3f}"
    )
    print(
        f"{args.mode}: RepF1={feature_total['rep_f1']:.4f} Exact={feature_total['exact_count_acc']:.3f} "
        f"MAE={feature_total['mean_abs_count_error']:.2f} PhaseIoU={feature_total['phase_seg_iou_f1_50_avg']:.4f} "
        f"CE={feature_total['ce_ratio_mae']:.3f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
