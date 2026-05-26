"""9-fold LOSO: compare raw 6-axis CNN input with a raw axis subset."""
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
from scripts.new_c_pipeline.test_pca_input import (  # noqa: E402
    CausalCNN_PhaseOnly,
    EXCLUDED_SESSIONS,
    PhaseDataset,
    aggregate,
    evaluate_stream,
    extract_active_segments_data,
    normalize,
    predict_fast,
    set_seed,
    should_exclude,
    train_fast,
)
from train.micro_macro_recognition import _load_streams, truth_reps_from_labels  # noqa: E402


def train_subset_model(train_streams, train_labels_source, imu_columns, hidden, epochs, device, train_idx=None, val_idx=None):
    segments, labels = extract_active_segments_data(train_streams, imu_columns)
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
    model = CausalCNN_PhaseOnly(len(imu_columns), hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=device)
    return model, mean, std, train_idx, val_idx, len(segments)


def evaluate_fold(test_streams, cfg, active_models, active_scalers, raw_pack, subset_pack, subset_columns):
    raw_model, raw_mean, raw_std = raw_pack
    subset_model, subset_mean, subset_std = subset_pack

    raw_results = []
    subset_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)

        raw_probs = predict_fast(raw_model, df, active_segments, cfg.imu_columns, raw_mean, raw_std, pca=None)
        raw_results.append(evaluate_stream(stream_id, df, raw_probs, gt_reps, gt_phases))

        subset_probs = predict_fast(subset_model, df, active_segments, subset_columns, subset_mean, subset_std, pca=None)
        subset_results.append(evaluate_stream(stream_id, df, subset_probs, gt_reps, gt_phases))

    return raw_results, subset_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--subset-columns", default="ax,ay,az")
    parser.add_argument("--subset-name", default="acc3")
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/acc3_cnn_9fold_gpu_h64e20.json")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PhaseCompareConfig()
    subset_columns = [col.strip() for col in args.subset_columns.split(",") if col.strip()]

    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    subjects = sorted({sid.split("/")[0] for sid, _ in streams})

    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Remaining streams: {len(streams)}")
    print(f"Subjects: {subjects}")
    print(f"Settings: hidden={args.hidden}, epochs={args.epochs}, subset={args.subset_name}:{subset_columns}, device={device}")

    all_raw_results = []
    all_subset_results = []
    fold_summaries = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in streams if sid.startswith(f"{test_subject}/")]
        print(f"Train streams={len(train_streams)}, test streams={len(test_streams)}")

        print("Training raw 6-axis CNN...")
        raw_model, raw_mean, raw_std, train_idx, val_idx, raw_segments = train_subset_model(
            train_streams, None, cfg.imu_columns, args.hidden, args.epochs, device
        )
        print(f"Raw train segments={raw_segments}")

        print(f"Training {args.subset_name} CNN...")
        subset_model, subset_mean, subset_std, _, _, subset_segments = train_subset_model(
            train_streams, None, subset_columns, args.hidden, args.epochs, device, train_idx=train_idx, val_idx=val_idx
        )
        print(f"Subset train segments={subset_segments}")

        print("Training active detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)

        raw_results, subset_results = evaluate_fold(
            test_streams,
            cfg,
            active_models,
            active_scalers,
            (raw_model, raw_mean, raw_std),
            (subset_model, subset_mean, subset_std),
            subset_columns,
        )
        raw_agg = aggregate(raw_results)
        subset_agg = aggregate(subset_results)
        all_raw_results.extend(raw_results)
        all_subset_results.extend(subset_results)

        fold_summaries.append({
            "fold": fold_idx,
            "test_subject": test_subject,
            "test_streams": len(test_streams),
            "raw": raw_agg,
            args.subset_name: subset_agg,
        })

        print(
            f"RAW : RepF1={raw_agg['rep_f1']:.4f} Exact={raw_agg['exact_count_acc']:.3f} "
            f"MAE={raw_agg['mean_abs_count_error']:.2f} Over/Under={raw_agg['over_count']}/{raw_agg['under_count']} "
            f"PhaseF1={raw_agg['phase_macro_f1']:.4f}"
        )
        print(
            f"{args.subset_name}: RepF1={subset_agg['rep_f1']:.4f} Exact={subset_agg['exact_count_acc']:.3f} "
            f"MAE={subset_agg['mean_abs_count_error']:.2f} Over/Under={subset_agg['over_count']}/{subset_agg['under_count']} "
            f"PhaseF1={subset_agg['phase_macro_f1']:.4f}"
        )

    raw_total = aggregate(all_raw_results)
    subset_total = aggregate(all_subset_results)
    output = {
        "settings": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "subset_name": args.subset_name,
            "subset_columns": subset_columns,
            "decoder": "MA25 + Viterbi penalty=0.3",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw_total": raw_total,
        f"{args.subset_name}_total": subset_total,
        "folds": fold_summaries,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("TOTAL")
    print(f"{'=' * 72}")
    print(
        f"RAW : RepF1={raw_total['rep_f1']:.4f} Exact={raw_total['exact_count_acc']:.3f} "
        f"MAE={raw_total['mean_abs_count_error']:.2f} Over/Under={raw_total['over_count']}/{raw_total['under_count']} "
        f"PhaseF1={raw_total['phase_macro_f1']:.4f}"
    )
    print(
        f"{args.subset_name}: RepF1={subset_total['rep_f1']:.4f} Exact={subset_total['exact_count_acc']:.3f} "
        f"MAE={subset_total['mean_abs_count_error']:.2f} Over/Under={subset_total['over_count']}/{subset_total['under_count']} "
        f"PhaseF1={subset_total['phase_macro_f1']:.4f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
