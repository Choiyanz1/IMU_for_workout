"""
9-fold LOSO probe: raw 6-axis vs PCA-N input for 1D Causal CNN.

This is a full-subject follow-up to test_pca_input.py, but intentionally uses
the same fast settings as the probe by default, but can also run formal GPU
settings such as hidden=64 and epochs=20.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def train_raw_model(train_segments, train_labels, hidden, epochs, device):
    mean_raw, std_raw, norm_raw = normalize(train_segments)
    n_total = len(norm_raw)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    train_ds = PhaseDataset([norm_raw[i] for i in train_idx], [train_labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_raw[i] for i in val_idx], [train_labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = CausalCNN_PhaseOnly(6, hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=device)
    return model, mean_raw, std_raw, train_idx, val_idx


def train_pca_model(train_segments, train_labels, hidden, epochs, train_idx, val_idx, device, n_components):
    all_train_samples = np.concatenate(train_segments, axis=0)
    scaler = StandardScaler()
    all_train_std = scaler.fit_transform(all_train_samples)

    pca_full = PCA()
    pca_full.fit(all_train_std)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    pca = PCA(n_components=n_components)
    pca.fit(all_train_std)

    pca_segments = [pca.transform(scaler.transform(seg)) for seg in train_segments]
    mean_pca, std_pca, norm_pca = normalize(pca_segments)

    train_ds = PhaseDataset([norm_pca[i] for i in train_idx], [train_labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_pca[i] for i in val_idx], [train_labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = CausalCNN_PhaseOnly(n_components, hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=device)
    return model, mean_pca, std_pca, scaler, pca, cumvar.tolist()


def evaluate_fold(test_streams, cfg, active_models, active_scalers, raw_pack, pca_pack):
    raw_model, mean_raw, std_raw = raw_pack
    pca_model, mean_pca, std_pca, scaler, pca = pca_pack

    raw_results = []
    pca_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)

        raw_probs = predict_fast(raw_model, df, active_segments, cfg.imu_columns, mean_raw, std_raw, pca=None)
        raw_results.append(evaluate_stream(stream_id, df, raw_probs, gt_reps, gt_phases))

        pca_probs = predict_fast(
            pca_model,
            df,
            active_segments,
            cfg.imu_columns,
            mean_pca,
            std_pca,
            pca=(scaler, pca),
        )
        pca_results.append(evaluate_stream(stream_id, df, pca_probs, gt_reps, gt_phases))

    return raw_results, pca_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--pca-components", type=int, default=4)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/pca4_cnn_9fold_fast.json")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = PhaseCompareConfig()
    raw_cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    all_streams, subjects, _ = _load_streams(raw_cfg, ["sets"])
    streams = [(sid, df) for sid, df in all_streams if not should_exclude(sid)]
    subjects = sorted({sid.split("/")[0] for sid, _ in streams})

    print(f"Excluded sessions: {EXCLUDED_SESSIONS}")
    print(f"Remaining streams: {len(streams)}")
    print(f"Subjects: {subjects}")
    pca_label = f"PCA{args.pca_components}"
    print(f"Settings: hidden={args.hidden}, epochs={args.epochs}, PCA={args.pca_components}, device={device}")

    all_raw_results = []
    all_pca_results = []
    fold_summaries = []

    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")

        train_streams = [(sid, df) for sid, df in streams if not sid.startswith(f"{test_subject}/")]
        test_streams = [(sid, df) for sid, df in streams if sid.startswith(f"{test_subject}/")]
        train_segments, train_labels = extract_active_segments_data(train_streams, cfg.imu_columns)
        print(f"Train streams={len(train_streams)}, test streams={len(test_streams)}, train segments={len(train_segments)}")

        print("Training raw 6-axis CNN...")
        raw_model, mean_raw, std_raw, train_idx, val_idx = train_raw_model(
            train_segments, train_labels, args.hidden, args.epochs, device
        )

        print(f"Training {pca_label} CNN...")
        pca_model, mean_pca, std_pca, scaler, pca, cumvar = train_pca_model(
            train_segments, train_labels, args.hidden, args.epochs, train_idx, val_idx, device, args.pca_components
        )
        print(f"PCA cumulative variance: {[round(v, 3) for v in cumvar]}")

        print("Training active detector...")
        active_models, active_scalers = train_active_detector(train_streams, cfg)

        raw_results, pca_results = evaluate_fold(
            test_streams,
            cfg,
            active_models,
            active_scalers,
            (raw_model, mean_raw, std_raw),
            (pca_model, mean_pca, std_pca, scaler, pca),
        )
        raw_agg = aggregate(raw_results)
        pca_agg = aggregate(pca_results)
        all_raw_results.extend(raw_results)
        all_pca_results.extend(pca_results)

        fold_summary = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "test_streams": len(test_streams),
            "pca_cumulative_variance": cumvar,
            "raw": raw_agg,
            "pca": pca_agg,
        }
        fold_summaries.append(fold_summary)

        print(
            f"RAW : RepF1={raw_agg['rep_f1']:.4f} Exact={raw_agg['exact_count_acc']:.3f} "
            f"MAE={raw_agg['mean_abs_count_error']:.2f} Over/Under={raw_agg['over_count']}/{raw_agg['under_count']} "
            f"PhaseF1={raw_agg['phase_macro_f1']:.4f}"
        )
        print(
            f"{pca_label}: RepF1={pca_agg['rep_f1']:.4f} Exact={pca_agg['exact_count_acc']:.3f} "
            f"MAE={pca_agg['mean_abs_count_error']:.2f} Over/Under={pca_agg['over_count']}/{pca_agg['under_count']} "
            f"PhaseF1={pca_agg['phase_macro_f1']:.4f}"
        )

    raw_total = aggregate(all_raw_results)
    pca_total = aggregate(all_pca_results)
    output = {
        "settings": {
            "epochs": args.epochs,
            "hidden": args.hidden,
            "pca_components": args.pca_components,
            "decoder": "MA25 + Viterbi penalty=0.3",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "raw_total": raw_total,
        "pca_total": pca_total,
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
        f"{pca_label}: RepF1={pca_total['rep_f1']:.4f} Exact={pca_total['exact_count_acc']:.3f} "
        f"MAE={pca_total['mean_abs_count_error']:.2f} Over/Under={pca_total['over_count']}/{pca_total['under_count']} "
        f"PhaseF1={pca_total['phase_macro_f1']:.4f}"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
