"""Comprehensive 9-fold LOSO metrics for the raw 6-axis 1D Causal CNN."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def stream_subject(stream_id: str) -> str:
    return stream_id.split("/")[0]


def stream_action(stream_id: str) -> str:
    parts = [p for p in stream_id.split("/") if p]
    return parts[-2] if len(parts) >= 3 else "unknown"


def train_raw6_model(train_streams, imu_columns, hidden, epochs, device):
    segments, labels = extract_active_segments_data(train_streams, imu_columns)
    mean, std, norm_segments = normalize(segments)
    n_total = len(norm_segments)
    n_val = max(1, int(n_total * 0.15))
    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:-n_val], indices[-n_val:]

    train_ds = PhaseDataset([norm_segments[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = PhaseDataset([norm_segments[i] for i in val_idx], [labels[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    model = CausalCNN_PhaseOnly(len(imu_columns), hidden)
    model = train_fast(model, train_loader, val_loader, epochs=epochs, device=device)
    return model, mean, std, len(segments)


def aggregate_rich(results):
    if not results:
        return {}

    n = len(results)
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    rep_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pred_counts = np.array([r["pred_count"] for r in results], dtype=float)
    gt_counts = np.array([r["gt_count"] for r in results], dtype=float)
    signed_errors = pred_counts - gt_counts
    abs_errors = np.abs(signed_errors)
    trans = [r["transition_mae_ms"] for r in results if r.get("transition_mae_ms") is not None]
    ce = [r["ce_ratio_mae"] for r in results if r.get("ce_ratio_mae") is not None]

    c_seg = float(np.mean([r["concentric_seg_f1"] for r in results]))
    e_seg = float(np.mean([r["eccentric_seg_f1"] for r in results]))
    return {
        "streams": n,
        "rep_precision": float(precision),
        "rep_recall": float(recall),
        "rep_f1": float(rep_f1),
        "exact_count_acc": float(np.mean([r["exact_count"] for r in results])),
        "within_1_count_acc": float(np.mean(abs_errors <= 1.0)),
        "mean_abs_count_error": float(np.mean(abs_errors)),
        "median_abs_count_error": float(np.median(abs_errors)),
        "count_rmse": float(np.sqrt(np.mean(signed_errors ** 2))),
        "count_bias_pred_minus_gt": float(np.mean(signed_errors)),
        "mean_pred_count": float(np.mean(pred_counts)),
        "mean_gt_count": float(np.mean(gt_counts)),
        "over_count": int(sum(r["over"] for r in results)),
        "under_count": int(sum(r["under"] for r in results)),
        "over_rate": float(np.mean([r["over"] for r in results])),
        "under_rate": float(np.mean([r["under"] for r in results])),
        "phase_macro_f1": float(np.mean([r["phase_macro_f1"] for r in results])),
        "phase_accuracy": float(np.mean([r["phase_accuracy"] for r in results])),
        "transition_mae_ms": float(np.mean(trans)) if trans else None,
        "concentric_seg_iou_f1_50": c_seg,
        "eccentric_seg_iou_f1_50": e_seg,
        "phase_seg_iou_f1_50_avg": float((c_seg + e_seg) / 2.0),
        "ce_ratio_mae": float(np.mean(ce)) if ce else None,
        "ce_ratio_valid_streams": int(len(ce)),
    }


def group_aggregate(results, key_fn):
    grouped = defaultdict(list)
    for result in results:
        grouped[key_fn(result)].append(result)
    return {key: aggregate_rich(items) for key, items in sorted(grouped.items())}


def evaluate_fold(test_streams, train_streams, cfg, model, mean, std):
    active_models, active_scalers = train_active_detector(train_streams, cfg)
    fold_results = []
    for stream_id, df in test_streams:
        if "phase" not in df.columns:
            continue
        gt_phases = df["phase"].to_numpy()
        gt_reps = truth_reps_from_labels(gt_phases, min_phase_samples=3)
        active_probs = predict_active(active_models, active_scalers, stream_id, df, cfg)
        active_segments = extract_active_segments(active_probs, threshold=0.5, min_consecutive=3)
        phase_probs = predict_fast(model, df, active_segments, cfg.imu_columns, mean, std, pca=None)
        result = evaluate_stream_rich(stream_id, df, phase_probs, gt_reps, gt_phases)
        result["subject"] = stream_subject(stream_id)
        result["action"] = stream_action(stream_id)
        fold_results.append(result)
    return fold_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", default="artifacts/cnn_variant_comparison/raw6_cnn_comprehensive_9fold_gpu_h64e20.json")
    args = parser.parse_args()

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
    print(f"Settings: raw6 1D Causal CNN, hidden={args.hidden}, epochs={args.epochs}, device={device}")

    all_results = []
    folds = []
    for fold_idx, test_subject in enumerate(subjects, 1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(subjects)}: held-out subject = {test_subject}")
        print(f"{'=' * 72}")
        train_streams = [(sid, df) for sid, df in streams if stream_subject(sid) != test_subject]
        test_streams = [(sid, df) for sid, df in streams if stream_subject(sid) == test_subject]
        print(f"Train streams={len(train_streams)}, test streams={len(test_streams)}")

        print("Training raw 6-axis CNN...")
        model, mean, std, n_segments = train_raw6_model(train_streams, cfg.imu_columns, args.hidden, args.epochs, device)
        print(f"Train active segments={n_segments}")

        print("Evaluating rich metrics...")
        fold_results = evaluate_fold(test_streams, train_streams, cfg, model, mean, std)
        fold_agg = aggregate_rich(fold_results)
        folds.append({"fold": fold_idx, "test_subject": test_subject, "summary": fold_agg})
        all_results.extend(fold_results)
        print(
            f"RepF1={fold_agg['rep_f1']:.4f} Exact={fold_agg['exact_count_acc']:.3f} "
            f"MAE={fold_agg['mean_abs_count_error']:.2f} PhaseF1={fold_agg['phase_macro_f1']:.4f} "
            f"PhaseIoU={fold_agg['phase_seg_iou_f1_50_avg']:.4f} CE_MAE={fold_agg['ce_ratio_mae']:.4f}"
        )

    overall = aggregate_rich(all_results)
    output = {
        "settings": {
            "model": "raw6_global_2class_1d_causal_cnn",
            "input_columns": list(cfg.imu_columns),
            "input_shape": "[batch, 6, 300]",
            "epochs": args.epochs,
            "hidden": args.hidden,
            "decoder": "MA25 + Viterbi penalty=0.3",
            "excluded_sessions": EXCLUDED_SESSIONS,
        },
        "overall": overall,
        "per_subject": group_aggregate(all_results, lambda item: item["subject"]),
        "per_action": group_aggregate(all_results, lambda item: item["action"]),
        "folds": folds,
        "streams": all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("OVERALL")
    print(f"{'=' * 72}")
    print(json.dumps(overall, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
