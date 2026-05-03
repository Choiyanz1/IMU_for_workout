"""Knowledge distillation: use teacher soft labels to train the student transformer.

Usage:
    1. First run action classification training to generate soft labels:
       python -m train.action_classification --config config.yaml

    2. Then distill into the student model:
       python -m train.distillation --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from models.inertial_student import InertialStudent, ModelConfig
from preprocessing.window_pipeline import (
    WindowConfig,
    apply_zscore,
    build_window_dataset,
    compute_train_stats,
    set_seed,
    split_subjects,
)


@dataclass
class DistillConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Distillation temperature (higher = softer probabilities)
    temperature: float = 3.0
    # Weight of the soft-label loss vs hard-label loss (alpha * soft + (1-alpha) * hard)
    alpha: float = 0.7


class DistillDataset(Dataset):
    """Dataset that returns (window, hard_label, soft_label_probs)."""
    def __init__(self, windows: np.ndarray, labels: np.ndarray, soft_probs: np.ndarray) -> None:
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.soft_probs = soft_probs.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[index])
        y = torch.tensor(self.labels[index], dtype=torch.long)
        soft = torch.from_numpy(self.soft_probs[index])
        return x, y, soft


def distillation_loss(
    student_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    soft_probs: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Combined KD loss: alpha * KL-divergence(soft) + (1-alpha) * CrossEntropy(hard)."""
    # Hard label loss
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    # Soft label loss (KL divergence with temperature)
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    # soft_probs are already probabilities from AutoGluon; apply temperature scaling
    soft_targets = torch.pow(soft_probs, 1.0 / temperature)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

    kl_loss = F.kl_div(log_student, soft_targets, reduction="batchmean") * (temperature ** 2)

    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def run_distill_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    temperature: float,
    alpha: float,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds_all: List[int] = []
    y_all: List[int] = []

    for x, y, soft in loader:
        x = x.to(device)
        y = y.to(device)
        soft = soft.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = distillation_loss(logits, y, soft, temperature, alpha)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        y_all.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(y_all, preds_all) if y_all else 0.0
    return avg_loss, acc


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    preds_all: List[int] = []
    y_all: List[int] = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        logits = model(x)
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        y_all.extend(y.tolist())

    return {
        "accuracy": float(accuracy_score(y_all, preds_all)),
        "macro_f1": float(f1_score(y_all, preds_all, average="macro")),
    }


def _load_soft_labels(soft_labels_csv: Path, label_encoder, n_samples: int) -> np.ndarray:
    """Load AutoGluon soft labels and align columns to label_encoder order."""
    import pandas as pd
    df = pd.read_csv(soft_labels_csv)

    n_classes = len(label_encoder.class_to_index)
    probs = np.zeros((n_samples, n_classes), dtype=np.float32)

    for class_name, class_idx in label_encoder.class_to_index.items():
        if class_name in df.columns:
            probs[:, class_idx] = df[class_name].values[:n_samples].astype(np.float32)
        elif str(class_name) in df.columns:
            probs[:, class_idx] = df[str(class_name)].values[:n_samples].astype(np.float32)

    # Normalize rows
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    probs = probs / row_sums

    return probs


def _get_timestamped_dir(base_dir: Path) -> Path:
    """Create a timestamped subdirectory for organizing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


def _find_latest_autogluon_output(base_dir: Path) -> Path:
    """Find the most recent timestamped AutoGluon output directory."""
    if not base_dir.exists():
        return base_dir

    # Look for timestamped subdirectories (format: YYYYMMDD_HHMMSS)
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == '_']
    if not subdirs:
        # Fall back to base directory if no timestamped folders found
        return base_dir

    # Sort by directory name (timestamp format ensures chronological order)
    latest = sorted(subdirs)[-1]
    return latest


def distill_from_autogluon(config_path: Path, use_timestamp: bool = True) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    feature_cfg = FeatureConfig(**raw.get("feature", {}))
    window_cfg = WindowConfig(**raw.get("window", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))

    # Distill config from train section + distill overrides
    train_raw = raw.get("train", {})
    distill_raw = raw.get("distill", {})
    distill_cfg = DistillConfig(
        seed=distill_raw.get("seed", train_raw.get("seed", 42)),
        batch_size=distill_raw.get("batch_size", train_raw.get("batch_size", 64)),
        epochs=distill_raw.get("epochs", 50),
        lr=distill_raw.get("lr", train_raw.get("lr", 1e-4)),
        weight_decay=distill_raw.get("weight_decay", train_raw.get("weight_decay", 1e-5)),
        num_workers=distill_raw.get("num_workers", train_raw.get("num_workers", 0)),
        device=distill_raw.get("device", train_raw.get("device", "cpu")),
        temperature=distill_raw.get("temperature", 3.0),
        alpha=distill_raw.get("alpha", 0.7),
    )

    set_seed(distill_cfg.seed)

    data_cfg = raw.get("data", {})
    io_cfg = raw.get("io", {})

    data_dir = Path(data_cfg.get("data_dir", "./data"))
    csv_glob = data_cfg.get("csv_glob", "*.csv")
    exclude_patterns = data_cfg.get("exclude_patterns", None)
    include_actions = data_cfg.get("include_actions", None)
    base_ag_dir = Path(io_cfg.get("output_dir", "./artifacts/action_classification"))

    # Find the latest AutoGluon output (timestamped)
    ag_output_dir = _find_latest_autogluon_output(base_ag_dir)
    print(f"[INFO] Using AutoGluon output: {ag_output_dir}")

    # Create output directory for distillation
    base_output_dir = Path(io_cfg.get("output_dir", "./artifacts/action_classification"))
    if use_timestamp:
        distill_output_dir = _get_timestamped_dir(base_output_dir / "distilled")
    else:
        distill_output_dir = base_output_dir / "distilled"
    distill_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Distillation output directory: {distill_output_dir}")

    # Check that AutoGluon soft labels exist
    soft_labels_path = ag_output_dir / "train_soft_labels.csv"
    if not soft_labels_path.exists():
        raise FileNotFoundError(
            f"Soft labels not found: {soft_labels_path}\n"
            "Run `python -m train.action_classification --config config.yaml` first."
        )

    sequences, subjects = prepare_sequences_from_folder(
        data_dir=data_dir,
        feature_cfg=feature_cfg,
        sample_rate_hz=window_cfg.sample_rate_hz,
        exclude_patterns=exclude_patterns,
        include_actions=include_actions,
        csv_glob=csv_glob,
    )

    train_subj, val_subj, test_subj = split_subjects(subjects, seed=distill_cfg.seed)

    train_seqs = filter_sequences_by_subject(sequences, train_subj, feature_cfg.subject_column)
    val_seqs = filter_sequences_by_subject(sequences, val_subj, feature_cfg.subject_column)
    test_seqs = filter_sequences_by_subject(sequences, test_subj, feature_cfg.subject_column)

    stats = compute_train_stats(train_seqs, feature_cfg.imu_columns)
    stats.save(distill_output_dir / "zscore_stats.json")

    train_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in train_seqs]
    val_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in val_seqs]
    test_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in test_seqs]

    x_train, y_train, _, label_encoder = build_window_dataset(
        train_seqs,
        imu_columns=feature_cfg.imu_columns,
        label_column=feature_cfg.label_column,
        subject_column=feature_cfg.subject_column,
        window_cfg=window_cfg,
        label_encoder=None,
    )
    x_val, y_val, _, _ = build_window_dataset(
        val_seqs,
        imu_columns=feature_cfg.imu_columns,
        label_column=feature_cfg.label_column,
        subject_column=feature_cfg.subject_column,
        window_cfg=window_cfg,
        label_encoder=label_encoder,
    )
    x_test, y_test, _, _ = build_window_dataset(
        test_seqs,
        imu_columns=feature_cfg.imu_columns,
        label_column=feature_cfg.label_column,
        subject_column=feature_cfg.subject_column,
        window_cfg=window_cfg,
        label_encoder=label_encoder,
    )

    label_encoder.to_json(distill_output_dir / "label_map.json")

    # Load soft labels from AutoGluon
    soft_train = _load_soft_labels(soft_labels_path, label_encoder, len(x_train))
    print(f"[INFO] Loaded soft labels: {soft_train.shape}")
    print(f"[INFO] Temperature={distill_cfg.temperature}, Alpha={distill_cfg.alpha}")

    # For val set, create uniform soft labels (only use hard labels for eval)
    n_classes = len(label_encoder.class_to_index)
    soft_val = np.zeros((len(x_val), n_classes), dtype=np.float32)
    for i, label_idx in enumerate(y_val):
        soft_val[i, label_idx] = 1.0

    model_cfg.input_channels = len(feature_cfg.imu_columns)
    model_cfg.num_classes = n_classes

    train_loader = DataLoader(
        DistillDataset(x_train, y_train, soft_train),
        batch_size=distill_cfg.batch_size,
        shuffle=True,
        num_workers=distill_cfg.num_workers,
    )
    val_loader = DataLoader(
        DistillDataset(x_val, y_val, soft_val),
        batch_size=distill_cfg.batch_size,
        shuffle=False,
        num_workers=distill_cfg.num_workers,
    )
    test_loader = DataLoader(
        DistillDataset(x_test, y_test, np.zeros((len(x_test), n_classes), dtype=np.float32)),
        batch_size=distill_cfg.batch_size,
        shuffle=False,
        num_workers=distill_cfg.num_workers,
    )

    model = InertialStudent(model_cfg, window_cfg.window_size).to(distill_cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=distill_cfg.lr, weight_decay=distill_cfg.weight_decay)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, distill_cfg.epochs + 1):
        tr_loss, tr_acc = run_distill_epoch(
            model, train_loader, optimizer, distill_cfg.device,
            distill_cfg.temperature, distill_cfg.alpha,
        )
        # Evaluate with hard labels only (alpha=0)
        va_loss, va_acc = run_distill_epoch(
            model, val_loader, None, distill_cfg.device,
            distill_cfg.temperature, 0.0,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    torch.save(best_state, distill_output_dir / "student_distilled.pt")
    model.load_state_dict(best_state)
    model.to(distill_cfg.device)

    metrics = evaluate_model(model, test_loader, distill_cfg.device)
    print(f"\n{'=' * 60}")
    print("DISTILLED STUDENT - TEST METRICS")
    print(f"{'=' * 60}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  macro_f1:  {metrics['macro_f1']:.4f}")

    summary = {
        "method": "knowledge_distillation_from_autogluon",
        "temperature": distill_cfg.temperature,
        "alpha": distill_cfg.alpha,
        "epochs": distill_cfg.epochs,
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
        "window_size": window_cfg.window_size,
        "stride_size": window_cfg.stride_size,
        "sample_rate_hz": window_cfg.sample_rate_hz,
        "test_metrics": metrics,
    }
    (distill_output_dir / "distill_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved distilled model: {distill_output_dir / 'student_distilled.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Knowledge distillation: AutoGluon teacher -> Student transformer."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distill_from_autogluon(args.config, use_timestamp=not args.no_timestamp)


if __name__ == "__main__":
    main()
