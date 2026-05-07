from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from evaluation.reporting import write_standard_run_outputs
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
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class WindowedIMUDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[index])
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds_all: List[int] = []
    y_all: List[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

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

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        y_all.extend(y.tolist())

    return {
        "accuracy": float(accuracy_score(y_all, preds_all)),
        "macro_f1": float(f1_score(y_all, preds_all, average="macro")),
    }


def build_configs(config_path: Path) -> tuple[dict, FeatureConfig, WindowConfig, ModelConfig, TrainConfig]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    feature_cfg = FeatureConfig(**raw.get("feature", {}))
    window_cfg = WindowConfig(**raw.get("window", {}))
    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))

    return raw, feature_cfg, window_cfg, model_cfg, train_cfg


def _get_timestamped_dir(base_dir: Path) -> Path:
    """Create a timestamped subdirectory for organizing outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / timestamp


def train_student(config_path: Path, use_timestamp: bool = True) -> None:
    raw_cfg, feature_cfg, window_cfg, model_cfg, train_cfg = build_configs(config_path)
    set_seed(train_cfg.seed)

    data_cfg = raw_cfg.get("data", {})
    io_cfg = raw_cfg.get("io", {})

    data_dir = Path(data_cfg.get("data_dir", "./data"))
    csv_glob = data_cfg.get("csv_glob", "*.csv")
    exclude_patterns = data_cfg.get("exclude_patterns", None)
    include_actions = data_cfg.get("include_actions", None)
    base_output_dir = Path(io_cfg.get("output_dir", "./artifacts/action_classification"))

    # Create timestamped subdirectory
    if use_timestamp:
        output_dir = _get_timestamped_dir(base_output_dir)
    else:
        output_dir = base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_dir}")

    sequences, subjects = prepare_sequences_from_folder(
        data_dir=data_dir,
        feature_cfg=feature_cfg,
        sample_rate_hz=window_cfg.sample_rate_hz,
        csv_glob=csv_glob,
        exclude_patterns=exclude_patterns,
        include_actions=include_actions,
    )

    train_subj, val_subj, test_subj = split_subjects(subjects, seed=train_cfg.seed)

    train_seqs = filter_sequences_by_subject(sequences, train_subj, feature_cfg.subject_column)
    val_seqs = filter_sequences_by_subject(sequences, val_subj, feature_cfg.subject_column)
    test_seqs = filter_sequences_by_subject(sequences, test_subj, feature_cfg.subject_column)

    stats = compute_train_stats(train_seqs, feature_cfg.imu_columns)
    stats.save(output_dir / "zscore_stats.json")

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

    label_encoder.to_json(output_dir / "label_map.json")

    model_cfg.input_channels = len(feature_cfg.imu_columns)
    model_cfg.num_classes = len(label_encoder.class_to_index)

    train_loader = DataLoader(
        WindowedIMUDataset(x_train, y_train),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )
    val_loader = DataLoader(
        WindowedIMUDataset(x_val, y_val),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )
    test_loader = DataLoader(
        WindowedIMUDataset(x_test, y_test),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    model = InertialStudent(model_cfg, window_cfg.window_size).to(train_cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, train_cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train_cfg.device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, train_cfg.device)

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

    torch.save(best_state, output_dir / "student_best.pt")
    model.load_state_dict(best_state)
    model.to(train_cfg.device)

    metrics = evaluate_model(model, test_loader, train_cfg.device)
    print("test_metrics:", metrics)

    summary = {
        "train_subjects": train_subj,
        "val_subjects": val_subj,
        "test_subjects": test_subj,
        "window_size": window_cfg.window_size,
        "stride_size": window_cfg.stride_size,
        "sample_rate_hz": window_cfg.sample_rate_hz,
        "test_metrics": metrics,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_standard_run_outputs(
        output_dir,
        task="action_classification",
        model_name="inertial_student",
        title="Student Model Report",
        overall=metrics,
        details={"legacy_summary": summary},
        artifacts={
            "Legacy summary": (output_dir / "train_summary.json").as_posix(),
            "Checkpoint": (output_dir / "student_best.pt").as_posix(),
            "Z-score stats": (output_dir / "zscore_stats.json").as_posix(),
            "Label map": (output_dir / "label_map.json").as_posix(),
            "Effective config": (output_dir / "effective_config.json").as_posix(),
        },
        config={
            "train_subjects": train_subj,
            "val_subjects": val_subj,
            "test_subjects": test_subj,
            "window_size": window_cfg.window_size,
            "stride_size": window_cfg.stride_size,
        },
        config_path=config_path,
        notes=["Standardized report added for cross-model comparison."],
    )

    effective_cfg = {
        "feature": feature_cfg.__dict__,
        "window": {
            "sample_rate_hz": window_cfg.sample_rate_hz,
            "window_seconds": window_cfg.window_seconds,
            "stride_seconds": window_cfg.stride_seconds,
            "window_size": window_cfg.window_size,
            "stride_size": window_cfg.stride_size,
        },
        "model": model_cfg.__dict__,
        "train": train_cfg.__dict__,
        "data": data_cfg,
        "io": io_cfg,
    }
    (output_dir / "effective_config.json").write_text(json.dumps(effective_cfg, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inertial student model (action classification only).")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamped subfolder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_student(args.config, use_timestamp=not args.no_timestamp)


if __name__ == "__main__":
    main()
