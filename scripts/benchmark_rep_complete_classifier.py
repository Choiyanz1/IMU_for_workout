from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cb = _load_module(ROOT / "scripts" / "compare_baselines.py", "compare_baselines_mod")

from train.action_classification import _compute_rich_features, _compute_stats_features
from preprocessing.micro_macro_segments import truth_reps_from_labels

try:
    import xgboost as xgb
    _XGB_AVAIL = True
except ImportError:
    _XGB_AVAIL = False
try:
    import lightgbm as lgb
    _LGB_AVAIL = True
except Exception:
    _LGB_AVAIL = False
    print("[WARN] LightGBM import failed (pandas compat), skipping", flush=True)
try:
    import catboost as catb
    _CATB_AVAIL = True
except ImportError:
    _CATB_AVAIL = False

import torch
import torch.nn as nn
import torch.nn.functional as F


def _action_from_stream_id(stream_id: str) -> str:
    parts = [p for p in str(stream_id).split("/") if p]
    return parts[-2] if len(parts) >= 2 else "unknown"


def _extract_reps(
    train_streams: Sequence[tuple[str, pd.DataFrame]],
    test_streams: Sequence[tuple[str, pd.DataFrame]],
    imu_columns: Sequence[str],
    target_length: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows, train_raw, train_labels_raw = [], [], []
    test_rows, test_raw, test_labels_raw = [], [], []

    for sid, df in train_streams:
        truth = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=1,
        )
        action = _action_from_stream_id(sid)
        for rep in truth:
            data = df[list(imu_columns)].to_numpy(dtype=np.float32)[int(rep.start_idx):int(rep.end_idx)]
            if len(data) < 5:
                continue
            window = data[np.newaxis, :, :]
            feats = _compute_rich_features(window, imu_columns)
            train_rows.append({k: float(v[0]) for k, v in feats.items()})
            train_raw.append(data)
            train_labels_raw.append(action)

    for sid, df in test_streams:
        truth = truth_reps_from_labels(
            df["phase"].to_numpy(),
            actions=df["action_type"].astype(str).to_numpy() if "action_type" in df.columns else None,
            min_phase_samples=1,
        )
        action = _action_from_stream_id(sid)
        for rep in truth:
            data = df[list(imu_columns)].to_numpy(dtype=np.float32)[int(rep.start_idx):int(rep.end_idx)]
            if len(data) < 5:
                continue
            window = data[np.newaxis, :, :]
            feats = _compute_rich_features(window, imu_columns)
            test_rows.append({k: float(v[0]) for k, v in feats.items()})
            test_raw.append(data)
            test_labels_raw.append(action)

    train_df = pd.DataFrame(train_rows)
    train_df["label"] = train_labels_raw
    test_df = pd.DataFrame(test_rows)
    test_df["label"] = test_labels_raw

    if target_length is None:
        all_lens = [len(r) for r in train_raw + test_raw]
        target_length = int(np.median(all_lens)) if all_lens else 100
    target_length = max(10, int(target_length))

    train_raw_resampled = _resample_signals(train_raw, target_length, imu_columns)
    test_raw_resampled = _resample_signals(test_raw, target_length, imu_columns)

    return train_df, test_df, train_raw_resampled, test_raw_resampled


def _resample_signals(signals: list[np.ndarray], target_len: int, imu_columns: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(signals), target_len, len(imu_columns)), dtype=np.float32)
    for i, sig in enumerate(signals):
        t = sig.shape[0]
        if t == target_len:
            out[i] = sig
        elif t > target_len:
            idx = np.linspace(0, t - 1, target_len).astype(np.int64)
            out[i] = sig[idx]
        else:
            out[i, :t] = sig
    return out


def _make_dl_datasets(
    train_raw: np.ndarray, train_labels: list[str],
    test_raw: np.ndarray, test_labels: list[str],
    batch_size: int = 64,
):
    class _DS(torch.utils.data.Dataset):
        def __init__(self, x, y, classes):
            self.x = torch.from_numpy(x).permute(0, 2, 1)
            self.y = torch.tensor([classes.index(l) for l in y], dtype=torch.long)
        def __len__(self):
            return len(self.y)
        def __getitem__(self, i):
            return self.x[i], self.y[i]

    classes = sorted(set(train_labels))
    train_ds = _DS(train_raw, train_labels, classes)
    test_ds = _DS(test_raw, test_labels, classes)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, classes


class CNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, filters: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, filters, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(filters)
        self.conv3 = nn.Conv1d(filters, filters, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(filters)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class BiLSTM(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=(kernel - 1) * dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))[:, :, :x.size(-1)]


class TCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, filters: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TCNBlock(in_channels if i == 0 else filters, filters, 3, 2 ** i))
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(filters, num_classes)
    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def _train_dl_model(model, device, train_loader, test_loader, epochs: int, lr: float, weight_decay: float):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds.extend(model(x).argmax(dim=1).cpu().numpy().tolist())
            trues.extend(y.numpy().tolist())
    return np.array(trues), np.array(preds)


def _eval(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _action_label_from_rep(rep_path: str) -> str:
    return _action_from_stream_id(rep_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark rep-complete action classifiers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/baseline_comparison/rep_complete_classifier_benchmark")
    parser.add_argument("--outer-subject", required=True)
    parser.add_argument("--include-actions", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dl-batch-size", type=int, default=64)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    feature_cfg = raw.get("feature", {}) or {}
    window_cfg = raw.get("window", {}) or {}
    train_raw_cfg = raw.get("train", {}) or {}
    mm_raw = raw.get("micro_macro", {}) or {}

    mm_cfg = cb.MicroMacroConfig(**{k: v for k, v in mm_raw.items() if k in cb.MicroMacroConfig.__dataclass_fields__})
    train_cfg = cb.TrainConfig(**{k: v for k, v in train_raw_cfg.items() if k in cb.TrainConfig.__dataclass_fields__})
    cb.set_seed(train_cfg.seed)

    imu_columns = list(feature_cfg.get("imu_columns", ["ax", "ay", "az", "gx", "gy", "gz"]))
    time_column = str(feature_cfg.get("time_column", "sensor_ts"))
    target_sample_rate = int(window_cfg.get("sample_rate_hz", 100))
    subject_column = str(feature_cfg.get("subject_column", "subject_id"))
    include_actions = [a for a in str(args.include_actions).split(",") if a.strip()]

    modes = list(mm_cfg.train_on_modes) if mm_cfg.train_on_modes else ["sets"]
    streams, subjects, _ = cb._load_streams(raw, modes)
    if mm_cfg.resample_to_window_rate:
        streams = cb._resample_streams_to_rate(streams, imu_columns, time_column, target_sample_rate)
    if include_actions:
        streams = [(sid, df) for sid, df in streams if _action_from_stream_id(sid) in include_actions]

    outer_subject = str(args.outer_subject)
    train_subjects = [s for s in sorted(set(subjects)) if s != outer_subject]
    train_streams = cb._filter_subjects(streams, train_subjects, subject_column)
    test_streams = cb._filter_subjects(streams, [outer_subject], subject_column)

    stats = cb.compute_train_stats([df for _, df in train_streams], imu_columns)
    train_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in train_streams]
    test_streams = [(sid, cb.apply_zscore(df, imu_columns, stats)) for sid, df in test_streams]

    print(f"[INFO] outer_subject={outer_subject} train_streams={len(train_streams)} test_streams={len(test_streams)}", flush=True)

    t0 = time.time()
    train_df, test_df, train_raw, test_raw = _extract_reps(
        train_streams, test_streams, imu_columns, target_length=None,
    )
    extract_time = time.time() - t0

    actions = [a for a in sorted(set(train_df["label"]))]

    label_col = "label"
    x_train = train_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_train = train_df[label_col].astype(str).to_numpy()
    x_test = test_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_test = test_df[label_col].astype(str).to_numpy()

    feature_names = list(train_df.drop(columns=[label_col]).columns)
    n_features = len(feature_names)
    n_train_reps = len(train_df)
    n_test_reps = len(test_df)
    n_classes = len(actions)

    print(f"[INFO] train_reps={n_train_reps} test_reps={n_test_reps} features={n_features} classes={n_classes}", flush=True)
    print(f"[INFO] class_distribution_train={dict(train_df[label_col].value_counts())}", flush=True)
    print(f"[INFO] class_distribution_test={dict(test_df[label_col].value_counts())}", flush=True)

    results: list[dict] = []

    # === 1. Logistic Regression ===
    name = "Logistic Regression"
    t1 = time.time()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, multi_class="auto", random_state=42))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append(_eval(name, y_test, y_pred, actions))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 2. Random Forest ===
    name = "Random Forest"
    t1 = time.time()
    model = RandomForestClassifier(n_estimators=400, max_depth=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append(_eval(name, y_test, y_pred, actions))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 3. SVM (RBF) ===
    name = "SVM (RBF)"
    t1 = time.time()
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=42))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append(_eval(name, y_test, y_pred, actions))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 4. k-NN ===
    name = "k-NN (k=5)"
    t1 = time.time()
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append(_eval(name, y_test, y_pred, actions))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 5. XGBoost ===
    if _XGB_AVAIL:
        name = "XGBoost"
        t1 = time.time()
        le = LabelEncoder()
        y_train_int = le.fit_transform(y_train)
        y_test_int = le.transform(y_test)
        xgb_cls = xgb.XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        xgb_cls.fit(x_train, y_train_int)
        y_pred_int = xgb_cls.predict(x_test)
        y_pred = le.inverse_transform(y_pred_int.astype(np.int64))
        results.append(_eval(name, y_test, y_pred, actions))
        print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)
    else:
        print("  [SKIP] XGBoost not installed", flush=True)

    # === 6. LightGBM ===
    if _LGB_AVAIL:
        name = "LightGBM"
        t1 = time.time()
        lgb_cls = lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=-1, n_jobs=-1,
        )
        lgb_cls.fit(x_train, y_train)
        y_pred = lgb_cls.predict(x_test)
        results.append(_eval(name, y_test, y_pred, actions))
        print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)
    else:
        print("  [SKIP] LightGBM not installed", flush=True)

    # === 7. CatBoost ===
    if _CATB_AVAIL:
        name = "CatBoost"
        t1 = time.time()
        le = LabelEncoder()
        y_train_int = le.fit_transform(y_train)
        y_test_int = le.transform(y_test)
        catb_cls = catb.CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.1,
            random_seed=42, verbose=0,
        )
        catb_cls.fit(x_train, y_train_int)
        y_pred_int = catb_cls.predict(x_test)
        y_pred = le.inverse_transform(y_pred_int.ravel().astype(np.int64))
        results.append(_eval(name, y_test, y_pred, actions))
        print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)
    else:
        print("  [SKIP] CatBoost not installed", flush=True)

    # ===============================================================
    # Deep Learning models (raw signal)
    # ===============================================================
    dl_train_labels = [str(l) for l in train_df[label_col].tolist()]
    dl_test_labels = [str(l) for l in test_df[label_col].tolist()]
    n_channels = len(imu_columns)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] DL device={device}", flush=True)

    train_loader, test_loader, dl_classes = _make_dl_datasets(
        train_raw, dl_train_labels, test_raw, dl_test_labels, batch_size=int(args.dl_batch_size),
    )

    # === 8. 1D CNN ===
    name = "1D CNN (raw signal)"
    t1 = time.time()
    model = CNN1D(in_channels=n_channels, num_classes=len(dl_classes))
    y_true, y_pred = _train_dl_model(model, device, train_loader, test_loader, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
    results.append(_eval(name, y_true, y_pred, dl_classes))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 9. BiLSTM ===
    name = "BiLSTM (raw signal)"
    t1 = time.time()
    model = BiLSTM(in_channels=n_channels, num_classes=len(dl_classes))
    y_true, y_pred = _train_dl_model(model, device, train_loader, test_loader, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
    results.append(_eval(name, y_true, y_pred, dl_classes))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # === 10. TCN ===
    name = "TCN (raw signal)"
    t1 = time.time()
    model = TCN(in_channels=n_channels, num_classes=len(dl_classes))
    y_true, y_pred = _train_dl_model(model, device, train_loader, test_loader, epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
    results.append(_eval(name, y_true, y_pred, dl_classes))
    print(f"  [{name}] acc={results[-1]['accuracy']:.4f} macro_f1={results[-1]['macro_f1']:.4f} time={time.time()-t1:.1f}s", flush=True)

    # ===============================================================
    # Summary table
    # ===============================================================
    summary_rows = []
    for r in results:
        summary_rows.append({
            "model": r["model"],
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "weighted_f1": round(r["weighted_f1"], 4),
        })

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "benchmark": "rep_complete_classifier_benchmark",
        "outer_subject": str(args.outer_subject),
        "actions": actions,
        "feature_columns": int(n_features),
        "train_rep_count": int(n_train_reps),
        "test_rep_count": int(n_test_reps),
        "train_class_distribution": dict(train_df[label_col].value_counts()),
        "test_class_distribution": dict(test_df[label_col].value_counts()),
        "extract_time_s": round(extract_time, 1),
        "models_available": {
            "xgboost": _XGB_AVAIL,
            "lightgbm": _LGB_AVAIL,
            "catboost": _CATB_AVAIL,
        },
        "summary": summary_rows,
        "detail": results,
    }

    (out_dir / "results.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'Model':30s} {'Accuracy':>10s} {'Macro F1':>10s} {'Wtd F1':>10s}")
    print("=" * 70)
    for s in summary_rows:
        print(f"{s['model']:30s} {s['accuracy']:>10.4f} {s['macro_f1']:>10.4f} {s['weighted_f1']:>10.4f}")
    print("=" * 70)
    print(f"[OK] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
