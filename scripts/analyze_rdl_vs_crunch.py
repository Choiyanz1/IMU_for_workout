from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "raw_data"
ACTIONS = ["db_rdl", "db_weighted_crunch"]
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]


def contiguous_lengths(labels: np.ndarray, target: str) -> list[int]:
    lengths: list[int] = []
    run = 0
    for label in labels:
        if label == target:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def sample_rate_hz(sensor_ts: pd.Series) -> float:
    diffs = np.diff(sensor_ts.to_numpy(dtype=np.float64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100.0
    return float(1000.0 / np.median(diffs))


def file_signature(path: Path, subject: str, action: str) -> dict[str, object]:
    df = pd.read_csv(path)
    rate = sample_rate_hz(df["sensor_ts"])
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
    gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=np.float64)
    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    phases = df["phase"].astype(str).to_numpy()
    conc = contiguous_lengths(phases, "concentric")
    ecc = contiguous_lengths(phases, "eccentric")
    none_ratio = float(np.mean(phases == "none"))
    conc_ratio = float(np.mean(phases == "concentric"))
    ecc_ratio = float(np.mean(phases == "eccentric"))
    out: dict[str, object] = {
        "subject": subject,
        "action": action,
        "path": path.as_posix(),
        "samples": int(len(df)),
        "sample_rate_hz": rate,
        "duration_s": float(len(df) / rate),
        "none_ratio": none_ratio,
        "concentric_ratio": conc_ratio,
        "eccentric_ratio": ecc_ratio,
        "concentric_segments": int(len(conc)),
        "eccentric_segments": int(len(ecc)),
        "concentric_len_mean": float(np.mean(conc)) if conc else 0.0,
        "eccentric_len_mean": float(np.mean(ecc)) if ecc else 0.0,
        "acc_norm_mean": float(acc_norm.mean()),
        "acc_norm_std": float(acc_norm.std()),
        "acc_norm_max": float(acc_norm.max()),
        "gyro_norm_mean": float(gyro_norm.mean()),
        "gyro_norm_std": float(gyro_norm.std()),
        "gyro_norm_max": float(gyro_norm.max()),
    }
    for col in IMU_COLS:
        values = df[col].to_numpy(dtype=np.float64)
        out[f"{col}_mean"] = float(values.mean())
        out[f"{col}_std"] = float(values.std())
        out[f"{col}_abs_mean"] = float(np.abs(values).mean())
    return out


def collect_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_dir in sorted(DATA_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        for action in ACTIONS:
            action_dir = subject_dir / action
            if not action_dir.exists():
                continue
            for csv_path in sorted(action_dir.glob("set*/*.csv")):
                rows.append(file_signature(csv_path, subject_dir.name, action))
    return pd.DataFrame(rows)


def grouped_summary(df: pd.DataFrame) -> dict[str, object]:
    cols = [
        "samples",
        "duration_s",
        "none_ratio",
        "concentric_ratio",
        "eccentric_ratio",
        "concentric_len_mean",
        "eccentric_len_mean",
        "acc_norm_mean",
        "acc_norm_std",
        "gyro_norm_mean",
        "gyro_norm_std",
    ]
    summary: dict[str, object] = {}
    for (subject, action), group in df.groupby(["subject", "action"]):
        summary[f"{subject}/{action}"] = {
            "n_reps": int(len(group)),
            "stats": {
                col: {
                    "mean": float(group[col].mean()),
                    "std": float(group[col].std(ddof=0)),
                }
                for col in cols
            },
        }
    return summary


def centroid_distance_report(df: pd.DataFrame) -> dict[str, float]:
    feature_cols = [
        c
        for c in df.columns
        if c.endswith("_mean") or c.endswith("_std") or c.endswith("_abs_mean") or c.endswith("_ratio")
    ]
    train = df[df["subject"] != "kevin"]
    scale = train[feature_cols].std(ddof=0).replace(0.0, 1.0)

    def centroid(subject: str | None, action: str) -> pd.Series:
        mask = df["action"].eq(action)
        if subject is not None:
            mask &= df["subject"].eq(subject)
        else:
            mask &= df["subject"].ne("kevin")
        return df.loc[mask, feature_cols].mean()

    k_rdl = centroid("kevin", "db_rdl")
    k_crunch = centroid("kevin", "db_weighted_crunch")
    t_rdl = centroid(None, "db_rdl")
    t_crunch = centroid(None, "db_weighted_crunch")

    def dist(a: pd.Series, b: pd.Series) -> float:
        z = (a - b) / scale
        return float(np.sqrt(np.square(z).mean()))

    return {
        "kevin_rdl_to_train_rdl": dist(k_rdl, t_rdl),
        "kevin_rdl_to_train_crunch": dist(k_rdl, t_crunch),
        "kevin_crunch_to_train_crunch": dist(k_crunch, t_crunch),
        "kevin_crunch_to_train_rdl": dist(k_crunch, t_rdl),
    }


def classifier_report(df: pd.DataFrame) -> dict[str, object]:
    feature_cols = [
        c
        for c in df.columns
        if c.endswith("_mean") or c.endswith("_std") or c.endswith("_abs_mean") or c.endswith("_ratio")
    ]
    train = df[df["subject"] != "kevin"].copy()
    test = df[df["subject"] == "kevin"].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(train[feature_cols], train["action"])
    pred = model.predict(test[feature_cols])
    report = classification_report(test["action"], pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(test["action"], pred, labels=ACTIONS)
    return {
        "classification_report": report,
        "confusion_matrix": {
            "labels": ACTIONS,
            "matrix": cm.tolist(),
        },
    }


def main() -> None:
    df = collect_rows()
    print(f"rows={len(df)}")
    print("\n## Counts by subject/action")
    counts = df.groupby(["subject", "action"]).agg(n_reps=("path", "count"), total_samples=("samples", "sum"))
    print(counts.to_string())

    print("\n## Kevin vs train centroid distances")
    print(json.dumps(centroid_distance_report(df), indent=2))

    print("\n## Simple rep-signature classifier train!=kevin test=kevin")
    print(json.dumps(classifier_report(df), indent=2))

    print("\n## Selected grouped summary")
    summary = grouped_summary(df)
    keys = [
        "kevin/db_rdl",
        "kevin/db_weighted_crunch",
        "thomas/db_rdl",
        "thomas/db_weighted_crunch",
        "thomas0506workout/db_rdl",
        "thomas0506workout/db_weighted_crunch",
        "1000/db_rdl",
        "1000/db_weighted_crunch",
    ]
    filtered = {k: summary[k] for k in keys if k in summary}
    print(json.dumps(filtered, indent=2))


if __name__ == "__main__":
    main()
