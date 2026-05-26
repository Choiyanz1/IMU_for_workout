from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.custom_resistance_dataset import FeatureConfig, filter_sequences_by_subject, prepare_sequences_from_folder
from preprocessing.micro_macro_segments import match_segments, truth_reps_from_labels
from preprocessing.window_pipeline import apply_zscore, compute_train_stats
from train.action_classification import _build_rep_features


def _fit_best_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str) -> tuple[str, object, dict[str, object]]:
    x_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col].astype(str)
    x_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col].astype(str)

    candidates = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, multi_class="auto", random_state=42),
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
    }

    best_name = ""
    best_model = None
    best_score = -1.0
    report_bundle: dict[str, object] = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        score = float(f1_score(y_test, pred, average="macro"))
        report_bundle[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": score,
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    assert best_model is not None
    return best_name, best_model, report_bundle


def _coarse_label(label: str) -> str:
    if label in {"db_bench_press", "db_weighted_crunch"}:
        return "bench_or_crunch"
    return label


def _fit_hierarchical_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str) -> tuple[dict[str, object], dict[str, object]]:
    feature_cols = [c for c in train_df.columns if c != label_col]
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["coarse_label"] = train_df[label_col].astype(str).map(_coarse_label)
    test_df["coarse_label"] = test_df[label_col].astype(str).map(_coarse_label)

    coarse_name, coarse_model, coarse_reports = _fit_best_classifier(
        train_df[feature_cols + ["coarse_label"]].rename(columns={"coarse_label": label_col}),
        test_df[feature_cols + ["coarse_label"]].rename(columns={"coarse_label": label_col}),
        label_col,
    )

    bc_train = train_df[train_df[label_col].isin(["db_bench_press", "db_weighted_crunch"])].copy()
    bc_test = test_df[test_df[label_col].isin(["db_bench_press", "db_weighted_crunch"])].copy()
    bc_classes = bc_train[label_col].nunique() if len(bc_train) else 0
    if bc_classes >= 2:
        fine_name, fine_model, fine_reports = _fit_best_classifier(bc_train[feature_cols + [label_col]], bc_test[feature_cols + [label_col]], label_col)
    else:
        fine_name, fine_model, fine_reports = "skip", None, {}
        bc_fallback_label = str(bc_train[label_col].iloc[0]) if len(bc_train) else "db_bench_press"

    def predict(df_features: pd.DataFrame) -> np.ndarray:
        coarse_pred = coarse_model.predict(df_features[feature_cols])
        out = []
        for i, pred in enumerate(coarse_pred):
            pred = str(pred)
            if pred == "bench_or_crunch":
                if fine_model is not None:
                    out.append(str(fine_model.predict(df_features.iloc[[i]][feature_cols])[0]))
                else:
                    out.append(bc_fallback_label)
            else:
                out.append(pred)
        return np.asarray(out, dtype=object)

    y_true = test_df[label_col].astype(str).to_numpy()
    y_pred = predict(test_df)
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    model_bundle = {
        "coarse_name": coarse_name,
        "coarse_model": coarse_model,
        "coarse_reports": coarse_reports,
        "fine_name": fine_name,
        "fine_model": fine_model,
        "fine_reports": fine_reports,
        "feature_cols": feature_cols,
        "bc_fallback_label": bc_fallback_label if fine_model is None else None,
    }
    return model_bundle, overall


def _segment_dfs_from_detections(merged_df: pd.DataFrame, detections_df: pd.DataFrame) -> list[pd.DataFrame]:
    seqs = []
    for row in detections_df.itertuples(index=False):
        start = max(0, int(row.start_idx))
        end = min(len(merged_df), int(row.end_idx))
        seg = merged_df.iloc[start:end].copy().reset_index(drop=True)
        if len(seg) == 0:
            continue
        if "action_type" not in seg.columns:
            seg["action_type"] = "unknown"
        seqs.append(seg)
    return seqs


def _hierarchical_predict(model_bundle: dict[str, object], feature_df: pd.DataFrame) -> np.ndarray:
    feature_cols = list(model_bundle["feature_cols"])
    coarse_model = model_bundle["coarse_model"]
    fine_model = model_bundle["fine_model"]
    bc_fallback = model_bundle.get("bc_fallback_label", "db_bench_press")
    coarse_pred = coarse_model.predict(feature_df[feature_cols])
    out = []
    for i, pred in enumerate(coarse_pred):
        pred = str(pred)
        if pred == "bench_or_crunch":
            if fine_model is not None:
                out.append(str(fine_model.predict(feature_df.iloc[[i]][feature_cols])[0]))
            else:
                out.append(bc_fallback)
        else:
            out.append(pred)
    return np.asarray(out, dtype=object)


def _read_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)


def _is_all_subjects_mode(test_subject: object) -> bool:
    value = str(test_subject or "").strip().lower()
    return value in {"all", "__all__", "*"}


def _collect_rep_eval_dirs(explicit_dirs: list[str], root_dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for rep_dir in explicit_dirs:
        path = Path(rep_dir)
        resolved = path.resolve().as_posix()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    for root_dir in root_dirs:
        root = Path(root_dir)
        if not root.exists():
            continue
        for summary_path in sorted(root.rglob("streaming_summary.json")):
            path = summary_path.parent
            resolved = path.resolve().as_posix()
            if resolved not in seen:
                seen.add(resolved)
                out.append(path)
    return out


def _evaluate_stream(
    rep_eval_dir: Path,
    model,
    hierarchical_bundle: dict[str, object],
    trusted_flat_labels: set[str],
    feature_cfg: FeatureConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    merged_df = pd.read_csv(rep_eval_dir / "merged_set_input.csv")
    online_df = _read_optional_csv(
        rep_eval_dir / "online_rep_detections.csv",
        ["start_idx", "end_idx", "pred_action_type", "action_confidence"],
    )
    event_df = (
        _read_optional_csv(rep_eval_dir / "online_rep_events.csv", ["emit_sample_idx"])
        if (rep_eval_dir / "online_rep_events.csv").exists()
        else pd.DataFrame(columns=["emit_sample_idx"])
    )

    truth_reps = truth_reps_from_labels(
        merged_df["phase"].to_numpy(),
        actions=merged_df["action_type"].astype(str).to_numpy() if "action_type" in merged_df.columns else None,
        min_phase_samples=3,
    )
    pred_segments = [(int(r.start_idx), int(r.end_idx)) for r in online_df.itertuples(index=False)]
    truth_segments = [(int(r.start_idx), int(r.end_idx)) for r in truth_reps]
    matches = match_segments(pred_segments, truth_segments, iou_threshold=0.5)

    pred_seq_dfs = _segment_dfs_from_detections(merged_df, online_df)
    if pred_seq_dfs:
        pred_feat = _build_rep_features(pred_seq_dfs, feature_cfg.imu_columns, feature_cfg.label_column, "rich")
        pred_labels = model.predict(pred_feat.drop(columns=["label"])) if len(pred_feat) else np.asarray([], dtype=object)
        pred_labels_hier = _hierarchical_predict(hierarchical_bundle, pred_feat.drop(columns=["label"])) if len(pred_feat) else np.asarray([], dtype=object)
    else:
        pred_labels = np.asarray([], dtype=object)
        pred_labels_hier = np.asarray([], dtype=object)

    clf_true = []
    clf_pred = []
    clf_pred_hier = []
    agg_true = []
    agg_pred = []
    hybrid_true = []
    hybrid_pred = []
    conf_hybrid_true = []
    conf_hybrid_pred = []
    delays_ms = []
    sample_rate = float(summary.get("sample_rate_hz", 100.0))
    for pi, ti, _ in matches:
        truth_label = str(truth_reps[ti].pred_action_type)
        flat_label = str(pred_labels[pi])
        hier_label = str(pred_labels_hier[pi])
        agg_label = str(online_df.iloc[pi]["pred_action_type"])
        clf_true.append(truth_label)
        clf_pred.append(flat_label)
        clf_pred_hier.append(hier_label)
        agg_true.append(truth_label)
        agg_pred.append(agg_label)
        hybrid_true.append(truth_label)
        hybrid_pred.append(flat_label if flat_label in trusted_flat_labels else agg_label)
        conf_hybrid_true.append(truth_label)
        # Confidence-based hybrid: use macro if its confidence is high, else classifier
        agg_conf = float(online_df.iloc[pi]["action_confidence"]) if "action_confidence" in online_df.columns else 0.0
        conf_hybrid_pred.append(agg_label if agg_conf >= 0.7 else flat_label)
        if not event_df.empty and pi < len(event_df):
            emit_idx = int(event_df.iloc[pi]["emit_sample_idx"])
            delay_ms = max(0, emit_idx - (int(truth_reps[ti].end_idx) - 1)) / sample_rate * 1000.0
            delays_ms.append(delay_ms)

    def pack(y_true, y_pred):
        if not y_true:
            return {"matched_reps": 0, "accuracy": float("nan"), "macro_f1": float("nan")}
        return {
            "matched_reps": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

    return {
        "stream_dir": rep_eval_dir.as_posix(),
        "stream_summary": summary,
        "rep_complete_classifier": pack(clf_true, clf_pred),
        "rep_complete_hierarchical": pack(clf_true, clf_pred_hier),
        "online_macro_aggregation": pack(agg_true, agg_pred),
        "hybrid_routing": pack(hybrid_true, hybrid_pred),
        "confidence_hybrid": pack(conf_hybrid_true, conf_hybrid_pred),
        "mean_emit_delay_ms": float(np.mean(delays_ms)) if delays_ms else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rep-complete action classifier vs online macro aggregation.")
    parser.add_argument("--config", type=Path, default=Path("configs/micro_macro_recognition_stage3_40ep.yaml"))
    parser.add_argument("--test-subject", default="kevin")
    parser.add_argument("--rep-eval-dir", action="append", default=[], help="Streaming eval directory with online rep outputs. Repeatable.")
    parser.add_argument("--rep-eval-root", action="append", default=[], help="Root directory to recursively scan for streaming eval outputs.")
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/rep_complete_action_compare.json"))
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    feature_cfg = FeatureConfig(**(raw.get("feature", {}) or {}))
    data_cfg = raw.get("data", {}) or {}
    window_cfg = raw.get("window", {}) or {}

    data_dir = Path(data_cfg.get("data_dir", "./datasets/raw_data"))
    sequences, subjects = prepare_sequences_from_folder(
        data_dir=data_dir,
        feature_cfg=feature_cfg,
        sample_rate_hz=int(window_cfg.get("sample_rate_hz", 100)),
        csv_glob=data_cfg.get("csv_glob", "*.csv"),
        exclude_patterns=data_cfg.get("exclude_patterns", None),
        include_actions=data_cfg.get("include_actions", None),
        subject_aliases=data_cfg.get("subject_aliases", None),
    )
    if _is_all_subjects_mode(args.test_subject):
        train_subjects = sorted(set(subjects))
        train_seqs = list(sequences)
        test_seqs = list(sequences)
        evaluation_protocol = "train_all_in_sample"
    else:
        train_subjects = [s for s in sorted(set(subjects)) if s != str(args.test_subject)]
        train_seqs = filter_sequences_by_subject(sequences, train_subjects, feature_cfg.subject_column)
        test_seqs = filter_sequences_by_subject(sequences, [str(args.test_subject)], feature_cfg.subject_column)
        evaluation_protocol = "subject_holdout"

    stats = compute_train_stats(train_seqs, feature_cfg.imu_columns)
    train_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in train_seqs]
    test_seqs = [apply_zscore(seq, feature_cfg.imu_columns, stats) for seq in test_seqs]

    train_df = _build_rep_features(train_seqs, feature_cfg.imu_columns, feature_cfg.label_column, "rich")
    test_df = _build_rep_features(test_seqs, feature_cfg.imu_columns, feature_cfg.label_column, "rich")
    best_name, model, model_reports = _fit_best_classifier(train_df, test_df, "label")
    hierarchical_bundle, hierarchical_overall = _fit_hierarchical_classifier(train_df, test_df, "label")
    _summary_keys = {"accuracy", "macro avg", "weighted avg"}
    trusted_flat_labels = {
        label
        for label, report in model_reports[best_name]["classification_report"].items()
        if isinstance(report, dict) and label not in _summary_keys
        and float(report.get("precision", 0.0)) >= 0.95
    }

    rep_eval_dirs = _collect_rep_eval_dirs(args.rep_eval_dir, args.rep_eval_root)
    if not rep_eval_dirs:
        raise FileNotFoundError("No rep eval directories were provided or discovered.")
    stream_results = []
    for rep_eval_dir in rep_eval_dirs:
        summary = json.loads((rep_eval_dir / "streaming_summary.json").read_text(encoding="utf-8"))
        stream_results.append(_evaluate_stream(rep_eval_dir, model, hierarchical_bundle, trusted_flat_labels, feature_cfg, summary))

    output = {
        "evaluation_protocol": evaluation_protocol,
        "train_subjects": train_subjects,
        "test_subject": str(args.test_subject),
        "best_model": best_name,
        "trusted_flat_labels": sorted(trusted_flat_labels),
        "heldout_rep_classifier_reports": model_reports,
        "heldout_hierarchical_report": hierarchical_overall,
        "stream_results": stream_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    import yaml

    main()
