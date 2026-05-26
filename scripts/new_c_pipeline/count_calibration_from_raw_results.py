"""LOSO count calibration from existing raw6 CNN stream-level predictions.

This is a post-hoc, real-time-safe probe: calibrators may use only fields that
are available at inference time after raw rep parsing (`action`, `pred_count`).
It does not alter rep boundaries, so it should be interpreted as final displayed
count correction rather than a replacement for boundary-level rep F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def count_summary(rows, key="calibrated_count"):
    pred = np.array([row[key] for row in rows], dtype=float)
    gt = np.array([row["gt_count"] for row in rows], dtype=float)
    signed = pred - gt
    abs_err = np.abs(signed)
    return {
        "streams": int(len(rows)),
        "exact_count_acc": float(np.mean(abs_err == 0.0)),
        "within_1_count_acc": float(np.mean(abs_err <= 1.0)),
        "mean_abs_count_error": float(np.mean(abs_err)),
        "median_abs_count_error": float(np.median(abs_err)),
        "count_rmse": float(np.sqrt(np.mean(signed ** 2))),
        "count_bias_pred_minus_gt": float(np.mean(signed)),
        "mean_pred_count": float(np.mean(pred)),
        "mean_gt_count": float(np.mean(gt)),
        "over_rate": float(np.mean(signed > 0.0)),
        "under_rate": float(np.mean(signed < 0.0)),
    }


def group_summary(rows, group_key, count_key="calibrated_count"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    return {key: count_summary(items, count_key) for key, items in sorted(grouped.items())}


def rounded_nonnegative(value):
    return int(max(0, round(float(value))))


class IdentityCalibrator:
    name = "raw_identity"

    def fit(self, rows):
        return self

    def predict(self, row):
        return rounded_nonnegative(row["pred_count"])

    def params(self):
        return {}


class GlobalBiasCalibrator:
    name = "global_bias"


    def fit(self, rows):
        errors = [row["pred_count"] - row["gt_count"] for row in rows]
        self.bias_ = float(np.mean(errors)) if errors else 0.0
        return self

    def predict(self, row):
        return rounded_nonnegative(row["pred_count"] - self.bias_)

    def params(self):
        return {"bias": self.bias_}


class ActionBiasCalibrator:
    name = "action_bias"

    def fit(self, rows):
        by_action = defaultdict(list)
        errors = []
        for row in rows:
            err = row["pred_count"] - row["gt_count"]
            errors.append(err)
            by_action[row["action"]].append(err)
        self.global_bias_ = float(np.mean(errors)) if errors else 0.0
        self.bias_by_action_ = {
            action: float(np.mean(vals)) for action, vals in by_action.items()
        }
        return self

    def predict(self, row):
        bias = self.bias_by_action_.get(row["action"], self.global_bias_)
        return rounded_nonnegative(row["pred_count"] - bias)

    def params(self):
        return {"global_bias": self.global_bias_, "bias_by_action": self.bias_by_action_}


class ActionLinearCalibrator:
    name = "action_linear"

    def fit_line(self, rows):
        x = np.array([row["pred_count"] for row in rows], dtype=float)
        y = np.array([row["gt_count"] for row in rows], dtype=float)
        if len(rows) < 2 or np.std(x) < 1e-8:
            return 1.0, float(np.mean(y - x)) if len(rows) else 0.0
        slope, intercept = np.polyfit(x, y, deg=1)
        return float(slope), float(intercept)

    def fit(self, rows):
        self.global_line_ = self.fit_line(rows)
        by_action = defaultdict(list)
        for row in rows:
            by_action[row["action"]].append(row)
        self.line_by_action_ = {
            action: self.fit_line(items) for action, items in by_action.items()
        }
        return self

    def predict(self, row):
        slope, intercept = self.line_by_action_.get(row["action"], self.global_line_)
        return rounded_nonnegative(slope * row["pred_count"] + intercept)

    def params(self):
        return {"global_line": self.global_line_, "line_by_action": self.line_by_action_}


class ActionLinearDurationCalibrator:
    name = "action_linear_duration"

    def features(self, row):
        return [float(row["pred_count"]), float(row.get("duration_sec", 0.0)), 1.0]

    def fit_line(self, rows):
        if len(rows) < 3 or not any(row.get("duration_sec", 0.0) > 0 for row in rows):
            fallback = ActionLinearCalibrator().fit_line(rows)
            return [fallback[0], 0.0, fallback[1]]
        x = np.array([self.features(row) for row in rows], dtype=float)
        y = np.array([row["gt_count"] for row in rows], dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        return [float(v) for v in coef]

    def fit(self, rows):
        self.global_coef_ = self.fit_line(rows)
        by_action = defaultdict(list)
        for row in rows:
            by_action[row["action"]].append(row)
        self.coef_by_action_ = {
            action: self.fit_line(items) for action, items in by_action.items()
        }
        return self

    def predict(self, row):
        coef = self.coef_by_action_.get(row["action"], self.global_coef_)
        return rounded_nonnegative(np.dot(np.array(coef, dtype=float), np.array(self.features(row), dtype=float)))

    def params(self):
        return {"global_coef": self.global_coef_, "coef_by_action": self.coef_by_action_}


class ActionMedianLookupCalibrator:
    name = "action_predcount_median_lookup"

    def fit(self, rows):
        self.action_bias_ = ActionBiasCalibrator().fit(rows)
        buckets = defaultdict(list)
        for row in rows:
            buckets[(row["action"], int(row["pred_count"]))].append(row["gt_count"])
        self.lookup_ = {f"{action}|{pred}": float(np.median(vals)) for (action, pred), vals in buckets.items()}
        return self

    def predict(self, row):
        key = f"{row['action']}|{int(row['pred_count'])}"
        if key in self.lookup_:
            return rounded_nonnegative(self.lookup_[key])
        return self.action_bias_.predict(row)

    def params(self):
        return {"fallback": self.action_bias_.params(), "lookup": self.lookup_}


def make_calibrator(name):
    calibrators = {
        IdentityCalibrator.name: IdentityCalibrator,
        GlobalBiasCalibrator.name: GlobalBiasCalibrator,
        ActionBiasCalibrator.name: ActionBiasCalibrator,
        ActionLinearCalibrator.name: ActionLinearCalibrator,
        ActionLinearDurationCalibrator.name: ActionLinearDurationCalibrator,
        ActionMedianLookupCalibrator.name: ActionMedianLookupCalibrator,
    }
    return calibrators[name]()


def score_rows(rows, metric):
    summary = count_summary(rows)
    if metric == "mae":
        return summary["mean_abs_count_error"]
    if metric == "exact":
        return -summary["exact_count_acc"]
    raise ValueError(f"Unknown metric: {metric}")


class NestedActionSelectorCalibrator:
    def __init__(self, metric):
        self.metric = metric
        self.candidate_names = [
            "raw_identity",
            "action_linear",
            "action_linear_duration",
            "action_predcount_median_lookup",
        ]

    @property
    def name(self):
        return f"nested_action_select_{self.metric}"

    def fit(self, rows):
        subjects = sorted({row["subject"] for row in rows})
        actions = sorted({row["action"] for row in rows})
        candidate_rows = {name: [] for name in self.candidate_names}

        for subject in subjects:
            inner_train = [row for row in rows if row["subject"] != subject]
            inner_test = [row for row in rows if row["subject"] == subject]
            for name in self.candidate_names:
                calibrator = make_calibrator(name).fit(inner_train)
                for row in inner_test:
                    out = dict(row)
                    out["calibrated_count"] = calibrator.predict(row)
                    candidate_rows[name].append(out)

        self.selected_by_action_ = {}
        for action in actions:
            best_name = None
            best_score = float("inf")
            for name, rows_for_name in candidate_rows.items():
                action_rows = [row for row in rows_for_name if row["action"] == action]
                if not action_rows:
                    continue
                score = score_rows(action_rows, self.metric)
                if score < best_score:
                    best_score = score
                    best_name = name
            self.selected_by_action_[action] = best_name or "raw_identity"

        self.fitted_candidates_ = {
            name: make_calibrator(name).fit(rows) for name in self.candidate_names
        }
        return self

    def predict(self, row):
        name = self.selected_by_action_.get(row["action"], "raw_identity")
        return self.fitted_candidates_[name].predict(row)

    def params(self):
        return {
            "metric": self.metric,
            "candidate_names": self.candidate_names,
            "selected_by_action": self.selected_by_action_,
            "fitted_candidate_params": {
                name: calibrator.params() for name, calibrator in self.fitted_candidates_.items()
            },
        }


def evaluate_loso(streams, calibrator_name):
    subjects = sorted({row["subject"] for row in streams})
    calibrated_rows = []
    folds = []
    fold_params = {}
    for subject in subjects:
        train_rows = [row for row in streams if row["subject"] != subject]
        test_rows = [row for row in streams if row["subject"] == subject]
        if calibrator_name == "nested_action_select_mae":
            calibrator = NestedActionSelectorCalibrator("mae").fit(train_rows)
        elif calibrator_name == "nested_action_select_exact":
            calibrator = NestedActionSelectorCalibrator("exact").fit(train_rows)
        else:
            calibrator = make_calibrator(calibrator_name).fit(train_rows)
        fold_rows = []
        for row in test_rows:
            out = dict(row)
            out["raw_count"] = int(row["pred_count"])
            out["calibrated_count"] = calibrator.predict(row)
            out["calibrated_error"] = abs(out["calibrated_count"] - out["gt_count"])
            fold_rows.append(out)
        calibrated_rows.extend(fold_rows)
        folds.append({"test_subject": subject, "summary": count_summary(fold_rows)})
        fold_params[subject] = calibrator.params()
    return calibrated_rows, folds, fold_params


def add_stream_durations(streams, config_path, sample_rate):
    import yaml
    from scripts.new_c_pipeline.test_pca_input import should_exclude
    from train.micro_macro_recognition import _load_streams

    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    all_streams, _, _ = _load_streams(raw_cfg, ["sets"])
    durations = {
        stream_id: len(df) / sample_rate
        for stream_id, df in all_streams
        if not should_exclude(stream_id)
    }
    for row in streams:
        row["duration_sec"] = float(durations.get(row["stream_id"], 0.0))
    return streams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/cnn_variant_comparison/raw6_cnn_comprehensive_9fold_gpu_h64e20.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/cnn_variant_comparison/count_calibration_from_raw6_loso.json",
    )
    parser.add_argument("--include-duration", action="store_true", help="Add total stream duration as an inference-time feature.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-rate", type=float, default=100.0)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    streams = data["streams"]
    if args.include_duration:
        streams = add_stream_durations(streams, args.config, args.sample_rate)

    raw_rows = [dict(row, calibrated_count=int(row["pred_count"])) for row in streams]
    methods = [
        "raw_identity",
        "global_bias",
        "action_bias",
        "action_linear",
        "action_linear_duration",
        "action_predcount_median_lookup",
        "nested_action_select_mae",
        "nested_action_select_exact",
    ]
    results = {
        "settings": {
            "source": args.input,
            "protocol": "post-hoc subject-wise LOSO over existing raw6 CNN held-out predictions",
            "allowed_features": ["action", "pred_count"] + (["duration_sec"] if args.include_duration else []),
            "note": "Calibrated counts do not change rep boundaries or Rep F1.",
        },
        "raw_identity": {
            "overall": count_summary(raw_rows),
            "per_action": group_summary(raw_rows, "action"),
            "per_subject": group_summary(raw_rows, "subject"),
        },
        "methods": {},
    }

    for method in methods[1:]:
        rows, folds, fold_params = evaluate_loso(streams, method)
        results["methods"][method] = {
            "overall": count_summary(rows),
            "per_action": group_summary(rows, "action"),
            "per_subject": group_summary(rows, "subject"),
            "folds": folds,
            "fold_params": fold_params,
            "streams": rows,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Count calibration from raw6 stream predictions")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("\nMethod                         Exact  Within1   MAE   RMSE   Bias   Over  Under")
    print("-" * 82)
    raw = results["raw_identity"]["overall"]
    print(
        f"{'raw_identity':<30s} {raw['exact_count_acc']:.3f}   {raw['within_1_count_acc']:.3f}  "
        f"{raw['mean_abs_count_error']:.3f}  {raw['count_rmse']:.3f}  {raw['count_bias_pred_minus_gt']:+.3f}  "
        f"{raw['over_rate']:.3f}  {raw['under_rate']:.3f}"
    )
    for method in methods[1:]:
        overall = results["methods"][method]["overall"]
        print(
            f"{method:<30s} {overall['exact_count_acc']:.3f}   {overall['within_1_count_acc']:.3f}  "
            f"{overall['mean_abs_count_error']:.3f}  {overall['count_rmse']:.3f}  {overall['count_bias_pred_minus_gt']:+.3f}  "
            f"{overall['over_rate']:.3f}  {overall['under_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
