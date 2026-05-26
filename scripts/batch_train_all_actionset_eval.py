from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _find_set_dirs(data_dir: Path, include_actions: list[str], exclude_patterns: list[str]) -> list[Path]:
    include = set(str(x) for x in include_actions)
    out: list[Path] = []
    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            for action_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
                if action_dir.name not in include:
                    continue
                for set_dir in sorted(p for p in action_dir.iterdir() if p.is_dir() and p.name.startswith("set")):
                    rel_parts = set_dir.relative_to(data_dir).parts
                    if any(fnmatch.fnmatch(part, pattern) for part in rel_parts for pattern in exclude_patterns):
                        continue
                    has_csv = any(set_dir.glob("rep*.csv")) or any(set_dir.glob("*.csv"))
                    if not has_csv:
                        continue
                    out.append(set_dir)
    return out


def _run_command(cmd: list[str], description: str) -> None:
    result = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with code {result.returncode}: {' '.join(cmd)}")


def _aggregate_streaming(stream_dirs: list[Path], include_actions: list[str]) -> dict:
    totals = {
        "n_pred": 0,
        "n_true": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "exact_count_streams": 0,
        "over_segmented_streams": 0,
        "under_segmented_streams": 0,
        "zero_tp_streams": 0,
        "streams": len(stream_dirs),
    }
    per_action: dict[str, dict[str, int]] = {
        action: {
            "streams": 0,
            "n_pred": 0,
            "n_true": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "exact_count_streams": 0,
            "over_segmented_streams": 0,
            "under_segmented_streams": 0,
            "zero_tp_streams": 0,
        }
        for action in include_actions
    }
    stream_rows = []
    for stream_dir in stream_dirs:
        summary = json.loads((stream_dir / "streaming_summary.json").read_text(encoding="utf-8"))
        input_path = Path(summary["input_path"])
        action = input_path.parent.name if input_path.is_dir() else input_path.parent.parent.name
        n_pred = int(summary.get("n_pred", 0))
        n_true = int(summary.get("n_true", 0))
        tp = int(summary.get("tp", 0))
        fp = int(summary.get("fp", 0))
        fn = int(summary.get("fn", 0))
        diff = n_pred - n_true
        exact = int(diff == 0)
        over = int(diff > 0)
        under = int(diff < 0)
        zero_tp = int(tp == 0)
        for key, value in (("n_pred", n_pred), ("n_true", n_true), ("tp", tp), ("fp", fp), ("fn", fn)):
            totals[key] += value
            per_action[action][key] += value
        totals["exact_count_streams"] += exact
        totals["over_segmented_streams"] += over
        totals["under_segmented_streams"] += under
        totals["zero_tp_streams"] += zero_tp
        per_action[action]["streams"] += 1
        per_action[action]["exact_count_streams"] += exact
        per_action[action]["over_segmented_streams"] += over
        per_action[action]["under_segmented_streams"] += under
        per_action[action]["zero_tp_streams"] += zero_tp
        stream_rows.append(
            {
                "stream_dir": stream_dir.as_posix(),
                "action": action,
                "n_pred": n_pred,
                "n_true": n_true,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "count_diff": diff,
                "rep_f1": summary.get("f1"),
                "rep_precision": summary.get("precision"),
                "rep_recall": summary.get("recall"),
            }
        )
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    for action, stats in per_action.items():
        ap = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0.0
        ar = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) else 0.0
        stats["precision"] = ap
        stats["recall"] = ar
        stats["f1"] = (2 * ap * ar / (ap + ar)) if (ap + ar) else 0.0
    return {
        "overall": {
            **totals,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "per_action": per_action,
        "per_stream": stream_rows,
    }


def _aggregate_action_compare(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method_names = [
        "online_macro_aggregation",
        "rep_complete_classifier",
        "rep_complete_hierarchical",
        "hybrid_routing",
        "confidence_hybrid",
    ]
    overall: dict[str, dict[str, float | int]] = {}
    per_action: dict[str, dict[str, dict[str, int | float]]] = {}
    for method in method_names:
        total_correct = 0
        total_matched = 0
        for stream in payload.get("stream_results", []):
            section = stream.get(method, {}) or {}
            matched = int(section.get("matched_reps", 0) or 0)
            acc = section.get("accuracy")
            if matched and acc is not None:
                total_matched += matched
                total_correct += int(round(float(acc) * matched))
            report = section.get("classification_report", {}) or {}
            for action, stats in report.items():
                if not isinstance(stats, dict) or action in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                bucket = per_action.setdefault(action, {}).setdefault(method, {"support": 0, "correct": 0})
                support = int(stats.get("support", 0) or 0)
                recall = float(stats.get("recall", 0.0) or 0.0)
                bucket["support"] += support
                bucket["correct"] += int(round(recall * support))
        overall[method] = {
            "matched_reps": total_matched,
            "correct": total_correct,
            "accuracy": (total_correct / total_matched) if total_matched else 0.0,
        }
    for action, method_map in per_action.items():
        for method, stats in method_map.items():
            support = int(stats["support"])
            correct = int(stats["correct"])
            stats["accuracy"] = (correct / support) if support else 0.0
    return {
        "metadata": {
            "evaluation_protocol": payload.get("evaluation_protocol"),
            "best_model": payload.get("best_model"),
            "trusted_flat_labels": payload.get("trusted_flat_labels", []),
        },
        "overall_methods": overall,
        "per_action_methods": per_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on all subjects, run streaming eval on all sets, and aggregate results.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/micro_macro_recognition"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-existing-streams", action="store_true")
    args = parser.parse_args()

    raw = _load_config(args.config)
    data_cfg = raw.get("data", {}) or {}
    data_dir = (ROOT / data_cfg.get("data_dir", "./datasets/raw_data")).resolve()
    include_actions = [str(x) for x in data_cfg.get("include_actions", [])]
    exclude_patterns = list(data_cfg.get("exclude_patterns") or [])
    if not include_actions:
        raise ValueError("Config must define data.include_actions for batch train-all evaluation.")

    run_dir = (args.output_root / args.run_stamp / "tcn").resolve()
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "train.micro_macro_recognition",
            "--config",
            str(args.config),
            "--micro-source",
            "tcn",
            "--mode",
            "sets",
            "--no-timestamp",
            "--run-stamp",
            args.run_stamp,
            "--output-dir",
            str(args.output_root),
        ]
        _run_command(train_cmd, "train.micro_macro_recognition")
    elif not run_dir.exists():
        raise FileNotFoundError(f"Cannot use --skip-train because run_dir does not exist: {run_dir}")

    set_dirs = _find_set_dirs(data_dir, include_actions, exclude_patterns)
    if not set_dirs:
        raise FileNotFoundError(f"No matching set dirs found under {data_dir}")

    streaming_root = run_dir / "streaming_eval_all"
    stream_dirs: list[Path] = []
    skipped_streams: list[dict[str, str]] = []
    for set_dir in set_dirs:
        rel = set_dir.relative_to(data_dir)
        output_dir = streaming_root.joinpath(*rel.parts)
        if args.skip_existing_streams and (output_dir / "streaming_summary.json").exists():
            stream_dirs.append(output_dir)
            continue
        stream_cmd = [
            sys.executable,
            "-m",
            "evaluation.streaming_micro_macro",
            "--run-dir",
            str(run_dir),
            "--csv",
            str(set_dir),
            "--config",
            str(args.config),
            "--stream-id",
            rel.as_posix(),
            "--output-dir",
            str(output_dir),
            "--method",
            "fast",
            "--max-samples",
            "0",
            "--no-hybrid-action",
        ]
        try:
            _run_command(stream_cmd, f"streaming eval for {rel.as_posix()}")
            stream_dirs.append(output_dir)
        except RuntimeError as exc:
            skipped_streams.append({"stream": rel.as_posix(), "reason": str(exc)})
            print(f"[WARN] skipping stream after failure: {rel.as_posix()}", flush=True)

    compare_json = run_dir / "rep_action_compare_all.json"
    compare_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_rep_complete_action_classifier.py"),
        "--config",
        str(args.config),
        "--test-subject",
        "__all__",
        "--rep-eval-root",
        str(streaming_root),
        "--output-json",
        str(compare_json),
    ]
    _run_command(compare_cmd, "evaluate_rep_complete_action_classifier")

    aggregate = {
        "config": args.config.as_posix(),
        "run_dir": run_dir.as_posix(),
        "include_actions": include_actions,
        "evaluation_protocol": "train_all_in_sample",
        "evaluated_stream_count": len(stream_dirs),
        "skipped_streams": skipped_streams,
        "streaming": _aggregate_streaming(stream_dirs, include_actions),
        "action_compare": _aggregate_action_compare(compare_json),
    }
    out_path = run_dir / "aggregate_train_all_eval.json"
    out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"[OK] Wrote aggregate summary to {out_path}")


if __name__ == "__main__":
    main()
