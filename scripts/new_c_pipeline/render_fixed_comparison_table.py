"""Render a fixed-baseline comparison table with current project results.

This script keeps literature and legacy baseline rows fixed in
docs/experiments/2026-05-19-fixed-baseline-registry.json, then refreshes only
the current project rows from result artifacts. Use it after rerunning our model
so the comparison table can improve without moving the baseline goalposts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "docs" / "experiments" / "2026-05-19-fixed-baseline-registry.json"
DEFAULT_OUTPUT = ROOT / "docs" / "experiments" / "2026-05-19-fixed-baseline-comparison-table.md"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(value: float | int | None) -> str:
    if value is None:
        return "NR"
    return f"{float(value) * 100:.1f}%"


def num(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "NR"
    return f"{float(value):.3f}{suffix}"


def rel(path_text: str) -> Path:
    return ROOT / path_text


def row(
    group: str,
    method: str,
    protocol: str,
    recognition: str,
    count_exact_or_within1: str,
    count_mae: str,
    segmentation: str,
    phase_or_ce: str,
    real_time: str,
    status: str,
) -> Dict[str, str]:
    return {
        "group": group,
        "method": method,
        "protocol": protocol,
        "recognition": recognition,
        "count_exact_or_within1": count_exact_or_within1,
        "count_mae": count_mae,
        "segmentation": segmentation,
        "phase_or_ce": phase_or_ce,
        "real_time": real_time,
        "status": status,
    }


def load_current_rows(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    sources = registry["current_ours_sources"]
    rows: List[Dict[str, str]] = []

    raw_path = rel(sources["raw6_comprehensive"])
    if raw_path.exists():
        raw = load_json(raw_path)["overall"]
        rows.append(
            row(
                "Current ours",
                "Raw6 1D Causal CNN + MA25/Viterbi",
                "9-fold LOSO, 220 streams, 8 actions",
                "action recognition not integrated",
                f"Exact {pct(raw.get('exact_count_acc'))}; within-1 {pct(raw.get('within_1_count_acc'))}",
                f"{num(raw.get('mean_abs_count_error'))} reps/set",
                f"Rep P/R/F1 {pct(raw.get('rep_precision'))}/{pct(raw.get('rep_recall'))}/{pct(raw.get('rep_f1'))}",
                f"Phase acc {pct(raw.get('phase_accuracy'))}; Phase macro F1 {pct(raw.get('phase_macro_f1'))}; C/E MAE {num(raw.get('ce_ratio_mae'))}",
                "causal 3s window; streaming-style replay available",
                "dynamic_current_result",
            )
        )

    selective_path = rel(sources["selective_duration_merge"])
    if selective_path.exists():
        selective = load_json(selective_path)["config_totals"]["top5_p5"]
        rows.append(
            row(
                "Current ours",
                "Raw6 1D Causal CNN + top5_p5 selective merge",
                "9-fold LOSO, 220 streams, 8 actions",
                "action recognition not integrated; assumes action context",
                f"Exact {pct(selective.get('exact_count_acc'))}; within-1 {pct(selective.get('within_1_count_acc'))}",
                f"{num(selective.get('mean_abs_count_error'))} reps/set",
                f"Rep P/R/F1 {pct(selective.get('rep_precision'))}/{pct(selective.get('rep_recall'))}/{pct(selective.get('rep_f1'))}",
                f"Phase acc {pct(selective.get('phase_accuracy'))}; Phase macro F1 {pct(selective.get('phase_macro_f1'))}; C/E MAE {num(selective.get('ce_ratio_mae'))}",
                "causal 3s window; decoder-only merge; streaming-style replay available",
                "dynamic_current_best_structured",
            )
        )

    calibration_path = rel(sources["count_calibration"])
    if calibration_path.exists():
        calibration = load_json(calibration_path)["methods"]["action_linear"]["overall"]
        rows.append(
            row(
                "Current ours",
                "Raw6 CNN + action-linear count calibration",
                "post-hoc subject-wise LOSO over existing raw6 predictions",
                "action recognition not integrated; assumes action context",
                f"Exact {pct(calibration.get('exact_count_acc'))}; within-1 {pct(calibration.get('within_1_count_acc'))}",
                f"{num(calibration.get('mean_abs_count_error'))} reps/set",
                "same raw boundaries; calibration does not change Rep F1",
                "same raw phase labels; does not improve C/E boundaries",
                "tiny display layer; count-only correction",
                "dynamic_optional_count_display",
            )
        )

    active_path = rel(sources["active_detector_thr06"])
    if active_path.exists():
        active = load_json(active_path)["overall"]
        rows.append(
            row(
                "Current ours",
                "Rest-aware active detector probe, threshold 0.6",
                "held-out yushuan, 4 set+20s-rest snippets",
                "active/rest detection only",
                "N/A",
                "N/A",
                f"Active P/R/F1 {pct(active.get('precision'))}/{pct(active.get('recall'))}/{pct(active.get('f1'))}",
                f"false-active rest {num(active.get('false_active_rest_sec'), 's')}/20s; missed active {num(active.get('missed_active_sec'), 's')}",
                "1.0s RF window, 0.1s stride",
                "dynamic_probe_not_full_9fold",
            )
        )

    return rows


def load_same_dataset_baseline_rows(registry: Dict[str, Any]) -> List[Dict[str, str]]:
    sources = [
        registry.get("same_dataset_baselines_source"),
        registry.get("same_dataset_deep_baselines_source"),
    ]
    sources = [str(source) for source in sources if source]
    if not sources:
        return []
    descriptions = {
        "peak_acc": {
            "method": "Peak detection, accel magnitude",
            "protocol": "9-fold LOSO, same 220 streams",
            "recognition": "N/A; assumes stream/action context",
            "real_time": "simple causal-friendly signal processing",
            "status": "fixed_same_dataset_baseline",
            "seg_note": "Rep P/R/F1 {rep_prf}; Phase IoU-F1@50 N/A (no C/E prediction)",
            "ce_note": "C/E metrics N/A: peak does not predict C/E transition",
        },
        "peak_6axis": {
            "method": "Peak detection, 6-axis magnitude",
            "protocol": "9-fold LOSO, same 220 streams",
            "recognition": "N/A; assumes stream/action context",
            "real_time": "simple causal-friendly signal processing",
            "status": "fixed_same_dataset_baseline",
            "seg_note": "Rep P/R/F1 {rep_prf}; Phase IoU-F1@50 N/A (no C/E prediction)",
            "ce_note": "C/E metrics N/A: peak does not predict C/E transition",
        },
        "rf_phase": {
            "method": "Per-action RF active detector + RF C/E phase classifier",
            "protocol": "9-fold LOSO, same 220 streams",
            "recognition": "N/A; assumes stream/action context",
            "real_time": "1.0s windows, 0.1s stride",
            "status": "fixed_same_dataset_baseline",
        },
        "tcn_lite": {
            "method": "Causal TCN-lite phase segmenter + shared active detector",
            "protocol": "9-fold LOSO, same 220 streams, 20 epochs",
            "recognition": "N/A; assumes stream/action context",
            "real_time": "causal conv sequence model; MA25+Viterbi decoding",
            "status": "fixed_same_dataset_deep_baseline",
        },
        "bilstm": {
            "method": "BiLSTM phase segmenter + shared active detector",
            "protocol": "9-fold LOSO, same 220 streams, 20 epochs",
            "recognition": "N/A; assumes stream/action context",
            "real_time": "non-causal bidirectional sequence baseline",
            "status": "fixed_same_dataset_deep_baseline",
        },
    }
    rows: List[Dict[str, str]] = []
    for source in sources:
        source_path = rel(source)
        if not source_path.exists():
            continue
        data = load_json(source_path)
        for name, payload in data.get("models", {}).items():
            overall = payload.get("overall", {})
            desc = descriptions.get(
                name,
                {
                    "method": name,
                    "protocol": "9-fold LOSO, same streams",
                    "recognition": "N/A",
                    "real_time": "NR",
                    "status": "fixed_same_dataset_baseline",
                },
            )
            rows.append(
                row_phase_or_ce := row(
                    "Same-dataset baseline",
                    desc["method"],
                    desc["protocol"],
                    desc["recognition"],
                    f"Exact {pct(overall.get('exact_count_acc'))}; within-1 {pct(overall.get('within_1_count_acc'))}",
                    f"{num(overall.get('mean_abs_count_error'))} reps/set",
                    (
                        desc.get("seg_note", "").format(
                            rep_prf=f"{pct(overall.get('rep_precision'))}/{pct(overall.get('rep_recall'))}/{pct(overall.get('rep_f1'))}"
                        )
                        if desc.get("seg_note")
                        else f"Rep P/R/F1 {pct(overall.get('rep_precision'))}/{pct(overall.get('rep_recall'))}/{pct(overall.get('rep_f1'))}; Phase IoU-F1@50 {pct(overall.get('phase_seg_iou_f1_50_avg'))}"
                    ),
                    desc.get("ce_note") or f"Phase acc {pct(overall.get('phase_accuracy'))}; Phase macro F1 {pct(overall.get('phase_macro_f1'))}; C/E MAE {num(overall.get('ce_ratio_mae'))}",
                    desc["real_time"],
                    desc["status"],
                )
            )
    return rows


def escape_cell(text: str) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def render_markdown(rows: Iterable[Dict[str, str]], registry: Dict[str, Any]) -> str:
    columns = [
        "group",
        "method",
        "protocol",
        "recognition",
        "count_exact_or_within1",
        "count_mae",
        "segmentation",
        "phase_or_ce",
        "real_time",
        "status",
    ]
    headers = [
        "Group",
        "Method",
        "Protocol",
        "Recognition",
        "Exact / Within-1",
        "Count MAE",
        "Segmentation",
        "Phase / C-E",
        "Real-Time",
        "Status",
    ]
    lines = [
        "# Fixed Baseline Comparison Table",
        "",
        f"Registry date: {registry.get('registry_date', 'unknown')}",
        "",
        "Dynamic rows are refreshed from current project artifacts. Fixed literature and legacy baseline rows should not change unless a new locked protocol is created.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for item in rows:
        lines.append("| " + " | ".join(escape_cell(item.get(col, "")) for col in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fixed-baseline comparison table.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = load_json(args.registry)
    rows = load_current_rows(registry) + load_same_dataset_baseline_rows(registry) + registry["fixed_rows"]
    output = render_markdown(rows, registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
