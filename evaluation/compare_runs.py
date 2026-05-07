from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.reporting import primary_metric_table


def _flatten_summary(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    overall = payload.get("overall") or payload.get("test_metrics") or {}
    if "test_metrics" in payload and not payload.get("overall"):
        overall = payload["test_metrics"]
    primary = payload.get("primary_metrics") or primary_metric_table(overall)
    run_dir = path.parent.parent if path.parent.name == "metrics" else path.parent
    manifest_path = run_dir / "metadata" / "run_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    row: Dict[str, Any] = {
        "run_dir": run_dir.as_posix(),
        "summary_path": path.as_posix(),
        "task": payload.get("task") or manifest.get("task"),
        "model_name": payload.get("model_name") or manifest.get("model_name"),
    }
    for key, value in primary.items():
        row[key] = value
    for key in (
        "micro_source",
        "mode",
        "configured_micro_source",
        "resolved_micro_source",
        "test_subject",
        "iou_threshold",
    ):
        if key in payload:
            row[key] = payload[key]
        elif key in manifest:
            row[key] = manifest[key]
    return row


def collect_runs(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("summary.json")):
        if path.parent.name != "metrics":
            standardized = path.parent / "metrics" / "summary.json"
            if standardized.exists():
                continue
        try:
            rows.append(_flatten_summary(path))
        except Exception as exc:
            rows.append({"summary_path": path.as_posix(), "error": str(exc)})
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md_table(df: pd.DataFrame, columns: List[str]) -> str:
    present = [col for col in columns if col in df.columns]
    if not present or df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(present) + " |", "| " + " | ".join(["---"] * len(present)) + " |"]
    for _, row in df[present].iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in present) + " |")
    return "\n".join(lines)


def write_comparison_report(
    df: pd.DataFrame,
    output: Path,
    title: str = "Model Comparison Report",
    csv_path: Path | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_cols = ["task", "model_name", "accuracy", "macro_f1", "precision", "recall", "f1", "iou_f1_50"]
    link_cols = metric_cols + ["run_dir"]
    sort_cols = [col for col in ("task", "model_name") if col in df.columns]
    display = df.sort_values(sort_cols) if sort_cols else df

    sections = [
        f"# {title}",
        "",
        "## Summary",
        "",
        f"- Runs compared: {len(df)}",
        f"- CSV: `{(csv_path or output.with_suffix('.csv')).as_posix()}`",
        "",
        "## Key Metrics",
        "",
        _md_table(display, metric_cols),
        "",
        "## Run Links",
        "",
        _md_table(display, link_cols),
        "",
    ]

    if "task" in display.columns:
        for task, group in display.groupby("task", dropna=False):
            sections.extend(
                [
                    f"## {task}",
                    "",
                    _md_table(group, link_cols),
                    "",
                ]
            )
    output.write_text("\n".join(sections), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect standardized run summaries into one comparison table")
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run_comparison.csv"))
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_runs(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    report_path = args.report or args.output.with_suffix(".md")
    write_comparison_report(df, report_path, csv_path=args.output)
    print(f"[OK] Wrote {len(df)} runs to {args.output}")
    print(f"[OK] Wrote comparison report to {report_path}")
    if not df.empty:
        cols = [c for c in ("task", "model_name", "accuracy", "macro_f1", "precision", "recall", "f1", "iou_f1_50", "run_dir") if c in df.columns]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
