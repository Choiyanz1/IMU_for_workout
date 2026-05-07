from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


MODEL_ALIASES = {
    "all": ["ds_ms_tcn_tcn", "ds_ms_tcn_dtw", "sdtw", "hybrid"],
    "ds_ms_tcn": ["ds_ms_tcn_tcn", "ds_ms_tcn_dtw"],
    "micro_macro": ["ds_ms_tcn_tcn", "ds_ms_tcn_dtw"],
}


def _expand_models(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        for part in parts:
            expanded = MODEL_ALIASES.get(part, [part])
            for item in expanded:
                if item not in out:
                    out.append(item)
    return out


def _command_for_model(args: argparse.Namespace, model: str, run_dir: Path) -> List[str]:
    py = sys.executable
    common_mode = ["--mode", args.mode]
    if model == "ds_ms_tcn_tcn":
        return [
            py,
            "-m",
            "train.micro_macro_recognition",
            "--config",
            str(args.micro_macro_config),
            "--micro-source",
            "tcn",
            *common_mode,
            "--output-dir",
            str(run_dir),
            "--run-stamp",
            "ds_ms_tcn",
        ]
    if model == "ds_ms_tcn_dtw":
        return [
            py,
            "-m",
            "train.micro_macro_recognition",
            "--config",
            str(args.micro_macro_config),
            "--micro-source",
            "dtw",
            *common_mode,
            "--output-dir",
            str(run_dir),
            "--run-stamp",
            "ds_ms_tcn",
        ]
    if model == "sdtw":
        cmd = [
            py,
            "-m",
            "evaluation.rep_segmentation",
            "--config",
            str(args.rep_config),
            *common_mode,
            "--out-dir",
            str(run_dir / "sdtw"),
            "--no-timestamp",
            "--iou-threshold",
            str(args.iou_threshold),
        ]
        if args.no_plots:
            cmd.append("--no-plots")
        if args.max_plots is not None:
            cmd.extend(["--max-plots", str(args.max_plots)])
        return cmd
    if model == "hybrid":
        cmd = [
            py,
            "-m",
            "train.hybrid_rep_segmentation",
            "--config",
            str(args.hybrid_config),
            *common_mode,
            "--out-dir",
            str(run_dir / "hybrid"),
            "--no-timestamp",
            "--iou-threshold",
            str(args.iou_threshold),
        ]
        if args.no_plots:
            cmd.append("--no-plots")
        if args.max_plots is not None:
            cmd.extend(["--max-plots", str(args.max_plots)])
        return cmd
    raise ValueError(f"Unknown model: {model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparable workout recognition models into one suite folder")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ds_ms_tcn"],
        help="Models to run: ds_ms_tcn_tcn, ds_ms_tcn_dtw, ds_ms_tcn, sdtw, hybrid, all",
    )
    parser.add_argument("--mode", choices=["sets", "whole", "both"], default="sets")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/model_suites"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--micro-macro-config", type=Path, default=Path("configs/micro_macro_recognition.yaml"))
    parser.add_argument("--rep-config", type=Path, default=Path("configs/rep_segmentation.yaml"))
    parser.add_argument("--hybrid-config", type=Path, default=Path("configs/hybrid_rep_segmentation.yaml"))
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-plots", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = _expand_models(args.models)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "models": models,
        "mode": args.mode,
        "commands": {},
        "results": {},
    }
    print(f"[INFO] suite output={run_dir}")
    for model in models:
        cmd = _command_for_model(args, model, run_dir)
        manifest["commands"][model] = cmd
        print("\n[RUN]", model)
        print(" ".join(cmd))
        if args.dry_run:
            manifest["results"][model] = {"status": "dry_run"}
            continue
        proc = subprocess.run(cmd, cwd=Path.cwd())
        manifest["results"][model] = {"status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode}
        if proc.returncode != 0 and not args.continue_on_error:
            break

    if args.dry_run:
        comparison_path = run_dir / "comparison.csv"
        manifest["comparison_csv"] = None
        (run_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n[DRY RUN] suite_manifest={run_dir / 'suite_manifest.json'}")
        return

    from evaluation.compare_runs import collect_runs, write_comparison_report

    comparison = collect_runs(run_dir)
    comparison_path = run_dir / "comparison.csv"
    comparison_report_path = run_dir / "comparison.md"
    comparison.to_csv(comparison_path, index=False)
    write_comparison_report(
        comparison,
        comparison_report_path,
        title=f"Model Suite Comparison ({run_id})",
        csv_path=comparison_path,
    )
    manifest["comparison_csv"] = comparison_path.as_posix()
    manifest["comparison_report_md"] = comparison_report_path.as_posix()
    (run_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[OK] comparison={comparison_path}")
    print(f"[OK] comparison_report={comparison_report_path}")
    if not comparison.empty:
        cols = [c for c in ("task", "model_name", "accuracy", "macro_f1", "precision", "recall", "f1", "iou_f1_50", "run_dir") if c in comparison.columns]
        print(comparison[cols].to_string(index=False))


if __name__ == "__main__":
    main()
