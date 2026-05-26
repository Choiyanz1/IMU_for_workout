from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


STANDARD_DIRS = ("metrics", "detections", "models", "plots", "metadata")


def ensure_standard_dirs(run_dir: Path, names: Sequence[str] = STANDARD_DIRS) -> Dict[str, Path]:
    dirs = {"root": Path(run_dir)}
    dirs["root"].mkdir(parents=True, exist_ok=True)
    for name in names:
        dirs[name] = dirs["root"] / name
        dirs[name].mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def copy_config_snapshot(config_path: Path | None, metadata_dir: Path) -> str | None:
    if config_path is None:
        return None
    try:
        target = metadata_dir / "config_snapshot.yaml"
        shutil.copy2(config_path, target)
        return target.as_posix()
    except Exception:
        return None


def _metric_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def _first_existing(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def primary_metric_table(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common metric names across tasks for easier comparison."""
    return {
        "start_mae_ms": _first_existing(metrics, ("start_mae_ms",)),
        "end_mae_ms": _first_existing(metrics, ("end_mae_ms",)),
        "transition_mae_ms": _first_existing(metrics, ("transition_mae_ms",)),
        "precision": _first_existing(metrics, ("precision",)),
        "recall": _first_existing(metrics, ("recall",)),
        "f1": _first_existing(metrics, ("f1",)),
        "n_pred": _first_existing(metrics, ("n_pred",)),
        "n_true": _first_existing(metrics, ("n_true",)),
        "tp": _first_existing(metrics, ("tp",)),
        "fp": _first_existing(metrics, ("fp",)),
        "fn": _first_existing(metrics, ("fn",)),
        "iou_f1_50": _first_existing(metrics, ("macro_f1_at_50", "micro_f1_at_50", "f1_at_50")),
        "accuracy": _first_existing(metrics, ("accuracy", "test_accuracy")),
        "macro_f1": _first_existing(metrics, ("macro_f1", "macro_sample_macro_f1")),
    }


def write_run_manifest(
    run_dir: Path,
    *,
    task: str,
    model_name: str,
    config_path: Path | None = None,
    command: str | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    dirs = ensure_standard_dirs(run_dir)
    manifest: Dict[str, Any] = {
        "task": task,
        "model_name": model_name,
        "run_dir": dirs["root"].as_posix(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path.as_posix() if config_path else None,
        "command": command,
    }
    if extras:
        manifest.update(dict(extras))
    write_json(dirs["metadata"] / "run_manifest.json", manifest)
    if config_path:
        copy_config_snapshot(config_path, dirs["metadata"])
    return manifest


def write_standard_summary(
    run_dir: Path,
    *,
    task: str,
    model_name: str,
    overall: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    dirs = ensure_standard_dirs(run_dir)
    payload: Dict[str, Any] = {
        "task": task,
        "model_name": model_name,
        "overall": dict(overall),
        "primary_metrics": primary_metric_table(overall),
    }
    if details:
        payload.update(dict(details))
    write_json(dirs["metrics"] / "summary.json", payload)
    return payload


def write_report(
    run_dir: Path,
    *,
    title: str,
    task: str,
    model_name: str,
    overall: Mapping[str, Any],
    artifacts: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> Path:
    dirs = ensure_standard_dirs(run_dir)
    primary = primary_metric_table(overall)
    metric_lines = ["| Metric | Value |", "|---|---:|"]
    for key, value in primary.items():
        if value is not None:
            metric_lines.append(f"| `{key}` | {_metric_value(value)} |")
    for key, value in overall.items():
        if key not in primary and isinstance(value, (int, float, str)):
            metric_lines.append(f"| `{key}` | {_metric_value(value)} |")

    config_lines = ["| Setting | Value |", "|---|---|", f"| Task | `{task}` |", f"| Model | `{model_name}` |"]
    for key, value in (config or {}).items():
        config_lines.append(f"| `{key}` | `{value}` |")

    artifact_lines = ["| Artifact | Path |", "|---|---|"]
    for key, value in (artifacts or {}).items():
        artifact_lines.append(f"| {key} | `{value}` |")

    note_lines = [f"- {note}" for note in (notes or [])]
    text = "\n".join(
        [
            f"# {title}",
            "",
            "## Key Metrics",
            "",
            "\n".join(metric_lines),
            "",
            "## Run",
            "",
            "\n".join(config_lines),
            "",
            "## Artifacts",
            "",
            "\n".join(artifact_lines),
            "",
            "## Notes",
            "",
            "\n".join(note_lines) if note_lines else "- None.",
            "",
        ]
    )
    path = dirs["root"] / "report.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_standard_run_outputs(
    run_dir: Path,
    *,
    task: str,
    model_name: str,
    title: str,
    overall: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
    manifest_extras: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> Dict[str, Any]:
    write_run_manifest(run_dir, task=task, model_name=model_name, config_path=config_path, extras=manifest_extras)
    summary = write_standard_summary(run_dir, task=task, model_name=model_name, overall=overall, details=details)
    write_report(
        run_dir,
        title=title,
        task=task,
        model_name=model_name,
        overall=overall,
        artifacts=artifacts,
        config=config,
        notes=notes,
    )
    return summary
