"""Batch streaming eval for all test-subject sets, then run rep-complete classifier comparison."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_ACTIONS = {"db_bench_press", "db_rdl", "one_arm_db_row"}
EXCLUDE_PATTERNS = ["*whole_session*", "*_w", "*rest_after*"]


def _matches_exclude(name: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_PATTERNS)


def find_test_set_dirs(data_dir: Path, subject: str) -> list[Path]:
    subject_dir = data_dir / subject
    dirs = []
    for session_dir in sorted(subject_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        for action_dir in sorted(session_dir.iterdir()):
            if not action_dir.is_dir() or action_dir.name not in INCLUDE_ACTIONS:
                continue
            for set_dir in sorted(action_dir.iterdir()):
                if not set_dir.is_dir() or not set_dir.name.startswith("set"):
                    continue
                if _matches_exclude(set_dir.name):
                    continue
                dirs.append(set_dir)
    return dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Trained TCN run dir")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-subject", required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "raw_data")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    set_dirs = find_test_set_dirs(args.data_dir, args.test_subject)
    print(f"[INFO] Found {len(set_dirs)} test set dirs for subject={args.test_subject}")

    streaming_output_dirs = []
    for i, set_dir in enumerate(set_dirs, 1):
        rel = set_dir.relative_to(args.data_dir)
        stream_id = str(rel).replace("\\", "/")
        output_dir = args.run_dir / "streaming_eval" / stream_id
        print(f"\n[{i}/{len(set_dirs)}] streaming eval: {stream_id}")

        cmd = [
            sys.executable, "-m", "evaluation.streaming_micro_macro",
            "--run-dir", str(args.run_dir),
            "--csv", str(set_dir),
            "--config", str(args.config),
            "--stream-id", stream_id,
            "--output-dir", str(output_dir),
            "--method", "fast",
        ]
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] failed: {result.stderr[-500:]}")
            continue
        streaming_output_dirs.append(str(output_dir))
        print(f"  [OK] -> {output_dir}")

    print(f"\n[INFO] Completed {len(streaming_output_dirs)}/{len(set_dirs)} streaming evals")

    # Run rep-complete action classifier comparison
    if not streaming_output_dirs:
        print("[ERROR] No streaming eval outputs, skipping comparison")
        return

    output_json = args.output_json or (args.run_dir / "rep_action_compare.json")
    compare_cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate_rep_complete_action_classifier.py"),
        "--config", str(args.config),
        "--test-subject", args.test_subject,
        "--output-json", str(output_json),
    ]
    for d in streaming_output_dirs:
        compare_cmd.extend(["--rep-eval-dir", d])

    print(f"\n[INFO] Running rep-complete classifier comparison...")
    result = subprocess.run(compare_cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Comparison failed:\n{result.stderr[-1000:]}")
        print(f"stdout:\n{result.stdout[-1000:]}")
    else:
        print(result.stdout)
        print(f"[OK] Results written to {output_json}")


if __name__ == "__main__":
    main()
