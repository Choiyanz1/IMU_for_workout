"""Run SDTW LOSO evaluation on 7 subjects (excluding tsenyu, ziho).

This temporarily renames excluded subject directories so that
evaluation/rep_segmentation.py only sees the 7 desired subjects.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


EXCLUDED = {"tsenyu", "ziho"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="artifacts/baseline_comparison/sdtw_7subjects_cleaned")
    parser.add_argument("--mode", default="sets", choices=["sets", "whole"])
    args = parser.parse_args()

    # Load config to find data dir
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg.get("data", {}).get("data_dir", "./datasets/raw_data"))

    # Verify excluded dirs exist
    excluded_dirs = []
    for name in EXCLUDED:
        p = data_dir / name
        if p.exists():
            excluded_dirs.append(p)

    # Temporarily rename excluded dirs
    renamed = []
    try:
        for p in excluded_dirs:
            new_name = p.parent / f"_{p.name}_temp"
            p.rename(new_name)
            renamed.append((new_name, p))
            print(f"[INFO] Temporarily excluded: {p.name} -> {new_name.name}")

        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "evaluation.rep_segmentation",
            "--config", args.config,
            "--mode", args.mode,
            "--out-dir", str(output),
            "--no-plots",
            "--no-timestamp",
        ]
        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    finally:
        # Restore excluded dirs
        for new_path, original_path in renamed:
            new_path.rename(original_path)
            print(f"[INFO] Restored: {original_path.name}")

    print(f"[OK] Results in {args.output}")


if __name__ == "__main__":
    main()
