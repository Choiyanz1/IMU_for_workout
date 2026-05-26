from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "datasets" / "raw_data"
TARGET_ACTIONS = ["db_rdl", "db_weighted_crunch", "db_bench_press", "one_arm_db_row"]


def active_runs(phases: list[str]) -> list[str]:
    out: list[str] = []
    prev = None
    for phase in phases:
        if phase not in {"concentric", "eccentric"}:
            continue
        if phase != prev:
            out.append(phase)
            prev = phase
    return out


def main() -> None:
    rows = []
    for subject_dir in sorted(DATA_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        for action in TARGET_ACTIONS:
            action_dir = subject_dir / action
            if not action_dir.exists():
                continue
            for csv_path in sorted(action_dir.glob("set*/*.csv")):
                df = pd.read_csv(csv_path)
                if "phase" not in df.columns:
                    continue
                runs = active_runs(df["phase"].astype(str).tolist())
                first = runs[0] if runs else "none"
                second = runs[1] if len(runs) > 1 else "missing"
                pair = f"{first}->{second}"
                rows.append(
                    {
                        "subject": subject_dir.name,
                        "action": action,
                        "path": csv_path.as_posix(),
                        "first_active": first,
                        "second_active": second,
                        "pair": pair,
                        "n_active_runs": len(runs),
                    }
                )
    df = pd.DataFrame(rows)
    print("## Phase pair counts by action")
    print(df.groupby(["action", "pair"]).size().to_string())
    print("\n## Kevin only")
    print(df[df["subject"] == "kevin"].groupby(["action", "pair"]).size().to_string())
    print("\n## Reps with >2 active runs by action")
    print(df.groupby("action")["n_active_runs"].apply(lambda s: int((s > 2).sum())).to_string())


if __name__ == "__main__":
    main()
