from __future__ import annotations

import argparse
import csv
from pathlib import Path


VALID_PHASES = {"concentric", "eccentric"}


def _first_valid_block(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    start = None
    end = None

    for idx, row in enumerate(rows):
        if row.get("phase") in VALID_PHASES:
            if start is None:
                start = idx
            end = idx
        elif start is not None:
            break

    if start is None or end is None:
        return []
    return rows[start : end + 1]


def clean_rep0_files(data_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
    trimmed = 0
    deleted = 0
    unchanged = 0

    for csv_path in sorted(data_dir.rglob("rep0_*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)

        if not fieldnames:
            unchanged += 1
            continue

        kept_rows = _first_valid_block(rows)

        if not kept_rows:
            deleted += 1
            print(f"DELETE {csv_path}")
            if not dry_run:
                csv_path.unlink()
            continue

        if len(kept_rows) == len(rows):
            unchanged += 1
            continue

        trimmed += 1
        print(f"TRIM {csv_path} rows {len(rows)} -> {len(kept_rows)}")
        if dry_run:
            continue

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    return trimmed, deleted, unchanged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean rep0 CSVs by keeping only the first contiguous concentric/eccentric block."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/raw_data"),
        help="Root raw-data directory to scan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without modifying files.",
    )
    args = parser.parse_args()

    trimmed, deleted, unchanged = clean_rep0_files(args.data_dir, dry_run=args.dry_run)
    print(
        f"Summary: trimmed={trimmed} deleted={deleted} unchanged={unchanged} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
