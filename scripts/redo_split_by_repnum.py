"""Redistribute rep files into proper sets sorted by rep number."""
from pathlib import Path
import re, shutil

for base_path in [
    "datasets/raw_data/thomas/thomas_2/db_bench_press",
    "datasets/raw_data/yanz/1000/db_shoulder_press",
]:
    base = Path(base_path)
    print(f"=== {base} ===")

    # 1. copy all csv files to a temp dir
    tmp = base / "_tmp"
    tmp.mkdir(exist_ok=True)
    entries = []
    for d in base.iterdir():
        if d.name.startswith("set"):
            for f in sorted(d.glob("*.csv")):
                m = re.search(r"rep(\d+)", f.stem)
                n = int(m.group(1)) if m else 0
                dst = tmp / f.name
                shutil.copy2(str(f), str(dst))
                entries.append((n, dst))
    entries.sort(key=lambda x: x[0])
    print(f"  total: {len(entries)} reps")

    # 2. delete old set dirs
    for d in list(base.iterdir()):
        if d.name.startswith("set"):
            shutil.rmtree(d)

    # 3. create new set dirs, each with 12 reps
    for i in range(0, len(entries), 12):
        chunk = entries[i : i + 12]
        sdir = base / f"set{i // 12}"
        sdir.mkdir()
        for _, src in chunk:
            src.rename(sdir / src.name)
        print(f"  set{i // 12}: {len(chunk)} reps")

    # 4. remove tmp if empty
    try:
        tmp.rmdir()
    except Exception:
        pass
