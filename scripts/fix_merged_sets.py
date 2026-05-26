import shutil
from pathlib import Path

pairs = [
    (Path("datasets/raw_data/thomas/thomas_2/db_bench_press/set0"), Path("datasets/raw_data/thomas/thomas_2/db_bench_press")),
    (Path("datasets/raw_data/yanz/1000/db_shoulder_press/set0"), Path("datasets/raw_data/yanz/1000/db_shoulder_press")),
]

for src_dir, parent_dir in pairs:
    src = Path(src_dir)
    rep_files = sorted(src.glob("rep*.csv"))
    timestamps = [int(f.stem.split("_")[1]) for f in rep_files]
    mid = (max(timestamps[:12]) + min(timestamps[12:])) // 2
    set0_files = [f for f, t in zip(rep_files, timestamps) if t < mid]
    set1_files = [f for f, t in zip(rep_files, timestamps) if t >= mid]
    set0_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    set1_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    set0_dir = parent_dir / "set0"
    set1_dir = parent_dir / "set1"
    set1_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(set0_files):
        if f.parent == set0_dir:
            continue
        dst = set0_dir / f.name
        shutil.move(str(f), str(dst))
        print(f"  kept {f.name} -> set0/")
    for i, f in enumerate(set1_files):
        dst = set1_dir / f.name
        shutil.move(str(f), str(dst))
        print(f"  moved {f.name} -> set1/")
    leftover = list(src.glob("*"))
    if leftover and src != set0_dir:
        for lf in leftover:
            (set0_dir / lf.name).write_bytes(lf.read_bytes()) if lf.is_file() else None
        print(f"  cleaned up {src}")
    print(f"[OK] {src} split into set0 ({len(set0_files)} reps) + set1 ({len(set1_files)} reps)\n")
