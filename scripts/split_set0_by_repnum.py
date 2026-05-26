from pathlib import Path
import re, shutil

# Split thomas_2/db_bench_press/set0 (24 reps) into set0 (rep0-11) + set3 (rep12-23)
base = Path("datasets/raw_data/thomas_2/db_bench_press/set0")
target = Path("datasets/raw_data/thomas_2/db_bench_press/set3")

def rep_num(f):
    m = re.search(r"rep(\d+)", f.stem)
    return int(m.group(1)) if m else 9999

files = sorted(base.glob("*.csv"), key=rep_num)
print(f"set0: {len(files)} files")
for f in files:
    n = rep_num(f)
    if n >= 12:
        target.mkdir(parents=True, exist_ok=True)
        dst = target / f.name
        shutil.move(str(f), str(dst))
        print(f"  moved {f.name} -> set3/")

# Verify
for s in ["set0", "set1", "set2", "set3"]:
    d = base.parent / s
    if d.exists():
        print(f"  {s}: {len(list(d.glob('*.csv')))} files")
