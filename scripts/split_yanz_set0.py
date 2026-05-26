from pathlib import Path
import re, shutil

base = Path("datasets/raw_data/yanz/1000/db_shoulder_press/set0")
target = Path("datasets/raw_data/yanz/1000/db_shoulder_press/set1")

def rep_num(f):
    m = re.search(r"rep(\d+)", f.stem)
    return int(m.group(1)) if m else 9999

files = sorted(base.glob("*.csv"), key=rep_num)
for f in files:
    n = rep_num(f)
    if n >= 12:
        target.mkdir(parents=True, exist_ok=True)
        dst = target / f.name
        shutil.move(str(f), str(dst))
        print(f"  {f.name} -> set1/")

for s in ["set0", "set1"]:
    d = base.parent / s
    print(f"  {s}: {len(list(d.glob('*.csv')))} files")
