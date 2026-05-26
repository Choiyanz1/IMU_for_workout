from pathlib import Path
import re, shutil

base = Path("datasets/raw_data/yanz/1000/db_shoulder_press")

# 1. find all files and their rep numbers
entries = []
for d in base.iterdir():
    if d.name.startswith("set"):
        for f in d.glob("*.csv"):
            m = re.search(r"rep(\d+)", f.stem)
            n = int(m.group(1)) if m else 0
            entries.append((n, f))

entries.sort(key=lambda x: x[0])
print(f"Total files: {len(entries)}")

# 2. copy everything to _tmp first
tmp = base / "_tmp"
tmp.mkdir(exist_ok=True)
for n, src in entries:
    shutil.copy2(str(src), str(tmp / src.name))

# 3. delete all old set dirs
for d in list(base.iterdir()):
    if d.name.startswith("set"):
        shutil.rmtree(d)

# 4. recreate sets from _tmp
for i in range(0, len(entries), 12):
    sdir = base / f"set{i//12}"
    sdir.mkdir()
    for j in range(i, min(i + 12, len(entries))):
        fname = entries[j][1].name
        shutil.move(str(tmp / fname), str(sdir / fname))
    print(f"  set{i//12}: {min(12, len(entries)-i)} files")

# 5. cleanup tmp
for leftover in tmp.iterdir():
    leftover.unlink()
tmp.rmdir()
print("Done")
