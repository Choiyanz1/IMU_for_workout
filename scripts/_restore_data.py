from pathlib import Path
import subprocess

git_root = Path("D:/ResistanceTraining_IMU/IMU_for_workout")
raw = subprocess.check_output(["git", "ls-files", "datasets/raw_data/"], cwd=str(git_root))
files = raw.decode().strip().splitlines()
print("Total tracked files:", len(files))
count = 0
for f in files:
    fp = git_root / f
    if fp.exists():
        continue
    fp.parent.mkdir(parents=True, exist_ok=True)
    content = subprocess.check_output(["git", "show", "HEAD:" + f], cwd=str(git_root))
    fp.write_bytes(content)
    count += 1
print("Restored:", count, "files")
