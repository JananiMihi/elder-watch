import mediapipe as mp
from pathlib import Path

print("mp.__file__ =", getattr(mp, "__file__", None))
print("mp.__version__ =", getattr(mp, "__version__", None))
print("has solutions =", hasattr(mp, "solutions"))
print("dir head =", dir(mp)[:30])

mp_dir = Path(mp.__file__).resolve().parent
task_files = list(mp_dir.rglob("*.task"))
print("task_files_count =", len(task_files))
for p in task_files[:30]:
    print("task_file =", str(p))

