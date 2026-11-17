
from pathlib import Path

ROOT = Path(r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE")
DATASET = ROOT / "DataSet"

train_dir = DATASET / "Train"
train_list = DATASET / "Train.txt"

# Include both video formats
extensions = {".avi", ".mp4"}

videos = [
    p for p in train_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

listed_names = {
    line.strip()
    for line in train_list.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
}

actual_names = {p.name for p in videos}

print("=" * 55)
print("SYNAPSENSE - TRAINING DATA VERIFICATION")
print("=" * 55)

print("Train.txt entries:", len(listed_names))
print("Actual video files:", len(videos))
print("AVI files:", sum(p.suffix.lower() == ".avi" for p in videos))
print("MP4 files:", sum(p.suffix.lower() == ".mp4" for p in videos))

print("Listed videos found:", len(listed_names & actual_names))
print("Listed videos missing:", len(listed_names - actual_names))
print("Unlisted videos found:", len(actual_names - listed_names))

print("\nSample missing filenames:")
for name in sorted(listed_names - actual_names)[:15]:
    print(name)

print("=" * 55)
